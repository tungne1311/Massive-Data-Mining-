"""
Bước 1 — Data Ingestion (Producer-Consumer & Micro-batch)
Sử dụng luồng chạy ngầm (Background Thread) và ThreadPool để kéo dữ liệu từ HuggingFace
song song với việc Spark ghi xuống MinIO. Không ghi file raw rác.
"""

import logging
import queue
import threading
import concurrent.futures
from datetime import datetime
from typing import Iterator, Optional

from datasets import load_dataset
from pyspark.sql.types import (
    BooleanType, FloatType, IntegerType, LongType,
    StringType, StructField, StructType,
)

logger = logging.getLogger(__name__)

SOURCE_NAME = "amazon_reviews_2023_electronics"
INGEST_DATE = datetime.now().strftime("%Y-%m-%d")

# ── 1. SCHEMA (LƯU Ý QUAN TRỌNG KHI GIỮ NGUYÊN DATA GỐC) ──────────────────────
# CẢNH BÁO: Vì bạn muốn giữ lại TOÀN BỘ dữ liệu gốc từ HuggingFace, 
# nếu bạn ép Spark dùng Schema cứng này lúc tạo DataFrame, Spark sẽ DROP các cột 
# không có tên trong đây.
# => KHUYẾN NGHỊ: Ở bước tạo DataFrame, hãy để Spark tự inferSchema (không truyền biến schema)
# hoặc bạn phải tự động gen schema dựa trên data thực tế. Ở đây mình giữ lại để tham khảo.

REVIEW_SCHEMA = StructType([
    StructField("reviewer_id",       StringType(),  False),
    StructField("parent_asin",       StringType(),  False),
    StructField("rating",            FloatType(),   False),
    StructField("review_text",       StringType(),  True),
    StructField("text_len",          IntegerType(), True),
    StructField("timestamp",         LongType(),    True),
    StructField("helpful_vote",      IntegerType(), True),
    StructField("verified_purchase", BooleanType(), True),
    StructField("ingest_date",       StringType(),  True),
    StructField("source_name",       StringType(),  True),
    StructField("year_month",        StringType(),  False), # <-- Partition key mới thêm
])

METADATA_SCHEMA = StructType([
    StructField("parent_asin",   StringType(), False),
    StructField("item_title",    StringType(), True),
    StructField("brand",         StringType(), True),
    StructField("main_category", StringType(), True),
    StructField("features",      StringType(), True),
    StructField("price_bucket",  StringType(), True),
    StructField("ingest_date",   StringType(), True),
    StructField("source_name",   StringType(), True),
    
])

# ── 2. NORMALIZERS (CHỈ THÊM YEAR_MONTH, GIỮ NGUYÊN GỐC) ──────────────────────

def normalize_review(raw: dict) -> Optional[dict]:
    """
    Giữ nguyên toàn bộ dữ liệu gốc của Review.
    Đồng bộ khóa chính và trích xuất timestamp để sinh ra cột phân mảnh year_month.
    """
    # 1. Đồng bộ các key định danh (để Spark không bị thiếu khóa Join sau này)
    raw["reviewer_id"] = raw.get("user_id") or raw.get("reviewer_id") or raw.get("reviewerID")
    raw["parent_asin"] = raw.get("parent_asin") or raw.get("asin")
    
    # Bỏ qua ngay lập tức các dòng rác không có thông tin định danh
    if not raw["reviewer_id"] or not raw["parent_asin"]:
        return None

    # 2. Xử lý thời gian và sinh year_month
    ts_raw = raw.get("timestamp") or raw.get("unixReviewTime") or 0
    try:
        ts = int(ts_raw)
        # Ép về định dạng giây (seconds) nếu đang ở dạng mili-giây (milliseconds)
        ts = ts // 1000 if ts > 1_000_000_000_000 else ts
        
        # Tính toán year_month và chặn các năm quá vô lý (bảo vệ hệ thống thư mục)
        dt = datetime.fromtimestamp(ts)
        if 1995 <= dt.year <= 2030:
            year_month = dt.strftime("%Y-%m")
        else:
            year_month = "unknown_date"
            
        raw["timestamp"] = ts  # Ghi đè lại bằng timestamp chuẩn
    except Exception:
        year_month = "unknown_date"

    # 3. Gán partition key và trả về toàn bộ dữ liệu gốc
    raw["year_month"] = year_month
    return raw


def normalize_metadata(raw: dict) -> Optional[dict]:
    """
    Giữ nguyên toàn bộ dữ liệu gốc của Metadata.
    Sinh ra cột phân mảnh year_month dựa trên ngày chạy pipeline (INGEST_DATE).
    """
    if not raw.get("parent_asin") and not raw.get("asin"):
        return None
    #Xử lý cột features (gom list thành chuỗi)
    feats_raw = raw.get("features") or []
    features = " | ".join(str(f) for f in feats_raw if f) if isinstance(feats_raw, list) else str(feats_raw).strip()
    raw["features"] = features
    raw["ingest_date"] = INGEST_DATE
    raw["source_name"] = SOURCE_NAME
    return raw


# ── 3. KIẾN TRÚC PRODUCER-CONSUMER (ĐA LUỒNG TRÊN RAM) ───────────────────────

def _fetch_and_process_worker(ds, batch_size, max_records, n_threads, out_queue, normalizer_func):
    """Hàm chạy ngầm: Kéo dữ liệu từ HF và đẩy vào ThreadPool để xử lý."""
    
    def _process(rb):
        # Chuyển dữ liệu cột thành dữ liệu dòng
        keys = rb.keys()
        row_oriented = [dict(zip(keys, vals)) for vals in zip(*rb.values())]
        # Xử lý (chỉ thêm year_month)
        return [r for r in (normalizer_func(raw) for raw in row_oriented) if r is not None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = set()
        batch_idx = 0
        
        for raw_batch in ds.iter(batch_size=batch_size):
            # Nhét batch thô vào ThreadPool xử lý
            futures.add(pool.submit(_process, raw_batch))
            batch_idx += 1
            
            # Cân bằng tải: Giữ số lượng tác vụ song song bằng n_threads
            if len(futures) >= n_threads:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    # Lệnh put này sẽ BỊ CHẶN (Block) nếu out_queue đầy do Spark xử lý chậm -> Chống tràn RAM
                    out_queue.put(f.result())
                    
            if max_records and (batch_idx * batch_size >= max_records):
                break
                
        # Nhét nốt các batch còn sót lại vào hàng đợi
        for f in concurrent.futures.as_completed(futures):
            out_queue.put(f.result())
            
    # Báo hiệu cho Spark (Consumer) biết là đã hết dữ liệu
    out_queue.put(None)


def _create_iterator(cfg: dict, dataset_name: str, subset: str, max_records: Optional[int], normalizer_func) -> Iterator[list[dict]]:
    hf = cfg["huggingface"]
    batch_size = hf["stream_batch_size"]
    n_threads = hf.get("staging_threads", 4)
    
    ds = load_dataset(dataset_name, subset, split="full", streaming=True, trust_remote_code=True)
    
    # Hàng đợi giao tiếp: Giới hạn maxsize = n_threads * 2 để bảo vệ RAM Driver
    out_queue = queue.Queue(maxsize=n_threads * 2)
    
    # Khởi chạy luồng ngầm (Producer)
    t = threading.Thread(
        target=_fetch_and_process_worker,
        args=(ds, batch_size, max_records, n_threads, out_queue, normalizer_func)
    )
    t.start()
    
    total_valid = 0
    while True:
        # Luồng chính (Consumer/Spark) lấy dữ liệu. Sẽ chờ nếu chưa có data.
        batch = out_queue.get()
        if batch is None:
            break # Tín hiệu kết thúc
            
        if batch:
            total_valid += len(batch)
            logger.info(f"  [QUEUE] Spark rút {len(batch):,} rows để ghi (Tổng đã rút: {total_valid:,})")
            yield batch
            
    t.join()


def iter_review_batches(cfg: dict) -> Iterator[list[dict]]:
    hf = cfg["huggingface"]
    logger.info(f"[INGESTION] Khởi động Kéo Reviews Song Song (Producer-Consumer)")
    return _create_iterator(cfg, hf["review_dataset"], hf["review_subset"], hf.get("max_review_records"), normalize_review)

def iter_metadata_batches(cfg: dict) -> Iterator[list[dict]]:
    hf = cfg["huggingface"]
    logger.info(f"[INGESTION] Khởi động Kéo Metadata Song Song (Producer-Consumer)")
    return _create_iterator(cfg, hf["metadata_dataset"], hf["metadata_subset"], hf.get("max_metadata_records"), normalize_metadata)