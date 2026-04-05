import logging
import queue
import threading
import time
from typing import Iterator, Optional
import pyarrow as pa
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Đã xóa datetime, INGEST_DATE và SOURCE_NAME vì không còn cần thiết

# ==========================================
# 1. ĐỊNH NGHĨA SCHEMA (SIÊU TINH GỌN)
# ==========================================

REVIEW_ARROW_SCHEMA = pa.schema([
    ('reviewer_id', pa.string()),
    ('parent_asin', pa.string()),
    ('rating', pa.float32()),
    ('review_text', pa.string()),
    ('timestamp', pa.int64()),
    ('helpful_vote', pa.int32()),
    ('verified_purchase', pa.bool_())
])

METADATA_ARROW_SCHEMA = pa.schema([
    ('parent_asin', pa.string()),
    ('title', pa.string()),
    ('main_category', pa.string()),
    ('store', pa.string()),
    ('price', pa.float32()),
    ('categories', pa.string()),
    ('features', pa.string()),
    ('bought_together', pa.string())
])

# ==========================================
# 2. HÀM CHUẨN HÓA DỮ LIỆU 
# ==========================================

def normalize_review(raw: dict) -> Optional[dict]:
    reviewer_id = raw.get("user_id") or raw.get("reviewer_id") or raw.get("reviewerID")
    parent_asin = raw.get("parent_asin") or raw.get("asin")
    if not reviewer_id or not parent_asin: return None

    # Xử lý timestamp bằng toán học (Cực nhanh)
    ts_raw = raw.get("timestamp") or raw.get("unixReviewTime") or 0
    try:
        ts = int(ts_raw)
        ts = ts // 1000 if ts > 1_000_000_000_000 else ts
    except:
        ts = 0

    return {
        "reviewer_id": str(reviewer_id), 
        "parent_asin": str(parent_asin),
        "rating": float(raw.get("rating") or raw.get("overall") or 0.0),
        "review_text": str(raw.get("text") or raw.get("review_text") or ""), # Đã mở khóa
        "timestamp": int(ts), 
        "helpful_vote": int(raw.get("helpful_vote") or 0),
        "verified_purchase": bool(raw.get("verified_purchase") or False)
    }

def normalize_metadata(raw: dict) -> Optional[dict]:
    parent_asin = raw.get("parent_asin") or raw.get("asin")
    if not parent_asin: return None
    
    # Hàm con hỗ trợ ép List thành String siêu tốc
    def _join_list(lst, sep):
        if isinstance(lst, list):
            return sep.join(str(i) for i in lst if i)
        return str(lst).strip() if lst else ""

    # Ép giá về float an toàn
    try:
        price = float(raw.get("price") or 0.0)
    except:
        price = 0.0

    return {
        "parent_asin": str(parent_asin),
        "title": str(raw.get("title") or ""),
        "main_category": str(raw.get("main_category") or ""),
        "store": str(raw.get("store") or ""),
        "price": price,
        "categories": _join_list(raw.get("categories"), " > "),
        "features": _join_list(raw.get("features"), " | "),
        "bought_together": _join_list(raw.get("bought_together"), ",")
    }

# ==========================================
# 3. LUỒNG XỬ LÝ ĐA NHIỆM (STREAMING)
# ==========================================

def _fetch_and_process_worker(ds, batch_size, max_records, out_queue, normalizer_func, schema):
    batch_idx = 0
    max_retries = 5  
    retry_delay = 30 
    
    ds_iter = iter(ds.iter(batch_size=batch_size))
    
    while True:
        try:
            raw_batch = next(ds_iter)
            
            keys = raw_batch.keys()
            row_oriented = [dict(zip(keys, vals)) for vals in zip(*raw_batch.values())]
            processed = [r for r in (normalizer_func(raw) for raw in row_oriented) if r is not None]
            
            if processed:
                arrow_table = pa.Table.from_pylist(processed, schema=schema)
                out_queue.put(arrow_table)
                
            batch_idx += 1
            if max_records and (batch_idx * batch_size >= max_records): 
                break
                
        except StopIteration:
            break 
        except Exception as e:
            logger.error(f"Lỗi khi kéo batch {batch_idx}: {e}")
            if max_retries > 0:
                logger.info(f"Đang thử lại sau {retry_delay}s... (Còn {max_retries} lần thử)")
                time.sleep(retry_delay)
                max_retries -= 1
                continue
            else:
                logger.error("Đã hết lượt thử lại. Dừng tiến trình.")
                break

    out_queue.put(None)

def _create_iterator(cfg: dict, dataset_name: str, subset: str, max_records: Optional[int], normalizer_func, schema) -> Iterator[pa.Table]:
    hf = cfg["huggingface"]
    # Tăng maxsize lên 30 để có không gian đệm lớn hơn, giúp ghi MinIO mượt hơn
    out_queue = queue.Queue(maxsize=int(hf.get("queue_maxsize", 30)))
    ds = load_dataset(dataset_name, subset, split="full", streaming=True, trust_remote_code=True)
    
    t = threading.Thread(target=_fetch_and_process_worker, args=(ds, hf["stream_batch_size"], max_records, out_queue, normalizer_func, schema))
    t.start()
    
    while True:
        table = out_queue.get()
        if table is None: break 
        if table.num_rows > 0: yield table
    t.join()

def iter_review_batches(cfg: dict):
    return _create_iterator(cfg, cfg["huggingface"]["review_dataset"], cfg["huggingface"]["review_subset"], cfg["huggingface"].get("max_review_records"), normalize_review, REVIEW_ARROW_SCHEMA)

def iter_metadata_batches(cfg: dict):
    return _create_iterator(cfg, cfg["huggingface"]["metadata_dataset"], cfg["huggingface"]["metadata_subset"], cfg["huggingface"].get("max_metadata_records"), normalize_metadata, METADATA_ARROW_SCHEMA)