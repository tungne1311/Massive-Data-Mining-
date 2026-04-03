import logging
import queue
import threading
from datetime import datetime
from typing import Iterator, Optional
import pyarrow as pa
from datasets import load_dataset

logger = logging.getLogger(__name__)

SOURCE_NAME = "amazon_reviews_2023_electronics"
INGEST_DATE = datetime.now().strftime("%Y-%m-%d")

# Định nghĩa Schema cho PyArrow (Loại bỏ pyspark.sql.types)
REVIEW_ARROW_SCHEMA = pa.schema([
    ('reviewer_id', pa.string()),
    ('parent_asin', pa.string()),
    ('rating', pa.float32()),
    ('review_text', pa.string()),
    ('text_len', pa.int32()),
    ('timestamp', pa.int64()),
    ('helpful_vote', pa.int32()),
    ('verified_purchase', pa.bool_()),
    ('ingest_date', pa.string()),
    ('source_name', pa.string()),
    ('year_month', pa.string())
])

METADATA_ARROW_SCHEMA = pa.schema([
    ('parent_asin', pa.string()),
    ('item_title', pa.string()),
    ('brand', pa.string()),
    ('main_category', pa.string()),
    ('features', pa.string()),
    ('price_bucket', pa.string()),
    ('ingest_date', pa.string()),
    ('source_name', pa.string()),
    ('year_month', pa.string())
])

def normalize_review(raw: dict) -> Optional[dict]:
    reviewer_id = raw.get("user_id") or raw.get("reviewer_id") or raw.get("reviewerID")
    parent_asin = raw.get("parent_asin") or raw.get("asin")
    if not reviewer_id or not parent_asin: return None

    ts_raw = raw.get("timestamp") or raw.get("unixReviewTime") or 0
    try:
        ts = int(ts_raw)
        ts = ts // 1000 if ts > 1_000_000_000_000 else ts
        dt = datetime.fromtimestamp(ts)
        year_month = dt.strftime("%Y-%m") if 1995 <= dt.year <= 2030 else "unknown_date"
    except:
        ts, year_month = 0, "unknown_date"

    return {
        "reviewer_id": str(reviewer_id), "parent_asin": str(parent_asin),
        "rating": float(raw.get("rating") or raw.get("overall") or 0.0),
        "review_text": str(raw.get("text") or raw.get("review_text") or ""),
        "text_len": int(raw.get("text_len") or len(str(raw.get("text") or ""))),
        "timestamp": int(ts), "helpful_vote": int(raw.get("helpful_vote") or 0),
        "verified_purchase": bool(raw.get("verified_purchase") or False),
        "ingest_date": INGEST_DATE, "source_name": SOURCE_NAME, "year_month": year_month
    }

def normalize_metadata(raw: dict) -> Optional[dict]:
    parent_asin = raw.get("parent_asin") or raw.get("asin")
    if not parent_asin: return None
    feats = raw.get("features") or []
    return {
        "parent_asin": str(parent_asin),
        "item_title": str(raw.get("title") or raw.get("item_title") or ""),
        "brand": str(raw.get("brand") or ""),
        "main_category": str(raw.get("main_category") or ""),
        "features": " | ".join(str(f) for f in feats if f) if isinstance(feats, list) else str(feats).strip(),
        "price_bucket": str(raw.get("price_bucket") or ""),
        "ingest_date": INGEST_DATE, "source_name": SOURCE_NAME, "year_month": INGEST_DATE[:7]
    }

import time # Thêm ở đầu file

def _fetch_and_process_worker(ds, batch_size, max_records, out_queue, normalizer_func, schema):
    batch_idx = 0
    max_retries = 5  # Số lần thử lại tối đa khi HF sập
    retry_delay = 30 # Chờ 30 giây trước khi thử lại
    
    # Sử dụng iterator để có thể tiếp tục nếu lỗi (tùy thuộc vào khả năng hỗ trợ của HF stream)
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
            break # Hết dữ liệu thật sự
        except Exception as e:
            logger.error(f"Lỗi khi kéo batch {batch_idx}: {e}")
            if max_retries > 0:
                logger.info(f"Đang thử lại sau {retry_delay}s... (Còn {max_retries} lần thử)")
                time.sleep(retry_delay)
                max_retries -= 1
                # Lưu ý: Streaming đôi khi không resume được đúng vị trí, 
                # nhưng ít nhất nó không làm crash toàn bộ pipeline của bạn.
                continue
            else:
                logger.error("Đã hết lượt thử lại. Dừng tiến trình.")
                break

    out_queue.put(None)

def _create_iterator(cfg: dict, dataset_name: str, subset: str, max_records: Optional[int], normalizer_func, schema) -> Iterator[pa.Table]:
    hf = cfg["huggingface"]
    out_queue = queue.Queue(maxsize=int(hf.get("queue_maxsize", 3)))
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