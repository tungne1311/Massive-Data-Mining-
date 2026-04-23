"""
ste1.py – Ingestion & Normalization (HuggingFace → Arrow Batches)

TỐI ƯU v2:
  - BỎ: bought_together (metadata), verified_purchase (review)
  - Vectorized normalization: dùng pyarrow.compute thay vì lặp từng dòng Python
    → Tăng tốc 10-50× cho mỗi batch
  - Giữ nguyên kiến trúc streaming + exponential backoff
"""

import logging
import os
import queue
import threading
import time
from typing import Iterator, Optional
import pyarrow as pa
import pyarrow.compute as pc
from datasets import load_dataset

# Tăng timeout cho HuggingFace requests (mặc định 10s quá ngắn cho dataset lớn)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("REQUESTS_TIMEOUT", "120")

logger = logging.getLogger(__name__)

# ==========================================
# 1. ĐỊNH NGHĨA SCHEMA
# ==========================================

REVIEW_ARROW_SCHEMA = pa.schema([
    ('reviewer_id',       pa.string()),
    ('parent_asin',       pa.string()),
    ('rating',            pa.float32()),
    ('review_title',      pa.string()),
    ('review_text',       pa.string()),
    ('timestamp',         pa.int64()),
    ('helpful_vote',      pa.int32()),
])

METADATA_ARROW_SCHEMA = pa.schema([
    ('parent_asin',     pa.string()),
    ('title',           pa.string()),
    ('main_category',   pa.string()),
    ('store',           pa.string()),
    ('price',           pa.float32()),
    ('average_rating',  pa.float32()),
    ('rating_number',   pa.int32()),
    ('categories',      pa.string()),
    ('features',        pa.string()),
    ('description',     pa.string()),
    ('details',         pa.string()),
])


# ==========================================
# 2. VECTORIZED NORMALIZATION (pyarrow.compute)
#    Xử lý toàn bộ batch dạng cột — nhanh hơn Python loop 10-50×
# ==========================================

def _safe_string_column(raw_batch: dict, *keys: str, default: str = "") -> pa.Array:
    """Lấy cột string với fallback qua nhiều tên trường."""
    for key in keys:
        vals = raw_batch.get(key)
        if vals is not None:
            arr = pa.array(vals, type=pa.string())
            # Thay null bằng default
            return pc.if_else(pc.is_null(arr), default, arr)
    n = len(next(iter(raw_batch.values())))
    return pa.array([default] * n, type=pa.string())


def _safe_numeric_column(raw_batch: dict, key: str, target_type, default=0):
    """Lấy cột số với fallback về default khi null hoặc lỗi cast."""
    vals = raw_batch.get(key)
    if vals is None:
        n = len(next(iter(raw_batch.values())))
        return pa.array([default] * n, type=target_type)

    arr = pa.array(vals)
    # Fill null trước khi cast
    if arr.null_count > 0:
        arr = pc.if_else(pc.is_null(arr), default, arr)
    try:
        return pc.cast(arr, target_type)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        # Fallback: cast từng phần tử an toàn
        result = []
        for v in vals:
            try:
                result.append(target_type.to_pandas_dtype()(v) if v is not None else default)
            except (ValueError, TypeError):
                result.append(default)
        return pa.array(result, type=target_type)


def normalize_review_batch(raw_batch: dict) -> pa.Table:
    """
    Chuẩn hóa batch review hoàn toàn vectorized.
    Input: dict of lists (từ HuggingFace streaming)
    Output: pa.Table đã chuẩn hóa
    """
    n = len(next(iter(raw_batch.values())))
    if n == 0:
        return pa.table({}, schema=REVIEW_ARROW_SCHEMA)

    reviewer_id = _safe_string_column(raw_batch, "user_id", "reviewer_id", "reviewerID")
    parent_asin = _safe_string_column(raw_batch, "parent_asin", "asin")

    # Filter: bỏ dòng thiếu reviewer_id hoặc parent_asin
    valid_mask = pc.and_(
        pc.is_valid(reviewer_id),
        pc.is_valid(parent_asin),
    )
    # Thêm kiểm tra chuỗi rỗng
    valid_mask = pc.and_(
        valid_mask,
        pc.and_(
            pc.not_equal(reviewer_id, ""),
            pc.not_equal(parent_asin, ""),
        )
    )

    # Timestamp: xử lý millisecond → second
    ts_arr = _safe_numeric_column(raw_batch, "timestamp", pa.int64(), default=0)
    # Nếu không có "timestamp", thử "unixReviewTime"
    if raw_batch.get("timestamp") is None:
        ts_arr = _safe_numeric_column(raw_batch, "unixReviewTime", pa.int64(), default=0)
    # Chuyển millisecond về second nếu cần
    threshold = pa.scalar(1_000_000_000_000, type=pa.int64())
    ts_arr = pc.if_else(pc.greater(ts_arr, threshold), pc.divide(ts_arr, 1000), ts_arr)

    rating_arr = _safe_numeric_column(raw_batch, "rating", pa.float32(), default=0.0)

    # Khóa chặn OOM: Lọc rating >= 3.0 NGAY TỪ HUGGING FACE STREAM
    # Cắt giảm >30% dữ liệu rác (string reviews bị bỏ đi) trước khi nạp vào RAM bộ nhớ
    valid_mask = pc.and_(
        valid_mask,
        pc.greater_equal(rating_arr, 3.0)
    )

    table = pa.table({
        "reviewer_id":  reviewer_id,
        "parent_asin":  parent_asin,
        "rating":       rating_arr,
        "review_title": _safe_string_column(raw_batch, "title", default=""),
        "review_text":  _safe_string_column(raw_batch, "text", "review_text", default=""),
        "timestamp":    ts_arr,
        "helpful_vote": _safe_numeric_column(raw_batch, "helpful_vote", pa.int32(), default=0),
    })

    # Áp filter mask
    filtered = pc.filter(table, valid_mask)
    return filtered.cast(REVIEW_ARROW_SCHEMA)


def normalize_metadata_batch(raw_batch: dict) -> pa.Table:
    """
    Chuẩn hóa batch metadata hoàn toàn vectorized.
    Trường list (categories, features, description) → join thành string.
    Trường dict (details) → xử lý row-level (không thể vectorize dict phức tạp).
    """
    n = len(next(iter(raw_batch.values())))
    if n == 0:
        return pa.table({}, schema=METADATA_ARROW_SCHEMA)

    parent_asin = _safe_string_column(raw_batch, "parent_asin", "asin")

    # Filter rows thiếu parent_asin
    valid_mask = pc.and_(
        pc.is_valid(parent_asin),
        pc.not_equal(parent_asin, ""),
    )

    # Xử lý trường list → string (cần row-level vì pyarrow không join list natively)
    def _join_list_column(key: str, sep: str) -> pa.Array:
        vals = raw_batch.get(key)
        if vals is None:
            return pa.array([""] * n, type=pa.string())
        result = []
        for v in vals:
            if isinstance(v, list):
                result.append(sep.join(str(i) for i in v if i))
            elif v is not None:
                result.append(str(v).strip())
            else:
                result.append("")
        return pa.array(result, type=pa.string())

    # details: dict → "key: value | key: value" (row-level bắt buộc)
    def _flatten_details() -> pa.Array:
        vals = raw_batch.get("details")
        if vals is None:
            return pa.array([""] * n, type=pa.string())
        result = []
        for v in vals:
            if isinstance(v, dict):
                parts = [f"{k}: {val}" for k, val in v.items()
                         if isinstance(val, str) and len(val) < 200]
                result.append(" | ".join(parts))
            elif v is not None:
                result.append(str(v).strip())
            else:
                result.append("")
        return pa.array(result, type=pa.string())

    table = pa.table({
        "parent_asin":    parent_asin,
        "title":          _safe_string_column(raw_batch, "title", default=""),
        "main_category":  _safe_string_column(raw_batch, "main_category", default=""),
        "store":          _safe_string_column(raw_batch, "store", default=""),
        "price":          _safe_numeric_column(raw_batch, "price", pa.float32(), default=0.0),
        "average_rating": _safe_numeric_column(raw_batch, "average_rating", pa.float32(), default=0.0),
        "rating_number":  _safe_numeric_column(raw_batch, "rating_number", pa.int32(), default=0),
        "categories":     _join_list_column("categories", " > "),
        "features":       _join_list_column("features", " | "),
        "description":    _join_list_column("description", " "),
        "details":        _flatten_details(),
    })

    filtered = pc.filter(table, valid_mask)
    return filtered.cast(METADATA_ARROW_SCHEMA)


# ==========================================
# 3. LUỒNG XỬ LÝ ĐA NHIỆM (STREAMING)
#    Vectorized worker: nhận dict-of-lists → output Arrow Table trực tiếp
# ==========================================

def _fetch_and_process_worker(
    ds, batch_size: int, max_records: Optional[int],
    out_queue: queue.Queue, batch_normalizer, schema: pa.Schema
) -> None:
    max_retries    = 5
    base_delay     = 10
    current_delay  = base_delay
    total_records  = 0
    batch_idx      = 0
    ds_iter        = iter(ds.iter(batch_size=batch_size))

    while True:
        try:
            raw_batch = next(ds_iter)
        except StopIteration:
            break
        except Exception as e:
            logger.error(f"Lỗi khi kéo batch {batch_idx}: {e}")
            if max_retries > 0:
                logger.info(f"Exponential backoff: chờ {current_delay}s... (còn {max_retries} lần thử)")
                time.sleep(current_delay)
                max_retries   -= 1
                current_delay  = min(current_delay * 2, 300)
                continue
            else:
                logger.error("Đã hết lượt thử lại. Dừng tiến trình worker.")
                break

        # Reset delay sau khi thành công
        current_delay = base_delay

        # VECTORIZED: chuyển dict-of-lists thẳng sang Arrow Table
        try:
            arrow_table = batch_normalizer(raw_batch)
        except Exception as e:
            logger.warning(f"Lỗi chuẩn hóa batch {batch_idx}: {e}. Bỏ qua batch này.")
            batch_idx += 1
            continue

        if arrow_table.num_rows > 0:
            out_queue.put(arrow_table)
            total_records += arrow_table.num_rows

        batch_idx += 1

        if max_records and total_records >= max_records:
            logger.info(f"Đã đạt giới hạn max_records={max_records}. Dừng worker.")
            break

    logger.info(f"Worker hoàn tất: đã xử lý {total_records:,} bản ghi từ {batch_idx} batch.")
    out_queue.put(None)  # Sentinel


def _create_iterator(
    cfg: dict, dataset_name: str, subset: str,
    max_records: Optional[int], batch_normalizer, schema: pa.Schema
) -> Iterator[pa.Table]:
    hf         = cfg["huggingface"]
    queue_size = int(hf.get("queue_maxsize", 30))
    out_queue  = queue.Queue(maxsize=queue_size)

    # download_mode="force_redownload": tránh cache cũ/hỏng gây lỗi
    # FileNotFoundError: '/app/raw/review_categories/Electronics.jsonl'
    ds = load_dataset(
        dataset_name, subset,
        split="full", streaming=True, trust_remote_code=True,
        download_mode="force_redownload",
    )

    t = threading.Thread(
        target=_fetch_and_process_worker,
        args=(ds, hf["stream_batch_size"], max_records, out_queue, batch_normalizer, schema),
        daemon=True,
    )
    t.start()

    while True:
        table = out_queue.get()
        if table is None:
            break
        if table.num_rows > 0:
            yield table

    t.join()


def iter_review_batches(cfg: dict) -> Iterator[pa.Table]:
    hf = cfg["huggingface"]
    return _create_iterator(
        cfg,
        hf["review_dataset"],
        hf["review_subset"],
        hf.get("max_review_records"),
        normalize_review_batch,        # vectorized batch normalizer
        REVIEW_ARROW_SCHEMA,
    )


def iter_metadata_batches(cfg: dict) -> Iterator[pa.Table]:
    hf = cfg["huggingface"]
    return _create_iterator(
        cfg,
        hf["metadata_dataset"],
        hf["metadata_subset"],
        hf.get("max_metadata_records"),
        normalize_metadata_batch,      # vectorized batch normalizer
        METADATA_ARROW_SCHEMA,
    )