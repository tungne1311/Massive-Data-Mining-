"""
Bước 1 — Data Ingestion
Stream Amazon Reviews 2023 Electronics từ HuggingFace → normalize → Spark DataFrame.
"""

import logging
import re
from datetime import datetime
from typing import Iterator, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType, FloatType, IntegerType, LongType,
    StringType, StructField, StructType,
)

logger = logging.getLogger(__name__)

SOURCE_NAME = "amazon_reviews_2023_electronics"
INGEST_DATE = datetime.now().strftime("%Y-%m-%d")

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
])

METADATA_SCHEMA = StructType([
    StructField("parent_asin",   StringType(), False),
    StructField("item_title",    StringType(), True),
    StructField("brand",         StringType(), True),
    StructField("main_category", StringType(), True),
    StructField("description",   StringType(), True),
    StructField("features",      StringType(), True),
    StructField("price_bucket",  StringType(), True),
    StructField("ingest_date",   StringType(), True),
    StructField("source_name",   StringType(), True),
])


# ── Streaming ─────────────────────────────────────────────────────────────────

def _stream_batched(
    dataset_name: str,
    subset: str,
    batch_size: int,
    max_records: Optional[int],
) -> Iterator[list[dict]]:
    """Yield batches từ HuggingFace streaming — không load toàn bộ vào RAM."""
    from datasets import load_dataset
    logger.info(f"  Kết nối: {dataset_name} / {subset}")

    ds = load_dataset(dataset_name, subset, split="full",
                      streaming=True, trust_remote_code=True)
    batch: list[dict] = []
    total = 0

    for record in ds:
        batch.append(record)
        total += 1
        if len(batch) >= batch_size:
            logger.info(f"  {total:,} records streamed...")
            yield batch
            batch = []
        if max_records and total >= max_records:
            break

    if batch:
        yield batch
    logger.info(f"  Stream xong: {total:,} records")


# ── Normalizers ───────────────────────────────────────────────────────────────

def normalize_review(raw: dict) -> Optional[dict]:
    """Chuẩn hóa 1 record review. Trả về None nếu thiếu field bắt buộc."""
    reviewer_id = raw.get("user_id") or raw.get("reviewer_id") or raw.get("reviewerID")
    parent_asin = raw.get("parent_asin") or raw.get("asin")
    rating_raw  = raw.get("rating") or raw.get("overall")

    if not reviewer_id or not parent_asin or rating_raw is None:
        return None
    try:
        rating = float(rating_raw)
    except (TypeError, ValueError):
        return None
    if not (1.0 <= rating <= 5.0):
        return None

    review_text = str(raw.get("text") or raw.get("reviewText") or "").strip()

    ts_raw = raw.get("timestamp") or raw.get("unixReviewTime") or 0
    try:
        ts = int(ts_raw)
        ts = ts // 1000 if ts > 1_000_000_000_000 else ts  # ms → s
    except (TypeError, ValueError):
        ts = 0

    helpful_raw = raw.get("helpful_vote") or raw.get("helpful") or 0
    if isinstance(helpful_raw, list):   # dataset cũ có dạng [x, y]
        helpful_raw = helpful_raw[0] if helpful_raw else 0
    try:
        helpful_vote = int(helpful_raw)
    except (TypeError, ValueError):
        helpful_vote = 0

    return {
        "reviewer_id":       reviewer_id,
        "parent_asin":       parent_asin,
        "rating":            rating,
        "review_text":       review_text,
        "text_len":          len(review_text.split()) if review_text else 0,
        "timestamp":         ts,
        "helpful_vote":      helpful_vote,
        "verified_purchase": bool(raw.get("verified_purchase") or raw.get("verified") or False),
        "ingest_date":       INGEST_DATE,
        "source_name":       SOURCE_NAME,
    }


def normalize_metadata(raw: dict) -> Optional[dict]:
    """Chuẩn hóa 1 record metadata. Trả về None nếu thiếu parent_asin."""
    parent_asin = raw.get("parent_asin") or raw.get("asin")
    if not parent_asin:
        return None

    desc_raw = raw.get("description") or ""
    description = (
        " ".join(str(x) for x in desc_raw if x)
        if isinstance(desc_raw, list) else str(desc_raw).strip()
    )

    feats_raw = raw.get("features") or []
    features = (
        " | ".join(str(f) for f in feats_raw if f)
        if isinstance(feats_raw, list) else str(feats_raw).strip()
    )

    return {
        "parent_asin":   parent_asin,
        "item_title":    str(raw.get("title") or "").strip(),
        "brand":         str(raw.get("store") or raw.get("brand") or raw.get("manufacturer") or "").strip(),
        "main_category": str(raw.get("main_category") or "").strip(),
        "description":   description,
        "features":      features,
        "price_bucket":  _price_bucket(raw.get("price") or ""),
        "ingest_date":   INGEST_DATE,
        "source_name":   SOURCE_NAME,
    }


def _price_bucket(price_raw) -> str:
    match = re.search(r"[\d.]+", str(price_raw).replace(",", ""))
    if not match:
        return "unknown"
    try:
        val = float(match.group())
    except ValueError:
        return "unknown"
    if val < 25:   return "budget"
    if val < 100:  return "mid"
    if val < 300:  return "premium"
    return "high_end"


# ── Ingestion ─────────────────────────────────────────────────────────────────
def iter_review_batches(cfg: dict) -> Iterator[list[dict]]:
    hf = cfg["huggingface"]
    skipped = 0
    total_valid = 0

    for raw_batch in _stream_batched(
        hf["review_dataset"],
        hf["review_subset"],
        hf["stream_batch_size"],
        hf["max_review_records"],
    ):
        batch_rows: list[dict] = []

        for raw in raw_batch:
            r = normalize_review(raw)
            if r is None:
                skipped += 1
                continue
            batch_rows.append(r)

        if batch_rows:
            total_valid += len(batch_rows)
            logger.info(
                f"Review batch hợp lệ: {len(batch_rows):,} | "
                f"total={total_valid:,} | skipped={skipped:,}"
            )
            yield batch_rows

    logger.info(f"Reviews tổng hợp lệ: {total_valid:,} | tổng bỏ qua: {skipped:,}")


def iter_metadata_batches(cfg: dict) -> Iterator[list[dict]]:
    hf = cfg["huggingface"]
    skipped = 0
    total_valid = 0

    for raw_batch in _stream_batched(
        hf["metadata_dataset"],
        hf["metadata_subset"],
        hf["stream_batch_size"],
        hf["max_metadata_records"],
    ):
        batch_rows: list[dict] = []

        for raw in raw_batch:
            r = normalize_metadata(raw)
            if r is None:
                skipped += 1
                continue
            batch_rows.append(r)

        if batch_rows:
            total_valid += len(batch_rows)
            logger.info(
                f"Metadata batch hợp lệ: {len(batch_rows):,} | "
                f"total={total_valid:,} | skipped={skipped:,}"
            )
            yield batch_rows

    logger.info(f"Metadata tổng hợp lệ: {total_valid:,} | tổng bỏ qua: {skipped:,}")
