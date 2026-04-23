"""
silver_step1_popularity.py — Silver Layer: Item Popularity Classification

TỐI ƯU v2:
  - Nhận df_train từ orchestrator (không đọc lại từ HF/MinIO)
  - Projection pushdown: chỉ dùng cột parent_asin
  - collect() bảng nhỏ (~1M items) về driver để tính CDF Pareto
  - Output broadcast-ready cho downstream steps
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, FloatType

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

HEAD_RATIO = 0.20  # Top 20%
MID_RATIO  = 0.10  # Kế tiếp 10%
TAIL_RATIO = 0.70  # Dưới cùng 70%

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _s3a_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


# ─────────────────────────────────────────────────────────────────────────────
# CORE: TÍNH TRAIN_FREQ + PHÂN LOẠI
# ─────────────────────────────────────────────────────────────────────────────

def _compute_and_classify(df_train: DataFrame, spark: SparkSession) -> DataFrame:
    """
    Tính train_freq + phân loại HEAD/MID/TAIL trong một pipeline liền mạch.
    collect() bảng freq (~1M rows, ~40MB) về driver để tính CDF bằng Python thuần.
    """
    logger.info("⏳ [Step1] Tính train_freq cho mỗi item...")

    freq_df = (
        df_train
        .select("parent_asin")
        .groupBy("parent_asin")
        .agg(F.count("*").alias("train_freq"))
        .withColumn("train_freq", F.col("train_freq").cast(LongType()))
    )

    # Cache freq_df nhỏ (~1M rows) để tránh recompute
    freq_df = freq_df.cache()
    item_count = freq_df.count()
    logger.info(f"  📊 Tổng items: {item_count:,}")

    # ── Tính ngưỡng Pareto ────────────────────────────────────────────────────
    logger.info("⏳ [Step1] Xác định ngưỡng Pareto HEAD/MID/TAIL...")

    freq_rows = (
        freq_df
        .orderBy(F.col("train_freq").desc())
        .select("parent_asin", "train_freq")
        .collect()
    )

    head_idx = int(item_count * HEAD_RATIO)
    mid_idx  = int(item_count * (HEAD_RATIO + MID_RATIO))

    # Tránh out of bounds
    head_idx = min(head_idx, item_count - 1)
    mid_idx  = min(mid_idx, item_count - 1)

    head_freq_cutoff = freq_rows[head_idx]["train_freq"]
    tail_freq_cutoff = freq_rows[mid_idx]["train_freq"]

    # Đếm phân phối do tie-breaking
    head_count = sum(1 for r in freq_rows if r["train_freq"] >= head_freq_cutoff)
    tail_count = sum(1 for r in freq_rows if r["train_freq"] < tail_freq_cutoff)
    mid_count  = item_count - head_count - tail_count

    logger.info(f"  🔴 HEAD : {head_count:,} items | train_freq ≥ {head_freq_cutoff} | ~{head_count/max(item_count,1)*100:.1f}%")
    logger.info(f"  🟡 MID  : {mid_count:,} items | {tail_freq_cutoff} ≤ train_freq < {head_freq_cutoff} | ~{mid_count/max(item_count,1)*100:.1f}%")
    logger.info(f"  🟢 TAIL : {tail_count:,} items | train_freq < {tail_freq_cutoff} | ~{tail_count/max(item_count,1)*100:.1f}%")

    # ── Áp ngưỡng ────────────────────────────────────────────────────────────
    classified_df = freq_df.withColumn(
        "popularity_group",
        F.when(F.col("train_freq") >= head_freq_cutoff, F.lit("HEAD"))
         .when(F.col("train_freq") >= tail_freq_cutoff, F.lit("MID"))
         .otherwise(F.lit("TAIL"))
    )

    freq_df.unpersist()
    return classified_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def run(spark: SparkSession, cfg: dict, df_train: DataFrame) -> DataFrame:
    """
    Chạy Silver Step 1: Phân loại item popularity.

    Args:
        df_train: DataFrame bronze_train đã được đọc bởi orchestrator.

    Returns:
        DataFrame popularity đã cache, broadcast-ready.
    """
    silver_out = _s3a_path(cfg, "silver", "silver_item_popularity.parquet")
    write_mode = cfg.get("silver", {}).get("write_mode", "overwrite")

    # ── Tính và phân loại ─────────────────────────────────────────────────────
    popularity_df = _compute_and_classify(df_train, spark)

    # ── Ghi ra MinIO (ZSTD đã được set global bởi orchestrator) ──────────────
    logger.info(f"⏳ [Step1] Ghi silver_item_popularity → {silver_out}")

    popularity_df.coalesce(5) \
        .write.mode(write_mode) \
        .option("compression", "zstd") \
        .parquet(silver_out)

    # Cache lại để downstream dùng broadcast
    result_df = spark.read.parquet(silver_out).cache()
    count_result = result_df.count()
    logger.info(f"✅ [Step1] silver_item_popularity: {count_result:,} items")

    result_df.groupBy("popularity_group").count().orderBy("popularity_group").show()

    return result_df
