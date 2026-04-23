"""
silver_step3_user_profile.py — Silver Layer: User Text Profile

TỐI ƯU v2:
  - Nhận df_train, df_val từ orchestrator (không đọc lại)
  - Gộp Phase 2+3 thành 1 pass: groupBy → collect_list → sort → slice → join
    → Tiết kiệm 1 shuffle + loại bỏ checkpoint tạm ghi MinIO
  - Bỏ repartition khi ghi → dùng coalesce

NGUYÊN TẮC CHỐNG LEAKAGE:
  - Chỉ dùng bronze_train — reviews val/test bị loại hoàn toàn
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.column import Column
from .silver_utils import advanced_clean_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_REVIEWS      = 3
REVIEW_TITLE_CHARS = 120
REVIEW_TEXT_CHARS   = 220


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _s3a_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — TÍNH TRỌNG SỐ VÀ TẠO SNIPPET
# ─────────────────────────────────────────────────────────────────────────────

def _compute_review_weights(df_train: DataFrame) -> DataFrame:
    """
    Tính w(r) = 1 + log(1 + helpful_vote) và tạo snippet.
    Dùng Spark built-in log() — vectorized trên JVM.
    """
    # CHỈ NHẶT LỜI KHEN CỦA USER ĐỂ XÂY DỰNG SỞ THÍCH:
    df_train = df_train.filter(F.col("rating") >= 3.0)

    # ÁP DỤNG HÀM LÀM SẠCH TOÀN DIỆN CHO REVIEW TEXT
    clean_title = advanced_clean_text(F.coalesce(F.col("review_title"), F.lit("")))
    clean_text = advanced_clean_text(F.coalesce(F.col("review_text"), F.lit("")))

    title_part = F.coalesce(
        F.substring(clean_title, 1, REVIEW_TITLE_CHARS),
        F.lit("")
    )
    text_part = F.coalesce(
        F.substring(clean_text, 1, REVIEW_TEXT_CHARS),
        F.lit("")
    )

    snippet_col = F.when(
        (F.length(clean_title) > 0) &
        (F.length(clean_text) > 0),
        F.concat_ws(" - ", title_part, text_part)
    ).when(
        F.length(clean_text) > 0, text_part
    ).when(
        F.length(clean_title) > 0, title_part
    ).otherwise(F.lit(""))

    helpful = F.coalesce(F.col("helpful_vote").cast(FloatType()), F.lit(0.0))
    weight_col = F.lit(1.0) + F.log(F.lit(1.0) + helpful)

    return df_train.select(
        "reviewer_id",
        snippet_col.alias("review_snippet"),
        weight_col.alias("review_weight"),
        "rating",
        "timestamp",
    ).filter(
        F.length(F.col("review_snippet")) > 0
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — TOP-K + GHÉP TEXT (1 PASS DUY NHẤT)
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_profiles(df_weighted: DataFrame, shuffle_parts: int) -> DataFrame:
    """
    1 groupBy duy nhất: collect_list → sort → slice top-K → join text.

    So với v1 (2 groupBy + checkpoint):
      - Tiết kiệm 1 shuffle lớn (~35M rows)
      - Loại bỏ checkpoint tạm ghi MinIO (~9M rows write + read)
    """
    logger.info(f"[Step3] groupBy + collect_list + top-{TOP_K_REVIEWS} + join text (1 pass)...")

    # Đóng gói review thành struct (thời gian trước để lấy review mới nhất làm gốc)
    df_struct = df_weighted.withColumn(
        "review_struct",
        F.struct(
            F.col("timestamp").alias("t"),
            F.col("review_weight").alias("w"),
            F.col("review_snippet").alias("s"),
        )
    ).repartition(shuffle_parts, "reviewer_id")

    # 1 groupBy duy nhất
    df_user = df_struct.groupBy("reviewer_id").agg(
        F.collect_list("review_struct").alias("reviews_raw"),
        F.count("*").alias("review_count_train"),
        F.avg("rating").alias("avg_rating"),
        F.avg("review_weight").alias("avg_review_weight"),
    )

    # Sort → slice top-K → extract text → join
    df_user = df_user.withColumn(
        "reviews_topk",
        F.slice(F.sort_array(F.col("reviews_raw"), asc=False), 1, TOP_K_REVIEWS)
    ).withColumn(
        "user_text",
        F.array_join(
            F.transform(F.col("reviews_topk"), lambda x: x.getField("s")),
            " [SEP] "
        )
    ).withColumn(
        "user_text",
        F.when(F.length(F.col("user_text")) > 5, F.col("user_text"))
         .otherwise(F.lit("[NO_TEXT] User interaction profile"))
    ).select(
        "reviewer_id",
        "user_text",
        "review_count_train",
        "avg_rating",
        "avg_review_weight",
    )

    return df_user


# ─────────────────────────────────────────────────────────────────────────────
# VAL GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def _build_val_ground_truth(df_val: DataFrame, df_popularity: DataFrame) -> DataFrame:
    """Chỉ lấy (reviewer_id, parent_asin, popularity_group) — không lấy text."""
    logger.info("[Step3] Xây dựng val ground truth...")

    val_cols = ["reviewer_id", "parent_asin", "timestamp", "rating"]
    pop_slim = df_popularity.select("parent_asin", "popularity_group", "train_freq")

    return df_val.select(*val_cols).join(
        F.broadcast(pop_slim), on="parent_asin", how="left"
    ).withColumn(
        "popularity_group",
        F.coalesce(F.col("popularity_group"), F.lit("COLD_START"))
    ).withColumn(
        "train_freq",
        F.coalesce(F.col("train_freq"), F.lit(0).cast("long"))
    ).withColumn(
        "is_tail", (F.col("popularity_group") == "TAIL").cast("int")
    ).withColumn(
        "is_cold_start", (F.col("popularity_group") == "COLD_START").cast("int")
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    df_popularity: DataFrame,
    df_train: DataFrame,
    df_val: DataFrame,
) -> None:
    """
    Args:
        df_train: bronze_train (đã projection pushdown bởi orchestrator).
        df_val: bronze_val từ orchestrator.
    """
    silver_user_out = _s3a_path(cfg, "silver", "silver_user_text_profile.parquet")
    silver_val_gt   = _s3a_path(cfg, "silver", "silver_val_ground_truth.parquet")
    write_mode      = cfg.get("silver", {}).get("write_mode", "overwrite")
    shuffle_parts   = int(cfg.get("spark", {}).get("shuffle_partitions", 200))

    # ── Phase 1: Tính trọng số ────────────────────────────────────────────────
    logger.info("[Step3] Tính review weights...")
    df_weighted = _compute_review_weights(df_train)

    df_weighted = df_weighted.cache()
    w_count = df_weighted.count()
    logger.info(f"  Reviews có text hợp lệ: {w_count:,}")

    # ── Phase 2: Top-K + ghép text (1 pass) ───────────────────────────────────
    df_user_text = _build_user_profiles(df_weighted, shuffle_parts)
    df_weighted.unpersist()

    # ── Ghi silver_user_text_profile ──────────────────────────────────────────
    logger.info(f"[Step3] Ghi silver_user_text_profile → {silver_user_out}")
    df_user_text.repartition(20) \
                .write.mode(write_mode) \
                .option("compression", "zstd") \
                .parquet(silver_user_out)

    user_count = spark.read.parquet(silver_user_out).count()
    logger.info(f"[Step3] silver_user_text_profile: {user_count:,} users")

    # ── Ghi silver_val_ground_truth ───────────────────────────────────────────
    logger.info(f"[Step3] Ghi silver_val_ground_truth → {silver_val_gt}")
    df_val_gt = _build_val_ground_truth(df_val, df_popularity)
    df_val_gt.coalesce(5) \
             .write.mode(write_mode) \
             .option("compression", "zstd") \
             .parquet(silver_val_gt)

    logger.info(f"[Step3] silver_val_ground_truth ghi xong.")
