"""
silver_step2_item_profile.py — Silver Layer: Item Text Profile

TỐI ƯU v2:
  - Đọc bronze_meta trực tiếp từ MinIO (path truyền từ orchestrator)
  - Bỏ repartition trước partitionBy (tránh shuffle thừa)
  - ZSTD compression (set global bởi orchestrator)
  - Dùng Spark SQL built-ins cho string ops (không UDF)
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from .silver_utils import advanced_clean_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — Token Budget
# ─────────────────────────────────────────────────────────────────────────────

TITLE_MAX_CHARS       = 150
FEATURES_MAX_CHARS    = 450
CATEGORIES_MAX_CHARS  = 150
DESCRIPTION_MAX_CHARS = 300
DETAILS_MAX_CHARS     = 150
FEATURES_EXTENDED_CHARS = 750  # 450 (features) + 300 (description budget transferred)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _s3a_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


# ─────────────────────────────────────────────────────────────────────────────
# CORE: XÂY DỰNG ITEM TEXT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def _build_item_text(df_meta: DataFrame) -> DataFrame:
    """
    Ghép chuỗi văn bản item theo 4 cấp độ ưu tiên, dùng Spark SQL built-ins.
    Khi description rỗng → mở rộng features budget.
    """

    def clean_and_safe_col(col_name: str) -> F.Column:
        return advanced_clean_text(F.coalesce(F.col(col_name), F.lit("")))

    title_part = clean_and_safe_col("title")

    # Features: mở rộng khi description rỗng
    features_extended_part = F.when(
        (F.length(clean_and_safe_col("features")) > 0) & (F.length(clean_and_safe_col("description")) == 0),
        F.substring(clean_and_safe_col("features"), 1, FEATURES_EXTENDED_CHARS)
    ).when(
        F.length(clean_and_safe_col("features")) > 0,
        F.substring(clean_and_safe_col("features"), 1, FEATURES_MAX_CHARS)
    ).otherwise(F.lit(""))

    categories_part = F.when(
        F.length(clean_and_safe_col("categories")) > 0,
        F.substring(clean_and_safe_col("categories"), 1, CATEGORIES_MAX_CHARS)
    ).otherwise(F.lit(""))

    description_part = F.when(
        F.length(clean_and_safe_col("description")) > 0,
        F.substring(clean_and_safe_col("description"), 1, DESCRIPTION_MAX_CHARS)
    ).otherwise(F.lit(""))

    details_part = F.when(
        F.length(clean_and_safe_col("details")) > 0,
        F.substring(clean_and_safe_col("details"), 1, DETAILS_MAX_CHARS)
    ).otherwise(F.lit(""))

    # Ghép và loại bỏ separator thừa
    df = df_meta.withColumn(
        "item_text",
        F.regexp_replace(
            F.concat_ws(
                " | ",
                title_part, features_extended_part,
                categories_part, description_part, details_part,
            ),
            r"( \| )+", " | "
        )
    ).withColumn(
        "item_text",
        F.regexp_replace(F.col("item_text"), r"^\s*\|\s*|\s*\|\s*$", "")
    )

    def safe_col(col_name: str) -> F.Column:
        return F.coalesce(F.trim(F.col(col_name)), F.lit(""))

    # text_source_level cho debug
    df = df.withColumn(
        "text_source_level",
        F.when(
            (F.length(safe_col("description")) > 0) | (F.length(safe_col("details")) > 0), F.lit(4)
        ).when(F.length(safe_col("categories")) > 0, F.lit(3)
        ).when(F.length(safe_col("features")) > 0, F.lit(2)
        ).otherwise(F.lit(1)).cast(IntegerType())
    )

    # Ước lượng token count
    df = df.withColumn(
        "token_estimate",
        (F.size(F.split(F.col("item_text"), r"\s+")) * 1.3).cast(IntegerType())
    )

    return df.select(
        "parent_asin", "title", "main_category",
        "item_text", "text_source_level", "token_estimate",
        "average_rating", "rating_number",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    df_popularity: DataFrame,
    meta_path: str,
) -> DataFrame:
    """
    Args:
        df_popularity: broadcast-ready popularity DataFrame.
        meta_path: đường dẫn s3a:// tới bronze_meta (từ orchestrator).
    """
    silver_out = _s3a_path(cfg, "silver", "silver_item_text_profile.parquet")
    write_mode = cfg.get("silver", {}).get("write_mode", "overwrite")

    # ── Đọc bronze_meta từ MinIO ──────────────────────────────────────────────
    logger.info(f"⏳ [Step2] Đọc bronze_meta từ MinIO: {meta_path}")
    meta_cols = [
        "parent_asin", "title", "main_category", "store",
        "features", "categories", "description", "details",
        "average_rating", "rating_number",
    ]
    df_meta = spark.read.parquet(meta_path).select(*meta_cols)

    # ── Xây dựng item text ────────────────────────────────────────────────────
    logger.info("⏳ [Step2] Ghép item text profile...")
    df_text = _build_item_text(df_meta)

    # ── Broadcast join với popularity ─────────────────────────────────────────
    logger.info("⏳ [Step2] Broadcast join với popularity...")
    pop_slim = df_popularity.select(
        "parent_asin", "train_freq", "popularity_group"
    )
    df_joined = df_text.join(
        F.broadcast(pop_slim), on="parent_asin", how="left"
    ).withColumn(
        "popularity_group",
        F.coalesce(F.col("popularity_group"), F.lit("COLD_START"))
    ).withColumn(
        "train_freq",
        F.coalesce(F.col("train_freq"), F.lit(0).cast("long"))
    )

    # ── Ghi ra Silver ─────────────────────────────────────────────────────────
    # partitionBy đã tự phân chia file → không cần repartition trước (tránh shuffle thừa)
    logger.info(f"⏳ [Step2] Ghi silver_item_text_profile → {silver_out}")
    df_joined.sortWithinPartitions("train_freq", ascending=False) \
             .write.mode(write_mode) \
             .option("compression", "zstd") \
             .partitionBy("popularity_group") \
             .parquet(silver_out)

    # Đọc lại và cache
    result_df = spark.read.parquet(silver_out).cache()
    count = result_df.count()
    logger.info(f"✅ [Step2] silver_item_text_profile: {count:,} items")

    result_df.groupBy("text_source_level").count().orderBy("text_source_level").show()

    return result_df
