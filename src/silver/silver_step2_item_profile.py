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
from pyspark.sql.types import FloatType, IntegerType
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
STORE_MAX_CHARS       = 120

ITEM_REVIEW_TOP_K     = 2
REVIEW_TITLE_CHARS    = 100
REVIEW_TEXT_CHARS     = 180
RECENCY_HALF_LIFE_DAYS = 365.0


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _s3a_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


def _null_string() -> Column:
    return F.lit(None).cast("string")


def _tagged_field(label: str, value_col: Column) -> Column:
    clean_value = F.trim(F.coalesce(value_col, F.lit("")))
    return F.when(
        F.length(clean_value) > 0,
        F.concat(F.lit(f"{label}: "), clean_value),
    ).otherwise(_null_string())


def _timestamp_days_scale(max_ts: float) -> float:
    # Amazon sources may expose timestamps in seconds or milliseconds.
    return 86_400_000.0 if max_ts and max_ts > 10_000_000_000 else 86_400.0


def _build_item_review_profiles(
    df_train_reviews: DataFrame,
    shuffle_parts: int,
    top_k: int = ITEM_REVIEW_TOP_K,
) -> DataFrame:
    """
    Build train-only positive/negative item review snippets.

    These fields enrich warm items only. Cold items keep metadata-only text, so
    cold-start evaluation is not leaked by future val/test reviews.
    """
    logger.info(f"[Step2] Build train-only item review text profiles, top-{top_k} per sentiment...")

    clean_title = advanced_clean_text(F.coalesce(F.col("review_title"), F.lit("")))
    clean_text = advanced_clean_text(F.coalesce(F.col("review_text"), F.lit("")))

    title_part = F.substring(clean_title, 1, REVIEW_TITLE_CHARS)
    text_part = F.substring(clean_text, 1, REVIEW_TEXT_CHARS)
    snippet_col = F.when(
        (F.length(clean_title) > 0) & (F.length(clean_text) > 0),
        F.concat_ws(" - ", title_part, text_part),
    ).when(
        F.length(clean_text) > 0, text_part
    ).when(
        F.length(clean_title) > 0, title_part
    ).otherwise(F.lit(""))

    df_base = df_train_reviews.select(
        "parent_asin",
        "rating",
        "timestamp",
        "helpful_vote",
        snippet_col.alias("review_snippet"),
    ).filter(
        ((F.col("rating") >= 4.0) | (F.col("rating") <= 2.0))
        & (F.length(F.col("review_snippet")) > 0)
    ).withColumn(
        "sentiment",
        F.when(F.col("rating") >= 4.0, F.lit("pos")).otherwise(F.lit("neg"))
    )

    max_ts_row = df_base.agg(F.max(F.col("timestamp").cast("double")).alias("max_ts")).first()
    max_ts = float(max_ts_row["max_ts"] or 0.0)
    ts_per_day = _timestamp_days_scale(max_ts)

    timestamp_d = F.coalesce(F.col("timestamp").cast("double"), F.lit(0.0))
    age_days = F.greatest(
        (F.lit(max_ts) - timestamp_d) / F.lit(ts_per_day),
        F.lit(0.0),
    )
    recency_weight = F.exp(-age_days / F.lit(RECENCY_HALF_LIFE_DAYS))
    helpful_weight = F.lit(1.0) + F.log(
        F.lit(1.0) + F.coalesce(F.col("helpful_vote").cast(FloatType()), F.lit(0.0))
    )

    df_struct = df_base.withColumn(
        "review_score",
        recency_weight * helpful_weight,
    ).withColumn(
        "review_struct",
        F.struct(
            F.col("review_score").alias("score"),
            F.col("timestamp").alias("t"),
            F.col("review_snippet").alias("s"),
        )
    ).repartition(shuffle_parts, "parent_asin")

    df_grouped = df_struct.groupBy("parent_asin", "sentiment").agg(
        F.collect_list("review_struct").alias("reviews_raw"),
        F.count("*").alias("review_count"),
    ).withColumn(
        "reviews_topk",
        F.slice(F.sort_array(F.col("reviews_raw"), asc=False), 1, top_k)
    ).withColumn(
        "profile_text",
        F.array_join(
            F.transform(F.col("reviews_topk"), lambda x: x.getField("s")),
            " [SEP] ",
        )
    )

    df_pos = df_grouped.filter(F.col("sentiment") == "pos").select(
        "parent_asin",
        F.col("profile_text").alias("item_review_pos_text"),
        F.col("review_count").alias("item_review_pos_count"),
    )
    df_neg = df_grouped.filter(F.col("sentiment") == "neg").select(
        "parent_asin",
        F.col("profile_text").alias("item_review_neg_text"),
        F.col("review_count").alias("item_review_neg_count"),
    )

    return df_pos.join(df_neg, on="parent_asin", how="full")


# ─────────────────────────────────────────────────────────────────────────────
# CORE: XÂY DỰNG ITEM TEXT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def _build_item_text(df_meta: DataFrame, df_item_reviews: DataFrame | None = None) -> DataFrame:
    """
    Ghép chuỗi văn bản item theo 4 cấp độ ưu tiên, dùng Spark SQL built-ins.
    Khi description rỗng → mở rộng features budget.
    """

    def clean_and_safe_col(col_name: str) -> F.Column:
        return advanced_clean_text(F.coalesce(F.col(col_name), F.lit("")))

    df = df_meta
    if df_item_reviews is not None:
        df = df.join(df_item_reviews, on="parent_asin", how="left")
    else:
        df = (
            df.withColumn("item_review_pos_text", _null_string())
              .withColumn("item_review_neg_text", _null_string())
              .withColumn("item_review_pos_count", F.lit(0).cast("long"))
              .withColumn("item_review_neg_count", F.lit(0).cast("long"))
        )

    title_part = _tagged_field(
        "title",
        F.substring(clean_and_safe_col("title"), 1, TITLE_MAX_CHARS),
    )
    main_category_part = _tagged_field(
        "main_category",
        F.substring(clean_and_safe_col("main_category"), 1, CATEGORIES_MAX_CHARS),
    )
    store_part = _tagged_field(
        "store",
        F.substring(clean_and_safe_col("store"), 1, STORE_MAX_CHARS),
    )

    # Features: mở rộng khi description rỗng
    features_extended_part = F.when(
        (F.length(clean_and_safe_col("features")) > 0) & (F.length(clean_and_safe_col("description")) == 0),
        F.substring(clean_and_safe_col("features"), 1, FEATURES_EXTENDED_CHARS)
    ).when(
        F.length(clean_and_safe_col("features")) > 0,
        F.substring(clean_and_safe_col("features"), 1, FEATURES_MAX_CHARS)
    ).otherwise(F.lit(""))
    features_part = _tagged_field("features", features_extended_part)

    categories_part = _tagged_field("categories", F.when(
        F.length(clean_and_safe_col("categories")) > 0,
        F.substring(clean_and_safe_col("categories"), 1, CATEGORIES_MAX_CHARS)
    ).otherwise(F.lit("")))

    description_part = _tagged_field("description", F.when(
        F.length(clean_and_safe_col("description")) > 0,
        F.substring(clean_and_safe_col("description"), 1, DESCRIPTION_MAX_CHARS)
    ).otherwise(F.lit("")))

    details_part = _tagged_field("details", F.when(
        F.length(clean_and_safe_col("details")) > 0,
        F.substring(clean_and_safe_col("details"), 1, DETAILS_MAX_CHARS)
    ).otherwise(F.lit("")))

    review_pos_part = _tagged_field(
        "positive_reviews",
        F.coalesce(F.col("item_review_pos_text"), F.lit("")),
    ) if "item_review_pos_text" in df.columns else _null_string()
    review_neg_part = _tagged_field(
        "negative_reviews",
        F.coalesce(F.col("item_review_neg_text"), F.lit("")),
    ) if "item_review_neg_text" in df.columns else _null_string()

    # Field tags help the sentence encoder preserve each source's role.
    df = df.withColumn(
        "item_text",
        F.concat_ws(
            " [SEP] ",
            title_part,
            main_category_part,
            store_part,
            categories_part,
            features_part,
            description_part,
            details_part,
            review_pos_part,
            review_neg_part,
        )
    ).withColumn(
        "item_text",
        F.when(F.length(F.col("item_text")) > 5, F.col("item_text"))
         .otherwise(F.lit("[NO_TEXT] item metadata unavailable"))
    )

    def safe_col(col_name: str) -> F.Column:
        return F.coalesce(F.trim(F.col(col_name)), F.lit(""))

    # text_source_level cho debug
    df = df.withColumn(
        "text_source_level",
        F.when(
            (F.length(F.coalesce(F.col("item_review_pos_text"), F.lit(""))) > 0)
            | (F.length(F.coalesce(F.col("item_review_neg_text"), F.lit(""))) > 0),
            F.lit(5)
        ).when(
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
        "item_text", "item_review_pos_text", "item_review_neg_text",
        "item_review_pos_count", "item_review_neg_count",
        "text_source_level", "token_estimate",
        # NOTE: average_rating và rating_number được BỎ để tránh data leakage tiềm ẩn.
        # Cả hai tính từ TOÀN BỘ reviews (bao gồm val/test tương lai).
        # popularity_group + train_freq đã đủ cho model training.
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    df_popularity: DataFrame,
    meta_path: str,
    df_train_reviews: DataFrame | None = None,
) -> DataFrame:
    """
    Args:
        df_popularity: broadcast-ready popularity DataFrame.
        meta_path: đường dẫn s3a:// tới bronze_meta (từ orchestrator).
    """
    silver_out = _s3a_path(cfg, "silver", "silver_item_text_profile.parquet")
    write_mode = cfg.get("silver", {}).get("write_mode", "overwrite")
    shuffle_parts = int(cfg.get("spark", {}).get("shuffle_partitions", 200))
    use_item_review_text = cfg.get("silver", {}).get("use_item_review_text", True)
    item_review_top_k = int(cfg.get("silver", {}).get("item_review_top_k_reviews", ITEM_REVIEW_TOP_K))

    # ── Đọc bronze_meta từ MinIO ──────────────────────────────────────────────
    logger.info(f"⏳ [Step2] Đọc bronze_meta từ MinIO: {meta_path}")
    meta_cols = [
        "parent_asin", "title", "main_category", "store",
        "features", "categories", "description", "details",
        # average_rating + rating_number bị bỏ: chứa future signal (val/test reviews)
    ]
    df_meta = spark.read.parquet(meta_path).select(*meta_cols)

    # ── Xây dựng item text ────────────────────────────────────────────────────
    df_item_reviews = None
    if use_item_review_text and df_train_reviews is not None:
        df_item_reviews = _build_item_review_profiles(
            df_train_reviews,
            shuffle_parts,
            top_k=item_review_top_k,
        )

    logger.info("⏳ [Step2] Ghép item text profile...")
    df_text = _build_item_text(df_meta, df_item_reviews=df_item_reviews)

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
