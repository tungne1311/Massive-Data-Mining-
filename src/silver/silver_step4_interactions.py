"""
silver_step4_interactions.py — Silver Layer: Enriched Interactions

TỐI ƯU v2:
  - Nhận df_train, df_val từ orchestrator (không đọc lại)
  - Bỏ count() action thừa (tiết kiệm 5-10 phút mỗi split)
  - Dùng coalesce thay repartition (tránh shuffle khi ghi)
  - ZSTD compression
  - Không cache kết quả cuối (data đã trên MinIO, không dùng in-memory nữa)
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType  # FloatType đã bỏ (edge_weight không còn dùng)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _s3a_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


# ─────────────────────────────────────────────────────────────────────────────
# CORE: ENRICH INTERACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_interactions(
    df_interactions: DataFrame,
    df_popularity: DataFrame,
    split_name: str,
) -> DataFrame:
    """
    Broadcast join interactions với popularity.
    # edge_weight = rating / 5.0 đã bị comment — Gold Step 2 chỉ dùng (user_idx, item_idx),
    # LightGCN dùng GCN degree normalization (D^{-1/2}) tính riêng trong Colab.
    Không gọi count() — tránh trigger DAG không cần thiết.
    """
    logger.info(f"⏳ [Step4] Enrich {split_name} interactions...")

    # Projection pushdown
    interaction_cols = ["reviewer_id", "parent_asin", "rating", "timestamp", "helpful_vote"]
    df_slim = df_interactions.select(*interaction_cols)

    # Broadcast popularity (~50MB)
    pop_broadcast = df_popularity.select(
        "parent_asin", "train_freq", "popularity_group"
    )

    df_enriched = df_slim.join(
        F.broadcast(pop_broadcast), on="parent_asin", how="left"
    ).withColumn(
        "popularity_group",
        F.coalesce(F.col("popularity_group"), F.lit("COLD_START"))
    ).withColumn(
        "train_freq",
        F.coalesce(F.col("train_freq"), F.lit(0).cast(LongType()))
    # ).withColumn(
    #     "edge_weight",                                          # KHÔNG DÙNG:
    #     (F.col("rating").cast(FloatType()) / F.lit(5.0))        # Gold Step 2 drop cột này;
    #         .cast(FloatType())                                  # LightGCN dùng GCN normalization
    # ).withColumn(                                               # (D^{-1/2}) tính riêng ở Colab.
    ).withColumn(
        "year_month",
        F.date_format(
            F.from_unixtime(F.col("timestamp").cast(LongType())),
            "yyyy-MM"
        )
    )

    return df_enriched


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
        df_train: bronze_train từ orchestrator.
        df_val: bronze_val từ orchestrator.
    """
    silver_train_out = _s3a_path(cfg, "silver", "silver_interactions_train.parquet")
    silver_val_out   = _s3a_path(cfg, "silver", "silver_interactions_val.parquet")
    write_mode       = cfg.get("silver", {}).get("write_mode", "overwrite")

    # ── Enrich + Write Train ──────────────────────────────────────────────────
    df_train_enriched = _enrich_interactions(df_train, df_popularity, "train")

    logger.info(f"⏳ [Step4] Ghi silver_interactions_train → {silver_train_out}")
    df_train_enriched \
        .coalesce(20) \
        .sortWithinPartitions("reviewer_id", "timestamp") \
        .write.mode(write_mode) \
        .option("compression", "zstd") \
        .parquet(silver_train_out)
    logger.info("✅ [Step4] silver_interactions_train ghi xong.")

    # ── Enrich + Write Val ────────────────────────────────────────────────────
    df_val_enriched = _enrich_interactions(df_val, df_popularity, "val")

    logger.info(f"⏳ [Step4] Ghi silver_interactions_val → {silver_val_out}")
    df_val_enriched \
        .coalesce(5) \
        .sortWithinPartitions("reviewer_id", "timestamp") \
        .write.mode(write_mode) \
        .option("compression", "zstd") \
        .parquet(silver_val_out)
    logger.info("✅ [Step4] silver_interactions_val ghi xong.")
