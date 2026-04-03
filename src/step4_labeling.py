"""
Bước 4 — Labeling
"""

import logging
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

def _join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"

def _labeled_path(cfg: dict, config_id: str) -> str:
    return _join_uri(
        cfg["paths"]["silver_base"],
        "labeled_interactions",
        f"config_id={config_id}",
    )

def assign_interaction_type(df: DataFrame) -> DataFrame:
    vp  = F.col("verified_purchase")
    rat = F.col("rating")

    df = df.withColumn(
        "interaction_type",
        F.when(vp & (rat >= 4),                                    F.lit("strong_positive"))
         .when((~vp & (rat >= 4)) | (vp & (rat == 3)),             F.lit("weak_positive"))
         .when(vp & (rat == 2),                                     F.lit("medium_negative"))
         .when(vp & (rat == 1),                                     F.lit("hard_negative"))
         .otherwise(F.lit("neutral"))
    )
    return df

def assign_bpr_role(df: DataFrame) -> DataFrame:
    it = F.col("interaction_type")

    df = df.withColumn(
        "bpr_role",
        F.when(it.isin("strong_positive", "weak_positive"), F.lit("positive"))
         .when(it.isin("medium_negative", "hard_negative"), F.lit("hard_negative"))
         .otherwise(F.lit("excluded"))
    )
    return df

def assign_bpr_positive_weight(df: DataFrame) -> DataFrame:
    it = F.col("interaction_type")

    df = df.withColumn(
        "bpr_positive_weight",
        F.when(it == "strong_positive", F.lit(1.0))
         .when(it == "weak_positive",   F.lit(0.5))
         .otherwise(F.lit(0.0))
    )
    return df

def assign_relevance_label(df: DataFrame) -> DataFrame:
    it = F.col("interaction_type")

    df = df.withColumn(
        "relevance_label",
        F.when(it == "strong_positive", F.lit(3))
         .when(it == "weak_positive",   F.lit(2))
         .when(it == "neutral",         F.lit(1))
         .otherwise(F.lit(0))
    )
    return df

def write_labeled_interactions(
    df: DataFrame,
    cfg: dict,
    config_id: str,
) -> None:
    out_path = _labeled_path(cfg, config_id)
    logger.info(f"  Ghi labeled interactions → {out_path}")
    
    # 🚀 TỐI ƯU 1: Xóa repartition gây OOM, thay bằng sortWithinPartitions
    df_out = df.sortWithinPartitions("year_month")

    (
        df_out
          .write
          .mode("overwrite") # Ép ghi đè kết hợp với cơ chế dynamic ở hàm run
          .partitionBy("year_month")
          .option("maxRecordsPerFile", 250000) # Đảm bảo file nhỏ gọn như Step 3
          .option("compression", "zstd")
          .parquet(out_path)
    )
    logger.info(f"  ✓ Ghi xong labeled: {out_path}")

def run(
    spark: SparkSession,
    cfg: dict,
    config_id: str,
    df_silver: DataFrame = None,
) -> DataFrame:
    logger.info(f"=== Bước 4: Labeling (Memory Optimized) — config_id={config_id} ===")

    # 🚀 TỐI ƯU 2: Bật Dynamic Overwrite để an toàn khi chạy lại
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    if df_silver is None:
        silver_path = f"s3a://{cfg['minio']['bucket'].strip('/')}/silver/{config_id}"
        logger.info(f"  Đọc silver từ disk: {silver_path}")
        df_silver = spark.read.parquet(silver_path)

    df = assign_interaction_type(df_silver)
    df = assign_bpr_role(df)
    df = assign_bpr_positive_weight(df)
    df = assign_relevance_label(df)

    # Chặn lineage bằng cách đẩy xuống ổ đĩa, giải phóng RAM
    df.persist(StorageLevel.DISK_ONLY)

    logger.info("  Tính phân phối interaction_type...")
    dist = {
        r["interaction_type"]: r["count"]
        for r in df.groupBy("interaction_type").count()
                   .orderBy("interaction_type").collect()
    }
    for k, v in dist.items():
        logger.info(f"    {k}: {v:,}")

    write_labeled_interactions(df, cfg, config_id)
    
    # Dọn dẹp dung lượng tạm
    df.unpersist()

    logger.info(f"  ✓ Bước 4 xong: config_id={config_id}")
    return spark.read.parquet(_labeled_path(cfg, config_id))