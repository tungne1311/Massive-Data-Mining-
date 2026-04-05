import logging
import time
from typing import Optional
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

def _get_feature_base_path(cfg: dict, config_id: str, table: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{bucket}/feature_store_{config_id}/{table}"

def _write_feature_optimized(df: DataFrame, cfg: dict, config_id: str, table: str) -> None:
    out_path = _get_feature_base_path(cfg, config_id, table)
    logger.info(f"    -> Đang ghi xuống MinIO: {out_path}")
    
    (df.write.mode("overwrite")
       .option("compression", "zstd")
       .parquet(out_path))

def _top_k_string_optimized(df: DataFrame, group_col: str, value_col: str, k: int, out_col: str) -> DataFrame:
    df_clean = df.filter(F.col(value_col).isNotNull() & (F.col(value_col) != "") & (F.col(value_col) != "unknown"))
    df_cnt = df_clean.groupBy(group_col, value_col).count()
    window_spec = Window.partitionBy(group_col).orderBy(F.desc("count"))
    
    df_topk = df_cnt.withColumn("rank", F.row_number().over(window_spec)) \
                    .filter(F.col("rank") <= k) \
                    .groupBy(group_col) \
                    .agg(F.concat_ws("|", F.collect_list(value_col)).alias(out_col))
    return df_topk

def build_user_features_optimized(df_train: DataFrame, global_avg_rating: float) -> DataFrame:
    df_base = df_train.groupBy("reviewer_id").agg(
        F.count("*").alias("user_total_reviews"),
        F.sum(F.col("interaction_type").contains("positive").cast("int")).alias("user_total_pos"),
        F.round(F.avg("rating"), 4).alias("user_avg_rating_given"),
        F.round(F.avg("rating") - F.lit(global_avg_rating), 4).alias("user_strictness_score"),
        F.round(F.avg(F.col("verified_purchase").cast("int")), 4).alias("user_verified_ratio"),
        F.round(F.avg(F.col("is_short_review").cast("int")), 4).alias("user_short_review_ratio")
    )

    df_fav_cat = _top_k_string_optimized(df_train, "reviewer_id", "main_category", 3, "fav_categories")
    
    # Chỉ join với df_fav_cat, đã bỏ qua df_fav_store
    return df_base.join(df_fav_cat, "reviewer_id", "left")

def build_item_features_optimized(df_train: DataFrame) -> DataFrame:
    df_item = df_train.groupBy("parent_asin").agg(
        F.count("*").alias("item_cnt"),
        F.sum(F.when(F.col("interaction_type").contains("positive"), 1).otherwise(0)).alias("item_positive_cnt"),
        F.round(F.avg("rating"), 4).alias("item_avg_rating"),
        F.min("review_time").alias("first_review_date")
    )
    
    z = F.lit(1.96)
    z2 = z * z
    n = F.col("item_cnt")
    p = F.col("item_positive_cnt") / n
    wilson_score = (p + z2/(2*n) - z * F.sqrt((p*(1-p) + z2/(4*n))/n)) / (1 + z2/n)
    
    return df_item.withColumn("item_wilson_score", F.round(wilson_score, 4)).drop("item_positive_cnt")

def run(spark: SparkSession, cfg: dict, config_id: str, df_train: Optional[DataFrame] = None) -> dict:
    t_start = time.perf_counter()
    logger.info(f"=== Bước 6: Feature Store (Super Lean Sequential) ===")

    if df_train is None:
        splits_base = cfg["paths"]["splits_base"].rstrip("/")
        train_path = f"{splits_base}/config_id={config_id}/train"
        
        logger.info(f"  Đang load tập Train từ: {train_path}")
        df_train = spark.read.parquet(train_path)


    cols_to_keep = ["reviewer_id", "parent_asin", "rating", "interaction_type", 
                    "verified_purchase", "is_short_review", "main_category", "review_time"]
    
    df_train = df_train.select(cols_to_keep)
    df_train.persist(StorageLevel.DISK_ONLY)
    
    global_avg_rating = df_train.select(F.avg("rating")).first()[0]
    logger.info(f"  Global average rating: {global_avg_rating:.4f}")

    logger.info("  [1/2] Bắt đầu xử lý User Features...")
    df_user = build_user_features_optimized(df_train, global_avg_rating)
    _write_feature_optimized(df_user, cfg, config_id, "user_features")
    user_path = _get_feature_base_path(cfg, config_id, "user_features")
    logger.info("  ✓ Xong User Features!")

    logger.info("  [2/2] Bắt đầu xử lý Item Features...")
    df_item = build_item_features_optimized(df_train)
    _write_feature_optimized(df_item, cfg, config_id, "item_features")
    item_path = _get_feature_base_path(cfg, config_id, "item_features")
    logger.info("  ✓ Xong Item Features!")

    df_train.unpersist()

    elapsed = time.perf_counter() - t_start
    logger.info(f"  ✓ Bước 6 hoàn tất trong {elapsed:.2f}s")
    
    return {
        "user_path": user_path,
        "item_path": item_path
    }