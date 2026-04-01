"""
Bước 3 — Silver Cleaning + Signal Scoring

Đọc Bronze reviews + metadata → join → làm sạch → tính signal scores.
Output: silver/interactions/config_id=<id>/year_month=YYYY-MM/
        logs/silver_summary_<config_id>.json
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

logger = logging.getLogger(__name__)


# ── Helpers đường dẫn ─────────────────────────────────────────────────────────

def _join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"


def _silver_interactions_path(cfg: dict, config_id: str) -> str:
    """Path output silver interactions theo config_id."""
    return _join_uri(
        cfg["paths"]["silver_base"],
        "interactions",
        f"config_id={config_id}",
    )


# ── Đọc Bronze ────────────────────────────────────────────────────────────────

def read_bronze_reviews(spark: SparkSession, cfg: dict) -> DataFrame:
    """Đọc toàn bộ bronze reviews từ MinIO."""
    path = _join_uri(cfg["paths"]["bronze_base"], "reviews")
    logger.info(f"  Đọc bronze reviews: {path}")
    return spark.read.parquet(path)


def read_bronze_metadata(spark: SparkSession, cfg: dict) -> DataFrame:
    """Đọc bronze metadata từ MinIO."""
    path = _join_uri(cfg["paths"]["bronze_base"], "metadata")
    logger.info(f"  Đọc bronze metadata: {path}")
    return spark.read.parquet(path)


# ── Làm sạch + join ───────────────────────────────────────────────────────────

def clean_and_join(
    df_reviews: DataFrame,
    df_metadata: DataFrame,
    cfg: dict,
) -> DataFrame:
    """
    Join reviews với metadata theo parent_asin.
    Làm sạch nhẹ: fill null, lọc rating sai, chuẩn hóa timestamp.
    Tạo quality flags.
    """
    sv_cfg = cfg.get("silver", {})
    max_year = sv_cfg.get("max_year", 2025)
    short_len = sv_cfg.get("short_review_len", 20)

    # ── Metadata: bỏ các cột trùng không cần thiết ────────────────────
    meta_cols = ["parent_asin", "item_title", "brand",
                 "main_category", "description", "features", "price_bucket"]
    df_meta = df_metadata.select(meta_cols).dropDuplicates(["parent_asin"])

    # ── Reviews: fill null cơ bản ────────────────────────────────────
    df_rev = (
        df_reviews
        .withColumn("review_text",
                    F.when(F.col("review_text").isNull(), F.lit(""))
                     .otherwise(F.col("review_text")))
        .withColumn("helpful_vote",
                    F.when(F.col("helpful_vote").isNull(), F.lit(0))
                     .otherwise(F.col("helpful_vote")))
        .withColumn("verified_purchase",
                    F.when(F.col("verified_purchase").isNull(), F.lit(False))
                     .otherwise(F.col("verified_purchase")))
    )

    # ── Lọc rating hợp lệ [1, 5] ─────────────────────────────────────
    before_rating = df_rev  # giữ ref để log
    df_rev = df_rev.filter(
        F.col("rating").isNotNull() &
        (F.col("rating") >= 1.0) &
        (F.col("rating") <= 5.0)
    )

    # ── Parse timestamp → review_time ────────────────────────────────
    # timestamp đã được chuẩn hóa sang unix seconds ở bước 2
    df_rev = df_rev.withColumn(
        "review_time",
        F.to_timestamp(F.from_unixtime(F.col("timestamp")))
    )

    # Loại timestamp lỗi (null sau parse hoặc ngoài range hợp lý)
    df_rev = df_rev.filter(
        F.col("review_time").isNotNull() &
        (F.year("review_time") <= max_year)
    )

    # ── year_month ────────────────────────────────────────────────────
    df_rev = df_rev.withColumn(
        "year_month",
        F.date_format(F.col("review_time"), "yyyy-MM")
    )

    # ── Join với metadata ─────────────────────────────────────────────
    df = df_rev.join(df_meta, on="parent_asin", how="left")

    # ── Fill null metadata ────────────────────────────────────────────
    df = (
        df
        .withColumn("brand",
                    F.when(F.col("brand").isNull() | (F.col("brand") == ""),
                           F.lit("unknown")).otherwise(F.col("brand")))
        .withColumn("main_category",
                    F.when(F.col("main_category").isNull() | (F.col("main_category") == ""),
                           F.lit("unknown")).otherwise(F.col("main_category")))
        .withColumn("description",
                    F.when(F.col("description").isNull(), F.lit(""))
                     .otherwise(F.col("description")))
        .withColumn("features",
                    F.when(F.col("features").isNull(), F.lit(""))
                     .otherwise(F.col("features")))
        .withColumn("price_bucket",
                    F.when(F.col("price_bucket").isNull() | (F.col("price_bucket") == ""),
                           F.lit("unknown")).otherwise(F.col("price_bucket")))
        .withColumn("item_title",
                    F.when(F.col("item_title").isNull(), F.lit(""))
                     .otherwise(F.col("item_title")))
    )

    # ── Quality flags ─────────────────────────────────────────────────
    # is_short_review
    df = df.withColumn(
        "is_short_review",
        (F.col("text_len") < short_len).cast("boolean")
    )

    # is_unverified
    df = df.withColumn(
        "is_unverified",
        (~F.col("verified_purchase")).cast("boolean")
    )

    # is_duplicate_text: hash(review_text) trùng trong cùng parent_asin
    df = df.withColumn(
        "_text_hash",
        F.xxhash64(F.col("review_text"))
    )
    dup_window = Window.partitionBy("parent_asin", "_text_hash")
    df = df.withColumn(
        "is_duplicate_text",
        (F.count("*").over(dup_window) > 1).cast("boolean")
    ).drop("_text_hash")

    return df


# ── Signal Scores ─────────────────────────────────────────────────────────────

def compute_signal_scores(df: DataFrame, rel_cfg: dict) -> DataFrame:
    """
    Tính 6 signal scores theo config:
      verified_score, text_quality, helpful_score
      → reliability_score (weighted sum)
      + polarity_score, strength_score
    """
    w_verified       = float(rel_cfg["w_verified"])
    w_text           = float(rel_cfg["w_text"])
    w_helpful        = float(rel_cfg["w_helpful"])
    unverified_value = float(rel_cfg["unverified_value"])

    # ── A. verified_score ─────────────────────────────────────────────
    df = df.withColumn(
        "verified_score",
        F.when(F.col("verified_purchase") == True, F.lit(1.0))
         .otherwise(F.lit(unverified_value))
    )

    # ── B. text_quality ───────────────────────────────────────────────
    df = df.withColumn(
        "text_quality",
        F.least(F.col("text_len") / F.lit(100.0), F.lit(1.0))
    )

    # ── C. helpful_score ──────────────────────────────────────────────
    # max_helpful_vote tính bằng Spark aggregation (không kéo về local)
    max_helpful = df.agg(F.max("helpful_vote").alias("mx")).collect()[0]["mx"] or 0

    if max_helpful == 0:
        logger.info("  max_helpful_vote = 0 → helpful_score = 0.0 cho tất cả")
        df = df.withColumn("helpful_score", F.lit(0.0))
    else:
        log_max = math.log1p(max_helpful)
        df = df.withColumn(
            "helpful_score",
            F.log1p(F.col("helpful_vote").cast("double")) / F.lit(log_max)
        )

    # ── D. reliability_score ──────────────────────────────────────────
    df = df.withColumn(
        "reliability_score",
        F.lit(w_verified) * F.col("verified_score") +
        F.lit(w_text)     * F.col("text_quality") +
        F.lit(w_helpful)  * F.col("helpful_score")
    )

    # ── E. polarity_score ─────────────────────────────────────────────
    df = df.withColumn(
        "polarity_score",
        F.when(F.col("rating") == 5.0,  F.lit( 1.0))
         .when(F.col("rating") == 4.0,  F.lit( 0.5))
         .when(F.col("rating") == 3.0,  F.lit( 0.0))
         .when(F.col("rating") == 2.0,  F.lit(-0.5))
         .otherwise(F.lit(-1.0))
    )

    # ── F. strength_score ─────────────────────────────────────────────
    df = df.withColumn(
        "strength_score",
        F.abs(F.col("rating") - F.lit(3.0)) / F.lit(2.0)
    )

    return df


# ── Ghi output + summary ──────────────────────────────────────────────────────

def write_silver_interactions(
    df: DataFrame,
    cfg: dict,
    config_id: str,
) -> None:
    """Ghi parquet partition theo year_month dưới config_id folder."""
    out_path = _silver_interactions_path(cfg, config_id)
    logger.info(f"  Ghi silver interactions → {out_path}")
    n_parts = int(cfg["spark"].get("write_partitions", 0))
    mode    = cfg["silver"]["write_mode"]
    df_out  = df.repartition(n_parts) if n_parts > 0 else df

    (
        df_out
          .write
          .mode(mode)
          .partitionBy("year_month")
          .parquet(out_path)
    )
    logger.info(f"  ✓ Ghi xong silver: {out_path}")


def save_silver_summary(
    df: DataFrame,
    cfg: dict,
    config_id: str,
    rel_cfg: dict,
) -> dict:
    """
    Tính summary metrics và lưu JSON vào data/logs/.
    Trả về dict summary để dùng tiếp trong grid search.
    """
    # Một lần scan duy nhất để lấy tất cả scalar metrics
    agg_row = df.agg(
        F.count("*").alias("total_rows"),
        F.round(F.mean(F.col("verified_purchase").cast("int")), 4)
         .alias("verified_ratio"),
        F.round(F.mean(F.col("is_short_review").cast("int")), 4)
         .alias("short_review_ratio"),
        F.round(F.mean(F.col("is_duplicate_text").cast("int")), 4)
         .alias("duplicate_text_ratio"),
        F.round(F.mean("reliability_score"), 4).alias("avg_reliability_score"),
        F.min("year_month").alias("month_min"),
        F.max("year_month").alias("month_max"),
    ).collect()[0]

    # Rating distribution — scan thứ 2 (nhỏ, nhẹ)
    rating_dist = {
        str(int(r["rating"])): int(r["count"])
        for r in df.groupBy("rating").count().orderBy("rating").collect()
    }

    summary = {
        "config_id":            config_id,
        "weights": {
            "w_verified":       rel_cfg["w_verified"],
            "w_text":           rel_cfg["w_text"],
            "w_helpful":        rel_cfg["w_helpful"],
            "unverified_value": rel_cfg["unverified_value"],
        },
        "total_rows":           int(agg_row["total_rows"]),
        "verified_ratio":       float(agg_row["verified_ratio"] or 0),
        "short_review_ratio":   float(agg_row["short_review_ratio"] or 0),
        "duplicate_text_ratio": float(agg_row["duplicate_text_ratio"] or 0),
        "avg_reliability_score":float(agg_row["avg_reliability_score"] or 0),
        "rating_distribution":  rating_dist,
        "month_min":            agg_row["month_min"],
        "month_max":            agg_row["month_max"],
        "generated_at":         datetime.now().isoformat(),
    }

    # Ghi file JSON local
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    fpath = log_dir / f"silver_summary_{config_id}.json"
    fpath.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"  Summary ghi → {fpath}")

    # Upload lên MinIO nếu có thể
    _upload_summary_minio(summary, cfg, config_id)

    return summary


def _upload_summary_minio(summary: dict, cfg: dict, config_id: str) -> None:
    """Upload summary JSON lên MinIO (không crash nếu lỗi)."""
    try:
        import boto3
        from botocore.client import Config

        mn = cfg["minio"]
        s3 = boto3.client(
            "s3",
            endpoint_url          = mn["endpoint"],
            aws_access_key_id     = mn["access_key"],
            aws_secret_access_key = mn["secret_key"],
            config                = Config(signature_version="s3v4"),
            region_name           = "us-east-1",
        )
        content = json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8")
        key = f"silver/logs/silver_summary_{config_id}.json"
        s3.put_object(
            Bucket=mn["bucket"], Key=key,
            Body=content, ContentType="application/json"
        )
        logger.info(f"  Summary MinIO → s3a://{mn['bucket']}/{key}")
    except Exception as e:
        logger.warning(f"  Không upload summary lên MinIO: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    rel_cfg: dict,
    config_id: str,
) -> DataFrame:
    """
    Chạy bước 3 cho một config_id cụ thể.
    Trả về DataFrame silver interactions (chưa cache, lazy).
    """
    logger.info(f"=== Bước 3: Silver Cleaning — config_id={config_id} ===")

    df_reviews  = read_bronze_reviews(spark, cfg)
    df_metadata = read_bronze_metadata(spark, cfg)

    logger.info("  Làm sạch + join Bronze...")
    df = clean_and_join(df_reviews, df_metadata, cfg)

    logger.info(f"  Tính signal scores (w_v={rel_cfg['w_verified']}, "
                f"w_t={rel_cfg['w_text']}, w_h={rel_cfg['w_helpful']})...")
    df = compute_signal_scores(df, rel_cfg)

    write_silver_interactions(df, cfg, config_id)

    # Summary cần trigger action (count, agg) — làm sau khi ghi để tận dụng cache OS
    logger.info("  Tính summary metrics...")
    summary = save_silver_summary(df, cfg, config_id, rel_cfg)
    logger.info(
        f"  ✓ Bước 3 xong: {summary['total_rows']:,} rows | "
        f"avg_reliability={summary['avg_reliability_score']:.4f}"
    )

    # Trả về DataFrame để bước 4 dùng tiếp (đọc lại từ disk để đảm bảo nhất quán)
    return spark.read.parquet(_silver_interactions_path(cfg, config_id))
