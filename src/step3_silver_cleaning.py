"""
Bước 3 — Silver Cleaning + Signal Scoring (Optimized CS246 & MapReduce)

Áp dụng Sort-Merge Join, Phân tán Max Aggregation, và Hashing.
Đảm bảo không có action ẩn giữa DAG, summary tính trên dữ liệu Parquet đã ghi.
"""

import json
import logging
import math
import time  # <--- Thêm dòng này
from datetime import datetime, timedelta
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

logger = logging.getLogger(__name__)


def _join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"

def _silver_interactions_path(cfg: dict, config_id: str) -> str:
    return _join_uri(cfg["paths"]["silver_base"], "interactions", f"config_id={config_id}")


# ── Đọc Bronze (Tận dụng cơ chế Pushdown) ─────────────────────────────────────

def read_bronze_reviews(spark: SparkSession, cfg: dict) -> DataFrame:
    path = _join_uri(cfg["paths"]["bronze_base"], "reviews")
    logger.info(f"  Đọc bronze reviews: {path}")
    return spark.read.parquet(path)

def read_bronze_metadata(spark: SparkSession, cfg: dict) -> DataFrame:
    path = _join_uri(cfg["paths"]["bronze_base"], "metadata")
    logger.info(f"  Đọc bronze metadata: {path}")
    return spark.read.parquet(path)


# ── Làm sạch + Tối ưu Join (Sort-Merge Join) ──────────────────────────────────

def clean_and_join(df_reviews: DataFrame, df_metadata: DataFrame, cfg: dict) -> DataFrame:
    sv_cfg = cfg.get("silver", {})
    max_year = sv_cfg.get("max_year", 2025)
    short_len = sv_cfg.get("short_review_len", 20)

    # 1. Chuẩn bị Metadata & Drop Duplicates
    meta_cols = ["parent_asin", "item_title", "brand", "main_category", "features", "price_bucket"]
    df_meta = df_metadata.select(meta_cols).dropDuplicates(["parent_asin"])

    # 2. Làm sạch Reviews cơ bản
    df_rev = df_reviews.filter(
        F.col("rating").isNotNull() & (F.col("rating") >= 1.0) & (F.col("rating") <= 5.0) &
        F.col("timestamp").isNotNull()
    )
    
    # Tạo cột time (dữ liệu timestamp ở bước 1 đã chuẩn hóa về giây)
    df_rev = df_rev.withColumn("review_time", F.to_timestamp(F.from_unixtime(F.col("timestamp"))))
    df_rev = df_rev.filter(F.year("review_time") <= max_year)
    
    df_rev = (
        df_rev.withColumn("review_text", F.coalesce(F.col("review_text"), F.lit("")))
              .withColumn("helpful_vote", F.coalesce(F.col("helpful_vote"), F.lit(0)))
              .withColumn("verified_purchase", F.coalesce(F.col("verified_purchase"), F.lit(False)))
    )

    # 3. KỸ THUẬT CS246: CHUẨN BỊ SORT-MERGE JOIN
    # Ép Spark băm (Hash) dữ liệu về các node theo khóa parent_asin trước khi nối
    # Điều này loại bỏ hoàn toàn các cú Shuffle ngẫu nhiên đắt đỏ.
    logger.info("  Thực hiện Repartition theo khóa để kích hoạt Sort-Merge Join...")
    df_rev = df_rev.repartition("parent_asin")
    df_meta = df_meta.repartition("parent_asin")
    
    df = df_rev.join(df_meta, on="parent_asin", how="left")

    # 4. Fill null sau Join
    df = (
        df.withColumn("brand", F.when(F.col("brand").isNull() | (F.col("brand") == ""), F.lit("unknown")).otherwise(F.col("brand")))
          .withColumn("main_category", F.when(F.col("main_category").isNull() | (F.col("main_category") == ""), F.lit("unknown")).otherwise(F.col("main_category")))
          .withColumn("features", F.coalesce(F.col("features"), F.lit("")))
          .withColumn("price_bucket", F.when(F.col("price_bucket").isNull() | (F.col("price_bucket") == ""), F.lit("unknown")).otherwise(F.col("price_bucket")))
          .withColumn("item_title", F.coalesce(F.col("item_title"), F.lit("")))
    )

    # 5. Quality flags (Sử dụng Hash để check duplicate)
    df = df.withColumn("is_short_review", (F.col("text_len") < short_len).cast("boolean"))
    df = df.withColumn("is_unverified", (~F.col("verified_purchase")).cast("boolean"))

    # Vì dữ liệu ĐÃ được chia theo parent_asin ở trên, Window function này sẽ KHÔNG gây shuffle I/O nữa
    df = df.withColumn("_text_hash", F.xxhash64(F.col("review_text")))
    dup_window = Window.partitionBy("parent_asin", "_text_hash")
    df = df.withColumn("is_duplicate_text", (F.count("*").over(dup_window) > 1).cast("boolean")).drop("_text_hash")

    return df


# ── Signal Scores (Phân tán phép Aggregation) ─────────────────────────────────

def compute_signal_scores(df: DataFrame, rel_cfg: dict) -> DataFrame:
    w_verified       = float(rel_cfg["w_verified"])
    w_text           = float(rel_cfg["w_text"])
    w_helpful        = float(rel_cfg["w_helpful"])
    unverified_value = float(rel_cfg["unverified_value"])

    df = df.withColumn("verified_score", F.when(F.col("verified_purchase") == True, F.lit(1.0)).otherwise(F.lit(unverified_value)))
    df = df.withColumn("text_quality", F.least(F.col("text_len") / F.lit(100.0), F.lit(1.0)))

    # TỐI ƯU SPARK: Tránh dùng Action (.collect) giữa luồng chạy
    # Ta tính max_helpful cục bộ theo từng tháng. Vừa phân tán được phép tính, vừa chuẩn hóa điểm công bằng hơn theo thời gian.
    w_month = Window.partitionBy("year_month")
    df = df.withColumn("max_helpful_month", F.max("helpful_vote").over(w_month))

    df = df.withColumn(
        "helpful_score",
        F.when(F.col("max_helpful_month") == 0, F.lit(0.0))
         .otherwise(F.log1p(F.col("helpful_vote").cast("double")) / F.log1p(F.col("max_helpful_month").cast("double")))
    ).drop("max_helpful_month")

    df = df.withColumn(
        "reliability_score",
        F.lit(w_verified) * F.col("verified_score") +
        F.lit(w_text) * F.col("text_quality") +
        F.lit(w_helpful) * F.col("helpful_score")
    )

    df = df.withColumn(
        "polarity_score",
        F.when(F.col("rating") == 5.0,  F.lit( 1.0))
         .when(F.col("rating") == 4.0,  F.lit( 0.5))
         .when(F.col("rating") == 3.0,  F.lit( 0.0))
         .when(F.col("rating") == 2.0,  F.lit(-0.5))
         .otherwise(F.lit(-1.0))
    )

    df = df.withColumn("strength_score", F.abs(F.col("rating") - F.lit(3.0)) / F.lit(2.0))

    return df


# ── Ghi output + summary (Tối ưu I/O) ─────────────────────────────────────────

def write_silver_interactions(df: DataFrame, cfg: dict, config_id: str) -> None:
    out_path = _silver_interactions_path(cfg, config_id)
    n_parts = int(cfg["spark"].get("write_partitions", 0))
    mode = cfg["silver"].get("write_mode", "overwrite")

    # Gom nhóm theo year_month để chống tạo ra nhiều file nhỏ
    logger.info(f"  Ghi silver interactions (ZSTD) → {out_path}")
    df_out = df.repartition(n_parts, "year_month") if n_parts > 0 else df.repartition("year_month")

    (
        df_out.write
        .mode(mode)
        .partitionBy("year_month")
        .option("compression", "zstd")
        .parquet(out_path)
    )
    logger.info(f"  ✓ Đã ghi xong Data xuống Silver.")


def save_silver_summary(df_saved: DataFrame, cfg: dict, config_id: str, rel_cfg: dict) -> dict:
    """Tính summary trên DataFrame đã được đọc trực tiếp từ ổ cứng (siêu nhanh)"""
    agg_row = df_saved.agg(
        F.count("*").alias("total_rows"),
        F.round(F.mean(F.col("verified_purchase").cast("int")), 4).alias("verified_ratio"),
        F.round(F.mean(F.col("is_short_review").cast("int")), 4).alias("short_review_ratio"),
        F.round(F.mean(F.col("is_duplicate_text").cast("int")), 4).alias("duplicate_text_ratio"),
        F.round(F.mean("reliability_score"), 4).alias("avg_reliability_score"),
        F.min("year_month").alias("month_min"),
        F.max("year_month").alias("month_max"),
    ).collect()[0]

    rating_dist = {str(int(r["rating"])): int(r["count"]) for r in df_saved.groupBy("rating").count().orderBy("rating").collect()}

    summary = {
        "config_id": config_id,
        "weights": {k: rel_cfg[k] for k in ["w_verified", "w_text", "w_helpful", "unverified_value"]},
        "total_rows": int(agg_row["total_rows"]),
        "verified_ratio": float(agg_row["verified_ratio"] or 0),
        "short_review_ratio": float(agg_row["short_review_ratio"] or 0),
        "duplicate_text_ratio": float(agg_row["duplicate_text_ratio"] or 0),
        "avg_reliability_score": float(agg_row["avg_reliability_score"] or 0),
        "rating_distribution": rating_dist,
        "month_min": agg_row["month_min"],
        "month_max": agg_row["month_max"],
        "generated_at": datetime.now().isoformat(),
    }

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    fpath = log_dir / f"silver_summary_{config_id}.json"
    fpath.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession, cfg: dict, rel_cfg: dict, config_id: str) -> DataFrame:
    t_start = time.perf_counter()
    logger.info(f"=== Bước 3: Silver Cleaning (CS246 Optimized) — config_id={config_id} ===")

    # 1. Định nghĩa DAG
    df_reviews = read_bronze_reviews(spark, cfg)
    df_metadata = read_bronze_metadata(spark, cfg)
    
    df = clean_and_join(df_reviews, df_metadata, cfg)
    df = compute_signal_scores(df, rel_cfg)

    # 2. Thực thi toàn bộ DAG và ghi xuống đĩa 1 lần duy nhất
    write_silver_interactions(df, cfg, config_id)

    # 3. Đọc ngược lại dữ liệu đã nén Parquet để tính Summary (Tận dụng sức mạnh metadata Parquet)
    out_path = _silver_interactions_path(cfg, config_id)
    df_saved = spark.read.parquet(out_path)
    
    logger.info("  Tính summary metrics từ file đã lưu...")
    summary = save_silver_summary(df_saved, cfg, config_id, rel_cfg)
    logger.info(f"  ✓ Bước 3 xong: {summary['total_rows']:,} rows | avg_reliability={summary['avg_reliability_score']:.4f}")

    # Kết thúc đếm thời gian và in kết quả
    t_end = time.perf_counter()
    elapsed_seconds = t_end - t_start
    elapsed_formatted = str(timedelta(seconds=int(elapsed_seconds)))
    logger.info(f"THỜI GIAN CHẠY BƯỚC 3: {elapsed_formatted} ({elapsed_seconds:.2f} giây)")
    return df_saved