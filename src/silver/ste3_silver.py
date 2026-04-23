"""
ste3_silver.py — Silver Layer Pipeline Orchestrator

TỐI ƯU v2:
  - Đọc bronze_train/val 1 LẦN DUY NHẤT, truyền DataFrame vào các steps
  - Đọc trực tiếp từ MinIO (không qua HuggingFace)
  - Thứ tự: Step1 → Step2 → Step4 → Step3 (Step3 tốn RAM nhất, chạy sau)
  - ZSTD compression cho toàn bộ Silver output

OUTPUT:
  s3a://recsys/silver/
    ├── silver_item_popularity.parquet
    ├── silver_item_text_profile.parquet
    ├── silver_user_text_profile.parquet
    ├── silver_interactions_train.parquet
    ├── silver_interactions_val.parquet
    └── silver_val_ground_truth.parquet
"""

import gc
import logging
import time
from datetime import timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from . import silver_step1_popularity  as step1
from . import silver_step2_item_profile as step2
from . import silver_step3_user_profile as step3
from . import silver_step4_interactions as step4

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _s3a_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


def _clear_memory(spark: SparkSession, label: str) -> None:
    logger.info(f"🧹 [Memory] Giải phóng RAM sau {label}...")
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def run_silver_pipeline(spark: SparkSession, cfg: dict) -> None:
    """
    Chạy toàn bộ Silver Pipeline.

    Tối ưu chính:
      - bronze_train đọc 1 lần, dùng cho Step 1, 3, 4
      - bronze_val đọc 1 lần, dùng cho Step 3, 4
      - bronze_meta đọc 1 lần cho Step 2
      - ZSTD compression cho tất cả output
    """
    t_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("🚀 BẮT ĐẦU SILVER PIPELINE (v2 — tối ưu I/O)")
    logger.info("=" * 60)

    # ── Đặt ZSTD compression cho toàn bộ Silver output ───────────────────────
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")

    # ── ĐỌC BRONZE 1 LẦN DUY NHẤT ───────────────────────────────────────────
    train_path = _s3a_path(cfg, "bronze", "bronze_train.parquet")
    val_path   = _s3a_path(cfg, "bronze", "bronze_val.parquet")
    meta_path  = _s3a_path(cfg, "bronze", "bronze_meta.parquet")

    # Data đã được lọc rating >= 3.0 trực tiếp từ tầng Bronze
    logger.info(f"📂 Đọc bronze_train từ MinIO: {train_path}")
    df_train_raw = spark.read.parquet(train_path)

    train_cols_light = ["reviewer_id", "parent_asin", "rating", "timestamp", "helpful_vote"]
    df_train_light = df_train_raw.select(*train_cols_light).cache()
    
    train_count = df_train_light.count()
    logger.info(f"  📦 bronze_train_light (positive only >= 3.0): {train_count:,} rows")

    logger.info(f"📂 Đọc bronze_val từ MinIO: {val_path}")
    df_val = spark.read.parquet(val_path)
    # Không cache val — chỉ dùng 2 lần (Step 3, 4) và đủ nhỏ

    # ── STEP 1: Item Popularity ───────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n📌 SILVER STEP 1: Item Popularity Classification")

    df_popularity = step1.run(spark, cfg, df_train=df_train_light)
    # df_popularity đã cache — dùng cho Step 2, 3, 4

    elapsed = timedelta(seconds=int(time.perf_counter() - t0))
    logger.info(f"⏱ Step 1 hoàn tất: {elapsed}")

    # ── STEP 2: Item Text Profile ─────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n📌 SILVER STEP 2: Item Text Profile")

    df_item_text = step2.run(spark, cfg, df_popularity=df_popularity,
                             meta_path=meta_path)
    df_item_text.unpersist()

    elapsed = timedelta(seconds=int(time.perf_counter() - t0))
    logger.info(f"⏱ Step 2 hoàn tất: {elapsed}")

    # ── STEP 4: Enrich Interactions (trước Step 3 vì nhẹ hơn) ─────────────────
    t0 = time.perf_counter()
    logger.info("\n📌 SILVER STEP 4: Enrich Interactions")

    step4.run(spark, cfg, df_popularity=df_popularity,
              df_train=df_train_light, df_val=df_val)

    elapsed = timedelta(seconds=int(time.perf_counter() - t0))
    logger.info(f"⏱ Step 4 hoàn tất: {elapsed}")

    # ── Giải phóng bronze_train (light) trước Step 3 (trả RAM cho shuffle) ────
    df_train_light.unpersist()
    _clear_memory(spark, "trước Step 3")

    # ── STEP 3: User Text Profile (tốn RAM nhất) ──────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n📌 SILVER STEP 3: User Text Profile (heavy step)")

    # Đọc lại train CHỈ CỘT CẦN cho Step 3 (giảm RAM)
    train_cols_text = ["reviewer_id", "parent_asin", "timestamp", "rating",
                       "review_title", "review_text", "helpful_vote"]
    df_train_text = spark.read.parquet(train_path).select(*train_cols_text)

    step3.run(spark, cfg, df_popularity=df_popularity,
              df_train=df_train_text, df_val=df_val)

    elapsed = timedelta(seconds=int(time.perf_counter() - t0))
    logger.info(f"⏱ Step 3 hoàn tất: {elapsed}")

    # ── CLEANUP ───────────────────────────────────────────────────────────────
    df_popularity.unpersist()
    _clear_memory(spark, "kết thúc Silver pipeline")

    total = timedelta(seconds=int(time.perf_counter() - t_start))
    logger.info("=" * 60)
    logger.info(f"🎉 SILVER PIPELINE HOÀN TẤT — ⏱ TỔNG: {total}")
    logger.info("=" * 60)
    logger.info("OUTPUT:")
    logger.info("  ✅ silver_item_popularity.parquet")
    logger.info("  ✅ silver_item_text_profile.parquet")
    logger.info("  ✅ silver_user_text_profile.parquet")
    logger.info("  ✅ silver_interactions_train.parquet")
    logger.info("  ✅ silver_interactions_val.parquet")
    logger.info("  ✅ silver_val_ground_truth.parquet")
