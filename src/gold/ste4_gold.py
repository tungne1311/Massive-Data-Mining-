"""
ste4_gold.py — Gold Layer Pipeline Orchestrator

Điều phối 4 bước Gold:
  Step 1:   Integer ID Mapping (string → int liên tục)
  Step 2:   Edge List + Temporal Decay (PyG format)
  Step 5:   Training Metadata (negative sampling, dataset stats)

CHIẾN LƯỢC BỘ NHỚ:
  - Đọc Silver data qua Spark, collect về driver cho xử lý numpy/torch
  - Giải phóng Spark DataFrame ngay sau khi collect
  - Embeddings logic removed from here (Perform on Colab/Kaggle with GPU)
  - Step 5:   Training Metadata (negative sampling, dataset stats)

OUTPUT:
  s3a://recsys/gold/
    ├── gold_item_id_map.parquet
    ├── gold_user_id_map.parquet
    ├── gold_edge_index.npy          [2, E]
    ├── gold_item_embeddings.npy     [N_items, 384]
    ├── gold_user_embeddings.npy     [N_users, 384]
    ├── gold_item_train_freq.npy     [N_items]
    ├── gold_item_popularity_group.npy [N_items]
    ├── gold_user_train_freq.npy     [N_users]
    ├── gold_user_activity_group.npy [N_users]
    ├── gold_negative_sampling_prob.npy [N_items]
    └── gold_dataset_stats.json
"""

import gc
import logging
import time
from datetime import timedelta

from pyspark.sql import SparkSession

from . import gold_step1_id_mapping    as step1
from . import gold_step2_edge_list     as step2
# step34 (embeddings) moved to notebook/colab
from . import gold_step5_training_meta as step5

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

def run_gold_pipeline(spark: SparkSession, cfg: dict) -> None:
    """
    Chạy toàn bộ Gold Pipeline.

    Đọc Silver outputs → tạo Gold artifacts sẵn sàng cho PyG training.
    """
    t_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("🚀 BẮT ĐẦU GOLD PIPELINE")
    logger.info("=" * 60)

    # ── Đọc Silver data ───────────────────────────────────────────────────────
    silver_paths = {
        "popularity":        _s3a_path(cfg, "silver", "silver_item_popularity.parquet"),
        "item_text":         _s3a_path(cfg, "silver", "silver_item_text_profile.parquet"),
        "interactions_train": _s3a_path(cfg, "silver", "silver_interactions_train.parquet"),
        "interactions_val":  _s3a_path(cfg, "silver", "silver_interactions_val.parquet"),
    }

    logger.info("📂 Đọc Silver data từ MinIO...")
    for name, path in silver_paths.items():
        logger.info(f"  {name}: {path}")

    df_popularity        = spark.read.parquet(silver_paths["popularity"])
    df_item_text         = spark.read.parquet(silver_paths["item_text"])
    df_interactions_train = spark.read.parquet(silver_paths["interactions_train"])
    df_interactions_val  = spark.read.parquet(silver_paths["interactions_val"])

    # ── STEP 1: Integer ID Mapping ────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n📌 GOLD STEP 1: Integer ID Mapping")

    step1_info = step1.run(
        spark, cfg,
        df_interactions_train=df_interactions_train,
        df_interactions_val=df_interactions_val,
        df_item_text=df_item_text,
        df_popularity=df_popularity,
    )
    
    elapsed = timedelta(seconds=int(time.perf_counter() - t0))
    logger.info(f"⏱ Step 1 hoàn tất: {elapsed}")
    _clear_memory(spark, "Step 1")

    # ── STEP 3+4: LLM Embeddings (MOVED TO COLAB) ────────────────────────────
    # Bước này tiêu tốn GPU và RAM lớn (1.6 triệu records), nên được thực hiện 
    # trực tiếp trên Colab bằng GPU T4/L4 để đạt hiệu suất cao nhất.

    # ── STEP 2: Edge List + Temporal Decay & Semantic Edges ──────────────────
    t0 = time.perf_counter()
    logger.info("\n📌 GOLD STEP 2: Edge List + Semantic Edges")

    edge_info = step2.run(
        spark, cfg,
        df_interactions_train=df_interactions_train,
        step1_info=step1_info,
    )

    elapsed = timedelta(seconds=int(time.perf_counter() - t0))
    logger.info(f"⏱ Step 2 hoàn tất: {elapsed}")

    del df_interactions_train, df_interactions_val
    _clear_memory(spark, "Step 2")

    # ── STEP 5: Training Metadata ─────────────────────────────────────────────
    t0 = time.perf_counter()
    logger.info("\n📌 GOLD STEP 5: Training Metadata")

    step5.run(
        spark, cfg,
        step1_info=step1_info,
        edge_info=edge_info,
    )

    elapsed = timedelta(seconds=int(time.perf_counter() - t0))
    logger.info(f"⏱ Step 5 hoàn tất: {elapsed}")

    del df_popularity, df_item_text
    _clear_memory(spark, "Step 5")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = timedelta(seconds=int(time.perf_counter() - t_start))
    logger.info("=" * 60)
    logger.info(f"🎉 GOLD PIPELINE HOÀN TẤT — ⏱ TỔNG: {total}")
    logger.info("=" * 60)
    logger.info("OUTPUT:")
    logger.info("  ✅ gold_item_id_map.parquet")
    logger.info("  ✅ gold_user_id_map.parquet")
    logger.info("  ✅ gold_edge_index.npy")
    logger.info("  ✅ gold_item_train_freq.npy")
    logger.info("  ✅ gold_item_popularity_group.npy")
    logger.info("  ✅ gold_user_train_freq.npy")
    logger.info("  ✅ gold_user_activity_group.npy")
    logger.info("  ✅ gold_negative_sampling_prob.npy")
    logger.info("  ✅ gold_dataset_stats.json")
