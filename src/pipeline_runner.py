"""
pipeline_runner.py – Điều phối toàn bộ pipeline TA-RecMind

PIPELINE:
  Step 1+2: Bronze — HuggingFace ingestion + Spark processing + chronological split
  Step 3:   Silver — Item popularity, text profiles, enriched interactions
  Step 4:   Gold   — ID mapping, edge list, LLM embeddings, training metadata

CẬP NHẬT v3:
  - Loại bỏ legacy steps (3-6 cũ) — đã thay bằng Silver pipeline + Gold pipeline
  - Thêm Gold pipeline (ste4_gold)
  - Đơn giản hóa CLI: chỉ còn 1_2, 3_silver, 4_gold
"""

import argparse
import gc
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import yaml

# Thêm thư mục hiện tại vào sys.path để import các module cùng cấp
sys.path.insert(0, str(Path(__file__).parent))

import bronze.ste1 as step1   # noqa: F401
import bronze.ste2 as step2
import silver.ste3_silver as step3_silver
import gold.ste4_gold     as step4_gold


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = {
        "SPARK_DRIVER_MEMORY":    ("spark",       "driver_memory"),
        "SPARK_EXECUTOR_MEMORY":  ("spark",       "executor_memory"),
        "SPARK_EXECUTOR_CORES":   ("spark",       "executor_cores"),
        "SPARK_MASTER":           ("spark",       "master"),
        "MAX_REVIEW_RECORDS":     ("huggingface", "max_review_records"),
        "MAX_METADATA_RECORDS":   ("huggingface", "max_metadata_records"),
        "HF_STREAM_BATCH_SIZE":   ("huggingface", "stream_batch_size"),
        "BRONZE_BASE_PATH":       ("paths",       "bronze_base"),
        "MINIO_ENDPOINT":         ("minio",       "endpoint"),
        "MINIO_ACCESS_KEY":       ("minio",       "access_key"),
        "MINIO_SECRET_KEY":       ("minio",       "secret_key"),
        "MINIO_BUCKET":           ("minio",       "bucket"),
        "HF_TOKEN":               ("huggingface", "token"),
    }
    for env_key, (section, field) in overrides.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        if field in ("max_review_records", "max_metadata_records", "stream_batch_size"):
            val = int(val) if val.lower() != "null" else None
        cfg.setdefault(section, {})[field] = val

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# SPARK BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_spark(cfg: dict):
    from pyspark.sql import SparkSession

    sc = cfg["spark"]
    mn = cfg["minio"]

    shuffle_parts = str(sc.get("shuffle_partitions", 24))

    builder = (
        SparkSession.builder
        .appName(sc["app_name"])
        .master(sc.get("master", "local[*]"))
        .config("spark.driver.memory",   sc["driver_memory"])
        .config("spark.executor.memory", sc["executor_memory"])
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "100s")
        .config("spark.driver.maxResultSize", "8g") # Quan trọng khi collect embeddings
        .config("spark.sql.autoBroadcastJoinThreshold", "256MB") # Đẩy mạnh broadcast join
        .config("spark.sql.shuffle.partitions",                  shuffle_parts)
        .config("spark.sql.adaptive.enabled",                    "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled",           "true")
        .config("spark.sql.autoBroadcastJoinThreshold",          "104857600")
        .config("spark.sql.parquet.compression.codec",           sc["parquet_compression"])
        .config("spark.hadoop.fs.s3a.endpoint",                  mn["endpoint"])
        .config("spark.hadoop.fs.s3a.access.key",                mn["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key",                mn["secret_key"])
        .config("spark.hadoop.fs.s3a.path.style.access",         "true")
        .config("spark.hadoop.fs.s3a.impl",                      "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",  "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled",    "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ──────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGER
# ──────────────────────────────────────────────────────────────────────────────

def clear_memory(spark, logger: logging.Logger) -> None:
    logger.info("  🧹 [Memory Manager] Giải phóng RAM và Spark Cache...")
    if spark is not None:
        spark.catalog.clearCache()
    gc.collect()
    time.sleep(2)


# ──────────────────────────────────────────────────────────────────────────────
# STEP RUNNERS
# ──────────────────────────────────────────────────────────────────────────────

def run_steps_1_2(spark, cfg: dict, logger: logging.Logger) -> None:
    t0 = time.perf_counter()
    logger.info("=== Bước 1 & 2: Ingestion & Bronze Processing ===")
    step2.run_bronze_pipeline(spark, cfg)
    elapsed = time.perf_counter() - t0
    logger.info(f"✓ Bước 1+2 hoàn tất.  ⏱ {timedelta(seconds=int(elapsed))}")


def run_step_3_silver(spark, cfg: dict, logger: logging.Logger) -> None:
    t0 = time.perf_counter()
    logger.info("=== Bước 3: Silver Pipeline ===")
    step3_silver.run_silver_pipeline(spark, cfg)
    elapsed = time.perf_counter() - t0
    logger.info(f"✓ Bước 3 (Silver) hoàn tất.  ⏱ {timedelta(seconds=int(elapsed))}")


def run_step_4_gold(spark, cfg: dict, logger: logging.Logger) -> None:
    t0 = time.perf_counter()
    logger.info("=== Bước 4: Gold Pipeline ===")
    step4_gold.run_gold_pipeline(spark, cfg)
    elapsed = time.perf_counter() - t0
    logger.info(f"✓ Bước 4 (Gold) hoàn tất.  ⏱ {timedelta(seconds=int(elapsed))}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_global = time.perf_counter()

    parser = argparse.ArgumentParser(description="TA-RecMind Pipeline Runner")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument(
        "--step",
        choices=["1_2", "3_silver", "4_gold"],
        default=None,
        help="Chạy một bước cụ thể: 1_2 (Bronze), 3_silver (Silver), 4_gold (Gold)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Chạy tuần tự toàn bộ pipeline: Bronze → Silver → Gold",
    )
    args = parser.parse_args()

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "pipeline.log", mode="a"),
        ],
    )
    logger = logging.getLogger("pipeline_runner")

    cfg = load_config(args.config)

    try:
        from py4j.protocol import Py4JJavaError  # noqa: PLC0415

        # Khởi tạo Spark dùng chung cho tất cả các bước
        spark = build_spark(cfg)

        if args.all:
            # ── Full pipeline: Bronze → Silver → Gold ─────────────────────────
            logger.info("=== FULL PIPELINE: Bronze → Silver → Gold ===")

            run_steps_1_2(spark, cfg, logger)
            clear_memory(spark, logger)

            run_step_3_silver(spark, cfg, logger)
            clear_memory(spark, logger)

            run_step_4_gold(spark, cfg, logger)

        elif args.step == "1_2":
            run_steps_1_2(spark, cfg, logger)

        elif args.step == "3_silver":
            run_step_3_silver(spark, cfg, logger)

        elif args.step == "4_gold":
            run_step_4_gold(spark, cfg, logger)

        else:
            parser.print_help()
            logger.info("Vui lòng chọn --step hoặc --all.")

    except Py4JJavaError as e:
        err_msg = str(e.java_exception)
        if "OutOfMemoryError" in err_msg or "GC overhead limit exceeded" in err_msg:
            logger.error("!" * 70)
            logger.error("🚨 [OOM] Spark tràn bộ nhớ!")
            logger.error("  1. Kiểm tra F.broadcast() trên bảng lớn.")
            logger.error("  2. Giảm spark.sql.shuffle.partitions trong config.")
            logger.error("  3. Tăng driver/executor memory.")
            logger.error("!" * 70)
        else:
            logger.error(f"❌ Py4JJavaError: {err_msg}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Pipeline lỗi ngoài dự kiến: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if 'spark' in locals():
            spark.stop()
        elapsed = time.perf_counter() - t_global
        logger.info(f"⏱ TỔNG THỜI GIAN: {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main()