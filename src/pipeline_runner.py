"""
Pipeline Runner — Bước 1 → 6

Cách chạy (trong Docker, qua run_pipeline.sh):
  docker compose run --rm pipeline --step 1_2
  docker compose run --rm pipeline --step 3_6 --config-id baseline
  docker compose run --rm pipeline --step all --config-id baseline
  docker compose run --rm pipeline --run-grid-search

Cách chạy local (ngoài Docker, cần pyspark cài sẵn):
  python src/pipeline_runner.py --step 3_6 --config-id baseline
"""

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent))

import step1_ingestion as step1
import step2_bronze_storage as step2
import step3_silver_cleaning as step3
import step4_labeling as step4
import step5_temporal_split as step5
import step6_feature_store as step6
import grid_search_reliability as grid_search


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Env overrides — dùng cho Docker / CI
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
    }
    for env_key, (section, field) in overrides.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        if field in ("max_review_records", "max_metadata_records", "stream_batch_size"):
            val = int(val) if val.lower() != "null" else None
        cfg[section][field] = val

    return cfg


def get_rel_cfg(cfg: dict, config_id: str) -> dict:
    """Lấy reliability config theo config_id."""
    for c in cfg["reliability_tuning"]["configs"]:
        if c["config_id"] == config_id:
            return c
    available = [c["config_id"] for c in cfg["reliability_tuning"]["configs"]]
    raise ValueError(
        f"config_id '{config_id}' không tồn tại. Có sẵn: {available}"
    )


# ── Spark Session ─────────────────────────────────────────────────────────────

def build_spark(cfg: dict):
    """
    Tạo SparkSession.
    """
    from pyspark.sql import SparkSession

    sc = cfg["spark"]
    mn = cfg["minio"]

    builder = (
        SparkSession.builder
        .appName(sc["app_name"])
        .master(sc.get("master", "local[*]"))
        # ── Memory ────────────────────────────────────────────────
        .config("spark.driver.memory",    sc["driver_memory"])
        .config("spark.executor.memory",  sc["executor_memory"])
        # ── Shuffle & AQE ─────────────────────────────────────────
        .config("spark.sql.shuffle.partitions",             sc.get("shuffle_partitions", "100"))
        .config("spark.sql.adaptive.enabled",               sc.get("aqe_enabled", "true"))
        .config("spark.sql.adaptive.coalescePartitions.enabled",
                sc.get("aqe_coalesce_partitions", "true"))
        .config("spark.sql.adaptive.skewJoin.enabled",      sc.get("aqe_skew_join", "true"))
        # ── Parquet ───────────────────────────────────────────────
        .config("spark.sql.parquet.compression.codec",      sc["parquet_compression"])
        # ── MinIO / S3A ───────────────────────────────────────────
        .config("spark.hadoop.fs.s3a.endpoint",             mn["endpoint"])
        .config("spark.hadoop.fs.s3a.access.key",           mn["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key",           mn["secret_key"])
        .config("spark.hadoop.fs.s3a.path.style.access",    "true")
        .config("spark.hadoop.fs.s3a.impl",                 "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.fast.upload",           "true")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _n_parts(cfg: dict) -> int:
    """Trả về số partitions khi ghi (0 = không coalesce)."""
    return int(cfg["spark"].get("write_partitions", 0))


# ── Các khối chạy ─────────────────────────────────────────────────────────────

def run_steps_1_2(spark, cfg: dict, logger: logging.Logger) -> None:
    """
    Chạy Bước 1+2 theo kiến trúc MapReduce:
    - Giai đoạn 1 (Driver): Parallel Staging đẩy raw file lên MinIO.
    - Giai đoạn 2 (Spark): Đọc song song, gom nhóm và ghi zstd xuống Bronze.
    """
    logger.info("=== Bước 1: Giai đoạn Parallel Staging (Python ThreadPool) ===")
    staging_paths = step1.run_staging(cfg)
    
    logger.info("=== Bước 2: Giai đoạn Spark Distributed Processing ===")
    step2.run(spark, cfg, staging_paths)

    logger.info("✓ Bước 1+2 hoàn tất toàn diện")


def run_steps_3_6(spark, cfg: dict, config_id: str, logger: logging.Logger) -> None:
    """Chạy bước 3→6 cho một config_id — fully distributed trên cluster."""
    rel_cfg = get_rel_cfg(cfg, config_id)

    logger.info(f"=== Chạy bước 3–6 | config_id={config_id} ===")

    df_silver  = step3.run(spark, cfg, rel_cfg, config_id)
    df_labeled = step4.run(spark, cfg, config_id, df_silver=df_silver)
    step5.run(spark, cfg, config_id, df_labeled=df_labeled)
    step6.run(spark, cfg, config_id)

    logger.info(f"✓ Bước 3–6 hoàn tất | config_id={config_id}")


def run_single_step(spark, cfg: dict, step_num: str, config_id: str, logger: logging.Logger) -> None:
    rel_cfg = get_rel_cfg(cfg, config_id)
    if step_num == "3":
        step3.run(spark, cfg, rel_cfg, config_id)
    elif step_num == "4":
        step4.run(spark, cfg, config_id)
    elif step_num == "5":
        step5.run(spark, cfg, config_id)
    elif step_num == "6":
        step6.run(spark, cfg, config_id)
    else:
        raise ValueError(f"Bước không hợp lệ: {step_num}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="RecSys Pipeline Runner — Bước 1–6",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Ví dụ:
  # Trong Docker (qua run_pipeline.sh / spark-submit):
  docker compose run --rm pipeline --step 3_6 --config-id baseline
  docker compose run --rm pipeline --run-grid-search

  # Local mode:
  python src/pipeline_runner.py --step 3_6 --config-id baseline
""",
    )
    parser.add_argument("--config", default="config/config.yaml",
                        help="Đường dẫn file config YAML")
    parser.add_argument(
        "--step",
        choices=["1_2", "3", "4", "5", "6", "3_6", "all"],
        default=None,
        help="Bước cần chạy: 1_2 | 3 | 4 | 5 | 6 | 3_6 | all",
    )
    parser.add_argument("--run-grid-search", action="store_true",
                        help="Chạy grid search toàn bộ configs (bước 3–6)")
    parser.add_argument("--config-id", default=None,
                        help="reliability config_id (mặc định: selected_config_id trong YAML)")
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────
    if not args.step and not args.run_grid_search:
        parser.error("Cần chỉ định --step hoặc --run-grid-search")

    # ── Logging ───────────────────────────────────────────────────
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "pipeline.log", mode="a"),
        ],
    )
    logger = logging.getLogger("pipeline_runner")

    # ── Load config ───────────────────────────────────────────────
    cfg = load_config(args.config)
    config_id = args.config_id or cfg["reliability_tuning"]["selected_config_id"]

    logger.info(f"Config file : {args.config}")
    logger.info(f"Config ID   : {config_id}")
    logger.info(f"Spark master: {cfg['spark'].get('master', 'not set')}")

    # ── Build Spark ───────────────────────────────────────────────
    spark = build_spark(cfg)

    # Log Spark context info
    sc = spark.sparkContext
    logger.info(f"Spark version   : {sc.version}")
    logger.info(f"Spark master    : {sc.master}")
    logger.info(f"App ID          : {sc.applicationId}")
    logger.info(f"Default parallelism: {sc.defaultParallelism}")

    try:
        if args.run_grid_search:
            logger.info("=== Chế độ: Grid Search (bước 3–6) ===")
            grid_search.run_grid_search(spark, cfg)
            return

        step = args.step

        if step in ("1_2", "all"):
            run_steps_1_2(spark, cfg, logger)

        if step in ("3", "4", "5", "6"):
            run_single_step(spark, cfg, step, config_id, logger)

        if step in ("3_6", "all"):
            run_steps_3_6(spark, cfg, config_id, logger)

        logger.info("=== Pipeline hoàn tất ===")

    except Exception as e:
        logger.error(f"Pipeline lỗi: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()
        t_end = time.perf_counter()
        elapsed_seconds = t_end - t_start
        elapsed_formatted = str(timedelta(seconds=int(elapsed_seconds)))
        logger.info(f"⏱ TỔNG THỜI GIAN CHẠY: {elapsed_formatted} ({elapsed_seconds:.2f} giây)")

if __name__ == "__main__":
    main()