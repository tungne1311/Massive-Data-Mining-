import argparse
import logging
import os
import sys
import time
import gc
from datetime import timedelta
from pathlib import Path
import yaml
import concurrent.futures
from py4j.protocol import Py4JJavaError

sys.path.insert(0, str(Path(__file__).parent))

import ste1 as step1
import ste2 as step2

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
    for c in cfg["reliability_tuning"]["configs"]:
        if c["config_id"] == config_id:
            return c
    raise ValueError(f"config_id '{config_id}' không tồn tại.")

def build_spark(cfg: dict):
    from pyspark.sql import SparkSession
    sc = cfg["spark"]
    mn = cfg["minio"]

    # Đã giảm shuffle partitions xuống 24 cho máy Single-Node
    shuffle_parts = "24" 

    builder = (
        SparkSession.builder
        .appName(sc["app_name"])
        .master(sc.get("master", "local[*]"))
        .config("spark.driver.memory",    sc["driver_memory"])
        .config("spark.executor.memory",  sc["executor_memory"])
        .config("spark.sql.shuffle.partitions",             shuffle_parts)
        .config("spark.sql.adaptive.enabled",               "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled",      "true")
        
        # Nâng ngưỡng Broadcast lên 100MB để chống OOM
        .config("spark.sql.autoBroadcastJoinThreshold",     "104857600")
        
        .config("spark.sql.parquet.compression.codec",      sc["parquet_compression"])
        .config("spark.hadoop.fs.s3a.endpoint",             mn["endpoint"])
        .config("spark.hadoop.fs.s3a.access.key",           mn["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key",           mn["secret_key"])
        .config("spark.hadoop.fs.s3a.path.style.access",    "true")
        .config("spark.hadoop.fs.s3a.impl",                 "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def run_steps_1_2(cfg: dict, logger: logging.Logger) -> None:
    t_start = time.perf_counter()
    logger.info("=== Bước 1 & 2: Kéo dữ liệu (PyArrow) và ghi TRỰC TIẾP xuống Landing Zone (PARALLEL MODE) ===")
    
    max_workers = 4 

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 1. Tải Metadata
        logger.info("  -> Đang tải METADATA...")
        meta_futures = []
        for batch in step1.iter_metadata_batches(cfg):
            future = executor.submit(step2.write_metadata_batch, batch, cfg)
            meta_futures.append(future)
        
        concurrent.futures.wait(meta_futures)
        logger.info("  -> Xong METADATA!")

        # 2. Tải Reviews
        logger.info("  -> Đang tải REVIEWS...")
        review_futures = []
        for batch in step1.iter_review_batches(cfg):
            future = executor.submit(step2.write_review_batch, batch, cfg)
            review_futures.append(future)
            
            # Cân bằng luồng: Tránh nghẽn bộ nhớ
            active_futures = [f for f in review_futures if not f.done()]
            if len(active_futures) > 20:
                time.sleep(1)

        concurrent.futures.wait(review_futures)
        logger.info("  -> Xong REVIEWS!")
        
    t_end = time.perf_counter()
    elapsed_seconds = t_end - t_start
    elapsed_formatted = str(timedelta(seconds=int(elapsed_seconds)))
    
    logger.info("✓ Bước 1+2 hoàn tất. Dữ liệu đã nằm an toàn trên MinIO (Landing Zone).")
    logger.info(f"⏱ THỜI GIAN CHẠY BƯỚC 1 & 2: {elapsed_formatted} ({elapsed_seconds:.2f} giây)")


# ── CÁC HÀM CHẠY TÁCH RỜI ──────────────────────────────────────────────

def run_step_3(spark, cfg: dict, config_id: str, logger: logging.Logger) -> None:
    import step3 as step3
    rel_cfg = get_rel_cfg(cfg, config_id)
    logger.info(f"=== Bắt đầu CHẠY RIÊNG Bước 3 (Silver) cho config_id={config_id} ===")
    step3.run(spark, cfg, rel_cfg, config_id, cleanup_landing=False)

def run_step_4(spark, cfg: dict, config_id: str, logger: logging.Logger) -> None:
    import step4_labeling as step4
    logger.info(f"=== Bắt đầu CHẠY RIÊNG Bước 4 (Labeling) cho config_id={config_id} ===")
    step4.run(spark, cfg, config_id, df_silver=None)

def run_step_5(spark, cfg: dict, config_id: str, logger: logging.Logger) -> None:
    import step5_temporal_split as step5
    logger.info(f"=== Bắt đầu CHẠY RIÊNG Bước 5 (Temporal Split) cho config_id={config_id} ===")
    step5.run(spark, cfg, config_id, df_labeled=None)

def run_step_6(spark, cfg: dict, config_id: str, logger: logging.Logger) -> None:
    import step6_feature_store as step6
    logger.info(f"=== Bắt đầu CHẠY RIÊNG Bước 6 (Feature Store) cho config_id={config_id} ===")
    step6.run(spark, cfg, config_id, df_train=None)


# ── HÀM QUẢN LÝ BỘ NHỚ ────────────────────────────────────────────────

def clear_memory(spark, logger: logging.Logger):
    """Hàm dọn dẹp bộ nhớ chuyên sâu giữa các Step để tránh OOM tích luỹ"""
    logger.info("  🧹 [Memory Manager] Đang ép giải phóng RAM và Spark Cache...")
    if spark:
        spark.catalog.clearCache() 
    gc.collect() 
    time.sleep(2) 


# ── HÀM ĐIỀU PHỐI CHÍNH ───────────────────────────────────────────────

def main():
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="RecSys Pipeline Runner")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--step", choices=["1_2", "3", "4", "5", "6"], default=None)
    parser.add_argument("--all", action="store_true", help="Chạy tuần tự từ Step 1 đến Step 6")
    parser.add_argument("--config-id", default=None)
    args = parser.parse_args()

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "pipeline.log", mode="a")])
    logger = logging.getLogger("pipeline_runner")

    cfg = load_config(args.config)
    config_id = args.config_id or cfg["reliability_tuning"]["selected_config_id"]

    try:
        # Bước lấy dữ liệu bằng PyArrow (Không tốn RAM Spark)
        if args.all or args.step == "1_2":
            run_steps_1_2(cfg, logger)
            clear_memory(None, logger)

        # Khởi tạo Spark cho các bước xử lý sau
        if args.all or args.step in ("3", "4", "5", "6"):
            spark = build_spark(cfg)
            
            try:
                if args.all:
                    logger.info(f"=== BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE (3 -> 6) CHO CONFIG: {config_id} ===")
                    
                    run_step_3(spark, cfg, config_id, logger)
                    clear_memory(spark, logger)

                    run_step_4(spark, cfg, config_id, logger)
                    clear_memory(spark, logger)

                    run_step_5(spark, cfg, config_id, logger)
                    clear_memory(spark, logger)

                    run_step_6(spark, cfg, config_id, logger)
                    clear_memory(spark, logger)
                    
                else:
                    if args.step == "3":
                        run_step_3(spark, cfg, config_id, logger)
                    elif args.step == "4":
                        run_step_4(spark, cfg, config_id, logger)
                    elif args.step == "5":
                        run_step_5(spark, cfg, config_id, logger)
                    elif args.step == "6":
                        run_step_6(spark, cfg, config_id, logger)
                    
                    clear_memory(spark, logger)

            except Py4JJavaError as e:
                err_msg = str(e.java_exception)
                if "java.lang.OutOfMemoryError" in err_msg or "GC overhead limit exceeded" in err_msg:
                    logger.error("\n" + "!"*70)
                    logger.error("🚨 [CRITICAL OOM] HỆ THỐNG SPARK ĐÃ BỊ TRÀN BỘ NHỚ (OUT OF MEMORY)!")
                    logger.error("💡 Cách khắc phục gợi ý:")
                    logger.error("  1. Gỡ bỏ F.broadcast() đối với các bảng quá lớn.")
                    logger.error("  2. Hạ cấu hình --executor-memory nếu RAM vật lý không đủ để gánh.")
                    logger.error("  3. Check log lịch sử để xem bước nào bị crash do partition quá lớn.")
                    logger.error("!"*70 + "\n")
                else:
                    logger.error(f"❌ Lỗi thực thi Spark (Py4JJavaError): {err_msg}")
                sys.exit(1)
                
            finally:
                spark.stop()

    except Exception as e:
        logger.error(f"❌ Pipeline lỗi ngoài dự kiến: {e}", exc_info=True)
        sys.exit(1)
    finally:
        t_end = time.perf_counter()
        logger.info(f"⏱ TỔNG THỜI GIAN CHẠY MÔ ĐUN NÀY: {str(timedelta(seconds=int(t_end - t_start)))}")

if __name__ == "__main__":
    main()