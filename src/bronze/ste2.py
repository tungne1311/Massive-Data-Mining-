"""
ste2.py – Bronze Layer Pipeline

TỐI ƯU v2:
  - BỎ cột: bought_together (metadata), verified_purchase (review)
  - Streaming metadata: ghi từng batch trực tiếp bằng ParquetWriter
    → Không tích lũy toàn bộ metadata vào RAM (tránh OOM Driver)
  - Giảm shuffle: dùng coalesce() thay repartition() khi ghi output
    (coalesce chỉ merge partitions, không shuffle toàn bộ)
  - Parallel batch upload: dùng ThreadPool để đồng thời ghi nhiều batch
    vào Landing Zone (I/O bound → thread = cải thiện đáng kể)
  - Giữ nguyên logic Double Max Join cho chronological split
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.sql import functions as F
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

import bronze.ste1 as step1

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_s3_filesystem(cfg: dict) -> s3fs.S3FileSystem:
    mn       = cfg["minio"]
    endpoint = mn["endpoint"].replace("http://", "").replace("https://", "")
    return s3fs.S3FileSystem(
        key=mn["access_key"],
        secret=mn["secret_key"],
        client_kwargs={"endpoint_url": f"http://{endpoint}"},
    )

def _s3_path(cfg: dict, *parts: str) -> str:
    """Tạo đường dẫn S3 chuẩn (dùng cho pyarrow s3fs)"""
    bucket = cfg["minio"]["bucket"].strip("/")
    return "/".join([bucket, *parts])

def _s3a_path(cfg: dict, *parts: str) -> str:
    """Tạo đường dẫn S3A chuẩn (dùng cho PySpark)"""
    return f"s3a://{_s3_path(cfg, *parts)}"

def _cleanup_landing_zone(landing_dir: str, fs: s3fs.S3FileSystem) -> None:
    try:
        if fs.exists(landing_dir):
            fs.rm(landing_dir, recursive=True)
            logger.info(f"🗑 Đã xóa landing zone: s3://{landing_dir}")
    except Exception as e:
        logger.warning(f"Không thể xóa landing zone (bỏ qua): {e}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP A: METADATA — Streaming Write (không tích lũy vào RAM)
# ──────────────────────────────────────────────────────────────────────────────

def process_bronze_metadata(cfg: dict) -> None:
    """
    Ghi metadata trực tiếp từng batch bằng ParquetWriter.
    Không cần list() toàn bộ → tiết kiệm RAM đáng kể cho dataset lớn.
    """
    logger.info("⏳ [Metadata] Bắt đầu kéo dữ liệu metadata (streaming write)...")
    fs       = _get_s3_filesystem(cfg)
    out_path = _s3_path(cfg, "bronze", "bronze_meta.parquet")

    total_rows   = 0
    batch_count  = 0
    writer       = None

    try:
        for batch in step1.iter_metadata_batches(cfg):
            if writer is None:
                # Mở ParquetWriter lần đầu khi có schema thực tế
                s3_file = fs.open(out_path, "wb")
                writer  = pq.ParquetWriter(s3_file, batch.schema, compression="zstd")

            writer.write_table(batch)
            total_rows  += batch.num_rows
            batch_count += 1

            if batch_count % 5 == 0:
                logger.info(f"  📦 Metadata: {total_rows:,} dòng ({batch_count} batches)")

    finally:
        if writer is not None:
            writer.close()
            s3_file.close()

    if total_rows == 0:
        logger.warning("⚠ Không có dữ liệu metadata! Bỏ qua.")
        return

    logger.info(f"✅ [Metadata] Hoàn tất: {total_rows:,} sản phẩm → s3://{out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# STEP B: REVIEWS – Parallel Upload + PySpark Native Split (giảm Shuffle)
# ──────────────────────────────────────────────────────────────────────────────

def _upload_batch(batch: pa.Table, landing_dir: str, batch_idx: int,
                  fs: s3fs.S3FileSystem) -> int:
    """Upload một batch lên Landing Zone (chạy trong thread)."""
    temp_path = f"{landing_dir}/batch_{batch_idx:04d}_{uuid.uuid4().hex[:6]}.parquet"
    with fs.open(temp_path, "wb") as f:
        pq.write_table(batch, f, compression="zstd")
    return batch.num_rows


def process_bronze_reviews_and_split(spark, cfg: dict, cleanup_landing: bool = True) -> None:
    fs          = _get_s3_filesystem(cfg)
    landing_fs  = _s3_path(cfg, "landing", "reviews_temp")
    landing_s3a = _s3a_path(cfg, "landing", "reviews_temp")

    # ── Phase 1: MAP — Parallel Upload (HuggingFace Stream → Landing Zone S3) ─
    logger.info("⏳ [Phase 1 – MAP] Streaming batches → Landing Zone (parallel upload)...")
    batch_count  = 0
    total_rows   = 0
    max_workers  = 4    # I/O bound → 4 threads song song

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in step1.iter_review_batches(cfg):
            future = executor.submit(_upload_batch, batch, landing_fs, batch_count, fs)
            futures.append(future)
            batch_count += 1

        # Thu kết quả
        for future in as_completed(futures):
            try:
                total_rows += future.result()
            except Exception as e:
                logger.error(f"Lỗi upload batch: {e}")

    logger.info(f"✅ [Phase 1] Đã ghi {batch_count} batch ({total_rows:,} dòng) vào Landing Zone.")

    # ── Safety check: không có dữ liệu thì dừng sớm ─────────────────────────
    if batch_count == 0 or total_rows == 0:
        logger.error(
            "❌ Không tải được batch nào từ HuggingFace!\n"
            "  → Kiểm tra mạng và chạy lại: docker compose run --rm pipeline --step 1_2"
        )
        raise RuntimeError("Không có dữ liệu review. Dừng pipeline.")

    # ── Phase 2: REDUCE — PySpark Native Processing (giảm Shuffle) ────────────
    logger.info("⏳ [Phase 2 – REDUCE] Chạy thuật toán Spark Native (tối ưu Shuffle)...")

    # 1. Đọc dữ liệu từ Landing Zone
    raw_df = spark.read.parquet(landing_s3a)

    # LỌC RATING >= 3.0 (Positive-only filtering from the start to prevent data leakage/ghost users)
    raw_df = raw_df.filter(F.col("rating") >= 3.0)

    # 2. Deduplication
    df = raw_df.dropDuplicates(["reviewer_id", "parent_asin"])

    # 3. Core-5 Filter bằng Broadcast Semi-Join
    #    groupBy reviewer_id chỉ tạo 1 shuffle nhỏ (chỉ đếm).
    #    Broadcast kết quả nhỏ (chỉ reviewer_id + count) để tránh shuffle lớn.
    user_counts = df.groupBy("reviewer_id").count()
    valid_users = user_counts.filter(F.col("count") >= 5).select("reviewer_id")

    # Broadcast valid_users (nhỏ: chỉ ~1.8M reviewer_id strings ≈ 30MB)
    # left_semi join + broadcast = không shuffle df lớn
    df_core5 = df.join(F.broadcast(valid_users), "reviewer_id", "left_semi").cache()

    # Materialize cache để tránh recompute trong các bước sau
    core5_count = df_core5.count()
    logger.info(f"  📦 Core-5: {core5_count:,} interactions")

    # 4. CHRONOLOGICAL SPLIT bằng Double Max Join (không Window)
    logger.info("⏳ Đang chia tập Test / Val / Train bằng Double Max Join...")

    # --- TÌM TEST SET (review mới nhất mỗi user) ---
    #   groupBy chỉ tạo 1 shuffle, kết quả max_ts nhỏ → broadcast
    max_ts_df = df_core5.groupBy("reviewer_id").agg(F.max("timestamp").alias("max_ts"))
    test_df = df_core5.join(
        F.broadcast(max_ts_df),
        (df_core5["reviewer_id"] == max_ts_df["reviewer_id"]) &
        (df_core5["timestamp"] == max_ts_df["max_ts"]),
        "inner"
    ).drop(max_ts_df["reviewer_id"]).drop("max_ts") \
     .dropDuplicates(["reviewer_id"])

    # Trừ tập Test khỏi tập gốc
    remaining_df = df_core5.join(test_df, ["reviewer_id", "parent_asin"], "left_anti")

    # --- TÌM VAL SET (review mới thứ 2 mỗi user) ---
    max_ts_val = remaining_df.groupBy("reviewer_id").agg(F.max("timestamp").alias("max_ts"))
    val_df = remaining_df.join(
        F.broadcast(max_ts_val),
        (remaining_df["reviewer_id"] == max_ts_val["reviewer_id"]) &
        (remaining_df["timestamp"] == max_ts_val["max_ts"]),
        "inner"
    ).drop(max_ts_val["reviewer_id"]).drop("max_ts") \
     .dropDuplicates(["reviewer_id"])

    # --- TÌM TRAIN SET (Phần còn lại) ---
    train_df = remaining_df.join(val_df, ["reviewer_id", "parent_asin"], "left_anti")

    # ── Phase 3: SINK — nén ZSTD, coalesce (không shuffle thêm) ──────────────
    logger.info("⏳ [Phase 3 – SINK] Ghi ra MinIO (ZSTD, coalesce)...")

    # Đặt compression = zstd cho toàn bộ output bronze
    spark.conf.set("spark.sql.parquet.compression.codec", "zstd")

    def write_optimized(df_target, name, n_files=10):
        out_s3a = _s3a_path(cfg, "bronze", f"{name}.parquet")
        df_target.coalesce(n_files) \
                 .sortWithinPartitions("reviewer_id", "timestamp") \
                 .write \
                 .mode("overwrite") \
                 .option("compression", "zstd") \
                 .parquet(out_s3a)
        logger.info(f"✅ Ghi thành công {name}.")

    # Test/Val ~1.8M rows → 5 files (~360K/file)
    # Train ~35M rows → 30 files (~1.2M/file)
    write_optimized(test_df, "bronze_test", n_files=5)
    write_optimized(val_df, "bronze_val", n_files=5)
    write_optimized(train_df, "bronze_train", n_files=30)

    df_core5.unpersist()

    # ── Phase 4: CLEANUP ─────────────────────────────────────────────────────
    if cleanup_landing:
        _cleanup_landing_zone(landing_fs, fs)

    logger.info("🎉 Bronze Pipeline hoàn tất!")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def run_bronze_pipeline(spark, cfg: dict) -> None:
    process_bronze_metadata(cfg)
    process_bronze_reviews_and_split(
        spark,
        cfg,
        cleanup_landing=cfg.get("bronze", {}).get("cleanup_landing", True),
    )