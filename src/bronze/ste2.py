"""
ste2.py – Bronze Layer Pipeline

TỐI ƯU v3:
  - Thêm pre-run cleanup landing zone: tự động xóa batch cũ còn sót từ lần chạy bị ngắt giữa chừng
    → Ngăn chặn Spark đọc nhầm dữ liệu bị nhân đôi (đầu vào cũ + mới) ở Phase 2
  - Streaming metadata: ghi từng batch trực tiếp bằng ParquetWriter
  - Parallel batch upload: ThreadPool đồng thời ghi nhiều batch vào Landing Zone
  - Double Max Join cho chronological split (không Window Function)

IMPLICIT FEEDBACK PROTOCOL (v3):
  - KHÔNG lọc theo rating tại tầng Bronze — giữ lại MỌI interactions
  - Phù hợp với bài báo RecMind: core-5 filter được áp trên TOÀN BỘ interactions
"""

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
            logger.info(f"Deleted landing zone: s3://{landing_dir}")
    except Exception as e:
        logger.warning(f"Cannot delete landing zone (skipped): {e}")


_SUCCESS_PATH_SUFFIX = "_SUCCESS"   # tên file marker

def _write_success_marker(landing_dir: str, fs: s3fs.S3FileSystem,
                          batch_count: int, total_rows: int) -> None:
    """
    Ghi file JSON ~50 bytes sau khi Phase 1 hoàn tất thành công.
    Chi phí: 1 PUT request S3 (~vài ms). Không tốn RAM.
    """
    marker_path = f"{landing_dir}/{_SUCCESS_PATH_SUFFIX}"
    payload = json.dumps({
        "status":      "SUCCESS",
        "batch_count": batch_count,
        "total_rows":  total_rows,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    })
    with fs.open(marker_path, "w") as f:
        f.write(payload)
    logger.info(f"[Phase 1] Wrote _SUCCESS marker ({total_rows:,} rows, {batch_count} batches).")


def _read_success_marker(landing_dir: str, fs: s3fs.S3FileSystem) -> dict | None:
    """
    Đọc _SUCCESS marker. Trả về dict nếu tồn tại và hợp lệ, None nếu không.
    Chi phí: 1 HEAD + 1 GET request S3 (~vài ms). Không tốn RAM.
    """
    marker_path = f"{landing_dir}/{_SUCCESS_PATH_SUFFIX}"
    try:
        if fs.exists(marker_path):
            with fs.open(marker_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


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

    logger.info(f"[Metadata] Hoàn tất: {total_rows:,} sản phẩm → s3://{out_path}")


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

    # ── PRE-RUN: Kiểm tra _SUCCESS marker để quyết định có bỏ qua Phase 1 ─────
    # Cơ chế:
    #   - _SUCCESS tồn tại  → Phase 1 đã hoàn tất trước đó → bỏ qua, đi thẳng Phase 2
    #   - _SUCCESS không có → xóa stale files (nếu có) → chạy Phase 1 → ghi _SUCCESS
    # Chi phí check: 1 HEAD request S3 (~vài ms), không tốn RAM.
    skip_phase1 = False
    marker = _read_success_marker(landing_fs, fs)
    if marker:
        logger.info(
            f"[PRE-RUN] Tìm thấy _SUCCESS marker (Phase 1 đã hoàn tất lần trước): "
            f"{marker.get('total_rows', '?'):,} rows | {marker.get('batch_count', '?')} batches | "
            f"lúc {marker.get('timestamp', '?')}"
        )
        logger.info("[PRE-RUN] Bỏ qua Phase 1 (Resume). Đi thẳng vào Phase 2.")
        skip_phase1 = True
        batch_count = marker.get("batch_count", 0)
        total_rows  = marker.get("total_rows", 0)
    else:
        # _SUCCESS không có → có thể có stale files từ lần chạy bị ngắt
        if fs.exists(landing_fs):
            stale_files = fs.glob(f"{landing_fs}/**/*.parquet")
            if stale_files:
                logger.warning(
                    f"[PRE-RUN] Phát hiện {len(stale_files)} file chưa hoàn tất "
                    f"(không có _SUCCESS marker). Xóa để bắt đầu lại sạch."
                )
                _cleanup_landing_zone(landing_fs, fs)
        logger.info("[PRE-RUN] Chạy Phase 1 từ đầu.")
        batch_count = 0
        total_rows  = 0

    if not skip_phase1:
        # ── Phase 1: MAP — Parallel Upload ───────────────────────────────────
        logger.info("[Phase 1 – MAP] Streaming batches → Landing Zone (parallel upload)...")
        max_workers = 4

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch in step1.iter_review_batches(cfg):
                future = executor.submit(_upload_batch, batch, landing_fs, batch_count, fs)
                futures.append(future)
                batch_count += 1

            for future in as_completed(futures):
                try:
                    total_rows += future.result()
                except Exception as e:
                    logger.error(f"Lỗi upload batch: {e}")

        logger.info(f"[Phase 1] Đã ghi {batch_count} batch ({total_rows:,} dòng) vào Landing Zone.")

        # Safety check
        if batch_count == 0 or total_rows == 0:
            logger.error(
                "Không tải được batch nào từ HuggingFace!\n"
                "  → Kiểm tra mạng và chạy lại: docker compose run --rm pipeline --step 1_2"
            )
            raise RuntimeError("Không có dữ liệu review. Dừng pipeline.")

        # Ghi _SUCCESS marker — đánh dấu Phase 1 hoàn tất an toàn
        _write_success_marker(landing_fs, fs, batch_count, total_rows)

    # ── Phase 2: REDUCE — PySpark Native Processing (giảm Shuffle) ────────────
    logger.info("[Phase 2 – REDUCE] Chạy thuật toán Spark Native (tối ưu Shuffle)...")

    # 1. Đọc dữ liệu từ Landing Zone
    raw_df = spark.read.parquet(landing_s3a)

    # 2. Deduplication
    df = raw_df.dropDuplicates(["reviewer_id", "parent_asin"])

    # 2b. Data Validation — loại các dòng có thể phá vỡ downstream joins
    # reviewer_id/parent_asin null → gây data skew trong Semi/Anti Join
    # timestamp ≤ 0     → phá vỡ Double Max Join (chon sai Test/Val)
    # rating ngoài [1,5] → ste1 đã clip, đây là lưới bảo vệ thứ 2
    df = df.filter(
        F.col("reviewer_id").isNotNull() & (F.col("reviewer_id") != "") &
        F.col("parent_asin").isNotNull() & (F.col("parent_asin") != "") &
        F.col("timestamp").isNotNull() & (F.col("timestamp") > 0) &
        F.col("rating").between(1, 5)
    )

    # 3. Core-5 Filter bằng Broadcast Semi-Join
    user_counts = df.groupBy("reviewer_id").count()
    valid_users = user_counts.filter(F.col("count") >= 5).select("reviewer_id")
    df_core5_raw = df.join(F.broadcast(valid_users), "reviewer_id", "left_semi")

    # ── [Cải thiện 1] Checkpoint: ghi/đọc tạm → bẻ gãy DAG lineage ─────────────
    # .cache() vẫn giữ toàn bộ lineage (HF → landing → dedup → core5).
    # Nếu executor chết khi chạy Double Max Join, Spark recompute từ đầu (~đọc
    # lại toàn bộ landing zone). Checkpoint giải phóng 100% RAM lineage:
    # các bước sau chỉ cần đọc từ file tuyết (I/O đơn giản).
    core5_ckpt_s3a = _s3a_path(cfg, "landing", "core5_ckpt.parquet")
    logger.info("[CKPT] Ghi core5 ra S3 (bẻ gãy DAG lineage)...")
    df_core5_raw.write.mode("overwrite").option("compression", "zstd").parquet(core5_ckpt_s3a)
    df_core5 = spark.read.parquet(core5_ckpt_s3a)

    core5_count = df_core5.count()
    logger.info(f"Core-5: {core5_count:,} interactions (lineage đã bị cắt)")

    # 4. CHRONOLOGICAL SPLIT bằng Double Max Join (không Window)
    logger.info("Đang chia tập Test / Val / Train bằng Double Max Join...")

    # --- TÌM TEST SET (review mới nhất mỗi user) ---
    max_ts_df = df_core5.groupBy("reviewer_id").agg(F.max("timestamp").alias("max_ts"))
    test_df = df_core5.join(
        F.broadcast(max_ts_df),
        (df_core5["reviewer_id"] == max_ts_df["reviewer_id"]) &
        (df_core5["timestamp"] == max_ts_df["max_ts"]),
        "inner"
    ).drop(max_ts_df["reviewer_id"]).drop("max_ts") \
     .dropDuplicates(["reviewer_id"])

    # [Cải thiện 4] Ép Broadcast Anti-Join — test_df nhỏ (~1.8M rows = 1/user)
    # Không broadcast → Spark có thể chọn SortMergeJoin → shuffle toàn bộ df_core5
    remaining_df = df_core5.join(
        F.broadcast(test_df.select("reviewer_id", "parent_asin")),
        ["reviewer_id", "parent_asin"],
        "left_anti"
    )

    # --- TÌM VAL SET (review mới thứ 2 mỗi user) ---
    max_ts_val = remaining_df.groupBy("reviewer_id").agg(F.max("timestamp").alias("max_ts"))
    val_df = remaining_df.join(
        F.broadcast(max_ts_val),
        (remaining_df["reviewer_id"] == max_ts_val["reviewer_id"]) &
        (remaining_df["timestamp"] == max_ts_val["max_ts"]),
        "inner"
    ).drop(max_ts_val["reviewer_id"]).drop("max_ts") \
     .dropDuplicates(["reviewer_id"])

    # [Cải thiện 4] Ép Broadcast Anti-Join — val_df nhỏ (~1.8M rows)
    train_df = remaining_df.join(
        F.broadcast(val_df.select("reviewer_id", "parent_asin")),
        ["reviewer_id", "parent_asin"],
        "left_anti"
    )

    # ── Phase 3: SINK — nén ZSTD, coalesce (không shuffle thêm) ──────────────
    logger.info("[Phase 3 – SINK] Ghi ra MinIO (ZSTD, coalesce)...")

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
        logger.info(f"Ghi thành công {name}.")

    # Test/Val ~1.8M rows → 5 files (~360K/file)
    # Train ~35M rows → 30 files (~1.2M/file)
    write_optimized(test_df, "bronze_test", n_files=5)
    write_optimized(val_df, "bronze_val", n_files=5)
    write_optimized(train_df, "bronze_train", n_files=30)

    df_core5.unpersist() if hasattr(df_core5, 'unpersist') else None

    # Xóa file checkpoint tạm sau khi đã ghi xong output cuối
    try:
        fs.rm(_s3_path(cfg, "landing", "core5_ckpt.parquet"), recursive=True)
        logger.info("[CKPT] Đã xóa file checkpoint tạm.")
    except Exception:
        pass  # Không bắt buộc phải xóa thành công

    # ── Phase 4: CLEANUP ─────────────────────────────────────────────────────
    if cleanup_landing:
        _cleanup_landing_zone(landing_fs, fs)

    logger.info("Bronze Pipeline hoàn tất!")


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