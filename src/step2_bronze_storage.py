"""
Bước 2 — Bronze Storage (Micro-batch Append & Compaction)
Ghi từng batch thô dạng part-files, sau đó gom nhóm và nén zstd.
Cuối cùng tự động xóa sạch thư mục tạm để giải phóng dung lượng.
"""

import logging
from pyspark.sql import DataFrame, SparkSession
import step1_ingestion as step1

logger = logging.getLogger(__name__)


def join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"

# ── HÀM TIỆN ÍCH XÓA THƯ MỤC BẰNG HADOOP API (HỖ TRỢ MINIO & LOCAL) ───────────
def delete_path(spark: SparkSession, path: str) -> None:
    """Xóa hoàn toàn một thư mục (kể cả có chứa file bên trong)"""
    sc = spark.sparkContext
    # Gọi xuống tầng Java JVM của Hadoop để thao tác xóa file
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    
    fs = FileSystem.get(URI(path), sc._jsc.hadoopConfiguration())
    hadoop_path = Path(path)
    if fs.exists(hadoop_path):
        fs.delete(hadoop_path, True)  # True = Xóa đệ quy (recursive) toàn bộ file bên trong


# ── 1. GHI LẺ TỪNG BATCH LÊN THƯ MỤC TẠM ──────────────────────────────────────

def write_review_batch(spark: SparkSession, batch_rows: list[dict], cfg: dict, write_mode: str = "append") -> None:
    """Ghi trực tiếp không nén vào temp_reviews"""
    temp_path = join_uri(cfg["paths"]["bronze_base"], "temp_reviews")
    df = spark.createDataFrame(batch_rows, schema=step1.REVIEW_SCHEMA)
    df.write.mode(write_mode).partitionBy("year_month").parquet(temp_path)

def write_metadata_batch(spark: SparkSession, batch_rows: list[dict], cfg: dict, write_mode: str = "append") -> None:
    """Ghi trực tiếp không nén vào temp_metadata"""
    temp_path = join_uri(cfg["paths"]["bronze_base"], "temp_metadata")
    df = spark.createDataFrame(batch_rows, schema=step1.METADATA_SCHEMA)
    df.write.mode(write_mode).partitionBy("year_month").parquet(temp_path)


# ── 2. GOM NHÓM COMPACTION VÀ NÉN ZSTD ────────────────────────────────────────

def run_compaction(spark: SparkSession, cfg: dict) -> None:
    """Gom các file nhỏ từ thư mục tạm và nén ZSTD vào Bronze đích"""
    logger.info("=== Bắt đầu Compaction & Compress (ZSTD) ===")
    
    bronze_base = cfg["paths"]["bronze_base"]
    n_partitions = int(cfg["spark"].get("write_partitions", 0))

    temp_rev = join_uri(bronze_base, "temp_reviews")
    final_rev = join_uri(bronze_base, "reviews")
    temp_meta = join_uri(bronze_base, "temp_metadata")
    final_meta = join_uri(bronze_base, "metadata")
    
    rev_success = False
    meta_success = False
    ## --- Gom nhóm Reviews ---
    logger.info(f"[COMPACT] Gom nhóm Reviews: {temp_rev} -> {final_rev}")
    try:
        df_rev = spark.read.parquet(temp_rev)
        df_rev = df_rev.repartition(n_partitions, "year_month") if n_partitions > 0 else df_rev.repartition("year_month")
            
        (
            df_rev.write
            .mode("overwrite")
            .partitionBy("year_month")
            .option("compression", "zstd")
            .parquet(final_rev)
        )
        logger.info("  ✓ Đã gom nhóm và nén ZSTD thành công cho Reviews.")
        rev_success = True  # <--- Bật cờ thành công
    except Exception as e:
        logger.error(f"  ✗ Lỗi gom file Reviews: {e}")

    # --- Gom nhóm Metadata ---
    logger.info(f"[COMPACT] Gom nhóm Metadata: {temp_meta} -> {final_meta}")
    try:
        df_meta = spark.read.parquet(temp_meta)
        df_meta = df_meta.repartition(n_partitions, "year_month") if n_partitions > 0 else df_meta.repartition("year_month")
            
        (
            df_meta.write
            .mode("overwrite")
            .partitionBy("year_month")
            .option("compression", "zstd")
            .parquet(final_meta)
        )
        logger.info("  ✓ Đã gom nhóm và nén ZSTD thành công cho Metadata.")
        meta_success = True # <--- Bật cờ thành công
    except Exception as e:
        logger.error(f"  ✗ Lỗi gom file Metadata: {e}")
        
    # --- DỌN DẸP THƯ MỤC TẠM (CHỈ KHI THÀNH CÔNG) ---
    logger.info("=== Đang dọn dẹp xóa sạch thư mục rác (temp_reviews & temp_metadata) ===")
    try:
        if rev_success:
            delete_path(spark, temp_rev)
            logger.info("  ✓ Đã xóa sạch temp_reviews.")
        else:
            logger.warning("  ! Bỏ qua xóa temp_reviews vì quá trình gom nhóm bị lỗi.")
            
        if meta_success:
            delete_path(spark, temp_meta)
            logger.info("  ✓ Đã xóa sạch temp_metadata.")
        else:
            logger.warning("  ! Bỏ qua xóa temp_metadata vì quá trình gom nhóm bị lỗi.")
            
    except Exception as e:
        logger.error(f"  ✗ Lỗi khi xóa thư mục tạm: {e}")

    logger.info("=== Hoàn tất toàn bộ Bước 2 ===")