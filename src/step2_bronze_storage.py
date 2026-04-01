"""
Bước 2 — Bronze Storage (Tối giản & MapReduce)
Đọc dữ liệu từ Staging (MinIO) -> Gom file nhỏ -> Ghi Parquet (zstd) xuống Bronze.
Không lưu Cache, Không chạy Quality Check. Ghi ngay lập tức từng luồng.
"""

import logging
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"


def process_and_write_reviews(spark: SparkSession, staging_path: str, cfg: dict) -> None:
    """Đọc Reviews từ Staging, tạo year_month, gom file và ghi ngay xuống Bronze."""
    logger.info(f"[SPARK MAP] Đọc reviews từ staging: {staging_path}")
    
    df = spark.read.parquet(staging_path)
    
    # 1. Lọc và tạo partition key (year_month)
    df = df.filter((F.col("timestamp").isNotNull()) & (F.col("timestamp") > 0))
    df = df.withColumn("review_time", F.to_timestamp(F.from_unixtime(F.col("timestamp"))))
    df = df.filter((F.year("review_time") >= 1995) & (F.year("review_time") <= 2030))
    df = df.withColumn("year_month", F.date_format(F.col("review_time"), "yyyy-MM"))
    
    out_path = join_uri(cfg["paths"]["bronze_base"], "reviews")
    n_partitions = int(cfg["spark"].get("write_partitions", 0))
    write_mode = cfg["bronze"].get("write_mode", "append")

    # 2. Giải quyết vấn đề "Nhiều file nhỏ" (Small Files Problem)
    # Sử dụng repartition theo cột phân mảnh (year_month)
    # Thao tác này buộc Spark phải Shuffle dữ liệu: tất cả các dòng của cùng một tháng 
    # sẽ được gom về chung 1 task/executor. Kết quả là mỗi thư mục tháng chỉ sinh ra 1 (hoặc rất ít) file lớn.
    if n_partitions > 0:
        df = df.repartition(n_partitions, "year_month")
    else:
        df = df.repartition("year_month")

    # 3. Lập tức kích hoạt Action: Ghi Parquet với chuẩn nén ZSTD
    logger.info(f"[SPARK REDUCE] Đang ghi Bronze reviews (ZSTD) → {out_path}")
    (
        df.write
        .mode(write_mode)
        .partitionBy("year_month")
        .option("compression", "zstd")
        .parquet(out_path)
    )
    logger.info("✓ Ghi xong Reviews.")


def process_and_write_metadata(spark: SparkSession, staging_path: str, cfg: dict) -> None:
    """Đọc Metadata từ Staging và ghi thẳng xuống Bronze (Không phân mảnh)."""
    logger.info(f"[SPARK MAP] Đọc metadata từ staging: {staging_path}")
    
    df = spark.read.parquet(staging_path)
    
    out_path = join_uri(cfg["paths"]["bronze_base"], "metadata")
    n_partitions = int(cfg["spark"].get("write_partitions", 0))
    write_mode = cfg["bronze"].get("write_mode", "append")

    # Chỉ gom file lại thành N file lớn, KHÔNG gom theo year_month nữa
    if n_partitions > 0:
        df = df.repartition(n_partitions)

    # Ghi Parquet ZSTD, KHÔNG dùng partitionBy("year_month") nữa
    logger.info(f"[SPARK REDUCE] Đang ghi Bronze metadata (ZSTD) → {out_path}")
    (
        df.write
        .mode(write_mode)
        .option("compression", "zstd")
        .parquet(out_path)
    )
    logger.info("✓ Ghi xong Metadata.")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(spark: SparkSession, cfg: dict, staging_paths: dict) -> None:
    """
    Điểm bắt đầu của Bước 2: Xử lý độc lập và ghi ngay lập tức (không gom chờ).
    """
    logger.info("=== Bước 2: Bronze Storage (MapReduce + ZSTD) ===")

    # Thực hiện luồng Reviews: Map -> Shuffle (Gom file) -> Reduce (Ghi ngay lập tức)
    if "reviews" in staging_paths and staging_paths["reviews"]:
        process_and_write_reviews(spark, staging_paths["reviews"], cfg)

    # Thực hiện luồng Metadata: Map -> Shuffle (Gom file) -> Reduce (Ghi ngay lập tức)
    if "metadata" in staging_paths and staging_paths["metadata"]:
        process_and_write_metadata(spark, staging_paths["metadata"], cfg)
        
    logger.info("=== Bước 2 hoàn tất ===")