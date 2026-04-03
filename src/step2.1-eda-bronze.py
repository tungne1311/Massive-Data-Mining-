"""
Bước 2.2 — Deep Exploratory Data Analysis (EDA)
Tác giả: Senior Data Scientist
Mục tiêu: Phân tích 44M Reviews & 1.6M Meta, xuất JSON Report phục vụ làm sạch Silver Layer.
Lưu ý: Không dùng .cache() để tránh OOM khi chạy Local với cấu hình RAM hạn chế.
"""

import json
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Deep_EDA")

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f: 
        return yaml.safe_load(f)

def create_spark(cfg: dict) -> SparkSession:
    sc = cfg["spark"]
    mn = cfg["minio"]
    return (SparkSession.builder.appName("Deep_EDA_Amazon")
            .config("spark.driver.memory", sc.get("driver_memory", "4g"))
            .config("spark.executor.memory", sc.get("executor_memory", "8g"))
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.hadoop.fs.s3a.endpoint", mn["endpoint"])
            .config("spark.hadoop.fs.s3a.access.key", mn["access_key"])
            .config("spark.hadoop.fs.s3a.secret.key", mn["secret_key"])
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .getOrCreate())

def get_missing_pct(df: DataFrame) -> dict:
    total = df.count()
    if total == 0: return {}
    exprs = []
    for c, dtype in df.dtypes:
        # Xử lý triệt để lỗi Datatype Mismatch khi kiểm tra missing values
        if dtype in ('float', 'double'):
            cond = F.col(c).isNull() | F.isnan(F.col(c))
        else:
            cond = F.col(c).isNull()
        exprs.append(F.round((F.sum(F.when(cond, 1).otherwise(0)) / total) * 100, 2).alias(c))
    
    return df.select(*exprs).collect()[0].asDict()

def run_eda(spark: SparkSession, cfg: dict) -> dict:
    bucket = cfg["minio"]["bucket"].strip("/")
    df_rev_raw = spark.read.parquet(f"s3a://{bucket}/landing/reviews")
    df_meta_raw = spark.read.parquet(f"s3a://{bucket}/landing/metadata")
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary_metrics": {},
        "detailed_metrics": {
            "reviews": {},
            "meta": {},
            "join": {}
        }
    }

    logger.info("=== ĐANG TÍNH TOÁN CÁC CHỈ SỐ CỐT LÕI ===")
    
    # 1. Deduplication (Lọc trùng) 
    # KHÔNG DÙNG CACHE() Ở ĐÂY ĐỂ TRÁNH OOM ERROR
    df_rev = df_rev_raw.dropDuplicates(["reviewer_id", "parent_asin", "timestamp"])
    df_meta = df_meta_raw.dropDuplicates(["parent_asin"])
    
    # 2. Count cơ bản
    clean_reviews = df_rev.count()
    clean_meta = df_meta.count()
    num_users = df_rev.select("reviewer_id").distinct().count()
    num_items = df_rev.select("parent_asin").distinct().count()
    
    # 3. Tính toán các chỉ số trung bình
    avg_rating = df_rev.select(F.avg("rating")).collect()[0][0]
    
    # 4. Join Analysis (Orphan Reviews)
    # KHÔNG DÙNG CACHE()
    df_joined = df_rev.join(df_meta, "parent_asin", "left")
    orphan_count = df_joined.filter(F.col("item_title").isNull()).count()

    # ==========================================
    # ĐƯA CHỈ SỐ VÀO BLOCK SUMMARY METRICS JSON
    # ==========================================
    report["summary_metrics"] = {
        "num_users": num_users,
        "num_products": clean_meta,
        "total_reviews_clean": clean_reviews,
        "avg_rating": round(avg_rating, 2) if avg_rating else 0,
        "avg_reviews_per_user": round(clean_reviews / num_users, 2) if num_users > 0 else 0,
        "avg_reviews_per_item": round(clean_reviews / num_items, 2) if num_items > 0 else 0,
        "sparsity_pct": round(100 * (1 - (clean_reviews / (num_users * num_items))), 5) if (num_users * num_items) > 0 else 100,
        "orphan_reviews_pct": round((orphan_count / clean_reviews) * 100, 2) if clean_reviews > 0 else 0
    }

    logger.info("=== ĐANG TÍNH TOÁN CÁC CHỈ SỐ CHI TIẾT (Phân phối, Missing) ===")
    
    # Missing %
    report["detailed_metrics"]["reviews"]["missing_pct"] = get_missing_pct(df_rev)
    report["detailed_metrics"]["meta"]["missing_pct"] = get_missing_pct(df_meta)
    
    # Rating Distribution
    rating_dist = df_rev.groupBy("rating").count().orderBy("rating").collect()
    report["detailed_metrics"]["reviews"]["rating_distribution"] = {str(r["rating"]): r["count"] for r in rating_dist}
    
    # Text Length Quantiles
    quantiles = df_rev.approxQuantile("text_len", [0.25, 0.5, 0.75, 0.95], 0.05)
    report["detailed_metrics"]["reviews"]["text_len_quantiles"] = {
        "Q1": quantiles[0], "Median": quantiles[1], "Q3": quantiles[2], "P95": quantiles[3]
    }
    
    # Verified Purchase
    verified_dist = df_rev.groupBy("verified_purchase").count().collect()
    report["detailed_metrics"]["reviews"]["verified_purchase_dist"] = {
        str(r["verified_purchase"]): r["count"] for r in verified_dist
    }

    return report

if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    spark = create_spark(cfg)
    
    try:
        final_report = run_eda(spark, cfg)
        
        # Lưu báo cáo ra file JSON
        out_dir = Path("data/logs")
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "eda_report.json"
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=4, ensure_ascii=False)
            
        logger.info(f"🎉 EDA hoàn tất! Báo cáo đã được lưu tại: {report_path}")
        
    except Exception as e:
        logger.error(f"Lỗi EDA: {e}", exc_info=True)
    finally:
        spark.stop()