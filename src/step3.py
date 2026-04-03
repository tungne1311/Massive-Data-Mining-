import logging
import time
from datetime import timedelta
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

def read_landing_zone(spark: SparkSession, cfg: dict, dataset_type: str) -> DataFrame:
    bucket = cfg["minio"]["bucket"].strip("/")
    path = f"s3a://{bucket}/landing/{dataset_type}"
    try:
        return spark.read.parquet(path)
    except Exception:
        return spark.createDataFrame([], schema=None)

def clean_and_join(df_reviews: DataFrame, df_metadata: DataFrame, cfg: dict) -> DataFrame:
    sv_cfg = cfg.get("silver", {})
    max_year = sv_cfg.get("max_year", 2025)
    
    # 1. METADATA: Lọc cột và ép kiểu trước khi broadcast
    cols_meta = ["parent_asin", "brand", "main_category", "price_bucket"]
    df_meta = df_metadata.select(*cols_meta) \
                         .dropDuplicates(["parent_asin"]) \
                         .dropna(subset=["parent_asin"])
                         
    # 2. REVIEW: Cắt gọn payload, nhưng GIỮ LẠI review_text
    cols_rev = [
        F.col("reviewer_id"),
        F.col("parent_asin"),
        F.col("rating").cast("float"),
        F.col("timestamp"),
        F.col("helpful_vote").cast("int"),
        F.col("verified_purchase").cast("boolean"),
        F.col("text_len").cast("int"),
        F.col("year_month"),
        F.col("review_text") # <-- GIỮ LẠI CỘT NÀY CHO MÔ HÌNH NLP SAU NÀY
    ]
    
    df_rev = df_reviews.select(*cols_rev) \
                       .dropna(subset=["reviewer_id", "parent_asin", "rating", "timestamp"])

    # 3. XỬ LÝ TEXT & HASH ĐỂ TÌM SPAM
    df_rev = df_rev.withColumn("review_text", F.coalesce(F.col("review_text"), F.lit("")))
    df_rev = df_rev.withColumn("_text_hash", F.when(F.col("text_len") > 50, F.xxhash64(F.col("review_text"))).otherwise(F.lit(None)))
    
    # Lọc trùng lặp dòng theo ID, ASIN và thời gian (Spill to disk nếu RAM đầy)
    df_rev = df_rev.dropDuplicates(["reviewer_id", "parent_asin", "timestamp"])

    # 4. CHẶN LINEAGE (Map-side Barrier): Đẩy xuống đĩa để xả RAM trước khi làm các bước phức tạp
    df_rev.persist(StorageLevel.DISK_ONLY)

    # 5. Xử lý thời gian và các default values
    df_rev = df_rev.withColumn("review_time", F.to_timestamp(F.from_unixtime(F.col("timestamp"))))
    df_rev = df_rev.filter(F.year("review_time") <= max_year)
    
    df_rev = (df_rev.withColumn("helpful_vote", F.coalesce(F.col("helpful_vote"), F.lit(0)))
                    .withColumn("verified_purchase", F.coalesce(F.col("verified_purchase"), F.lit(False))))

    # 6. BROADCAST JOIN BẢNG META (Hoàn toàn không có Shuffle mạng)
    df = df_rev.join(F.broadcast(df_meta), on="parent_asin", how="left")
    
    df = (df.withColumn("brand", F.coalesce(F.col("brand"), F.lit("unknown")))
            .withColumn("main_category", F.coalesce(F.col("main_category"), F.lit("unknown")))
            .withColumn("price_bucket", F.coalesce(F.col("price_bucket"), F.lit("unknown"))))
            
    df = df.withColumn("is_short_review", (F.col("text_len") < sv_cfg.get("short_review_len", 20)).cast("boolean"))
    
    # 7. MAP-REDUCE TÌM SPAM: Chỉ đếm và Broadcast các mã hash CÓ LẶP LẠI (>1) để chống nổ RAM Driver
    df_spam_count = df.filter(F.col("_text_hash").isNotNull()) \
                      .groupBy("_text_hash").count() \
                      .filter(F.col("count") > 1) \
                      .withColumnRenamed("count", "hash_count")
                      
    df = df.join(F.broadcast(df_spam_count), on="_text_hash", how="left")
    
    df = df.withColumn("is_duplicate_text", 
                       F.when(F.col("hash_count").isNotNull(), F.lit(True))
                        .otherwise(F.lit(False)))\
           .drop("_text_hash", "hash_count")
    
    df_rev.unpersist() # Giải phóng bản nháp
    return df

def compute_signal_scores(df: DataFrame, rel_cfg: dict) -> DataFrame:
    df = df.withColumn("verified_score", F.when(F.col("verified_purchase") == True, F.lit(1.0)).otherwise(F.lit(float(rel_cfg["unverified_value"]))))
    df = df.withColumn("text_quality", F.least(F.col("text_len").cast("float") / F.lit(100.0), F.lit(1.0)))
    
    # Map-Reduce cục bộ: Tìm max helpful vote của từng tháng rồi Broadcast
    df_max_helpful = df.groupBy("year_month").agg(F.max("helpful_vote").alias("max_helpful_month"))
    df = df.join(F.broadcast(df_max_helpful), on="year_month", how="left")
    
    df = df.withColumn("helpful_score", F.when(F.col("max_helpful_month") == 0, F.lit(0.0).cast("float"))
         .otherwise((F.log1p(F.col("helpful_vote").cast("float")) / F.log1p(F.col("max_helpful_month").cast("float"))).cast("float"))).drop("max_helpful_month")

    df = df.withColumn("reliability_score",
        (F.lit(float(rel_cfg["w_verified"])) * F.col("verified_score") +
        F.lit(float(rel_cfg["w_text"])) * F.col("text_quality") +
        F.lit(float(rel_cfg["w_helpful"])) * F.col("helpful_score")).cast("float"))
        
    return df

def run(spark: SparkSession, cfg: dict, rel_cfg: dict, config_id: str, cleanup_landing: bool = False) -> DataFrame:
    t_start = time.perf_counter()
    logger.info(f"=== Bước 3: Silver Cleaning (No-Shuffle MapReduce & Keep Review Text) ===")

    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    df_reviews = read_landing_zone(spark, cfg, "reviews")
    df_metadata = read_landing_zone(spark, cfg, "metadata")
    
    if df_reviews.isEmpty():
        return spark.createDataFrame([], schema=None)

    df = clean_and_join(df_reviews, df_metadata, cfg)
    df = compute_signal_scores(df, rel_cfg)

    bucket = cfg["minio"]["bucket"].strip("/")
    
    # Đường dẫn xuất ra thư mục silver/<config_id>
    out_path = f"s3a://{bucket}/silver/{config_id}"
    
    # 🚀 TỐI ƯU CỐT LÕI: MAP-SIDE LOCAL SORT
    df_out = df.sortWithinPartitions("year_month")

    (df_out.write.mode("overwrite")  
        .partitionBy("year_month")
        .option("maxRecordsPerFile", 250000) 
        .option("compression", "zstd")
        .parquet(out_path))

    logger.info(f"THỜI GIAN CHẠY BƯỚC 3: {str(timedelta(seconds=int(time.perf_counter() - t_start)))}")
    return df_out