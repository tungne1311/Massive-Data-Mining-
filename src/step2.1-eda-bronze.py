import logging
import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# ==============================================================================
# LẮP ĐẶT LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Bronze_EDA")

# ==============================================================================
# 1. KHỞI TẠO SPARK & LOAD DATA
# ==============================================================================
def create_spark_session(app_name: str = "Bronze_EDA") -> SparkSession:
    """Khởi tạo SparkSession chuẩn production."""
    logger.info("Khởi tạo SparkSession...")
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark: SparkSession, review_path: str, meta_path: str) -> tuple[DataFrame, DataFrame]:
    """Load dữ liệu từ Parquet (Bronze layer)."""
    logger.info(f"Đọc dữ liệu REVIEW từ: {review_path}")
    df_rev = spark.read.parquet(review_path)
    
    logger.info(f"Đọc dữ liệu METADATA từ: {meta_path}")
    df_meta = spark.read.parquet(meta_path)
    
    return df_rev, df_meta


# ==============================================================================
# 2. TỔNG QUAN DỮ LIỆU
# ==============================================================================
def basic_info(df: DataFrame, name: str, keys: list[str]) -> None:
    """Phân tích thông tin cơ bản: Schema, count, duplicate."""
    logger.info(f"--- BASIC INFO: {name.upper()} ---")
    
    # Schema
    df.printSchema()
    
    # Total count
    total_count = df.count()
    logger.info(f"Tổng số dòng ({name}): {total_count:,}")
    
    if total_count == 0:
        logger.warning(f"Bảng {name} rỗng!")
        return

    # Check duplicate toàn bộ dòng
    distinct_count = df.dropDuplicates().count()
    full_duplicates = total_count - distinct_count
    logger.info(f"Số dòng duplicate (toàn bộ cột): {full_duplicates:,} ({full_duplicates/total_count*100:.2f}%)")
    
    # Check duplicate theo keys
    if keys:
        distinct_keys_count = df.dropDuplicates(keys).count()
        key_duplicates = total_count - distinct_keys_count
        logger.info(f"Số dòng duplicate theo key {keys}: {key_duplicates:,} ({key_duplicates/total_count*100:.2f}%)")


# ==============================================================================
# 3. PHÂN TÍCH MISSING VALUES
# ==============================================================================
def missing_analysis(df: DataFrame, name: str) -> None:
    """Tính tỷ lệ % Null cho từng cột."""
    logger.info(f"--- MISSING VALUES: {name.upper()} ---")
    
    total_count = df.count()
    if total_count == 0:
        return
        
    # Tạo biểu thức tính tỷ lệ % null cho tất cả các cột trong 1 lần duyệt (tránh collect nhiều lần)
    exprs = [
        F.round((F.sum(F.when(F.col(c).isNull() | F.isnan(F.col(c)), 1).otherwise(0)) / total_count) * 100, 2).alias(c)
        for c in df.columns
    ]
    
    df_missing = df.select(*exprs)
    df_missing.show(truncate=False)


# ==============================================================================
# 4. UNIVARIATE ANALYSIS
# ==============================================================================
def univariate_analysis(df_rev: DataFrame, df_meta: DataFrame) -> None:
    """Phân tích đơn biến cho Numeric, Categorical và Time variables."""
    logger.info("--- UNIVARIATE ANALYSIS ---")
    
    # --- 4.1. Numeric Analysis (REVIEW) ---
    logger.info("[Numeric] Phân phối Rating, Text_len, Helpful_vote")
    num_cols = ["rating", "text_len", "helpful_vote"]
    df_rev.select(num_cols).summary("count", "mean", "stddev", "min", "max").show()
    
    # Tính Percentiles và IQR (Phát hiện Outlier)
    for c in num_cols:
        # Dùng approxQuantile để tối ưu cho Big Data
        quantiles = df_rev.approxQuantile(c, [0.25, 0.5, 0.75], 0.05)
        if len(quantiles) == 3:
            q1, median, q3 = quantiles
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_count = df_rev.filter((F.col(c) < lower_bound) | (F.col(c) > upper_bound)).count()
            logger.info(f"[{c}] Q1: {q1}, Median: {median}, Q3: {q3}, IQR: {iqr}")
            logger.info(f"[{c}] Outliers (ngoài [{lower_bound}, {upper_bound}]): {outliers_count:,} dòng")

    # --- 4.2. Categorical Analysis ---
    logger.info("[Categorical] Top values & Cardinality")
    
    # Review: verified_purchase
    logger.info("Phân phối Verified Purchase:")
    df_rev.groupBy("verified_purchase").count().orderBy(F.desc("count")).show()
    
    # Meta: brand, main_category, price_bucket
    cat_cols_meta = ["brand", "main_category", "price_bucket"]
    for c in cat_cols_meta:
        cardinality = df_meta.select(F.approx_count_distinct(c)).collect()[0][0]
        logger.info(f"[{c}] Số lượng giá trị duy nhất (ước tính): {cardinality:,}")
        logger.info(f"[{c}] Top 5 phổ biến nhất:")
        df_meta.groupBy(c).count().orderBy(F.desc("count")).limit(5).show(truncate=False)

    # --- 4.3. Time Analysis (REVIEW) ---
    logger.info("[Time] Phân tích theo thời gian")
    # Convert timestamp (chuẩn unix) sang datetime
    df_time = df_rev.filter(F.col("timestamp").isNotNull()) \
                    .withColumn("review_date", F.to_timestamp(F.from_unixtime(F.col("timestamp")))) \
                    .withColumn("year", F.year("review_date")) \
                    .withColumn("month", F.month("review_date"))
    
    logger.info("Phân phối lượt Review theo Năm (Top 10):")
    df_time.groupBy("year").count().orderBy(F.desc("year")).limit(10).show()


# ==============================================================================
# 5. DATA QUALITY CHECKS
# ==============================================================================
def data_quality_checks(df_rev: DataFrame) -> None:
    """Kiểm tra các quy tắc nghiệp vụ (Business rules) và chất lượng dữ liệu."""
    logger.info("--- DATA QUALITY CHECKS (REVIEW) ---")
    
    # 1. Rating ngoài khoảng [1-5]
    invalid_rating = df_rev.filter(~F.col("rating").between(1.0, 5.0)).count()
    logger.info(f"Số lượng Rating không hợp lệ (ngoài 1-5): {invalid_rating:,}")
    
    # 2. Text_len <= 0
    invalid_text_len = df_rev.filter(F.col("text_len") <= 0).count()
    logger.info(f"Số lượng review có text_len <= 0: {invalid_text_len:,}")
    
    # 3. Helpful_vote < 0
    invalid_votes = df_rev.filter(F.col("helpful_vote") < 0).count()
    logger.info(f"Số lượng review có helpful_vote < 0: {invalid_votes:,}")
    
    # 4. Độ lệch phân phối Rating (Skewness)
    skewness_val = df_rev.select(F.skewness("rating")).collect()[0][0]
    logger.info(f"Độ lệch (Skewness) của Rating: {skewness_val:.4f} (Âm: lệch phải/nhiều 5 sao, Dương: lệch trái)")


# ==============================================================================
# 6. JOIN ANALYSIS
# ==============================================================================
def join_analysis(df_rev: DataFrame, df_meta: DataFrame) -> DataFrame:
    """Thực hiện join bảng Review và Metadata, kiểm tra tỷ lệ khớp."""
    logger.info("--- JOIN ANALYSIS ---")
    
    count_rev = df_rev.count()
    
    # Thực hiện Left Join để xem có bao nhiêu review tìm thấy metadata
    df_joined = df_rev.alias("r").join(
        df_meta.alias("m"),
        F.col("r.parent_asin") == F.col("m.parent_asin"),
        "left"
    )
    
    # Cache bảng join để dùng cho các bước Bivariate tiếp theo
    df_joined.cache()
    count_joined = df_joined.count()
    
    # Tính toán tỷ lệ match (bao nhiêu review có item_title từ metadata)
    matched_meta = df_joined.filter(F.col("m.parent_asin").isNotNull()).count()
    orphan_reviews = count_joined - matched_meta
    
    logger.info(f"Tổng số Reviews gốc: {count_rev:,}")
    logger.info(f"Số Reviews khớp với Metadata: {matched_meta:,} ({(matched_meta/count_rev)*100:.2f}%)")
    logger.info(f"Số Reviews KHÔNG có Metadata (Orphan): {orphan_reviews:,} ({(orphan_reviews/count_rev)*100:.2f}%)")
    
    return df_joined


# ==============================================================================
# 7. BIVARIATE ANALYSIS
# ==============================================================================
def bivariate_analysis(df_joined: DataFrame) -> None:
    """Phân tích đa biến (tương quan giữa Rating và các yếu tố khác)."""
    logger.info("--- BIVARIATE ANALYSIS ---")
    
    # --- 7.1 Correlation Matrix ---
    logger.info("[Correlation] Tương quan tuyến tính (Pearson)")
    corr_len = df_joined.stat.corr("rating", "text_len")
    corr_vote = df_joined.stat.corr("rating", "helpful_vote")
    logger.info(f"Correlation(Rating, Text_len)    = {corr_len:.4f}")
    logger.info(f"Correlation(Rating, Helpful_vote) = {corr_vote:.4f}")

    # --- 7.2 Rating vs Categorical (Dùng Sample để tối ưu cho Data lớn) ---
    logger.info("[Bivariate] Trung bình Rating theo Verified Purchase")
    df_joined.groupBy("verified_purchase").agg(
        F.round(F.avg("rating"), 2).alias("avg_rating"),
        F.count("*").alias("count")
    ).show()

    logger.info("[Bivariate] Trung bình Rating theo Price Bucket")
    df_joined.filter(F.col("price_bucket").isNotNull()) \
             .groupBy("price_bucket") \
             .agg(F.round(F.avg("rating"), 2).alias("avg_rating"), F.count("*").alias("count")) \
             .orderBy(F.desc("avg_rating")) \
             .show()

    logger.info("[Bivariate] Top 5 Category có lượng Review cao nhất và Avg Rating")
    df_joined.filter(F.col("main_category").isNotNull()) \
             .groupBy("main_category") \
             .agg(F.round(F.avg("rating"), 2).alias("avg_rating"), F.count("*").alias("count")) \
             .orderBy(F.desc("count")) \
             .limit(5) \
             .show(truncate=False)


# ==============================================================================
# 8. FEATURE INSIGHTS
# ==============================================================================
def feature_insights(df_joined: DataFrame) -> None:
    """Phân tích nhãn và đề xuất đặc trưng cho Machine Learning."""
    logger.info("--- FEATURE INSIGHTS & ENGINEERING RECOMMENDATIONS ---")
    
    # Phân phối nhãn (Rating)
    logger.info("Label Distribution (Rating %):")
    total = df_joined.count()
    df_joined.groupBy("rating").count() \
             .withColumn("percentage", F.round((F.col("count") / total) * 100, 2)) \
             .orderBy(F.desc("rating")) \
             .show()
             
    logger.info("""
    [💡 ĐỀ XUẤT FEATURE ENGINEERING]
    1. Từ Cột Review:
       - Tạo feature 'is_long_review' (nếu text_len > Q3).
       - Khai thác NLP (TF-IDF, Embeddings) từ cột 'review_text'.
       - Trích xuất 'hour_of_day' hoặc 'day_of_week' từ cột 'timestamp' để đánh giá hành vi user.
       
    2. Từ Cột Metadata:
       - Tách các từ khóa từ cột 'features' (nếu có) thành cờ boolean (has_feature_X).
       - Dùng 'price_bucket' kết hợp 'main_category' để tạo feature cross (vd: High_End_Electronics).
       
    3. Tương tác User-Item:
       - Tính user_avg_rating, item_avg_rating để làm feature base cho mô hình RecSys.
    """)


# ==============================================================================
# MAIN EXECUTION THREAD
# ==============================================================================
if __name__ == "__main__":
    # GIẢ ĐỊNH PATH CHO BRONZE LAYER (Sửa lại đường dẫn thực tế khi chạy)
    REVIEW_PATH = "s3a://bronze/reviews"
    META_PATH = "s3a://bronze/metadata"
    
    # 1. Khởi tạo Spark
    spark_sess = create_spark_session()
    
    try:
        # Dùng dữ liệu dummy rỗng có schema để test script nếu path không tồn tại, 
        # Trong thực tế, hàm load_data sẽ tự động đọc schema từ Parquet
        logger.info("Bắt đầu tiến trình EDA Bronze Layer...")
        
        # NOTE: Bỏ comment dòng dưới và thay path khi chạy thực tế
        # df_reviews, df_metadata = load_data(spark_sess, REVIEW_PATH, META_PATH)
        
        # -- ĐOẠN CODE DƯỚI ĐÂY LÀ GIẢ LẬP ĐỂ SCRIPT CHẠY ĐƯỢC MÀ KHÔNG BỊ CRASH NẾU KHÔNG CÓ FILE --
        from pyspark.sql.types import StructType, StructField, StringType, FloatType, LongType, BooleanType
        rev_schema = StructType([
            StructField("reviewer_id", StringType(), False), StructField("parent_asin", StringType(), False),
            StructField("rating", FloatType(), False), StructField("review_text", StringType(), True),
            StructField("text_len", IntegerType(), True), StructField("timestamp", LongType(), True),
            StructField("helpful_vote", IntegerType(), True), StructField("verified_purchase", BooleanType(), True),
            StructField("ingest_date", StringType(), True), StructField("source_name", StringType(), True),
            StructField("year_month", StringType(), False)
        ])
        meta_schema = StructType([
            StructField("parent_asin", StringType(), False), StructField("item_title", StringType(), True),
            StructField("brand", StringType(), True), StructField("main_category", StringType(), True),
            StructField("features", StringType(), True), StructField("price_bucket", StringType(), True),
            StructField("ingest_date", StringType(), True), StructField("source_name", StringType(), True)
        ])
        df_reviews = spark_sess.createDataFrame(spark_sess.sparkContext.emptyRDD(), rev_schema)
        df_metadata = spark_sess.createDataFrame(spark_sess.sparkContext.emptyRDD(), meta_schema)
        # -----------------------------------------------------------------------------------------
        
        # Cache Raw DataFrame để tối ưu các bước quét nhiều lần
        df_reviews.cache()
        df_metadata.cache()

        # 2. Basic Info
        basic_info(df_reviews, "Reviews", ["reviewer_id", "parent_asin"])
        basic_info(df_metadata, "Metadata", ["parent_asin"])
        
        # 3. Missing Analysis
        missing_analysis(df_reviews, "Reviews")
        missing_analysis(df_metadata, "Metadata")
        
        # 4. Data Quality
        data_quality_checks(df_reviews)
        
        # 5. Univariate
        univariate_analysis(df_reviews, df_metadata)
        
        # 6. Join Tables
        df_joined_full = join_analysis(df_reviews, df_metadata)
        
        # 7. Bivariate
        bivariate_analysis(df_joined_full)
        
        # 8. Feature Insights
        feature_insights(df_joined_full)

    except Exception as e:
        logger.error(f"Lỗi trong quá trình chạy EDA: {str(e)}")
        
    finally:
        logger.info("Dọn dẹp bộ nhớ và tắt Spark...")
        # Unpersist các RDD/DataFrame đã cache
        try:
            df_reviews.unpersist()
            df_metadata.unpersist()
            df_joined_full.unpersist()
        except NameError:
            pass # Bỏ qua nếu DF chưa được tạo do lỗi trước đó
            
        spark_sess.stop()
        logger.info("HOÀN TẤT EDA PIPELINE!")