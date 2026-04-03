import sys
from pathlib import Path
import pyspark.sql.functions as F

# Đảm bảo import được hàm từ pipeline_runner
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_runner import load_config, build_spark

def main():
    # 1. Load cấu hình và khởi tạo Spark
    cfg = load_config("config/config.yaml")
    spark = build_spark(cfg)
    
    bucket = cfg["minio"]["bucket"].strip("/")
    config_id = cfg["reliability_tuning"]["selected_config_id"]
    
    # Đường dẫn output của Step 3 mà chúng ta vừa sửa
    path = f"s3a://{bucket}/silver/{config_id}"
    
    print(f"\n[{path}] ĐANG ĐỌC DỮ LIỆU...\n" + "="*60)
    
    try:
        df = spark.read.parquet(path)
        
        # 1. Xem số lượng file vật lý (Partition files)
        files = df.inputFiles()
        print(f"📁 Tổng số file Parquet vật lý: {len(files):,}")
        
        # 2. Đếm tổng số dòng
        total_rows = df.count()
        print(f"📊 Tổng số dòng dữ liệu: {total_rows:,}\n")
        
        # 3. In cấu trúc (Schema)
        print("--- CẤU TRÚC BẢNG (SCHEMA) ---")
        df.printSchema()
        
        # 4. Xem mẫu 5 dòng dữ liệu
        print("--- MẪU DỮ LIỆU (5 DÒNG) ---")
        df.show(5, truncate=False)
        
        # 5. Phân phối theo tháng (Để kiểm tra Data Skew)
        print("--- THỐNG KÊ SỐ LƯỢNG THEO THÁNG (Top 20 tháng gần nhất) ---")
        df.groupBy("year_month").count().orderBy(F.desc("year_month")).show(20)
        
    except Exception as e:
        print(f"❌ Lỗi khi đọc dữ liệu: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()