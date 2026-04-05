"""
Bước 5 — Temporal Split (Super Lean Version)
"""
import logging
from typing import Optional
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

def _join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"

def _split_path(cfg: dict, config_id: str, split: str) -> str:
    return _join_uri(cfg["paths"]["splits_base"], f"config_id={config_id}", split)

def _add_months(year_month: str, delta: int) -> str:
    year, month = int(year_month[:4]), int(year_month[5:7])
    total_months = year * 12 + (month - 1) + delta
    y, m = divmod(total_months, 12)
    return f"{y:04d}-{m+1:02d}"

def resolve_split_months(df_labeled: DataFrame, cfg: dict) -> tuple[list[str], list[str], list[str]]:
    valid_months_df = df_labeled.filter(F.col("year_month") != "unknown_date")
    row = valid_months_df.agg(
        F.max("year_month").alias("mx"),
        F.collect_set("year_month").alias("all_months")
    ).collect()[0]

    max_month = row["mx"]
    all_months = sorted(list(row["all_months"]))
    T = max_month
    T_m1, T_m2, T_m3 = _add_months(T, -1), _add_months(T, -2), _add_months(T, -3)

    train_months = [m for m in all_months if m <= T_m3]
    if "unknown_date" not in train_months: train_months.append("unknown_date")
    val_months = [m for m in all_months if T_m2 <= m <= T_m1]
    test_months = [m for m in all_months if m == T]

    return train_months, val_months, test_months

def _write_and_stats(df: DataFrame, cfg: dict, config_id: str, split_name: str) -> dict:
    # 🚀 CHIẾN THUẬT: Tính stats trước khi ghi để tận dụng dữ liệu đang trong RAM/Disk
    logger.info(f"  Đang tính thống kê tập {split_name}...")
    stats_row = df.agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("reviewer_id").alias("n_users"),
        F.countDistinct("parent_asin").alias("n_items"),
    ).collect()[0]
    
    stats = {
        "n_rows": int(stats_row["n_rows"]),
        "n_users": int(stats_row["n_users"]),
        "n_items": int(stats_row["n_items"]),
    }

    out_path = _split_path(cfg, config_id, split_name)
    logger.info(f"  Ghi {split_name} → {out_path}")
    (df.sortWithinPartitions("year_month")
       .write.mode("overwrite")
       .option("compression", "zstd")
       .parquet(out_path))
    
    return stats

def run(spark: SparkSession, cfg: dict, config_id: str, df_labeled: Optional[DataFrame] = None) -> dict:
    logger.info(f"=== Bước 5: Temporal Split (Super Lean) — config_id={config_id} ===")

    if df_labeled is None:
        labeled_path = _join_uri(cfg["paths"]["silver_base"], "labeled_interactions", f"config_id={config_id}")
        df_labeled = spark.read.parquet(labeled_path)

    # 1. Xác định tháng (chạy nhanh, không tốn RAM)
    train_months, val_months, test_months = resolve_split_months(df_labeled, cfg)
    split_stats = {}

    # 2. XỬ LÝ TUẦN TỰ: Xong tập nào, giải phóng tập đó ngay (Sequential Processing)
    
    # --- TẬP TRAIN ---
    df_train = df_labeled.filter(F.col("year_month").isin(train_months)).persist(StorageLevel.DISK_ONLY)
    split_stats["train"] = _write_and_stats(df_train, cfg, config_id, "train")
    
    # Lấy sẵn danh sách User/Item để lọc Val/Test (Dùng Broadcast để tránh Shuffle Join)
    train_users = df_train.select("reviewer_id").distinct().persist(StorageLevel.DISK_ONLY)
    train_items = df_train.select("parent_asin").distinct().persist(StorageLevel.DISK_ONLY)
    
    df_train.unpersist() # 🚀 GIẢI PHÓNG TRAIN NGAY LẬP TỨC

    # --- TẬP VAL ---
    df_val = df_labeled.filter(F.col("year_month").isin(val_months)) \
                       .join(train_users, "reviewer_id", "left_semi") \
                       .join(train_items, "parent_asin", "left_semi") \
                       .persist(StorageLevel.DISK_ONLY)
    split_stats["val"] = _write_and_stats(df_val, cfg, config_id, "val")
    df_val.unpersist() # 🚀 GIẢI PHÓNG VAL

    # --- TẬP TEST ---
    df_test = df_labeled.filter(F.col("year_month").isin(test_months)) \
                        .join(train_users, "reviewer_id", "left_semi") \
                        .join(train_items, "parent_asin", "left_semi") \
                        .persist(StorageLevel.DISK_ONLY)
    split_stats["test"] = _write_and_stats(df_test, cfg, config_id, "test")
    df_test.unpersist() # 🚀 GIẢI PHÓNG TEST
    
    train_users.unpersist()
    train_items.unpersist()

    logger.info("  ✓ Bước 5 xong.")
    return {"stats": split_stats}