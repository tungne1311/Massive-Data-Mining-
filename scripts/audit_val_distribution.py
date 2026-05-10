"""
scripts/audit_val_distribution.py — Chẩn đoán phân phối Val Ground Truth

Chạy standalone (không cần pipeline chạy lại) để kiểm tra:
  1. Phân phối popularity_group trong val_gt vs train
  2. So sánh tỉ lệ COLD_START/TAIL/HEAD giữa hai tập
  3. Cảnh báo nếu vượt ngưỡng an toàn

USAGE:
  docker compose run --rm pipeline python scripts/audit_val_distribution.py
  hoặc trực tiếp:
  python src/scripts/audit_val_distribution.py
"""

import logging
import os
import sys
from pathlib import Path

import yaml

# ─── Bootstrap sys.path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("val_audit")

# ─── THRESHOLDS ──────────────────────────────────────────────────────────────
THRESHOLDS = {
    "HEAD":       {"warn_below": 10.0,  "direction": "below"},
    "MID":        {"warn_below":  3.0,  "direction": "below"},
    "TAIL":       {"warn_above": 85.0,  "direction": "above"},
    "COLD_START": {"warn_above": 35.0,  "direction": "above"},
}

SEPARATOR = "=" * 70


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _s3a(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


def build_spark(cfg: dict):
    from pyspark.sql import SparkSession
    sc = cfg["spark"]
    mn = cfg["minio"]
    return (
        SparkSession.builder
        .appName("val_distribution_audit")
        .master(sc.get("master", "local[4]"))
        .config("spark.driver.memory",               sc.get("driver_memory", "8g"))
        .config("spark.executor.memory",             sc.get("executor_memory", "8g"))
        .config("spark.sql.shuffle.partitions",      "24")
        .config("spark.hadoop.fs.s3a.endpoint",      mn["endpoint"])
        .config("spark.hadoop.fs.s3a.access.key",    mn["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key",    mn["secret_key"])
        .config("spark.hadoop.fs.s3a.path.style.access",         "true")
        .config("spark.hadoop.fs.s3a.impl",          "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled",    "false")
        .getOrCreate()
    )


def audit_val_distribution(spark, cfg: dict) -> dict:
    """
    Đọc val_ground_truth + train popularity → so sánh distribution.
    Returns dict với summary để có thể dùng trong CI/testing.
    """
    from pyspark.sql import functions as F

    logger.info(SEPARATOR)
    logger.info("🔍 AUDIT: Val Ground Truth Distribution")
    logger.info(SEPARATOR)

    val_gt_path   = _s3a(cfg, "silver", "silver_val_ground_truth.parquet")
    train_pop_path = _s3a(cfg, "silver", "silver_item_popularity.parquet")
    train_inter_path = _s3a(cfg, "silver", "silver_interactions_train.parquet")

    # ── Đọc val_gt ───────────────────────────────────────────────────────────
    logger.info(f"📂 Đọc: {val_gt_path}")
    df_val_gt = spark.read.parquet(val_gt_path)
    total_val = df_val_gt.count()
    logger.info(f"  Tổng interactions val: {total_val:,}")

    val_dist_rows = df_val_gt.groupBy("popularity_group").count().collect()
    val_dist = {row["popularity_group"]: row["count"] for row in val_dist_rows}

    # Đếm số lượng mặt hàng duy nhất (unique items) trong tập Val
    val_unique_items_rows = df_val_gt.select("parent_asin", "popularity_group").distinct().groupBy("popularity_group").count().collect()
    val_unique_items = {row["popularity_group"]: row["count"] for row in val_unique_items_rows}

    # ── Đọc train popularity ─────────────────────────────────────────────────
    logger.info(f"📂 Đọc: {train_pop_path}")
    df_train_pop = spark.read.parquet(train_pop_path)
    total_train_items = df_train_pop.count()
    logger.info(f"  Tổng items trong train: {total_train_items:,}")

    train_dist_rows = df_train_pop.groupBy("popularity_group").count().collect()
    train_dist = {row["popularity_group"]: row["count"] for row in train_dist_rows}

    # ── Đọc train interactions để tính interaction-level distribution ─────────
    logger.info(f"📂 Đọc: {train_inter_path}")
    df_train_inter = spark.read.parquet(train_inter_path)
    total_train_inter = df_train_inter.count()
    train_inter_dist_rows = df_train_inter.groupBy("popularity_group").count().collect()
    train_inter_dist = {row["popularity_group"]: row["count"] for row in train_inter_dist_rows}

    # ── Bảng so sánh ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("📊 Distribution Comparison:")
    logger.info(f"  {'Group':12s} | {'Train items':>12s} | {'Train %':>8s} | {'Train inter':>12s} | {'Train inter%':>12s} | {'Val GT':>10s} | {'Val %':>7s} | {'Val items':>9s} | Status")
    logger.info(f"  {'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*7}-+-{'-'*9}-+--------")

    results = {}
    any_warning = False

    for group in ["HEAD", "MID", "TAIL", "COLD_START"]:
        t_item_cnt   = train_dist.get(group, 0)
        t_item_pct   = t_item_cnt / max(total_train_items, 1) * 100
        t_inter_cnt  = train_inter_dist.get(group, 0)
        t_inter_pct  = t_inter_cnt / max(total_train_inter, 1) * 100
        v_cnt        = val_dist.get(group, 0)
        v_pct        = v_cnt / max(total_val, 1) * 100
        v_item_cnt   = val_unique_items.get(group, 0)

        thresh = THRESHOLDS.get(group, {})
        status = "✅ OK"
        if thresh.get("direction") == "above" and v_pct > thresh.get("warn_above", 999):
            status = "⚠️  HIGH"
            any_warning = True
        elif thresh.get("direction") == "below" and v_pct < thresh.get("warn_below", 0):
            status = "⚠️  LOW"
            any_warning = True

        logger.info(
            f"  {group:12s} | {t_item_cnt:>12,} | {t_item_pct:>7.1f}% "
            f"| {t_inter_cnt:>12,} | {t_inter_pct:>11.1f}% "
            f"| {v_cnt:>10,} | {v_pct:>6.1f}% | {v_item_cnt:>9,} | {status}"
        )

        results[group] = {
            "train_item_count":   t_item_cnt,
            "train_item_pct":     round(t_item_pct, 2),
            "train_inter_count":  t_inter_cnt,
            "train_inter_pct":    round(t_inter_pct, 2),
            "val_count":          v_cnt,
            "val_pct":            round(v_pct, 2),
            "val_item_count":     v_item_cnt,
            "status":             status,
        }

    # ── Leakage check: val items trong train ────────────────────────────────
    logger.info("")
    logger.info("🔍 Leakage Check: Val items có trong train không?")
    val_items   = df_val_gt.select("reviewer_id", "parent_asin").distinct()
    train_items = df_train_inter.select("reviewer_id", "parent_asin").distinct()
    leaked = val_items.join(train_items, ["reviewer_id", "parent_asin"], "inner").count()
    if leaked == 0:
        logger.info(f"  ✅ Không có leakage: 0 cặp (reviewer_id, parent_asin) trùng nhau.")
    else:
        logger.error(f"  ❌ LEAKAGE DETECTED: {leaked:,} cặp (reviewer_id, parent_asin) trùng giữa val và train!")

    results["leakage_count"] = leaked

    # ── 1 item per user check ────────────────────────────────────────────────
    logger.info("")
    logger.info("🔍 LOO Check: Val có đúng 1 item/user không?")
    multi_item_users = (
        df_val_gt.groupBy("reviewer_id").count()
        .filter("count > 1")
        .count()
    )
    if multi_item_users == 0:
        logger.info("  ✅ Đúng 1 item/user trong val (LOO chính xác).")
    else:
        logger.warning(f"  ⚠️  {multi_item_users:,} users có > 1 item trong val — LOO có thể bị lỗi!")

    results["multi_item_users"] = multi_item_users

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info(SEPARATOR)
    if any_warning:
        logger.warning(
            "⚠️  Val distribution LỆCH so với ngưỡng an toàn.\n"
            "   → Khi evaluate, ưu tiên report Recall@K_core (loại COLD_START)\n"
            "     và Recall@K theo từng nhóm HEAD/MID/TAIL riêng biệt."
        )
    else:
        logger.info("✅ Val distribution trong ngưỡng an toàn. Recall@K phản ánh đúng chất lượng mô hình.")
    logger.info(SEPARATOR)

    return results


def main():
    cfg_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    cfg = load_config(cfg_path)

    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    try:
        results = audit_val_distribution(spark, cfg)
        # Exit code 1 nếu có leakage (để dùng trong CI)
        if results.get("leakage_count", 0) > 0:
            sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
