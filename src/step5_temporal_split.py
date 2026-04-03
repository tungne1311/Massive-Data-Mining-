"""
Bước 5 — Temporal Split
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

def resolve_split_months(
    df_labeled: DataFrame,
    cfg: dict,
) -> tuple[list[str], list[str], list[str]]:
    ts_cfg = cfg.get("temporal_split", {})
    mode   = ts_cfg.get("mode", "auto")

    if mode == "explicit":
        return ts_cfg["train_months"], ts_cfg["val_months"], ts_cfg["test_months"]

    row = df_labeled.agg(
        F.max("year_month").alias("mx"),
        F.collect_set("year_month").alias("all_months")
    ).collect()[0]

    max_month = row["mx"]
    all_months = sorted(list(row["all_months"]))

    if max_month is None:
        raise ValueError("Không tìm thấy year_month hợp lệ trong dữ liệu!")

    T      = max_month
    T_m1   = _add_months(T, -1)
    T_m2   = _add_months(T, -2)
    T_m3   = _add_months(T, -3)

    train_months = [m for m in all_months if m <= T_m3]
    val_months   = [m for m in all_months if T_m2 <= m <= T_m1]
    test_months  = [m for m in all_months if m == T]

    return train_months, val_months, test_months

def _write_split(
    df: DataFrame,
    cfg: dict,
    config_id: str,
    split_name: str,
) -> None:
    out_path = _split_path(cfg, config_id, split_name)
    n_parts  = int(cfg["spark"].get("write_partitions", 24))
    mode     = cfg["silver"]["write_mode"]
    
    df_out = df.repartition(n_parts) if n_parts > 0 else df
    logger.info(f"  Ghi {split_name} → {out_path}")
    (
        df_out
          .write
          .mode(mode)
          .parquet(out_path)
    )

def _log_split_stats(df: DataFrame, split_name: str) -> dict:
    stats_row = df.agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("reviewer_id").alias("n_users"),
        F.countDistinct("parent_asin").alias("n_items"),
    ).collect()[0]

    stats = {
        "n_rows":  int(stats_row["n_rows"]),
        "n_users": int(stats_row["n_users"]),
        "n_items": int(stats_row["n_items"]),
    }
    return stats

def run(
    spark: SparkSession,
    cfg: dict,
    config_id: str,
    df_labeled: Optional[DataFrame] = None,
) -> dict:
    logger.info(f"=== Bước 5: Temporal Split — config_id={config_id} ===")

    if df_labeled is None:
        labeled_path = _join_uri(
            cfg["paths"]["silver_base"],
            "labeled_interactions",
            f"config_id={config_id}",
        )
        df_labeled = spark.read.parquet(labeled_path)

    df_labeled.persist(StorageLevel.DISK_ONLY)
    train_months, val_months, test_months = resolve_split_months(df_labeled, cfg)

    df_train = df_labeled.filter(F.col("year_month").isin(train_months))
    df_train.persist(StorageLevel.DISK_ONLY)

    train_users = F.broadcast(df_train.select("reviewer_id").distinct())
    train_items = F.broadcast(df_train.select("parent_asin").distinct())

    df_val = df_labeled.filter(F.col("year_month").isin(val_months)) \
                       .join(train_users, "reviewer_id", "left_semi") \
                       .join(train_items, "parent_asin", "left_semi")

    df_test = df_labeled.filter(F.col("year_month").isin(test_months)) \
                       .join(train_users, "reviewer_id", "left_semi") \
                       .join(train_items, "parent_asin", "left_semi")

    split_stats = {}

    _write_split(df_train, cfg, config_id, "train")
    split_stats["train"] = _log_split_stats(spark.read.parquet(_split_path(cfg, config_id, "train")), "train")

    _write_split(df_val, cfg, config_id, "val")
    split_stats["val"] = _log_split_stats(spark.read.parquet(_split_path(cfg, config_id, "val")), "val")

    _write_split(df_test, cfg, config_id, "test")
    split_stats["test"] = _log_split_stats(spark.read.parquet(_split_path(cfg, config_id, "test")), "test")

    df_train.unpersist()
    df_labeled.unpersist()

    return {
        "train_path":   _split_path(cfg, config_id, "train"),
        "val_path":     _split_path(cfg, config_id, "val"),
        "test_path":    _split_path(cfg, config_id, "test"),
        "train_months": train_months,
        "val_months":   val_months,
        "test_months":  test_months,
        "stats":        split_stats,
    }