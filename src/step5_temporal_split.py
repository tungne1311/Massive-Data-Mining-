"""
Bước 5 — Temporal Split

Split labeled interactions theo thời gian (KHÔNG random).

Mode auto (mặc định):
  max_month = T
  train  : year_month <= T-3
  val    : T-2 <= year_month <= T-1
  test   : year_month == T

Mode explicit:
  khai báo train_months, val_months, test_months trong config.

Output:
  splits/<config_id>/train/
  splits/<config_id>/val/
  splits/<config_id>/test/
"""

import logging
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"


def _split_path(cfg: dict, config_id: str, split: str) -> str:
    return _join_uri(cfg["paths"]["splits_base"], f"config_id={config_id}", split)


def _add_months(year_month: str, delta: int) -> str:
    """Cộng/trừ số tháng vào chuỗi 'YYYY-MM'. Trả về 'YYYY-MM'."""
    year, month = int(year_month[:4]), int(year_month[5:7])
    total_months = year * 12 + (month - 1) + delta
    y, m = divmod(total_months, 12)
    return f"{y:04d}-{m+1:02d}"


# ── Xác định tháng split ──────────────────────────────────────────────────────

def resolve_split_months(
    df_labeled: DataFrame,
    cfg: dict,
) -> tuple[list[str], list[str], list[str]]:
    """
    Trả về (train_months, val_months, test_months) dạng list[str].
    Dựa vào config temporal_split.mode.
    """
    ts_cfg = cfg.get("temporal_split", {})
    mode   = ts_cfg.get("mode", "auto")

    if mode == "explicit":
        train_months = ts_cfg["train_months"]
        val_months   = ts_cfg["val_months"]
        test_months  = ts_cfg["test_months"]
        logger.info(f"  Mode explicit: train={train_months}, val={val_months}, test={test_months}")
        return train_months, val_months, test_months

    # ── Auto mode ─────────────────────────────────────────────────────
    max_month_row = df_labeled.agg(F.max("year_month").alias("mx")).collect()[0]
    max_month = max_month_row["mx"]

    if max_month is None:
        raise ValueError("Không tìm thấy year_month hợp lệ trong dữ liệu!")

    T      = max_month
    T_m1   = _add_months(T, -1)
    T_m2   = _add_months(T, -2)
    T_m3   = _add_months(T, -3)

    # Lấy tất cả tháng unique để xác định danh sách train
    all_months_rows = (
        df_labeled
        .select("year_month").distinct()
        .orderBy("year_month")
        .collect()
    )
    all_months = sorted([r["year_month"] for r in all_months_rows])

    train_months = [m for m in all_months if m <= T_m3]
    val_months   = [m for m in all_months if T_m2 <= m <= T_m1]
    test_months  = [m for m in all_months if m == T]

    logger.info(f"  Max month (T)  = {T}")
    logger.info(f"  Train months  : {train_months[:3]}{'...' if len(train_months) > 3 else ''} "
                f"({len(train_months)} tháng)")
    logger.info(f"  Val months    : {val_months}")
    logger.info(f"  Test months   : {test_months}")

    return train_months, val_months, test_months


# ── Ghi từng split ────────────────────────────────────────────────────────────

def _write_split(
    df: DataFrame,
    cfg: dict,
    config_id: str,
    split_name: str,
) -> None:
    out_path = _split_path(cfg, config_id, split_name)
    n_parts  = int(cfg["spark"].get("write_partitions", 0))
    mode     = cfg["silver"]["write_mode"]
    df_out   = df.repartition(n_parts) if n_parts > 0 else df
    logger.info(f"  Ghi {split_name} → {out_path}")
    (
        df_out
          .write
          .mode(mode)
          .parquet(out_path)
    )
    logger.info(f"  ✓ Ghi xong {split_name}")


def _log_split_stats(df: DataFrame, split_name: str) -> dict:
    """Log số dòng, user, item của một split. Trả về dict stats."""
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
    logger.info(
        f"  {split_name}: {stats['n_rows']:,} rows | "
        f"{stats['n_users']:,} users | {stats['n_items']:,} items"
    )
    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    config_id: str,
    df_labeled: Optional[DataFrame] = None,
) -> dict:
    """
    Chạy bước 5.
    Trả về dict {"train": path, "val": path, "test": path, "stats": {...}}
    """
    logger.info(f"=== Bước 5: Temporal Split — config_id={config_id} ===")

    if df_labeled is None:
        labeled_path = _join_uri(
            cfg["paths"]["silver_base"],
            "labeled_interactions",
            f"config_id={config_id}",
        )
        logger.info(f"  Đọc labeled từ disk: {labeled_path}")
        df_labeled = spark.read.parquet(labeled_path)

    train_months, val_months, test_months = resolve_split_months(df_labeled, cfg)

    if not train_months:
        logger.warning("  Không có tháng nào cho train split!")
    if not val_months:
        logger.warning("  Không có tháng nào cho val split!")
    if not test_months:
        logger.warning("  Không có tháng nào cho test split!")

    # ── Filter + ghi từng split ────────────────────────────────────────
    df_train = df_labeled.filter(F.col("year_month").isin(train_months))
    df_val   = df_labeled.filter(F.col("year_month").isin(val_months))
    df_test  = df_labeled.filter(F.col("year_month").isin(test_months))

    split_stats = {}

    _write_split(df_train, cfg, config_id, "train")
    split_stats["train"] = _log_split_stats(
        spark.read.parquet(_split_path(cfg, config_id, "train")), "train"
    )

    _write_split(df_val, cfg, config_id, "val")
    split_stats["val"] = _log_split_stats(
        spark.read.parquet(_split_path(cfg, config_id, "val")), "val"
    )

    _write_split(df_test, cfg, config_id, "test")
    split_stats["test"] = _log_split_stats(
        spark.read.parquet(_split_path(cfg, config_id, "test")), "test"
    )

    logger.info(f"  ✓ Bước 5 xong: config_id={config_id}")

    return {
        "train_path":   _split_path(cfg, config_id, "train"),
        "val_path":     _split_path(cfg, config_id, "val"),
        "test_path":    _split_path(cfg, config_id, "test"),
        "train_months": train_months,
        "val_months":   val_months,
        "test_months":  test_months,
        "stats":        split_stats,
    }
