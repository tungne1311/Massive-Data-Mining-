"""
Bước 6 — Feature Store

Tạo 3 bảng feature từ TRAIN split (để tránh data leakage):
  A. user_features
  B. item_features
  C. user_item_features

Output:
  feature_store/config_id=<id>/train/user_features/
  feature_store/config_id=<id>/train/item_features/
  feature_store/config_id=<id>/train/user_item_features/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"


def _feature_path(cfg: dict, config_id: str, table: str) -> str:
    return _join_uri(
        cfg["paths"]["feature_base"],
        f"config_id={config_id}",
        "train",
        table,
    )


def _write_feature(df: DataFrame, cfg: dict, config_id: str, table: str) -> None:
    out_path = _feature_path(cfg, config_id, table)
    n_parts  = int(cfg["spark"].get("write_partitions", 0))
    mode     = cfg["silver"]["write_mode"]
    df_out   = df.repartition(n_parts) if n_parts > 0 else df
    logger.info(f"  Ghi {table} → {out_path}")
    (
        df_out
          .write
          .mode(mode)
          .parquet(out_path)
    )
    logger.info(f"  ✓ Ghi xong {table}")


# ── A. User Features ──────────────────────────────────────────────────────────

def build_user_features(df_train: DataFrame) -> DataFrame:
    """
    Tính user-level features từ train split.

    Chỉ dùng Spark DataFrame API.
    Các top-K category/brand được tính bằng Window + rank, join lại.
    """
    # Cột tham chiếu phổ biến
    is_positive  = F.col("interaction_type").isin("strong_positive", "weak_positive")
    is_negative  = F.col("interaction_type").isin("medium_negative", "hard_negative")
    is_strong_pos = F.col("interaction_type") == "strong_positive"

    # ── Ngày mốc "30 ngày gần nhất" theo max review_time trong train ─
    max_ts_row = df_train.agg(F.max("review_time").alias("mx")).collect()[0]
    max_ts     = max_ts_row["mx"]
    cutoff_30d = F.date_sub(F.lit(max_ts).cast("date"), 30)

    # ── Scalar aggregations ────────────────────────────────────────────
    df_base = df_train.groupBy("reviewer_id").agg(
        F.sum(is_positive.cast("int")).alias("total_positive_count"),
        F.sum(is_negative.cast("int")).alias("total_negative_count"),
        F.sum(
            (is_positive & (F.to_date("review_time") >= cutoff_30d)).cast("int")
        ).alias("recent_positive_count_30d"),
        F.round(F.avg("rating"), 4).alias("average_rating_given"),
        F.round(F.avg(F.col("verified_purchase").cast("int")), 4)
         .alias("verified_ratio"),
        # preferred_price_bucket: bucket xuất hiện nhiều nhất
        # dùng first của sort trick (xem join bên dưới)
    )

    # preferred_price_bucket: mode của price_bucket theo reviewer_id
    w_price = Window.partitionBy("reviewer_id").orderBy(F.col("_cnt").desc())
    df_price_mode = (
        df_train
        .groupBy("reviewer_id", "price_bucket").count()
        .withColumnRenamed("count", "_cnt")
        .withColumn("_rank", F.rank().over(w_price))
        .filter(F.col("_rank") == 1)
        .select(
            F.col("reviewer_id"),
            F.col("price_bucket").alias("preferred_price_bucket")
        )
        .dropDuplicates(["reviewer_id"])   # tie-break: giữ 1 dòng
    )

    # ── Top-3 favorite_categories (theo strong_positive count) ────────
    df_fav_cat = _top_k_string(
        df_train.filter(is_strong_pos),
        group_col="reviewer_id",
        value_col="main_category",
        k=3,
        out_col="favorite_categories",
    )

    # ── Top-3 favorite_brands ─────────────────────────────────────────
    df_fav_brand = _top_k_string(
        df_train.filter(is_strong_pos),
        group_col="reviewer_id",
        value_col="brand",
        k=3,
        out_col="favorite_brands",
    )

    # ── recent_category_intent: top-2 category trong 30d ─────────────
    df_recent = df_train.filter(F.to_date("review_time") >= cutoff_30d)

    df_recent_cat = _top_k_string(
        df_recent,
        group_col="reviewer_id",
        value_col="main_category",
        k=2,
        out_col="recent_category_intent",
    )

    # ── recent_brand_intent: top-2 brand trong 30d ────────────────────
    df_recent_brand = _top_k_string(
        df_recent,
        group_col="reviewer_id",
        value_col="brand",
        k=2,
        out_col="recent_brand_intent",
    )

    # ── Join tất cả lại ───────────────────────────────────────────────
    df_user = (
        df_base
        .join(df_price_mode,    on="reviewer_id", how="left")
        .join(df_fav_cat,       on="reviewer_id", how="left")
        .join(df_fav_brand,     on="reviewer_id", how="left")
        .join(df_recent_cat,    on="reviewer_id", how="left")
        .join(df_recent_brand,  on="reviewer_id", how="left")
    )

    # Fill null cho các string features
    for col_name in ["preferred_price_bucket", "favorite_categories",
                     "favorite_brands", "recent_category_intent", "recent_brand_intent"]:
        df_user = df_user.withColumn(
            col_name,
            F.when(F.col(col_name).isNull(), F.lit("")).otherwise(F.col(col_name))
        )

    return df_user


def _top_k_string(
    df: DataFrame,
    group_col: str,
    value_col: str,
    k: int,
    out_col: str,
) -> DataFrame:
    """
    Tính top-k value_col theo count trong group_col,
    rồi join thành 1 chuỗi ngăn cách bởi '|'.
    """
    w = Window.partitionBy(group_col).orderBy(F.col("_cnt").desc())

    df_ranked = (
        df.groupBy(group_col, value_col).count()
          .withColumnRenamed("count", "_cnt")
          .filter(F.col(value_col).isNotNull() & (F.col(value_col) != "") &
                  (F.col(value_col) != "unknown"))
          .withColumn("_rank", F.rank().over(w))
          .filter(F.col("_rank") <= k)
    )

    df_agg = (
        df_ranked
        .groupBy(group_col)
        .agg(
            F.concat_ws("|", F.collect_list(value_col)).alias(out_col)
        )
    )
    return df_agg


# ── B. Item Features ──────────────────────────────────────────────────────────

def build_item_features(df_train: DataFrame) -> DataFrame:
    """
    Tính item-level features từ train split.
    """
    # Cột tiện dùng
    is_negative = F.col("interaction_type").isin("medium_negative", "hard_negative")

    max_ts_row = df_train.agg(
        F.max("review_time").alias("mx"),
        F.min("review_time").alias("mn"),
    ).collect()[0]
    max_ts    = max_ts_row["mx"]
    cutoff_7d  = F.date_sub(F.lit(max_ts).cast("date"), 7)
    cutoff_30d = F.date_sub(F.lit(max_ts).cast("date"), 30)

    df_item = df_train.groupBy("parent_asin").agg(
        F.count("*").alias("item_review_count"),
        F.round(F.avg(F.col("verified_purchase").cast("int")), 4)
         .alias("verified_review_ratio"),
        F.round(F.avg("rating"), 4).alias("avg_rating"),
        # recent_popularity_7d: số review trong 7 ngày gần nhất
        F.sum(
            (F.to_date("review_time") >= cutoff_7d).cast("int")
        ).alias("recent_popularity_7d"),
        # recent_popularity_30d
        F.sum(
            (F.to_date("review_time") >= cutoff_30d).cast("int")
        ).alias("recent_popularity_30d"),
        # recent_negative_ratio: tỉ lệ negative trong 30d
        F.round(
            F.sum(
                (is_negative & (F.to_date("review_time") >= cutoff_30d)).cast("int")
            ) /
            F.greatest(
                F.sum((F.to_date("review_time") >= cutoff_30d).cast("int")),
                F.lit(1)
            ),
            4
        ).alias("recent_negative_ratio"),
        # helpful_review_trend: avg helpful_vote (proxy trend)
        F.round(F.avg("helpful_vote"), 4).alias("helpful_review_trend"),
        # category, brand, price_bucket: lấy mode (first of sort trick)
        F.first("main_category").alias("category"),
        F.first("brand").alias("brand"),
        F.first("price_bucket").alias("price_bucket"),
        # age_of_item: khoảng cách (ngày) từ review đầu tiên đến max_ts
        F.datediff(
            F.lit(max_ts).cast("date"),
            F.min(F.to_date("review_time"))
        ).alias("age_of_item"),
        # recency_of_last_review: khoảng cách từ review cuối đến max_ts
        F.datediff(
            F.lit(max_ts).cast("date"),
            F.max(F.to_date("review_time"))
        ).alias("recency_of_last_review"),
    )

    return df_item


# ── C. User-Item Features ─────────────────────────────────────────────────────

def build_user_item_features(
    df_train: DataFrame,
    df_user: DataFrame,
    df_item: DataFrame,
) -> DataFrame:
    """
    Tính user-item cross features.
    Chỉ dựa trên train split để tránh leakage.
    """
    max_ts_row = df_train.agg(F.max("review_time").alias("mx")).collect()[0]
    max_ts     = max_ts_row["mx"]

    # has_seen_before: có tồn tại interaction trong train không
    # days_since_last_interaction: khoảng cách từ lần tương tác cuối
    df_ui_base = df_train.groupBy("reviewer_id", "parent_asin").agg(
        F.lit(True).alias("has_seen_before"),
        F.datediff(
            F.lit(max_ts).cast("date"),
            F.max(F.to_date("review_time"))
        ).alias("days_since_last_interaction"),
    )

    # Join user features để lấy favorite_brands, favorite_categories, preferred_price_bucket
    df_ui = df_ui_base.join(
        df_user.select(
            "reviewer_id",
            "favorite_brands",
            "favorite_categories",
            "preferred_price_bucket",
        ),
        on="reviewer_id",
        how="left",
    )

    # Join item features để lấy brand, category, price_bucket
    df_ui = df_ui.join(
        df_item.select(
            "parent_asin",
            F.col("brand").alias("item_brand"),
            F.col("category").alias("item_category"),
            F.col("price_bucket").alias("item_price_bucket"),
        ),
        on="parent_asin",
        how="left",
    )

    # same_brand_affinity: item_brand nằm trong favorite_brands không
    df_ui = df_ui.withColumn(
        "same_brand_affinity",
        (
            F.col("favorite_brands").isNotNull() &
            (F.col("favorite_brands") != "") &
            F.col("item_brand").isNotNull() &
            F.col("favorite_brands").contains(F.col("item_brand"))
        ).cast("boolean")
    )

    # category_affinity: item_category overlap với favorite_categories
    df_ui = df_ui.withColumn(
        "category_affinity",
        (
            F.col("favorite_categories").isNotNull() &
            (F.col("favorite_categories") != "") &
            F.col("item_category").isNotNull() &
            F.col("favorite_categories").contains(F.col("item_category"))
        ).cast("boolean")
    )

    # price_match: item_price_bucket khớp preferred_price_bucket
    df_ui = df_ui.withColumn(
        "price_match",
        (
            F.col("preferred_price_bucket").isNotNull() &
            (F.col("preferred_price_bucket") != "") &
            (F.col("preferred_price_bucket") == F.col("item_price_bucket"))
        ).cast("boolean")
    )

    # Giữ lại các cột cần thiết
    df_ui = df_ui.select(
        "reviewer_id",
        "parent_asin",
        "has_seen_before",
        "days_since_last_interaction",
        "same_brand_affinity",
        "category_affinity",
        "price_match",
    )

    return df_ui


# ── Summary Feature Store ─────────────────────────────────────────────────────

def save_feature_summary(
    df_user: DataFrame,
    df_item: DataFrame,
    df_ui: DataFrame,
    cfg: dict,
    config_id: str,
) -> dict:
    """Tính và ghi summary nhỏ cho feature store."""

    def _null_ratio(df: DataFrame, col_name: str) -> float:
        row = df.agg(
            F.round(F.mean(F.col(col_name).isNull().cast("int")), 4)
             .alias("nr")
        ).collect()[0]
        return float(row["nr"] or 0)

    n_user = df_user.count()
    n_item = df_item.count()
    n_ui   = df_ui.count()

    summary = {
        "config_id":      config_id,
        "n_user_rows":    int(n_user),
        "n_item_rows":    int(n_item),
        "n_ui_rows":      int(n_ui),
        "null_ratio": {
            "user_verified_ratio":           _null_ratio(df_user, "verified_ratio"),
            "user_average_rating_given":     _null_ratio(df_user, "average_rating_given"),
            "item_avg_rating":               _null_ratio(df_item, "avg_rating"),
            "item_verified_review_ratio":    _null_ratio(df_item, "verified_review_ratio"),
            "ui_days_since_last_interaction":_null_ratio(df_ui,   "days_since_last_interaction"),
        },
        "generated_at": datetime.now().isoformat(),
    }

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    fpath = log_dir / f"feature_summary_{config_id}.json"
    fpath.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"  Feature summary → {fpath}")

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    config_id: str,
    df_train: Optional[DataFrame] = None,
) -> dict:
    """
    Chạy bước 6.
    Trả về dict {"user_path", "item_path", "ui_path", "summary"}.
    """
    logger.info(f"=== Bước 6: Feature Store — config_id={config_id} ===")

    if df_train is None:
        train_path = _join_uri(
            cfg["paths"]["splits_base"],
            f"config_id={config_id}",
            "train",
        )
        logger.info(f"  Đọc train split từ disk: {train_path}")
        df_train = spark.read.parquet(train_path)

    # ── A. User features ──────────────────────────────────────────────
    logger.info("  Xây dựng user_features...")
    df_user = build_user_features(df_train)
    _write_feature(df_user, cfg, config_id, "user_features")

    # ── B. Item features ──────────────────────────────────────────────
    logger.info("  Xây dựng item_features...")
    df_item = build_item_features(df_train)
    _write_feature(df_item, cfg, config_id, "item_features")

    # ── C. User-Item features ─────────────────────────────────────────
    logger.info("  Xây dựng user_item_features...")
    # Đọc lại từ disk để tránh re-compute toàn bộ lineage dài
    df_user_r = spark.read.parquet(_feature_path(cfg, config_id, "user_features"))
    df_item_r = spark.read.parquet(_feature_path(cfg, config_id, "item_features"))
    df_ui     = build_user_item_features(df_train, df_user_r, df_item_r)
    _write_feature(df_ui, cfg, config_id, "user_item_features")

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("  Tính feature summary...")
    df_user_r2 = spark.read.parquet(_feature_path(cfg, config_id, "user_features"))
    df_item_r2 = spark.read.parquet(_feature_path(cfg, config_id, "item_features"))
    df_ui_r2   = spark.read.parquet(_feature_path(cfg, config_id, "user_item_features"))
    summary = save_feature_summary(df_user_r2, df_item_r2, df_ui_r2, cfg, config_id)

    logger.info(
        f"  ✓ Bước 6 xong: {summary['n_user_rows']:,} users | "
        f"{summary['n_item_rows']:,} items | {summary['n_ui_rows']:,} user-item pairs"
    )

    return {
        "user_path":   _feature_path(cfg, config_id, "user_features"),
        "item_path":   _feature_path(cfg, config_id, "item_features"),
        "ui_path":     _feature_path(cfg, config_id, "user_item_features"),
        "summary":     summary,
    }
