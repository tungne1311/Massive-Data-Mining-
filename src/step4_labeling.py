"""
Bước 4 — Labeling

Từ silver interactions → gán nhãn:
  - interaction_type (strong_positive / weak_positive / neutral /
                       medium_negative / hard_negative)
  - bpr_role (positive / hard_negative / excluded)
  - bpr_positive_weight
  - relevance_label (0–3)

Output: silver/labeled_interactions/config_id=<id>/year_month=YYYY-MM/
"""

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"


def _labeled_path(cfg: dict, config_id: str) -> str:
    return _join_uri(
        cfg["paths"]["silver_base"],
        "labeled_interactions",
        f"config_id={config_id}",
    )


# ── Labeling logic ────────────────────────────────────────────────────────────

def assign_interaction_type(df: DataFrame) -> DataFrame:
    """
    Gán interaction_type dựa trên verified_purchase + rating.

    strong_positive:   verified=True  AND rating >= 4
    weak_positive:     (verified=False AND rating >= 4)
                       OR (verified=True AND rating = 3)
    neutral:           rating = 3 AND verified=False
                       (hoặc các trường hợp mơ hồ còn lại)
    medium_negative:   verified=True  AND rating = 2
    hard_negative:     verified=True  AND rating = 1
    """
    vp  = F.col("verified_purchase")
    rat = F.col("rating")

    df = df.withColumn(
        "interaction_type",
        F.when(vp & (rat >= 4),                                    F.lit("strong_positive"))
         .when((~vp & (rat >= 4)) | (vp & (rat == 3)),             F.lit("weak_positive"))
         .when(vp & (rat == 2),                                     F.lit("medium_negative"))
         .when(vp & (rat == 1),                                     F.lit("hard_negative"))
         .otherwise(F.lit("neutral"))  # bao gồm: rating=3 & unverified, và edge cases
    )
    return df


def assign_bpr_role(df: DataFrame) -> DataFrame:
    """
    bpr_role:
      positive      → strong_positive, weak_positive
      hard_negative → medium_negative, hard_negative
      excluded      → neutral (loại khỏi BPR training)
    """
    it = F.col("interaction_type")

    df = df.withColumn(
        "bpr_role",
        F.when(it.isin("strong_positive", "weak_positive"), F.lit("positive"))
         .when(it.isin("medium_negative", "hard_negative"), F.lit("hard_negative"))
         .otherwise(F.lit("excluded"))
    )
    return df


def assign_bpr_positive_weight(df: DataFrame) -> DataFrame:
    """
    bpr_positive_weight:
      strong_positive → 1.0
      weak_positive   → 0.5
      còn lại         → 0.0
    """
    it = F.col("interaction_type")

    df = df.withColumn(
        "bpr_positive_weight",
        F.when(it == "strong_positive", F.lit(1.0))
         .when(it == "weak_positive",   F.lit(0.5))
         .otherwise(F.lit(0.0))
    )
    return df


def assign_relevance_label(df: DataFrame) -> DataFrame:
    """
    relevance_label (cho ranking metrics như NDCG):
      strong_positive   → 3
      weak_positive     → 2
      neutral           → 1
      medium/hard_neg   → 0
    """
    it = F.col("interaction_type")

    df = df.withColumn(
        "relevance_label",
        F.when(it == "strong_positive", F.lit(3))
         .when(it == "weak_positive",   F.lit(2))
         .when(it == "neutral",         F.lit(1))
         .otherwise(F.lit(0))
    )
    return df


# ── Ghi output ────────────────────────────────────────────────────────────────

def write_labeled_interactions(
    df: DataFrame,
    cfg: dict,
    config_id: str,
) -> None:
    out_path = _labeled_path(cfg, config_id)
    logger.info(f"  Ghi labeled interactions → {out_path}")
    n_parts = int(cfg["spark"].get("write_partitions", 0))
    mode    = cfg["silver"]["write_mode"]
    df_out  = df.repartition(n_parts) if n_parts > 0 else df

    (
        df_out
          .write
          .mode(mode)
          .partitionBy("year_month")
          .parquet(out_path)
    )
    logger.info(f"  ✓ Ghi xong labeled: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    config_id: str,
    df_silver: DataFrame = None,
) -> DataFrame:
    """
    Chạy bước 4.
    Nếu df_silver=None thì đọc từ disk theo config_id.
    Trả về labeled DataFrame (đọc lại từ disk).
    """
    logger.info(f"=== Bước 4: Labeling — config_id={config_id} ===")

    if df_silver is None:
        silver_path = _join_uri(
            cfg["paths"]["silver_base"],
            "interactions",
            f"config_id={config_id}",
        )
        logger.info(f"  Đọc silver từ disk: {silver_path}")
        df_silver = spark.read.parquet(silver_path)

    df = assign_interaction_type(df_silver)
    df = assign_bpr_role(df)
    df = assign_bpr_positive_weight(df)
    df = assign_relevance_label(df)

    # Log phân phối label nhỏ gọn
    logger.info("  Tính phân phối interaction_type...")
    dist = {
        r["interaction_type"]: r["count"]
        for r in df.groupBy("interaction_type").count()
                   .orderBy("interaction_type").collect()
    }
    for k, v in dist.items():
        logger.info(f"    {k}: {v:,}")

    write_labeled_interactions(df, cfg, config_id)

    logger.info(f"  ✓ Bước 4 xong: config_id={config_id}")
    return spark.read.parquet(_labeled_path(cfg, config_id))
