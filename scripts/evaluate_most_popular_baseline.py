"""
Evaluate an exact MostPopular full-ranking baseline on warm candidates.

The baseline ranks every warm item by train frequency, then applies the same
seen-item masking used by model evaluation:
  - val: mask train positives
  - test: mask train + val positives by default

For a MostPopular ranker, the exact rank after masking is:
  global_pop_rank(item) - count(seen_user_items ranked before item)

This avoids scoring every user against the full catalog while still matching
full warm-candidate ranking for Recall@K and NDCG@K.

Usage:
  docker compose run --rm pipeline \
    python scripts/evaluate_most_popular_baseline.py --split test --ks 20,40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = {
        "SPARK_DRIVER_MEMORY": ("spark", "driver_memory"),
        "SPARK_EXECUTOR_MEMORY": ("spark", "executor_memory"),
        "SPARK_MASTER": ("spark", "master"),
        "MINIO_ENDPOINT": ("minio", "endpoint"),
        "MINIO_ACCESS_KEY": ("minio", "access_key"),
        "MINIO_SECRET_KEY": ("minio", "secret_key"),
        "MINIO_BUCKET": ("minio", "bucket"),
    }
    for env_key, (section, field) in overrides.items():
        value = os.environ.get(env_key)
        if value is not None:
            cfg.setdefault(section, {})[field] = value
    return cfg


def _s3a(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


def build_spark(cfg: dict):
    from pyspark.sql import SparkSession

    sc = cfg["spark"]
    mn = cfg["minio"]
    return (
        SparkSession.builder
        .appName("most_popular_baseline")
        .master(sc.get("master", "local[*]"))
        .config("spark.driver.memory", sc.get("driver_memory", "4g"))
        .config("spark.executor.memory", sc.get("executor_memory", "4g"))
        .config("spark.sql.shuffle.partitions", str(sc.get("shuffle_partitions", 200)))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.hadoop.fs.s3a.endpoint", mn["endpoint"])
        .config("spark.hadoop.fs.s3a.access.key", mn["access_key"])
        .config("spark.hadoop.fs.s3a.secret.key", mn["secret_key"])
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .getOrCreate()
    )


def parse_ks(raw: str) -> list[int]:
    ks = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--ks must contain positive integers, e.g. 20,40")
    return ks


def _normal_group_expr(col):
    from pyspark.sql import functions as F

    return (
        F.when(col.isin("TAIL", "WARM_TAIL"), F.lit("TAIL"))
        .when(col.isin("COLD_START", "COLD_ITEM"), F.lit("COLD_START"))
        .otherwise(col)
    )


def _count_seen_before_udf():
    from pyspark.sql import functions as F
    from pyspark.sql.types import LongType

    @F.udf(LongType())
    def count_seen_before(seen_ranks, target_rank):
        if not seen_ranks:
            return 0
        return int(bisect_left(seen_ranks, int(target_rank)))

    return count_seen_before


def _metric_columns(df, ks: Iterable[int]):
    from pyspark.sql import functions as F

    scored = df
    rank_col = F.col("rank_after_mask").cast("double")
    discount = F.lit(1.0) / (F.log(rank_col + F.lit(1.0)) / F.log(F.lit(2.0)))
    for k in ks:
        hit_col = f"hit_{k}"
        ndcg_col = f"ndcg_{k}"
        scored = scored.withColumn(hit_col, (F.col("rank_after_mask") <= F.lit(k)).cast("double"))
        scored = scored.withColumn(ndcg_col, F.when(F.col(hit_col) > 0, discount).otherwise(F.lit(0.0)))
    return scored


def _agg_exprs(ks: Iterable[int]):
    from pyspark.sql import functions as F

    exprs = [F.count("*").alias("n")]
    for k in ks:
        exprs.append(F.sum(F.col(f"hit_{k}")).alias(f"hits_{k}"))
        exprs.append(F.sum(F.col(f"ndcg_{k}")).alias(f"ndcg_sum_{k}"))
    return exprs


def _row_to_metrics(row, ks: Iterable[int]) -> dict:
    n = int(row["n"] or 0)
    metrics = {"n": n}
    denom = max(n, 1)
    for k in ks:
        hits = float(row[f"hits_{k}"] or 0.0)
        ndcg_sum = float(row[f"ndcg_sum_{k}"] or 0.0)
        metrics[f"hits@{k}"] = int(hits)
        metrics[f"Recall@{k}"] = hits / denom
        metrics[f"NDCG@{k}"] = ndcg_sum / denom
        metrics[f"ndcg_sum@{k}"] = ndcg_sum
    return metrics


def _print_metrics(title: str, metrics: dict, ks: Iterable[int]) -> None:
    print(f"\n{title}")
    print(f"{'Group':<12} {'N':>12} " + " ".join(f"Recall@{k:>2} NDCG@{k:>2}" for k in ks))
    for group, values in metrics.items():
        cells = [f"{group:<12}", f"{values['n']:>12,}"]
        for k in ks:
            cells.append(f"{values[f'Recall@{k}']:.6f}")
            cells.append(f"{values[f'NDCG@{k}']:.6f}")
        print(" ".join(cells))


def evaluate_most_popular(spark, cfg: dict, split: str, ks: list[int], mask_validation: bool) -> dict:
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    logger = logging.getLogger("most_popular_baseline")

    item_map = (
        spark.read.parquet(_s3a(cfg, "gold", "gold_item_id_map.parquet"))
        .select("parent_asin", "item_idx", "train_freq", "popularity_group")
    )
    user_map = (
        spark.read.parquet(_s3a(cfg, "gold", "gold_user_id_map.parquet"))
        .select("reviewer_id", "user_idx")
    )

    rank_window = Window.orderBy(F.col("train_freq").desc(), F.col("item_idx").asc())
    ranked_items = (
        item_map
        .filter(F.col("train_freq") > 0)
        .withColumn("global_rank", F.row_number().over(rank_window).cast("long"))
        .withColumn("eval_group", _normal_group_expr(F.col("popularity_group")))
        .select("parent_asin", "item_idx", "train_freq", "eval_group", "global_rank")
        .cache()
    )
    warm_item_count = ranked_items.count()
    logger.info("Warm candidate items: %s", f"{warm_item_count:,}")

    gt_path = _s3a(cfg, "silver", f"silver_{split}_ground_truth.parquet")
    gt_raw = spark.read.parquet(gt_path).select("reviewer_id", "parent_asin")
    total_gt = gt_raw.count()

    gt_eval = (
        gt_raw
        .join(user_map, on="reviewer_id", how="inner")
        .join(ranked_items, on="parent_asin", how="inner")
        .select("user_idx", "item_idx", "global_rank", "eval_group")
        .cache()
    )
    eval_gt = gt_eval.count()
    skipped_gt = total_gt - eval_gt
    logger.info(
        "Ground truth rows: total=%s, warm_evaluated=%s, skipped_cold_or_unmapped=%s",
        f"{total_gt:,}",
        f"{eval_gt:,}",
        f"{skipped_gt:,}",
    )

    mask_splits = ["train"]
    if split == "test" and mask_validation:
        mask_splits.append("val")

    seen_raw = None
    for mask_split in mask_splits:
        part = (
            spark.read.parquet(_s3a(cfg, "silver", f"silver_interactions_{mask_split}.parquet"))
            .select("reviewer_id", "parent_asin")
        )
        seen_raw = part if seen_raw is None else seen_raw.unionByName(part)

    seen = (
        seen_raw
        .join(user_map, on="reviewer_id", how="inner")
        .join(ranked_items.select("parent_asin", "global_rank"), on="parent_asin", how="inner")
        .select("user_idx", "global_rank")
        .distinct()
    )
    seen_rank_lists = (
        seen
        .groupBy("user_idx")
        .agg(F.sort_array(F.collect_set("global_rank")).alias("seen_ranks"))
    )

    count_seen_before = _count_seen_before_udf()
    scored = (
        gt_eval
        .join(seen_rank_lists, on="user_idx", how="left")
        .withColumn("seen_before", count_seen_before(F.col("seen_ranks"), F.col("global_rank")))
        .withColumn("rank_after_mask", F.col("global_rank") - F.col("seen_before"))
        .drop("seen_ranks")
        .cache()
    )
    scored = _metric_columns(scored, ks).cache()

    overall_row = scored.agg(*_agg_exprs(ks)).collect()[0]
    overall = _row_to_metrics(overall_row, ks)

    by_group = {}
    group_rows = scored.groupBy("eval_group").agg(*_agg_exprs(ks)).collect()
    for row in group_rows:
        by_group[row["eval_group"]] = _row_to_metrics(row, ks)

    ordered_group_metrics = {"OVERALL": overall}
    for group in ["HEAD", "MID", "TAIL", "COLD_START"]:
        if group in by_group:
            ordered_group_metrics[group] = by_group[group]

    ranked_items.unpersist()
    gt_eval.unpersist()
    scored.unpersist()

    return {
        "baseline": "most_popular",
        "split": split,
        "candidate_scope": "full_warm_items",
        "mask_splits": mask_splits,
        "ks": ks,
        "warm_candidate_items": warm_item_count,
        "total_ground_truth": total_gt,
        "evaluated_ground_truth": eval_gt,
        "skipped_cold_or_unmapped_ground_truth": skipped_gt,
        "metrics": ordered_group_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--ks", default="20,40", help="Comma-separated K values, e.g. 20,40")
    parser.add_argument(
        "--mask-validation",
        dest="mask_validation",
        action="store_true",
        default=True,
        help="For test, mask validation positives in addition to train positives.",
    )
    parser.add_argument(
        "--no-mask-validation",
        dest="mask_validation",
        action="store_false",
        help="For test, mask only train positives.",
    )
    parser.add_argument("--output", default=None, help="JSON output path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args()
    ks = parse_ks(args.ks)
    output_path = args.output or f"data/evaluation/most_popular_{args.split}.json"

    cfg = load_config(args.config)
    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    try:
        result = evaluate_most_popular(
            spark=spark,
            cfg=cfg,
            split=args.split,
            ks=ks,
            mask_validation=args.mask_validation,
        )
        _print_metrics("MostPopular full warm baseline", result["metrics"], ks)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logging.info("Wrote baseline report: %s", out)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
