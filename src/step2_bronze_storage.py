"""
Bước 2 — Bronze Storage
Ghi Parquet partition theo year_month + quality check một lần quét.

Output:
    bronze/reviews/year_month=YYYY-MM/part-*.parquet
    bronze/metadata/year_month=YYYY-MM/part-*.parquet
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ── Ghi Bronze ────────────────────────────────────────────────────────────────
def join_uri(base: str, *parts: str) -> str:
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts)
    return f"{base}/{suffix}"
def prepare_review_batch(df: DataFrame) -> DataFrame:
    df = df.filter((F.col("timestamp").isNotNull()) & (F.col("timestamp") > 0))

    df = df.withColumn(
        "review_time",
        F.to_timestamp(F.from_unixtime(F.col("timestamp")))
    )

    df = df.filter(
        (F.year("review_time") >= 1995) &
        (F.year("review_time") <= 2030)
    )

    df = df.withColumn(
        "year_month",
        F.date_format(F.col("review_time"), "yyyy-MM")
    )

    return df


def write_review_batch(
    spark: SparkSession,
    batch_rows: list[dict],
    cfg: dict,
    review_schema,
) -> None:
    if not batch_rows:
        return

    df = spark.createDataFrame(batch_rows, schema=review_schema)
    df = prepare_review_batch(df)

    out_path = join_uri(cfg["paths"]["bronze_base"], "reviews")

    (
        df.write
        .mode("append")
        .partitionBy("year_month")
        .parquet(out_path)
    )

    logger.info(f"✓ Ghi xong review batch: {len(batch_rows):,} records → {out_path}")


def write_metadata_batch(
    spark: SparkSession,
    batch_rows: list[dict],
    cfg: dict,
    metadata_schema,
) -> None:
    if not batch_rows:
        return

    df = spark.createDataFrame(batch_rows, schema=metadata_schema)

    df = df.withColumn(
        "year_month",
        F.date_format(F.to_date(F.col("ingest_date"), "yyyy-MM-dd"), "yyyy-MM")
    )

    out_path = join_uri(cfg["paths"]["bronze_base"], "metadata")

    (
        df.write
        .mode("append")
        .partitionBy("year_month")
        .parquet(out_path)
    )

    logger.info(f"Ghi xong metadata batch: {len(batch_rows):,} records → {out_path}")

def _write_bronze(
    df: DataFrame,
    out_path: str,
    partition_col: str,
    n_partitions: int,
    mode: str,
) -> None:
    logger.info(f"Ghi → {out_path}")
    (
        (df.repartition(n_partitions) if n_partitions > 0 else df)
          .write
          .mode(mode)
          .partitionBy(partition_col)
          .parquet(out_path)
    )
    logger.info(f"✓ Ghi xong: {out_path}")


def write_reviews(df: DataFrame, cfg: dict) -> DataFrame:
    """Thêm year_month từ timestamp rồi ghi Parquet."""
    df = df.withColumn(
        "year_month",
        F.when(F.col("timestamp") == 0, F.lit("unknown"))
         .otherwise(F.date_format(F.from_unixtime(F.col("timestamp")), "yyyy-MM")),
    )
    _write_bronze(
        df,
        out_path= join_uri(cfg["paths"]["bronze_base"], "reviews"),
        partition_col= "year_month",
        n_partitions = cfg["spark"]["write_partitions"],
        mode= cfg["bronze"]["write_mode"],
    )
    return df


def write_metadata(df: DataFrame, cfg: dict) -> DataFrame:
    """Metadata dùng ingest_date làm partition key."""
    df = df.withColumn(
        "year_month",
        F.date_format(F.to_date(F.col("ingest_date"), "yyyy-MM-dd"), "yyyy-MM"),
    )
    _write_bronze(
        df,
        out_path= join_uri(cfg["paths"]["bronze_base"], "metadata"),
        partition_col= "year_month",
        n_partitions = cfg["spark"]["write_partitions"],
        mode         = cfg["bronze"]["write_mode"],
    )
    return df


# ── Quality Check ─────────────────────────────────────────────────────────────
# Mỗi bảng chỉ cần 3 Spark scans:
#   1. df.agg() — tất cả scalar metrics trong 1 lượt
#   2. groupBy("rating").count() — rating distribution
#   3. dropDuplicates().count() — duplicate check

def qc_reviews(df: DataFrame, thresholds: dict) -> dict:
    """Trả về dict metrics + list warnings. Dùng 3 scans thay vì 9+."""
    warnings: list[str] = []

    # ── Scan 1: tất cả scalar metrics ─────────────────────────────────
    s = df.agg(
        F.count("*").alias("total"),
        F.sum(F.col("reviewer_id").isNull().cast("int")).alias("null_reviewer"),
        F.sum(F.col("parent_asin").isNull().cast("int")).alias("null_asin"),
        F.sum(F.col("rating").isNull().cast("int")).alias("null_rating"),
        F.sum((F.col("verified_purchase") == True).cast("int")).alias("verified_count"),
        F.sum((F.col("rating") == 5.0).cast("int")).alias("five_star_count"),
        F.min("text_len").alias("text_len_min"),
        F.round(F.mean("text_len"), 1).alias("text_len_mean"),
        F.percentile_approx("text_len", 0.5).alias("text_len_median"),
        F.max("text_len").alias("text_len_max"),
        F.sum((F.col("text_len") < 20).cast("int")).alias("short_reviews"),
    ).collect()[0]

    total = s["total"]

    # ── Scan 2: rating distribution ────────────────────────────────────
    rating_dist = {
        str(r["rating"]): r["count"]
        for r in df.groupBy("rating").count().orderBy("rating").collect()
    }

    # ── Scan 3: duplicate (reviewer_id, parent_asin) ───────────────────
    n_dup = total - df.dropDuplicates(["reviewer_id", "parent_asin"]).count()
    dup_ratio = n_dup / total if total else 0

    # ── Warnings ───────────────────────────────────────────────────────
    verified_ratio = s["verified_count"] / total if total else 0
    five_star_ratio = s["five_star_count"] / total if total else 0
    null_reviewer_ratio = s["null_reviewer"] / total if total else 0
    null_asin_ratio = s["null_asin"] / total if total else 0

    if null_reviewer_ratio > thresholds["max_null_ratio"]:
        warnings.append(f"null reviewer_id = {null_reviewer_ratio:.2%}")
    if null_asin_ratio > thresholds["max_null_ratio"]:
        warnings.append(f"null parent_asin = {null_asin_ratio:.2%}")
    if dup_ratio > thresholds["max_duplicate_ratio"]:
        warnings.append(f"duplicate (reviewer, asin) = {dup_ratio:.2%}")
    if verified_ratio < thresholds["min_verified_ratio"]:
        warnings.append(f"verified ratio thấp = {verified_ratio:.2%} — ảnh hưởng reliability_score")
    if five_star_ratio > thresholds["max_five_star_ratio"]:
        warnings.append(f"5-sao bias = {five_star_ratio:.2%} — cần xem khi tính signal scores")

    metrics = {
        "total_records":            f"{total:,}",
        "null_reviewer_id":         s["null_reviewer"],
        "null_parent_asin":         s["null_asin"],
        "null_rating":              s["null_rating"],
        "duplicate_pairs":          f"{n_dup:,} ({dup_ratio:.2%})",
        "verified_purchase_ratio":  f"{verified_ratio:.2%}",
        "five_star_ratio":          f"{five_star_ratio:.2%}",
        "rating_distribution":      rating_dist,
        "text_len": {
            "min":    int(s["text_len_min"]),
            "mean":   float(s["text_len_mean"]),
            "median": int(s["text_len_median"]),
            "max":    int(s["text_len_max"]),
            "short<20": int(s["short_reviews"]),
        },
    }
    return {"table": "reviews", "passed": len(warnings) == 0,
            "warnings": warnings, "metrics": metrics}


def qc_metadata(df: DataFrame) -> dict:
    """Quality check metadata — 2 scans."""
    warnings: list[str] = []

    # ── Scan 1: null / empty counts ────────────────────────────────────
    s = df.agg(
        F.count("*").alias("total"),
        F.sum((F.col("item_title") == "").cast("int")).alias("empty_title"),
        F.sum((F.col("brand") == "").cast("int")).alias("empty_brand"),
        F.sum((F.col("main_category") == "").cast("int")).alias("empty_category"),
        F.sum((F.col("price_bucket") == "unknown").cast("int")).alias("unknown_price"),
    ).collect()[0]

    total = s["total"]

    # ── Scan 2: price bucket + category distribution ───────────────────
    price_dist = {
        r["price_bucket"]: r["count"]
        for r in df.groupBy("price_bucket").count().orderBy("price_bucket").collect()
    }

    empty_title_ratio  = s["empty_title"] / total if total else 0
    unknown_price_ratio = s["unknown_price"] / total if total else 0

    if empty_title_ratio > 0.1:
        warnings.append(f"item_title trống = {empty_title_ratio:.2%} — ảnh hưởng SBERT encoding")
    if unknown_price_ratio > 0.3:
        warnings.append(f"price_bucket='unknown' = {unknown_price_ratio:.2%} — kiểm tra field price")

    metrics = {
        "total_items":          f"{total:,}",
        "empty_item_title":     s["empty_title"],
        "empty_brand":          s["empty_brand"],
        "empty_main_category":  s["empty_category"],
        "price_bucket_distribution": price_dist,
    }
    return {"table": "metadata", "passed": len(warnings) == 0,
            "warnings": warnings, "metrics": metrics}


def _print_qc(report: dict) -> None:
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  QC — {report['table'].upper()}")
    print(sep)
    for k, v in report["metrics"].items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    if report["warnings"]:
        print(f"\n  ⚠  {len(report['warnings'])} warning(s):")
        for w in report["warnings"]:
            print(f"    → {w}")
    else:
        print("\n  ✓  Tất cả checks đạt")
    print(sep)


def _save_qc_report(reports: list[dict], log_dir: Path, cfg: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"bronze_qc_{ts}.json"
    fpath = log_dir / fname
    content = json.dumps({
        "run_time":   datetime.now().isoformat(),
        "all_passed": all(r["passed"] for r in reports),
        "reports":    reports,
    }, indent=2, ensure_ascii=False)
    fpath.write_text(content)
    logger.info(f"QC report local → {fpath}")

    # ── Upload lên MinIO ───────────────────────────────────────────────
    try:
        import boto3
        from botocore.client import Config

        mn = cfg["minio"]
        s3 = boto3.client(
            "s3",
            endpoint_url          = mn["endpoint"],
            aws_access_key_id     = mn["access_key"],
            aws_secret_access_key = mn["secret_key"],
            config                = Config(signature_version="s3v4"),
            region_name           = "us-east-1",
        )
        bucket   = mn["bucket"]
        s3_key   = f"bronze/logs/{fname}"
        s3.put_object(Bucket=bucket, Key=s3_key, Body=content.encode("utf-8"),
                      ContentType="application/json")
        logger.info(f"QC report MinIO → s3a://{bucket}/{s3_key}")
    except Exception as e:
        logger.warning(f"Không upload được QC report lên MinIO: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    df_reviews: DataFrame,
    df_metadata: DataFrame,
) -> None:
    logger.info("=== Bước 2: Bronze Storage ===")

    df_reviews  = write_reviews(df_reviews, cfg)
    df_metadata = write_metadata(df_metadata, cfg)

    logger.info("=== Bước 2: Quality Checks ===")
    rpt_rev  = qc_reviews(df_reviews,  cfg["quality_check"])
    rpt_meta = qc_metadata(df_metadata)

    _print_qc(rpt_rev)
    _print_qc(rpt_meta)
    _save_qc_report([rpt_rev, rpt_meta], Path(cfg["paths"]["log_dir"]), cfg)

    if not rpt_rev["passed"] or not rpt_meta["passed"]:
        logger.warning("⚠  Một số QC checks không đạt — xem báo cáo trong logs/")


# ── Reader helpers (cho các bước sau) ────────────────────────────────────────

def read_bronze_reviews(
    spark: SparkSession,
    cfg: dict,
    year_month: str = None,
) -> DataFrame:
    base = join_uri(cfg["paths"]["bronze_base"], "reviews")
    path = join_uri(base, f"year_month={year_month}") if year_month else base
    return spark.read.parquet(path)


def read_bronze_metadata(spark: SparkSession, cfg: dict) -> DataFrame:
    return spark.read.parquet(join_uri(cfg["paths"]["bronze_base"], "metadata"))
