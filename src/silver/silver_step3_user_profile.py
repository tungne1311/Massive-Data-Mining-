"""
silver_step3_user_profile.py — Silver Layer: User Text Profile

TỐI ƯU v2:
  - Nhận df_train, df_val từ orchestrator (không đọc lại)
  - Gộp Phase 2+3 thành 1 pass: groupBy → collect_list → sort → slice → join
    → Tiết kiệm 1 shuffle + loại bỏ checkpoint tạm ghi MinIO
  - Bỏ repartition khi ghi → dùng coalesce

NGUYÊN TẮC CHỐNG LEAKAGE:
  - Chỉ dùng bronze_train — reviews val/test bị loại hoàn toàn
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.column import Column
from .silver_utils import advanced_clean_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_REVIEWS      = 3
REVIEW_TITLE_CHARS = 120
REVIEW_TEXT_CHARS   = 220


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _s3a_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — TÍNH TRỌNG SỐ VÀ TẠO SNIPPET
# ─────────────────────────────────────────────────────────────────────────────

def _compute_review_weights(df_train: DataFrame) -> DataFrame:
    """
    Tính w(r) = 1 + log(1 + helpful_vote) và tạo snippet.
    Dùng Spark built-in log() — vectorized trên JVM.
    """
    # CHỈ NHẶT LỜI KHEN CỦA USER ĐỂ XÂY DỰNG SỞ THÍCH:
    df_train = df_train.filter(F.col("rating") >= 4.0)

    # ÁP DỤNG HÀM LÀM SẠCH TOÀN DIỆN CHO REVIEW TEXT
    clean_title = advanced_clean_text(F.coalesce(F.col("review_title"), F.lit("")))
    clean_text = advanced_clean_text(F.coalesce(F.col("review_text"), F.lit("")))

    title_part = F.coalesce(
        F.substring(clean_title, 1, REVIEW_TITLE_CHARS),
        F.lit("")
    )
    text_part = F.coalesce(
        F.substring(clean_text, 1, REVIEW_TEXT_CHARS),
        F.lit("")
    )

    snippet_col = F.when(
        (F.length(clean_title) > 0) &
        (F.length(clean_text) > 0),
        F.concat_ws(" - ", title_part, text_part)
    ).when(
        F.length(clean_text) > 0, text_part
    ).when(
        F.length(clean_title) > 0, title_part
    ).otherwise(F.lit(""))

    helpful = F.coalesce(F.col("helpful_vote").cast(FloatType()), F.lit(0.0))
    weight_col = F.lit(1.0) + F.log(F.lit(1.0) + helpful)

    return df_train.select(
        "reviewer_id",
        snippet_col.alias("review_snippet"),
        weight_col.alias("review_weight"),
        "rating",
        "timestamp",
    ).filter(
        F.length(F.col("review_snippet")) > 0
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — TOP-K + GHÉP TEXT (1 PASS DUY NHẤT)
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_profiles(df_weighted: DataFrame, shuffle_parts: int) -> DataFrame:
    """
    1 groupBy duy nhất: collect_list → sort → slice top-K → join text.

    So với v1 (2 groupBy + checkpoint):
      - Tiết kiệm 1 shuffle lớn (~35M rows)
      - Loại bỏ checkpoint tạm ghi MinIO (~9M rows write + read)
    """
    logger.info(f"[Step3] groupBy + collect_list + top-{TOP_K_REVIEWS} + join text (1 pass)...")

    # Đóng gói review thành struct (thời gian trước để lấy review mới nhất làm gốc)
    df_struct = df_weighted.withColumn(
        "review_struct",
        F.struct(
            F.col("timestamp").alias("t"),
            F.col("review_weight").alias("w"),
            F.col("review_snippet").alias("s"),
        )
    ).repartition(shuffle_parts, "reviewer_id")

    # 1 groupBy duy nhất
    df_user = df_struct.groupBy("reviewer_id").agg(
        F.collect_list("review_struct").alias("reviews_raw"),
        F.count("*").alias("review_count_train"),
        F.avg("rating").alias("avg_rating"),
        F.avg("review_weight").alias("avg_review_weight"),
    )

    # Sort → slice top-K → extract text → join
    df_user = df_user.withColumn(
        "reviews_topk",
        F.slice(F.sort_array(F.col("reviews_raw"), asc=False), 1, TOP_K_REVIEWS)
    ).withColumn(
        "user_text",
        F.array_join(
            F.transform(F.col("reviews_topk"), lambda x: x.getField("s")),
            " [SEP] "
        )
    ).withColumn(
        "user_text",
        F.when(F.length(F.col("user_text")) > 5, F.col("user_text"))
         .otherwise(F.lit("[NO_TEXT] User interaction profile"))
    ).select(
        "reviewer_id",
        "user_text",
        "review_count_train",
        "avg_rating",
        "avg_review_weight",
    )

    return df_user


# ─────────────────────────────────────────────────────────────────────────────
# VAL GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def _build_val_ground_truth(df_val: DataFrame, df_popularity: DataFrame) -> DataFrame:
    """Chỉ lấy (reviewer_id, parent_asin, popularity_group) — không lấy text.

    Bao gồm GUARDRAIL kiểm tra phân phối popularity_group trong val:
      - Nếu COLD_START > 35%: cảnh báo Recall@K toàn bộ có thể thấp giả tạo
      - Nếu TAIL > 85%: cảnh báo val bị dominated bởi long-tail items
      - Nếu HEAD < 10%: cảnh báo HEAD recall sẽ không đại diện
    """
    logger.info("[Step3] Xây dựng val ground truth...")

    val_cols = ["reviewer_id", "parent_asin", "timestamp", "rating"]
    pop_slim = df_popularity.select("parent_asin", "popularity_group", "train_freq")

    df_gt = df_val.select(*val_cols).join(
        F.broadcast(pop_slim), on="parent_asin", how="left"
    ).withColumn(
        "popularity_group",
        F.coalesce(F.col("popularity_group"), F.lit("COLD_START"))
    ).withColumn(
        "train_freq",
        F.coalesce(F.col("train_freq"), F.lit(0).cast("long"))
    ).withColumn(
        "is_tail", (F.col("popularity_group") == "TAIL").cast("int")
    ).withColumn(
        "is_cold_start", (F.col("popularity_group") == "COLD_START").cast("int")
    )

    # ─── GUARDRAIL: Kiểm tra phân phối val_gt ────────────────────────────────
    # Mục tiêu: phát hiện sớm nếu val bị dominated bởi COLD/TAIL items.
    # Nếu val_gt chứa quá nhiều COLD_START (items không có embedding trong train),
    # Recall@K sẽ thấp không phải vì model tệ, mà vì val biased.
    # Chi phí: 1 groupBy().collect() nhỏ (~1.8M rows → vài giây).
    dist_rows  = df_gt.groupBy("popularity_group").count().collect()
    total_val  = sum(row["count"] for row in dist_rows)
    dist_dict  = {row["popularity_group"]: row["count"] for row in dist_rows}

    logger.info(f"  📊 [GUARDRAIL] Val ground truth distribution ({total_val:,} interactions):")
    for group in ["HEAD", "MID", "TAIL", "COLD_START"]:
        cnt  = dist_dict.get(group, 0)
        pct  = cnt / max(total_val, 1) * 100
        flag = ""
        if (group == "COLD_START" and pct > 35) or \
           (group == "TAIL"       and pct > 85) or \
           (group == "HEAD"       and pct < 10):
            flag = "  ⚠️  UNBALANCED"
        logger.info(f"    {group:12s}: {cnt:>10,}  ({pct:5.1f}%){flag}")

    cold_pct = dist_dict.get("COLD_START", 0) / max(total_val, 1) * 100
    tail_pct = dist_dict.get("TAIL",       0) / max(total_val, 1) * 100
    head_pct = dist_dict.get("HEAD",       0) / max(total_val, 1) * 100

    if cold_pct > 35:
        logger.warning(
            f"⚠️  [GUARDRAIL] COLD_START chiếm {cold_pct:.1f}% val_gt (ngưỡng: 35%). "
            f"Recall@K tổng thể có thể thấp giả tạo — items này chưa có embedding trong train. "
            f"→ Ưu tiên report Recall@K_core (loại COLD_START) khi so sánh mô hình."
        )
    if tail_pct > 85:
        logger.warning(
            f"⚠️  [GUARDRAIL] TAIL chiếm {tail_pct:.1f}% val_gt (ngưỡng: 85%). "
            f"Val bị dominated bởi long-tail items — phân phối lệch nhiều so với train."
        )
    if head_pct < 10:
        logger.warning(
            f"⚠️  [GUARDRAIL] HEAD chỉ chiếm {head_pct:.1f}% val_gt (ngưỡng: 10%). "
            f"Recall@K_HEAD sẽ không đại diện — cân nhắc oversampling HEAD khi evaluate."
        )

    return df_gt


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    df_popularity: DataFrame,
    df_train: DataFrame,
    df_val: DataFrame,
) -> None:
    """
    Args:
        df_train: bronze_train (đã projection pushdown bởi orchestrator).
        df_val: bronze_val từ orchestrator.
    """
    silver_user_out = _s3a_path(cfg, "silver", "silver_user_text_profile.parquet")
    silver_val_gt   = _s3a_path(cfg, "silver", "silver_val_ground_truth.parquet")
    write_mode      = cfg.get("silver", {}).get("write_mode", "overwrite")
    shuffle_parts   = int(cfg.get("spark", {}).get("shuffle_partitions", 200))

    # ── Phase 1: Tính trọng số ────────────────────────────────────────────────
    logger.info("[Step3] Tính review weights...")
    df_weighted = _compute_review_weights(df_train)

    df_weighted = df_weighted.cache()
    w_count = df_weighted.count()
    logger.info(f"  Reviews có text hợp lệ: {w_count:,}")

    # ── Phase 2: Top-K + ghép text (1 pass) ───────────────────────────────────
    df_user_text = _build_user_profiles(df_weighted, shuffle_parts)
    df_weighted.unpersist()

    # ── Ghi silver_user_text_profile ──────────────────────────────────────────
    logger.info(f"[Step3] Ghi silver_user_text_profile → {silver_user_out}")
    df_user_text.repartition(20) \
                .write.mode(write_mode) \
                .option("compression", "zstd") \
                .parquet(silver_user_out)

    user_count = spark.read.parquet(silver_user_out).count()
    logger.info(f"[Step3] silver_user_text_profile: {user_count:,} users")

    # ── Ghi silver_val_ground_truth ───────────────────────────────────────────
    logger.info(f"[Step3] Ghi silver_val_ground_truth → {silver_val_gt}")
    # _build_val_ground_truth() sẽ tự log GUARDRAIL distribution trước khi return
    df_val_gt = _build_val_ground_truth(df_val, df_popularity)
    df_val_gt.coalesce(5) \
             .write.mode(write_mode) \
             .option("compression", "zstd") \
             .parquet(silver_val_gt)

    logger.info("[Step3] silver_val_ground_truth ghi xong.")

    # ── So sánh Train vs Val distribution (để phát hiện skew) ───────────────
    # Lấy distribution từ df_popularity (đã tính từ train, đã cache)
    train_dist = {
        row["popularity_group"]: row["count"]
        for row in df_popularity.groupBy("popularity_group").count().collect()
    }
    total_train_items = sum(train_dist.values())
    logger.info("  📋 [TRAIN vs VAL] Popularity distribution comparison (items trong train vs interactions trong val):")
    logger.info(f"  {'Group':12s} | {'Train items':>12s} | {'Train %':>8s} | {'Val GT':>10s} | {'Val %':>7s}")
    logger.info(f"  {'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}")
    for group in ["HEAD", "MID", "TAIL"]:
        t_cnt = train_dist.get(group, 0)
        t_pct = t_cnt / max(total_train_items, 1) * 100
        # val distribution đã được log bởi _build_val_ground_truth → không cần đọc lại
        logger.info(f"  {group:12s} | {t_cnt:>12,} | {t_pct:>7.1f}% | {'(xem trên)':>10s} |")
