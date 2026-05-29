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
RECENCY_HALF_LIFE_DAYS = 365.0


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _s3a_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return f"s3a://{'/'.join([bucket, *parts])}"


def _null_string() -> Column:
    return F.lit(None).cast("string")


def _tagged_field(label: str, value_col: Column) -> Column:
    clean_value = F.trim(F.coalesce(value_col, F.lit("")))
    return F.when(
        F.length(clean_value) > 0,
        F.concat(F.lit(f"{label}: "), clean_value),
    ).otherwise(_null_string())


def _timestamp_days_scale(max_ts: float) -> float:
    # Amazon sources may expose timestamps in seconds or milliseconds.
    return 86_400_000.0 if max_ts and max_ts > 10_000_000_000 else 86_400.0


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — TÍNH TRỌNG SỐ VÀ TẠO SNIPPET
# ─────────────────────────────────────────────────────────────────────────────

def _compute_review_weights(df_train: DataFrame) -> DataFrame:
    """
    Tạo train-only positive/negative preference snippets.

    review_weight combines recency and helpfulness:
      exp(-age_days / half_life) * (1 + log1p(helpful_vote))
    """
    df_train = df_train.filter((F.col("rating") >= 4.0) | (F.col("rating") <= 2.0))

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

    max_ts_row = df_train.agg(F.max(F.col("timestamp").cast("double")).alias("max_ts")).first()
    max_ts = float(max_ts_row["max_ts"] or 0.0)
    ts_per_day = _timestamp_days_scale(max_ts)

    timestamp_d = F.coalesce(F.col("timestamp").cast("double"), F.lit(0.0))
    age_days = F.greatest(
        (F.lit(max_ts) - timestamp_d) / F.lit(ts_per_day),
        F.lit(0.0),
    )
    recency_weight = F.exp(-age_days / F.lit(RECENCY_HALF_LIFE_DAYS))

    helpful = F.coalesce(F.col("helpful_vote").cast(FloatType()), F.lit(0.0))
    helpful_weight = F.lit(1.0) + F.log(F.lit(1.0) + helpful)
    weight_col = recency_weight * helpful_weight

    return df_train.select(
        "reviewer_id",
        F.when(F.col("rating") >= 4.0, F.lit("pos")).otherwise(F.lit("neg")).alias("sentiment"),
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

def _build_user_profiles(
    df_weighted: DataFrame,
    shuffle_parts: int,
    top_k: int = TOP_K_REVIEWS,
) -> DataFrame:
    """
    Build field-tagged user text from train-only positive and negative reviews.
    """
    logger.info(f"[Step3] groupBy sentiment + top-{top_k} + field-tagged user text...")

    # score first so sort_array(desc) ranks by recency/helpfulness, then timestamp.
    df_struct = df_weighted.withColumn(
        "review_struct",
        F.struct(
            F.col("review_weight").alias("score"),
            F.col("timestamp").alias("t"),
            F.col("review_snippet").alias("s"),
        )
    ).repartition(shuffle_parts, "reviewer_id")

    df_stats = df_struct.groupBy("reviewer_id").agg(
        F.count("*").alias("review_count_train"),
        F.sum(F.when(F.col("sentiment") == "pos", 1).otherwise(0)).alias("pos_review_count_train"),
        F.sum(F.when(F.col("sentiment") == "neg", 1).otherwise(0)).alias("neg_review_count_train"),
        F.avg("rating").alias("avg_rating"),
        F.avg("review_weight").alias("avg_review_weight"),
    )

    df_by_sentiment = df_struct.groupBy("reviewer_id", "sentiment").agg(
        F.collect_list("review_struct").alias("reviews_raw"),
    ).withColumn(
        "reviews_topk",
        F.slice(F.sort_array(F.col("reviews_raw"), asc=False), 1, top_k)
    ).withColumn(
        "profile_text",
        F.array_join(
            F.transform(F.col("reviews_topk"), lambda x: x.getField("s")),
            " [SEP] "
        )
    )

    df_pos = df_by_sentiment.filter(F.col("sentiment") == "pos").select(
        "reviewer_id",
        F.col("profile_text").alias("user_pos_text"),
    )
    df_neg = df_by_sentiment.filter(F.col("sentiment") == "neg").select(
        "reviewer_id",
        F.col("profile_text").alias("user_neg_text"),
    )

    df_user = df_stats.join(df_pos, on="reviewer_id", how="left") \
        .join(df_neg, on="reviewer_id", how="left") \
        .withColumn(
            "user_text",
            F.concat_ws(
                " [SEP] ",
                _tagged_field("positive_preferences", F.col("user_pos_text")),
                _tagged_field("negative_preferences", F.col("user_neg_text")),
            )
        ).withColumn(
        "user_text",
        F.when(F.length(F.col("user_text")) > 5, F.col("user_text"))
         .otherwise(F.lit("[NO_TEXT] User interaction profile"))
    ).select(
        "reviewer_id",
        "user_text",
        "user_pos_text",
        "user_neg_text",
        "review_count_train",
        "pos_review_count_train",
        "neg_review_count_train",
        "avg_rating",
        "avg_review_weight",
    )

    return df_user


# ─────────────────────────────────────────────────────────────────────────────
# VAL GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

def _build_eval_ground_truth(
    df_eval: DataFrame,
    df_popularity: DataFrame,
    split_name: str,
) -> DataFrame:
    """Chỉ lấy (reviewer_id, parent_asin, popularity_group) — không lấy text.

    Bao gồm GUARDRAIL kiểm tra phân phối popularity_group trong val:
      - Nếu COLD_START > 35%: cảnh báo Recall@K toàn bộ có thể thấp giả tạo
      - Nếu TAIL > 85%: cảnh báo val bị dominated bởi long-tail items
      - Nếu HEAD < 10%: cảnh báo HEAD recall sẽ không đại diện
    """
    logger.info(f"[Step3] Xây dựng {split_name} ground truth...")

    eval_cols = ["reviewer_id", "parent_asin", "timestamp", "rating"]
    pop_slim = df_popularity.select("parent_asin", "popularity_group", "train_freq")

    df_gt = df_eval.select(*eval_cols).join(
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

    # ─── GUARDRAIL: Kiểm tra phân phối eval_gt ───────────────────────────────
    # Mục tiêu: phát hiện sớm nếu eval bị dominated bởi COLD/TAIL items.
    # Nếu eval_gt chứa quá nhiều COLD_START (items không có embedding trong train),
    # Recall@K sẽ thấp không phải vì model tệ, mà vì val biased.
    # Chi phí: 1 groupBy().collect() nhỏ (~1.8M rows → vài giây).
    dist_rows  = df_gt.groupBy("popularity_group").count().collect()
    total_val  = sum(row["count"] for row in dist_rows)
    dist_dict  = {row["popularity_group"]: row["count"] for row in dist_rows}

    logger.info(f"  📊 [GUARDRAIL] {split_name} ground truth distribution ({total_val:,} interactions):")
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
            f"⚠️  [GUARDRAIL] COLD_START chiếm {cold_pct:.1f}% {split_name}_gt (ngưỡng: 35%). "
            f"Recall@K tổng thể có thể thấp giả tạo — items này chưa có embedding trong train. "
            f"→ Ưu tiên report Recall@K_core (loại COLD_START) khi so sánh mô hình."
        )
    if tail_pct > 85:
        logger.warning(
            f"⚠️  [GUARDRAIL] TAIL chiếm {tail_pct:.1f}% {split_name}_gt (ngưỡng: 85%). "
            f"{split_name} bị dominated bởi long-tail items — phân phối lệch nhiều so với train."
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
    df_test: DataFrame | None = None,
) -> None:
    """
    Args:
        df_train: bronze_train (đã projection pushdown bởi orchestrator).
        df_val: bronze_val từ orchestrator.
        df_test: bronze_test từ orchestrator.
    """
    silver_user_out = _s3a_path(cfg, "silver", "silver_user_text_profile.parquet")
    silver_val_gt   = _s3a_path(cfg, "silver", "silver_val_ground_truth.parquet")
    silver_test_gt  = _s3a_path(cfg, "silver", "silver_test_ground_truth.parquet")
    write_mode      = cfg.get("silver", {}).get("write_mode", "overwrite")
    shuffle_parts   = int(cfg.get("spark", {}).get("shuffle_partitions", 200))
    user_top_k      = int(cfg.get("silver", {}).get("user_profile_top_k_reviews", TOP_K_REVIEWS))

    # ── Phase 1: Tính trọng số ────────────────────────────────────────────────
    logger.info("[Step3] Tính review weights...")
    df_weighted = _compute_review_weights(df_train)

    df_weighted = df_weighted.cache()
    w_count = df_weighted.count()
    logger.info(f"  Reviews có text hợp lệ: {w_count:,}")

    # ── Phase 2: Top-K + ghép text (1 pass) ───────────────────────────────────
    df_user_text = _build_user_profiles(df_weighted, shuffle_parts, top_k=user_top_k)
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
    # _build_eval_ground_truth() sẽ tự log GUARDRAIL distribution trước khi return
    df_val_gt = _build_eval_ground_truth(df_val, df_popularity, "val")
    df_val_gt.coalesce(5) \
             .write.mode(write_mode) \
             .option("compression", "zstd") \
             .parquet(silver_val_gt)

    logger.info("[Step3] silver_val_ground_truth ghi xong.")

    if df_test is not None:
        logger.info(f"[Step3] Ghi silver_test_ground_truth → {silver_test_gt}")
        df_test_gt = _build_eval_ground_truth(df_test, df_popularity, "test")
        df_test_gt.coalesce(5) \
                  .write.mode(write_mode) \
                  .option("compression", "zstd") \
                  .parquet(silver_test_gt)
        logger.info("[Step3] silver_test_ground_truth ghi xong.")

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
