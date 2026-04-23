"""
gold_step1_id_mapping.py — Gold Layer: Integer ID Mapping

Tạo integer indices liên tục (0 → N-1) cho tất cả items và users.
LightGCN và PyG yêu cầu node indices là integer liên tục.

QUAN TRỌNG:
  - Item map bao gồm TẤT CẢ items từ train + val + metadata (kể cả cold-start)
  - Cold-start items cần embedding slot để LLM alignment loss hoạt động
  - User map lấy từ train (leave-one-out → tất cả users có trong train)

OUTPUT:
  gold/gold_item_id_map.parquet — parent_asin, item_idx, title, main_category
  gold/gold_user_id_map.parquet — reviewer_id, user_idx
"""

import logging

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_s3_filesystem(cfg: dict) -> s3fs.S3FileSystem:
    mn = cfg["minio"]
    endpoint = mn["endpoint"].replace("http://", "").replace("https://", "")
    return s3fs.S3FileSystem(
        key=mn["access_key"],
        secret=mn["secret_key"],
        client_kwargs={"endpoint_url": f"http://{endpoint}"},
    )


def _s3_path(cfg: dict, *parts: str) -> str:
    bucket = cfg["minio"]["bucket"].strip("/")
    return "/".join([bucket, *parts])


# ─────────────────────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    df_interactions_train: DataFrame,
    df_interactions_val: DataFrame,
    df_item_text: DataFrame,
    df_popularity: DataFrame,
) -> dict:
    """
    Tạo integer ID mappings cho items và users.

    Args:
        df_interactions_train: silver_interactions_train DataFrame.
        df_interactions_val:   silver_interactions_val DataFrame.
        df_item_text:          silver_item_text_profile DataFrame (bao gồm cold-start).

    Returns:
        dict:
          - item_to_idx:  dict {parent_asin: int}
          - user_to_idx:  dict {reviewer_id: int}
          - item_list:    list[str] — sorted, item_list[idx] = parent_asin
          - user_list:    list[str] — sorted, user_list[idx] = reviewer_id
          - n_items:      int
          - n_users:      int
    """
    fs = _get_s3_filesystem(cfg)

    # ── Collect all unique items ──────────────────────────────────────────────
    # Union: train items + val items + metadata items (bao gồm cold-start)
    logger.info("⏳ [Gold Step 1] Collecting all unique items...")

    items_train = df_interactions_train.select("parent_asin").distinct()
    items_val   = df_interactions_val.select("parent_asin").distinct()
    items_meta  = df_item_text.select("parent_asin").distinct()

    all_items = items_train.union(items_val).union(items_meta).distinct()
    item_list = sorted([row["parent_asin"] for row in all_items.collect()])
    item_to_idx = {asin: idx for idx, asin in enumerate(item_list)}
    n_items = len(item_list)
    logger.info(f"  📊 Total unique items: {n_items:,}")

    # ── Collect all unique users ──────────────────────────────────────────────
    # Leave-one-out → tất cả 1.8M users đều có trong train
    logger.info("⏳ [Gold Step 1] Collecting all unique users and computing frequency...")

    user_freq_df = df_interactions_train.groupBy("reviewer_id").agg(
        F.count("parent_asin").alias("user_train_freq")
    )
    user_freq_df = user_freq_df.withColumn(
        "user_activity_group",
        F.when(F.col("user_train_freq") < 5, F.lit("INACTIVE"))
         .when(F.col("user_train_freq") <= 20, F.lit("ACTIVE"))
         .otherwise(F.lit("SUPER_ACTIVE"))
    )
    all_users = user_freq_df.collect()

    user_list = sorted([row["reviewer_id"] for row in all_users])
    user_to_idx = {uid: idx for idx, uid in enumerate(user_list)}
    
    user_meta_dict = {}
    for row in all_users:
        user_meta_dict[row["reviewer_id"]] = {
            "user_train_freq": row["user_train_freq"] or 0,
            "user_activity_group": row["user_activity_group"] or "INACTIVE",
        }

    n_users = len(user_list)
    logger.info(f"  📊 Total unique users: {n_users:,}")

    # ── Ước tính kích thước embedding matrix ──────────────────────────────────
    embed_dim = cfg.get("gold", {}).get("embedding_dim", 384)
    total_nodes = n_items + n_users
    embed_size_mb = total_nodes * embed_dim * 4 / (1024 * 1024)
    logger.info(f"  📊 Embedding matrix estimate: {total_nodes:,} nodes × {embed_dim}d = {embed_size_mb:.0f} MB")

    # ── Enrich item map với title + category từ metadata ──────────────────────
    # ── Enrich item map với title + category + popularity ──────────────────────
    logger.info("⏳ [Gold Step 1] Enriching item map with metadata & popularity...")

    item_meta_rows = df_item_text.select(
        "parent_asin", "title", "main_category"
    ).join(
        df_popularity.select("parent_asin", "popularity_group", "train_freq"),
        on="parent_asin", how="left"
    ).collect()

    item_meta_dict = {}
    for row in item_meta_rows:
        item_meta_dict[row["parent_asin"]] = {
            "title": row["title"] or "",
            "main_category": row["main_category"] or "",
            "popularity_group": row["popularity_group"] or "COLD_START",
            "train_freq": row["train_freq"] or 0,
        }

    # MD5 Checksums
    import hashlib
    hash_md5 = hashlib.md5()
    hash_md5.update("".join(item_list).encode('utf-8'))
    hash_md5.update("".join(user_list).encode('utf-8'))
    checksum = hash_md5.hexdigest()

    # ── Save gold_item_id_map.parquet ─────────────────────────────────────────
    item_map_path = _s3_path(cfg, "gold", "gold_item_id_map.parquet")
    logger.info(f"⏳ [Gold Step 1] Writing item ID map → s3://{item_map_path}")

    item_table = pa.table({
        "parent_asin":   pa.array(item_list, type=pa.string()),
        "item_idx":      pa.array(list(range(n_items)), type=pa.int32()),
        "title":         pa.array(
            [item_meta_dict.get(a, {}).get("title", "") for a in item_list],
            type=pa.string(),
        ),
        "main_category": pa.array(
            [item_meta_dict.get(a, {}).get("main_category", "") for a in item_list],
            type=pa.string(),
        ),
        "popularity_group": pa.array(
            [item_meta_dict.get(a, {}).get("popularity_group", "COLD_START") for a in item_list],
            type=pa.string(),
        ),
        "train_freq": pa.array(
            [item_meta_dict.get(a, {}).get("train_freq", 0) for a in item_list],
            type=pa.int64(),
        ),
    })

    with fs.open(item_map_path, "wb") as f:
        pq.write_table(item_table, f, compression="zstd")

    # ── Save gold_user_id_map.parquet ─────────────────────────────────────────
    user_map_path = _s3_path(cfg, "gold", "gold_user_id_map.parquet")
    logger.info(f"⏳ [Gold Step 1] Writing user ID map → s3://{user_map_path}")

    user_table = pa.table({
        "reviewer_id":  pa.array(user_list, type=pa.string()),
        "user_idx":     pa.array(list(range(n_users)), type=pa.int32()),
        "user_train_freq": pa.array(
            [user_meta_dict.get(u, {}).get("user_train_freq", 0) for u in user_list],
            type=pa.int64(),
        ),
        "user_activity_group": pa.array(
            [user_meta_dict.get(u, {}).get("user_activity_group", "INACTIVE") for u in user_list],
            type=pa.string(),
        ),
    })

    with fs.open(user_map_path, "wb") as f:
        pq.write_table(user_table, f, compression="zstd")

    logger.info(f"✅ [Gold Step 1] ID Mapping complete: {n_items:,} items, {n_users:,} users")

    return {
        "n_items":     n_items,
        "n_users":     n_users,
        "checksum":    checksum,
    }
