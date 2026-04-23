"""
gold_step5_training_meta.py — Gold Layer: Training Metadata

Tạo các numpy arrays và JSON metadata phục vụ training loop.

NEGATIVE SAMPLING:
  P(j là negative) ∝ train_freq(j)^{-β_ns}
  β_ns = 0.75 mặc định
  → Head items có xác suất làm negative THẤP hơn
  → Tail items có xác suất làm negative CAO hơn
  → Mô hình học phân biệt tail items lẫn nhau

OUTPUT:
  gold/gold_item_train_freq.npy         — shape [N_items], dtype int64
  gold/gold_item_popularity_group.npy   — shape [N_items], dtype int8 (0=HEAD, 1=MID, 2=TAIL, 3=COLD)
  gold/gold_user_train_freq.npy         — shape [N_users], dtype int64
  gold/gold_user_activity_group.npy     — shape [N_users], dtype int8 (0=INACTIVE, 1=ACTIVE, 2=SUPER_ACTIVE)
  gold/gold_negative_sampling_prob.npy  — shape [N_items], dtype float32
  gold/gold_dataset_stats.json          — N, sparsity, tail_ratio, ...
"""

import gc
import json
import logging
import os

import numpy as np
import s3fs

from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)

# Mapping popularity_group string → integer
POP_GROUP_MAP = {
    "HEAD":       0,
    "MID":        1,
    "TAIL":       2,
    "COLD_START": 3,
}

# Mapping user_activity_group string → integer
USER_ACTIVITY_MAP = {
    "INACTIVE":     0,
    "ACTIVE":       1,
    "SUPER_ACTIVE": 2,
}


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


def _save_numpy_s3(
    arr: np.ndarray, filename: str, cfg: dict,
    fs: s3fs.S3FileSystem, tmp_dir: str,
) -> None:
    """Ghi numpy array → temp file → upload S3 → xóa temp."""
    s3_path = _s3_path(cfg, "gold", filename)
    local_path = os.path.join(tmp_dir, filename)
    np.save(local_path, arr)
    fs.put(local_path, s3_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    logger.info(f"  📤 {filename} → s3://{s3_path} ({size_mb:.1f} MB)")
    os.remove(local_path)


def _save_json_s3(data: dict, filename: str, cfg: dict, fs: s3fs.S3FileSystem) -> None:
    """Ghi JSON dict → S3."""
    s3_path = _s3_path(cfg, "gold", filename)
    with fs.open(s3_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"  📤 {filename} → s3://{s3_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    step1_info: dict,
    edge_info: dict,
) -> None:
    """
    Tạo training metadata arrays và dataset stats.

    Args:
        step1_info: dict từ Gold Step 1.
        edge_info: dict từ Gold Step 2 (chứa n_edges).
    """
    gold_cfg = cfg.get("gold", {})
    beta_ns  = float(gold_cfg.get("negative_sampling_beta", 0.75))
    tmp_dir  = gold_cfg.get("temp_dir", "data/gold_temp")
    os.makedirs(tmp_dir, exist_ok=True)

    fs = _get_s3_filesystem(cfg)

    n_items     = step1_info["n_items"]
    n_users     = step1_info["n_users"]
    n_edges     = edge_info["n_edges"]

    # ── Collect popularity data ───────────────────────────────────────────────
    logger.info("⏳ [Gold Step 5] Collecting popularity data from ID Map...")

    item_map_path = f"s3a://{_s3_path(cfg, 'gold', 'gold_item_id_map.parquet')}"
    pop_rows = (
        spark.read.parquet(item_map_path)
        .select("item_idx", "train_freq", "popularity_group")
        .collect()
    )
    logger.info(f"  📦 Collected {len(pop_rows):,} item popularity records")

    # ── Build arrays indexed by item_idx ──────────────────────────────────────
    train_freq_arr = np.zeros(n_items, dtype=np.int64)
    pop_group_arr  = np.full(n_items, POP_GROUP_MAP["COLD_START"], dtype=np.int8)

    for row in pop_rows:
        idx = row["item_idx"]
        train_freq_arr[idx] = int(row["train_freq"])
        group_str = row["popularity_group"] or "COLD_START"
        pop_group_arr[idx] = POP_GROUP_MAP.get(group_str, 3)

    del pop_rows
    gc.collect()

    # ── Log item distribution ──────────────────────────────────────────────────────
    for group_name, group_id in POP_GROUP_MAP.items():
        count = int(np.sum(pop_group_arr == group_id))
        pct = count / n_items * 100
        logger.info(f"  📊 ITEM {group_name}: {count:,} items ({pct:.1f}%)")

    # ── Collect USER activity data ───────────────────────────────────────────────
    logger.info("⏳ [Gold Step 5] Collecting user activity data from ID Map...")

    user_map_path = f"s3a://{_s3_path(cfg, 'gold', 'gold_user_id_map.parquet')}"
    user_rows = (
        spark.read.parquet(user_map_path)
        .select("user_idx", "user_train_freq", "user_activity_group")
        .collect()
    )
    logger.info(f"  📦 Collected {len(user_rows):,} user activity records")

    user_train_freq_arr = np.zeros(n_users, dtype=np.int64)
    user_activity_arr   = np.full(n_users, USER_ACTIVITY_MAP["INACTIVE"], dtype=np.int8)

    for row in user_rows:
        idx = row["user_idx"]
        user_train_freq_arr[idx] = int(row["user_train_freq"])
        group_str = row["user_activity_group"] or "INACTIVE"
        user_activity_arr[idx] = USER_ACTIVITY_MAP.get(group_str, 0)

    del user_rows
    gc.collect()

    # ── Log user distribution ──────────────────────────────────────────────────────
    for group_name, group_id in USER_ACTIVITY_MAP.items():
        count = int(np.sum(user_activity_arr == group_id))
        pct = count / n_users * 100
        logger.info(f"  📊 USER {group_name}: {count:,} users ({pct:.1f}%)")

    # ── Negative sampling probabilities ───────────────────────────────────────
    # P(j) ∝ train_freq(j)^{β_ns} (Positive alpha - Popularity-biased)
    # Head items có freq cao sẽ bị pick làm negative nhiều hơn, tránh nịnh hót Head.
    logger.info(f"⏳ [Gold Step 5] Computing negative sampling probs (β={beta_ns})...")

    freq_safe = np.maximum(train_freq_arr, 1).astype(np.float64)
    raw_prob  = np.power(freq_safe, beta_ns)  # Đã sửa -beta_ns thành beta_ns dương

    # Normalize về tổng = 1
    prob_sum  = raw_prob.sum()
    neg_sampling_prob = (raw_prob / prob_sum).astype(np.float32)

    # Clip tránh extreme values
    eps = 1e-7
    neg_sampling_prob = np.clip(neg_sampling_prob, eps, 1.0 - eps)
    neg_sampling_prob = neg_sampling_prob / neg_sampling_prob.sum()  # re-normalize

    logger.info(f"  📊 Sampling prob range: [{neg_sampling_prob.min():.2e}, {neg_sampling_prob.max():.2e}]")
    logger.info(f"  📊 Top-1 head item prob: {neg_sampling_prob.max():.2e} (cao → tăng cường làm negative)")

    # ── Dataset statistics ────────────────────────────────────────────────────
    sparsity = 1.0 - (n_edges / (n_users * n_items))

    n_head = int(np.sum(pop_group_arr == 0))
    n_mid  = int(np.sum(pop_group_arr == 1))
    n_tail = int(np.sum(pop_group_arr == 2))
    n_cold = int(np.sum(pop_group_arr == 3))

    n_inactive = int(np.sum(user_activity_arr == 0))
    n_active   = int(np.sum(user_activity_arr == 1))
    n_super    = int(np.sum(user_activity_arr == 2))

    dataset_stats = {
        "n_users":          n_users,
        "n_items":          n_items,
        "n_edges_train":    n_edges,
        "sparsity":         round(sparsity, 8),
        "sparsity_pct":     f"{sparsity * 100:.4f}%",
        "n_head_items":     n_head,
        "n_mid_items":      n_mid,
        "n_tail_items":     n_tail,
        "n_cold_items":     n_cold,
        "n_inactive_users": n_inactive,
        "n_active_users":   n_active,
        "n_super_active_users": n_super,
        "tail_ratio":       round(n_tail / max(n_items, 1), 4),
        "cold_ratio":       round(n_cold / max(n_items, 1), 4),
        "inactive_user_ratio": round(n_inactive / max(n_users, 1), 4),
        "avg_degree_user":  round(n_edges / max(n_users, 1), 2),
        "avg_degree_item":  round(n_edges / max(n_items, 1), 2),
        "max_train_freq_item": int(train_freq_arr.max()),
        "max_train_freq_user": int(user_train_freq_arr.max()),
        "median_train_freq_item": int(np.median(train_freq_arr[train_freq_arr > 0])) if np.any(train_freq_arr > 0) else 0,
        "median_train_freq_user": int(np.median(user_train_freq_arr[user_train_freq_arr > 0])) if np.any(user_train_freq_arr > 0) else 0,
        "embedding_dim":    int(cfg.get("gold", {}).get("embedding_dim", 384)),
        "negative_sampling_beta": beta_ns,
        "head_ratio_split": 0.20,
        "mid_ratio_split":  0.10,
        "tail_ratio_split": 0.70,
    }

    # ── Save all artifacts ────────────────────────────────────────────────────
    logger.info("⏳ [Gold Step 5] Saving training metadata...")

    _save_numpy_s3(train_freq_arr,    "gold_item_train_freq.npy",        cfg, fs, tmp_dir)
    _save_numpy_s3(pop_group_arr,     "gold_item_popularity_group.npy",  cfg, fs, tmp_dir)
    _save_numpy_s3(user_train_freq_arr, "gold_user_train_freq.npy",      cfg, fs, tmp_dir)
    _save_numpy_s3(user_activity_arr,   "gold_user_activity_group.npy",  cfg, fs, tmp_dir)
    _save_numpy_s3(neg_sampling_prob, "gold_negative_sampling_prob.npy", cfg, fs, tmp_dir)
    _save_json_s3(dataset_stats,      "gold_dataset_stats.json",         cfg, fs)

    # ── Log dataset stats ─────────────────────────────────────────────────────
    logger.info("📊 Dataset Statistics:")
    for key, val in dataset_stats.items():
        logger.info(f"  {key}: {val}")

    logger.info("✅ [Gold Step 5] Training metadata complete.")
