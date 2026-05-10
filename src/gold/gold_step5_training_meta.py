"""
gold_step5_training_meta.py — Gold Layer: Training Metadata

Tạo các numpy arrays và JSON metadata phục vụ training loop.

NEGATIVE SAMPLING — Blended Strategy:
  Mục tiêu: Tập trung long-tail NHUNG vẫn giữ được chất lượng head.

  Hai component được blend:
    A) Tail-focus:  P_tail(j) ∝ freq^{-β_ns}   → tail items có xác suất CAO
    B) Head-retain: P_head(j) ∝ freq^{+β_ns*0.4} → head items vẫn xuất hiện

  P(j) = α · P_tail(j) + (1-α) · P_head(j)
    α = neg_sampling_blend_alpha (mặc định 0.70)
    α = 1.0 → pure inverse-popularity (chỉ tập trung tail)
    α = 0.0 → pure popularity-biased (chỉ tập trung head, như word2vec)
    α = 0.7 → 70% tập trung tail + 30% giữ head (khuyến nghị)

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
    gold_cfg    = cfg.get("gold", {})
    beta_ns     = float(gold_cfg.get("negative_sampling_beta", 0.75))
    blend_alpha = float(gold_cfg.get("neg_sampling_blend_alpha", 0.70))
    tmp_dir     = gold_cfg.get("temp_dir", "data/gold_temp")
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

    # ── Negative sampling probabilities (Blended Strategy) ──────────────────────
    # Mục tiêu: Tập trung long-tail NHUNG vẫn giữ được chất lượng head.
    #
    # Vấn đề với pure inverse-popularity (α=1.0):
    #   Head items gần như không bao giờ là negative
    #   → Model không học được rằng “đây là negative” với head
    #   → Head recall giảm (model dễ bị đánh lừa bởi head items)
    #
    # Blended: P(j) = α · P_tail(j) + (1-α) · P_head(j)
    #   P_tail(j) ∝ freq^{-β}    → tail items được ưu tiên làm negative
    #   P_head(j) ∝ freq^{+β*0.4} → head items vẫn xuất hiện đủ để học
    #
    # Tham số tune:
    #   blend_alpha = 0.70 (mặc định): 70% tail-focus, 30% head-retain
    #   blend_alpha = 0.85: tập trung tail hơn nữa (có thể giảm head quality)
    #   blend_alpha = 0.50: cân bằng hơn
    logger.info(
        f"⏳ [Gold Step 5] Computing blended neg-sampling probs "
        f"(β={beta_ns}, α={blend_alpha})..."
    )

    freq_safe = np.maximum(train_freq_arr, 1).astype(np.float64)

    # Component A: Tail-focus — inverse-popularity
    p_tail = np.power(freq_safe, -beta_ns)
    p_tail = p_tail / p_tail.sum()

    # Component B: Head-retain — mild popularity-biased (exponent nhỏ hơn)
    p_head = np.power(freq_safe, beta_ns * 0.4)
    p_head = p_head / p_head.sum()

    # Blend
    raw_prob = blend_alpha * p_tail + (1.0 - blend_alpha) * p_head

    # Normalize về tổng = 1
    neg_sampling_prob = (raw_prob / raw_prob.sum()).astype(np.float32)

    # Clip tránh extreme values
    eps = 1e-7
    neg_sampling_prob = np.clip(neg_sampling_prob, eps, 1.0 - eps)
    neg_sampling_prob = neg_sampling_prob / neg_sampling_prob.sum()  # re-normalize

    logger.info(f"  📊 Sampling prob range: [{neg_sampling_prob.min():.2e}, {neg_sampling_prob.max():.2e}]")
    logger.info(f"  📊 Blend: {blend_alpha:.0%} tail-focus + {1-blend_alpha:.0%} head-retain")
    # xác suất cao nhất = tail/cold-start items; thấp nhất = head items
    logger.info(f"  📊 Max prob (tail): {neg_sampling_prob.max():.2e} | Min prob (head): {neg_sampling_prob.min():.2e}")


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
        "negative_sampling_beta":        beta_ns,
        "neg_sampling_blend_alpha":      blend_alpha,
        "neg_sampling_strategy":         "blended",   # α·freq^{-β} + (1-α)·freq^{+0.4β}
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
