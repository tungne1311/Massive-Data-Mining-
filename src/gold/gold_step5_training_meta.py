"""
gold_step5_training_meta.py — Gold Layer: Training Metadata

Tạo các numpy arrays và JSON metadata phục vụ training loop.

NEGATIVE SAMPLING:
  Tạo phân phối lấy mẫu negative theo cấu hình:
    - uniform
    - popularity
    - inverse_frequency
    - blended

  Sampling chỉ áp dụng trên warm items (train_freq > 0). Cold items có
  probability = 0 để không làm nhiễu warm long-tail protocol.

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

# Mapping popularity_group string → integer.
# WARM_TAIL/COLD_ITEM are kept as aliases for newer docs/artifacts.
POP_GROUP_MAP = {
    "HEAD":       0,
    "MID":        1,
    "TAIL":       2,
    "WARM_TAIL":  2,
    "COLD_START": 3,
    "COLD_ITEM":  3,
}

POP_GROUP_DISPLAY = [
    ("HEAD", 0),
    ("MID", 1),
    ("TAIL", 2),
    ("COLD_START", 3),
]

# Mapping user_activity_group string → integer
USER_ACTIVITY_MAP = {
    "INACTIVE":     0,
    "ACTIVE":       1,
    "SUPER_ACTIVE": 2,
}

VALID_NEGATIVE_SAMPLING_STRATEGIES = {
    "uniform",
    "popularity",
    "inverse_frequency",
    "blended",
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


def _normalize_warm_probs(raw_prob: np.ndarray, warm_mask: np.ndarray) -> np.ndarray:
    """Normalize probabilities on warm items only; cold items stay at zero."""
    prob = np.zeros_like(raw_prob, dtype=np.float64)
    prob[warm_mask] = raw_prob[warm_mask]
    total = float(prob.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Negative sampling probability has non-positive or non-finite mass.")
    return (prob / total).astype(np.float32)


def _build_negative_sampling_prob(
    train_freq_arr: np.ndarray,
    strategy: str,
    beta_ns: float,
    blend_alpha: float,
) -> np.ndarray:
    """
    Build item negative-sampling probabilities with zero mass for cold items.

    - uniform: equal probability over warm items.
    - popularity: P(j) proportional to freq(j)^beta.
    - inverse_frequency: P(j) proportional to freq(j)^(-beta).
    - blended: alpha * inverse_frequency + (1-alpha) * mild popularity.
    """
    strategy = (strategy or "uniform").strip().lower()
    if strategy not in VALID_NEGATIVE_SAMPLING_STRATEGIES:
        valid = ", ".join(sorted(VALID_NEGATIVE_SAMPLING_STRATEGIES))
        raise ValueError(f"Unknown negative_sampling_strategy={strategy!r}. Valid: {valid}")
    if beta_ns < 0:
        raise ValueError("negative_sampling_beta must be >= 0.")
    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError("neg_sampling_blend_alpha must be in [0, 1].")

    warm_mask = train_freq_arr > 0
    n_warm = int(warm_mask.sum())
    if n_warm == 0:
        raise ValueError("No warm items available for negative sampling.")

    freq = train_freq_arr.astype(np.float64)
    raw_prob = np.zeros_like(freq, dtype=np.float64)

    if strategy == "uniform":
        raw_prob[warm_mask] = 1.0
        return _normalize_warm_probs(raw_prob, warm_mask)

    if strategy == "popularity":
        raw_prob[warm_mask] = np.power(freq[warm_mask], beta_ns)
        return _normalize_warm_probs(raw_prob, warm_mask)

    if strategy == "inverse_frequency":
        raw_prob[warm_mask] = np.power(freq[warm_mask], -beta_ns)
        return _normalize_warm_probs(raw_prob, warm_mask)

    tail_component = np.zeros_like(freq, dtype=np.float64)
    tail_component[warm_mask] = np.power(freq[warm_mask], -beta_ns)
    p_tail = _normalize_warm_probs(tail_component, warm_mask).astype(np.float64)

    head_component = np.zeros_like(freq, dtype=np.float64)
    head_component[warm_mask] = np.power(freq[warm_mask], beta_ns * 0.4)
    p_head = _normalize_warm_probs(head_component, warm_mask).astype(np.float64)

    raw_prob = blend_alpha * p_tail + (1.0 - blend_alpha) * p_head
    return _normalize_warm_probs(raw_prob, warm_mask)


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
    neg_strategy = str(gold_cfg.get("negative_sampling_strategy", "uniform")).strip().lower()
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
    for group_name, group_id in POP_GROUP_DISPLAY:
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
    # Warm-only by design: cold items have train_freq = 0 and receive zero mass.
    # This keeps the training artifact aligned with warm long-tail evaluation.
    logger.info(
        f"⏳ [Gold Step 5] Computing {neg_strategy} neg-sampling probs "
        f"(β={beta_ns}, α={blend_alpha}, warm-only=True)..."
    )

    neg_sampling_prob = _build_negative_sampling_prob(
        train_freq_arr=train_freq_arr,
        strategy=neg_strategy,
        beta_ns=beta_ns,
        blend_alpha=blend_alpha,
    )
    warm_mask = train_freq_arr > 0
    warm_prob = neg_sampling_prob[warm_mask]
    cold_prob_mass = float(neg_sampling_prob[~warm_mask].sum())
    logger.info(f"  📊 Warm sampling prob range: [{warm_prob.min():.2e}, {warm_prob.max():.2e}]")
    logger.info(f"  📊 Warm items with prob > 0: {int(np.sum(neg_sampling_prob > 0)):,}")
    logger.info(f"  📊 Cold prob mass: {cold_prob_mass:.2e}")
    logger.info(
        "  📊 Prob mass by group: "
        f"HEAD={float(neg_sampling_prob[pop_group_arr == 0].sum()):.4f}, "
        f"MID={float(neg_sampling_prob[pop_group_arr == 1].sum()):.4f}, "
        f"TAIL={float(neg_sampling_prob[pop_group_arr == 2].sum()):.4f}, "
        f"COLD={float(neg_sampling_prob[pop_group_arr == 3].sum()):.4f}"
    )


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
        "negative_sampling_strategy":   neg_strategy,
        "negative_sampling_beta":        beta_ns,
        "neg_sampling_blend_alpha":      blend_alpha,
        "neg_sampling_strategy":         neg_strategy,
        "neg_sampling_warm_only":        True,
        "neg_sampling_warm_item_count":  int(np.sum(warm_mask)),
        "neg_sampling_cold_prob_mass":   round(cold_prob_mass, 10),
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
