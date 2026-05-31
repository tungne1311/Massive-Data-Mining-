"""
gold_step2_edge_list.py — Gold Layer: Edge List (PyG Format)

Tạo edge list chuẩn PyG: edge_index [2, E].
- Đã bỏ trọng số cạnh, mọi tương tác giữ lại coi như 1 cạnh (edge weight = 1 implicit)

OUTPUT:
  gold/gold_edge_index.npy   — shape [2, E], dtype int64
"""

import gc
import logging
import os

import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────────────────────

def run(
    spark: SparkSession,
    cfg: dict,
    df_interactions_train: DataFrame,
    step1_info: dict,
) -> dict:
    import pandas as pd
    
    gold_cfg = cfg.get("gold", {})
    tmp_dir  = gold_cfg.get("temp_dir", "data/gold_temp")
    os.makedirs(tmp_dir, exist_ok=True)

    fs = _get_s3_filesystem(cfg)

    # ── Đọc lại mapping từ Parquet để Spark xử lý song song trên cluster ────────────
    # Step 1 đã ghi ra S3, bây giờ nạp lại thành DataFrame để Broadcast Join
    item_map_path = f"s3a://{_s3_path(cfg, 'gold', 'gold_item_id_map.parquet')}"
    user_map_path = f"s3a://{_s3_path(cfg, 'gold', 'gold_user_id_map.parquet')}"

    df_item_map = spark.read.parquet(item_map_path).select("parent_asin", "item_idx")
    df_user_map = spark.read.parquet(user_map_path).select("reviewer_id", "user_idx")

    logger.info("⏳ [Gold Step 2] Joining interactions with ID maps on Spark (Parallel)...")

    # 1. Join với bảng ID (Sử dụng Broadcast để gửi bảng map nhỏ tới mọi node, tránh shuffle cực nặng)
    df_mapped = df_interactions_train.alias("inter") \
        .join(F.broadcast(df_user_map).alias("u"), F.col("inter.reviewer_id") == F.col("u.reviewer_id")) \
        .join(F.broadcast(df_item_map).alias("i"), F.col("inter.parent_asin") == F.col("i.parent_asin")) \
        .select("u.user_idx", "i.item_idx", "inter.rating", "inter.timestamp") \
        .dropna(subset=["user_idx", "item_idx", "rating", "timestamp"])

    # 2. Bỏ tính T_max, T_min, T_range vì Temporal Decay đã vào thẳng BPR Loss

    # 3. Không gán Edge Weight nữa để giảm bộ nhớ. (Mặc định edge=1 implicit cho LightGCN)
    df_final = df_mapped.select("user_idx", "item_idx")

    # 4. Workaround Socket Broken Pipe: Thay vì `toPandas()` truyền dữ liệu qua Spark RPC cực chậm,
    # chúng ta lưu kết quả xuống MinIO qua các Executors, rồi đọc trực tiếp về Driver bằng Pandas/PyArrow.
    temp_parquet_path = f"s3a://{_s3_path(cfg, 'gold', 'temp_edges.parquet')}"
    logger.info(f"⏳ [Gold Step 2] Writing temp parquet to MinIO to bypass Spark RPC limitations...")
    df_final.write.mode("overwrite").parquet(temp_parquet_path)

    logger.info("⏳ [Gold Step 2] Loading Parquet back to Driver natively via PyArrow Dataset...")
    import pyarrow.dataset as ds
    
    # Bypass Pandas' strict fsspec version check by using PyArrow directly với existing fs
    dataset_path = _s3_path(cfg, 'gold', 'temp_edges.parquet')
    dataset = ds.dataset(dataset_path, filesystem=fs, format="parquet")
    table = dataset.to_table()

    logger.info("⏳ [Gold Step 2] Converting to NumPy arrays...")
    user_arr    = table["user_idx"].to_numpy().astype(np.int64)
    item_arr    = table["item_idx"].to_numpy().astype(np.int64)

    # Giải phóng memory DataFrame lập tức
    del table
    gc.collect()

    n_edges = len(user_arr)
    logger.info(f"  📊 Total edges: {n_edges:,}")

    # Edge index [2, E]
    edge_index = np.stack([user_arr, item_arr], axis=0)
    
    # Dọn dẹp RAM tạm
    del user_arr, item_arr
    gc.collect()

    # ── Item-Item Semantic Edges (KNN) - MOVED TO COLAB ──────────────────────────
    # Logic nối cạnh Semantic dựa trên LLM Embedding yêu cầu GPU và FAISS cài đặt phức tạp.
    # Để tối ưu tài nguyên local, bước này được chuyển vào Notebook/Colab.
    # Step 2 tại đây chỉ tập trung vào cấu trúc User-Item Bipartite Graph chuẩn.

    # 5. Lưu vào MinIO
    logger.info("⏳ [Gold Step 2] Saving edge list to MinIO...")

    _save_numpy_s3(edge_index,  "gold_edge_index.npy",  cfg, fs, tmp_dir)

    # Cleanup temp parquet
    try:
        fs.rm(_s3_path(cfg, 'gold', 'temp_edges.parquet'), recursive=True)
    except Exception:
        pass

    logger.info(f"✅ [Gold Step 2] Edge list complete: {n_edges:,} edges")

    return {"n_edges": n_edges}
