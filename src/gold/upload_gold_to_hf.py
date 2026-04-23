"""
upload_gold_to_hf.py — Đẩy Gold Artifacts lên Hugging Face Hub

Hỗ trợ 2 chế độ:
  1. --mode full    : Upload toàn bộ Gold artifacts (bao gồm embeddings nặng)
  2. --mode partial : Chỉ upload Gold maps/edges/metadata (bỏ qua embeddings)

Cách dùng:
  # Trong Docker (MinIO accessible):
  docker compose run --rm pipeline python src/gold/upload_gold_to_hf.py --mode full

  # Hoặc local (cần .env):
  python src/gold/upload_gold_to_hf.py --mode partial --repo-id your-username/your-repo
  docker compose run --rm pipeline python src/gold/upload_gold_to_hf.py --mode full --repo-id chuongdo1104/amazon-2023-gold
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import s3fs
from huggingface_hub import HfApi


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Tất cả Gold artifacts (mode=full)
GOLD_FILES = [
    "gold/gold_item_id_map.parquet",
    "gold/gold_user_id_map.parquet",
    "gold/gold_edge_index.npy",
    # Đã chuyển step tạo Embeddings này lên Colab nên không còn file này
    # "gold/gold_item_embeddings.npy",       # Nặng (~1.8 GB)
    # "gold/gold_user_embeddings.npy",       # Nặng (~2.8 GB)
    "gold/gold_item_train_freq.npy",
    "gold/gold_item_popularity_group.npy",
    "gold/gold_user_train_freq.npy",
    "gold/gold_user_activity_group.npy",
    "gold/gold_negative_sampling_prob.npy",
    "gold/gold_dataset_stats.json",
]

# Mode partial: Bỏ qua embeddings để tiết kiệm thời gian/băng thông
PARTIAL_GOLD_FILES = [
    "gold/gold_item_id_map.parquet",
    "gold/gold_user_id_map.parquet",
    "gold/gold_edge_index.npy",
    "gold/gold_item_train_freq.npy",
    "gold/gold_item_popularity_group.npy",
    "gold/gold_user_train_freq.npy",
    "gold/gold_user_activity_group.npy",
    "gold/gold_negative_sampling_prob.npy",
    "gold/gold_dataset_stats.json",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_s3_filesystem() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        client_kwargs={
            "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        },
    )


def upload_file_to_hf(
    fs: s3fs.S3FileSystem,
    api: HfApi,
    repo_id: str,
    bucket: str,
    s3_relative_path: str,
    hf_path: str = None,
) -> None:
    """Download từ MinIO → upload lên HuggingFace từng file."""
    s3_full = f"{bucket}/{s3_relative_path}"
    hf_path = hf_path or s3_relative_path

    # Lấy kích thước file
    try:
        info = fs.info(s3_full)
        size_mb = info.get("size", 0) / (1024 * 1024)
    except Exception:
        size_mb = 0

    print(f"  {s3_relative_path} ({size_mb:.1f} MB) → {hf_path}")

    # File lớn (>100 MB): download xuống temp rồi upload
    if size_mb > 100:
        with tempfile.NamedTemporaryFile(
            suffix=Path(s3_relative_path).suffix, delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            print(f"     ⬇ Downloading to temp ({size_mb:.0f} MB)...")
            fs.get(s3_full, tmp_path)
            print(f"     ⬆ Uploading to HuggingFace...")
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=hf_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {hf_path} ({size_mb:.0f} MB)",
            )
        finally:
            os.remove(tmp_path)
    else:
        # File nhỏ: đọc vào memory
        with fs.open(s3_full, "rb") as f:
            data = f.read()
        api.upload_file(
            path_or_fileobj=data,
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {hf_path}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload Gold artifacts to HuggingFace Hub")
    parser.add_argument(
        "--mode",
        choices=["full", "partial"],
        default="partial",
        help=(
            "full: upload toàn bộ Gold (bao gồm embeddings ~4.6 GB). "
            "partial: CHỈ upload Gold maps/edges/metadata (bỏ qua embeddings)"
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_GOLD_REPO", "chuongdo1104/amazon-2023-gold"),
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("MINIO_BUCKET", "recsys"),
        help="MinIO bucket name",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN chưa được set! Export HF_TOKEN=hf_xxx trước khi chạy.")
        sys.exit(1)

    print("=" * 60)
    print(f"Upload Gold → HuggingFace: {args.repo_id}")
    print(f"   Mode: {args.mode}")
    print("=" * 60)

    fs  = get_s3_filesystem()
    api = HfApi(token=token)

    # 1. Tạo repo nếu chưa có
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    # -------------------------------------------------------------------------
    # BƯỚC MỚI: DỌN DẸP DỮ LIỆU CŨ TRÊN HUGGING FACE ĐỂ TRÁNH RÁC
    # -------------------------------------------------------------------------
    print("\nĐang dọn dẹp thư mục 'gold/' cũ trên Hugging Face...")
    try:
        # Xóa nguyên thư mục gold/ để đảm bảo sạch 100%
        api.delete_folder(
            path_in_repo="gold",
            repo_id=args.repo_id,
            repo_type="dataset"
        )
        print("  Đã xóa sạch thư mục gold/ cũ.")
    except Exception:
        print("  Thư mục gold/ chưa tồn tại trên repo (có thể là lần upload đầu tiên).")

    # 2. Xác định danh sách file cần upload
    if args.mode == "full":
        files_to_upload = GOLD_FILES
    else:
        files_to_upload = PARTIAL_GOLD_FILES

    # 3. Kiểm tra file tồn tại trên MinIO
    print("\nKiểm tra file trên MinIO...")
    valid_files = []
    for rel_path in files_to_upload:
        full_path = f"{args.bucket}/{rel_path}"
        try:
            fs.info(full_path)
            valid_files.append(rel_path)
        except FileNotFoundError:
            print(f" KHÔNG TÌM THẤY: {rel_path} — bỏ qua")

    if not valid_files:
        print("Không có file nào để upload!")
        sys.exit(1)

    print(f"\n Sẽ upload {len(valid_files)} files:")
    for f in valid_files:
        print(f"  • {f}")

    # 4. Upload từng file
    print(f"\n Đang upload {len(valid_files)} files...")
    for i, rel_path in enumerate(valid_files, 1):
        print(f"\n[{i}/{len(valid_files)}]")
        upload_file_to_hf(fs, api, args.repo_id, args.bucket, rel_path)

    print("\n" + "=" * 60)
    print(f"🎉 THÀNH CÔNG! Đã upload {len(valid_files)} files lên {args.repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()