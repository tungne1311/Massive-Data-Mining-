"""
upload_silver_to_hf.py — Đẩy toàn bộ Silver files (Parquet) lên Hugging Face Hub

Cách dùng:
  # Trong Docker (MinIO accessible):
  docker compose run --rm -e HF_TOKEN=hf_xxx pipeline python src/silver/upload_silver_to_hf.py

Output repo mặc định: chuongdo1104/amazon-2023-silver
"""

import argparse
import os
import sys

import s3fs
from huggingface_hub import HfApi, CommitOperationAdd


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload Silver data to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_SILVER_REPO", "chuongdo1104/amazon-2023-silver"),
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
        print(" HF_TOKEN chưa được set! Export HF_TOKEN=hf_xxx trước khi chạy.")
        sys.exit(1)

    print("=" * 60)
    print(f" Upload Silver → HuggingFace: {args.repo_id}")
    print("=" * 60)

    # 1. Kết nối MinIO
    fs = s3fs.S3FileSystem(
        key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        client_kwargs={
            "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        },
    )
    
    # 2. Khởi tạo/Check Repo trên HF
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # BƯỚC XÓA: Dọn sạch thư mục silver/ cũ trước khi upload mới
    # Dùng delete_folder("silver") thay vì xóa từng file — đảm bảo xóa sạch
    # kể cả partition sub-folders (popularity_group=HEAD/, popularity_group=TAIL/...)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n🗑  Đang dọn dẹp thư mục 'silver/' cũ trên Hugging Face...")
    try:
        api.delete_folder(
            path_in_repo="silver",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print("  ✅ Đã xóa sạch thư mục silver/ cũ.")
    except Exception:
        print("  ℹ️  Thư mục silver/ chưa tồn tại (lần upload đầu tiên).")

    # 3. Quét các file Parquet
    print(f"🔍 Đang quét danh sách file trong tầng Silver (s3://{args.bucket}/silver)...")
    all_files = fs.find(f"{args.bucket}/silver")
    
    # Lọc file hợp lệ, bỏ qua _SUCCESS, _crc checksums
    valid_files = [f for f in all_files if f.endswith(".parquet")]

    if not valid_files:
        print("❌ Không có file Parquet nào ở tầng Silver để upload!")
        sys.exit(1)

    # Tính kích thước
    total_size_mb = sum([fs.info(f)["size"] for f in valid_files]) / (1024 * 1024)
    print(f"📋 Tìm thấy {len(valid_files)} files (Tổng: {total_size_mb:.1f} MB)")

    # 4. Upload theo lô (Batching)
    operations = []
    commit_count = 1
    batch_size = 10  # Đẩy tối đa 10 file/batch để tránh quá tải RAM & network

    for i, file_path in enumerate(valid_files, 1):
        hf_path = file_path.replace(f"{args.bucket}/", "", 1)
        size_mb = fs.info(file_path)["size"] / (1024 * 1024)
        print(f"   [{i}/{len(valid_files)}] Đọc vào bộ nhớ: {hf_path} ({size_mb:.1f} MB)")
        
        # Đọc dữ liệu
        with fs.open(file_path, "rb") as f:
            file_data = f.read()
            
        operations.append(
            CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=file_data)
        )
        
        # Thực hiện commit nếu đủ số lượng
        if len(operations) >= batch_size:
            print(f"⏳ Đang đẩy lô {commit_count} ({len(operations)} files) lên Hugging Face...")
            api.create_commit(
                repo_id=args.repo_id,
                operations=operations,
                commit_message=f"Upload Silver Data - Batch {commit_count}",
                repo_type="dataset"
            )
            operations = []  # Reset lô
            commit_count += 1
            
    # Push nốt số file còn lại chưa đủ 1 batch
    if operations:
        print(f"⏳ Đang đẩy lô cuối cùng ({len(operations)} files)...")
        api.create_commit(
            repo_id=args.repo_id,
            operations=operations,
            commit_message="Upload Silver Data - Final Batch",
            repo_type="dataset"
        )
        
    print("\n" + "=" * 60)
    print(f"🎉 THÀNH CÔNG! Đã đóng gói {total_size_mb:.1f} MB tầng Silver lên {args.repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    main()
