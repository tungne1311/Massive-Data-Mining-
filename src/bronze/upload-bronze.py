"""
upload-bronze.py — Đẩy Bronze Parquet files lên Hugging Face Hub

Cách dùng:
  docker compose run --rm -e HF_TOKEN=hf_xxx pipeline python src/bronze/upload-bronze.py

Output repo mặc định: chuongdo1104/amazon-2023-bronze

Files được upload:
  bronze/bronze_meta.parquet
  bronze/bronze_train.parquet  (nhiều part file)
  bronze/bronze_val.parquet
  bronze/bronze_test.parquet
"""

import os
import sys

import s3fs
from huggingface_hub import HfApi, CommitOperationAdd


def main():
    TOKEN   = os.getenv("HF_TOKEN")
    REPO_ID = os.getenv("HF_BRONZE_REPO", "chuongdo1104/amazon-2023-bronze")
    BUCKET  = os.getenv("MINIO_BUCKET", "recsys")

    if not TOKEN:
        print("❌ HF_TOKEN chưa được set! Export HF_TOKEN=hf_xxx trước khi chạy.")
        sys.exit(1)

    print("=" * 60)
    print(f"🚀 Upload Bronze → HuggingFace: {REPO_ID}")
    print("=" * 60)

    # 1. Kết nối MinIO
    fs = s3fs.S3FileSystem(
        key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        client_kwargs={"endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")},
    )

    api = HfApi(token=TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # BƯỚC 1: XÓA THƯ MỤC BRONZE CŨ TRÊN HUGGING FACE
    # ──────────────────────────────────────────────────────────────────────────
    print("\n🗑  Đang dọn dẹp thư mục 'bronze/' cũ trên Hugging Face...")
    try:
        api.delete_folder(
            path_in_repo="bronze",
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print("  ✅ Đã xóa sạch thư mục bronze/ cũ.")
    except Exception:
        print("  ℹ️  Thư mục bronze/ chưa tồn tại (lần upload đầu tiên).")

    # ──────────────────────────────────────────────────────────────────────────
    # BƯỚC 2: QUÉT FILE PARQUET TRÊN MINIO
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n🔍 Đang quét file Parquet trong s3://{BUCKET}/bronze ...")
    all_files   = fs.find(f"{BUCKET}/bronze")
    valid_files = [f for f in all_files if f.endswith(".parquet")]

    if not valid_files:
        print("❌ Không có file Parquet nào trong Bronze để upload!")
        sys.exit(1)

    total_size_mb = sum(fs.info(f)["size"] for f in valid_files) / (1024 * 1024)
    print(f"📋 Tìm thấy {len(valid_files)} files (Tổng: {total_size_mb:.1f} MB)")
    for f in valid_files:
        sz = fs.info(f)["size"] / (1024 * 1024)
        print(f"   • {f.replace(BUCKET+'/', '')} ({sz:.1f} MB)")

    # ──────────────────────────────────────────────────────────────────────────
    # BƯỚC 3: UPLOAD THEO LÔ (10 files/batch)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n⏳ Bắt đầu upload {len(valid_files)} files...")
    BATCH_SIZE  = 10
    operations  = []
    commit_count = 1

    for i, file_path in enumerate(valid_files, 1):
        hf_path  = file_path.replace(f"{BUCKET}/", "", 1)
        size_mb  = fs.info(file_path)["size"] / (1024 * 1024)
        print(f"  [{i:>3}/{len(valid_files)}] Đọc: {hf_path} ({size_mb:.1f} MB)")

        with fs.open(file_path, "rb") as f:
            file_data = f.read()

        operations.append(
            CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=file_data)
        )

        if len(operations) >= BATCH_SIZE:
            print(f"  ⬆ Commit lô {commit_count} ({len(operations)} files)...")
            api.create_commit(
                repo_id=REPO_ID,
                operations=operations,
                commit_message=f"Upload Bronze Data — Batch {commit_count}",
                repo_type="dataset",
            )
            operations  = []
            commit_count += 1

    # Commit nốt phần còn lại
    if operations:
        print(f"  ⬆ Commit lô cuối ({len(operations)} files)...")
        api.create_commit(
            repo_id=REPO_ID,
            operations=operations,
            commit_message="Upload Bronze Data — Final Batch",
            repo_type="dataset",
        )

    print("\n" + "=" * 60)
    print(f"🎉 THÀNH CÔNG! Đã upload {len(valid_files)} files")
    print(f"   Repo: https://huggingface.co/datasets/{REPO_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()