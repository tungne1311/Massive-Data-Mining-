import s3fs
from huggingface_hub import HfApi, CommitOperationAdd
import os
import time

def main():
    TOKEN = TOKEN = os.getenv("HF_TOKEN")
    REPO_MAPPING = {
        "chuongdo1104/amazon-dataset-silver": ["recsys/silver"],
        "chuongdo1104/amazon-dataset-gold": ["recsys/splits", "recsys/feature_store_baseline"]
    }
    
    BATCH_SIZE = 300 
    COOLDOWN_TIME = 30 

    fs = s3fs.S3FileSystem(
        key="minioadmin",
        secret="minioadmin",
        client_kwargs={'endpoint_url': "http://minio:9000"}
    )
    api = HfApi(token=TOKEN)

    for repo_id, source_dirs in REPO_MAPPING.items():
        print(f"\n🚀 Kiểm tra Repo: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        
        # --- BƯỚC MỚI: Lấy danh sách file đã có trên HF để skip ---
        print("  📂 Đang lấy danh sách file hiện có trên Hugging Face...")
        try:
            existing_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        except:
            existing_files = set()

        operations = []
        total_uploaded_this_run = 0

        for root_dir in source_dirs:
            print(f"  🔍 Quét MinIO: {root_dir}...")
            all_files = fs.find(root_dir)
            
            for file_path in all_files:
                hf_path = file_path.replace("recsys/", "", 1)
                
                # Nếu file đã tồn tại trên HF, bỏ qua không đọc/đẩy lại nữa
                if hf_path in existing_files:
                    continue

                try:
                    with fs.open(file_path, "rb") as f:
                        file_data = f.read()
                    
                    operations.append(
                        CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=file_data)
                    )
                    total_uploaded_this_run += 1

                    if len(operations) >= BATCH_SIZE:
                        execute_commit_with_retry(api, repo_id, operations, total_uploaded_this_run)
                        operations = []
                        time.sleep(COOLDOWN_TIME)

                except Exception as e:
                    print(f"  ❌ Lỗi file {file_path}: {e}")

        if operations:
            execute_commit_with_retry(api, repo_id, operations, total_uploaded_this_run)
            
        print(f"  ✅ Hoàn tất Repo {repo_id}. Đã đẩy thêm {total_uploaded_this_run} file mới.")

def execute_commit_with_retry(api, repo_id, operations, current_total):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"  ⏳ Đang commit {len(operations)} file mới...")
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=f"Bổ sung dữ liệu - đợt {current_total}",
                repo_type="dataset"
            )
            print(f"  ✨ Thành công!")
            return
        except Exception as e:
            if "429" in str(e):
                wait = 180 * (attempt + 1)
                print(f"  ⚠️ Chạm giới hạn. Nghỉ {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Lỗi: {e}")
                break

if __name__ == "__main__":
    main()