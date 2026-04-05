import s3fs
from huggingface_hub import HfApi, CommitOperationAdd
import os

def main():
    TOKEN = TOKEN = os.getenv("HF_TOKEN")
    REPO_ID = "chuongdo1104/landing-amazon"
    
    print(f"🚀 Đang kết nối luồng trực tiếp: MinIO -> Hugging Face...")
    
    # 1. Kết nối MinIO
    fs = s3fs.S3FileSystem(
        key="minioadmin",
        secret="minioadmin",
        client_kwargs={'endpoint_url': "http://minio:9000"}
    )

    api = HfApi(token=TOKEN)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # 2. Quét danh sách file trong landing
    print("🔍 Đang quét danh sách file trong thư mục landing...")
    all_files = fs.find("recsys/landing")
    
    operations = []
    for file_path in all_files:
        # Tạo đường dẫn đích trên Hugging Face
        hf_path = file_path.replace("recsys/", "", 1)
        
        # Đọc dữ liệu dưới dạng bytes trực tiếp từ RAM
        with fs.open(file_path, "rb") as f:
            file_data = f.read()
            
        # Thêm vào danh sách các thao tác chờ commit
        operations.append(
            CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=file_data)
        )
        if len(operations) % 50 == 0:
            print(f"  -> Đã chuẩn bị {len(operations)} files...")

    # 3. THỰC HIỆN COMMIT TỔNG (GOM TẤT CẢ TRONG 1 LẦN)
    print(f"⏳ Đang thực hiện ONE-TIME COMMIT cho {len(operations)} files lên {REPO_ID}...")
    try:
        api.create_commit(
            repo_id=REPO_ID,
            operations=operations,
            commit_message="Bơm toàn bộ dữ liệu landing từ MinIO",
            repo_type="dataset"
        )
        print(f"🎉 THÀNH CÔNG! Đã bơm toàn bộ thư mục landing chỉ với 1 commit duy nhất.")
    except Exception as e:
        print(f"❌ Lỗi khi thực hiện commit: {e}")

if __name__ == "__main__":
    os.system("pip install -q huggingface_hub s3fs")
    main()