import logging
import uuid
import pyarrow.parquet as pq
import s3fs
import ste1 as step1

logger = logging.getLogger(__name__)

def _get_s3_filesystem(cfg: dict) -> s3fs.S3FileSystem:
    """Tạo kết nối s3fs độc lập với Spark"""
    mn = cfg["minio"]
    endpoint = mn["endpoint"].replace("http://", "").replace("https://", "")
    return s3fs.S3FileSystem(
        key=mn["access_key"],
        secret=mn["secret_key"],
        client_kwargs={'endpoint_url': f"http://{endpoint}"}
    )

def _write_pyarrow_to_landing(arrow_table, dataset_type: str, cfg: dict):
    """Ghi trực tiếp Table dưới dạng 1 file Parquet (ZSTD) vào cấu trúc phẳng"""
    fs = _get_s3_filesystem(cfg)
    
    # Cấu trúc phẳng: bucket/landing/dataset_type
    bucket = cfg["minio"]["bucket"].strip("/")
    file_name = f"chunk_{uuid.uuid4().hex}.parquet"
    
    # Đối với s3fs, đường dẫn bắt đầu bằng tên bucket
    full_path = f"{bucket}/landing/{dataset_type}/{file_name}"
    
    pq.write_table(
        arrow_table,
        full_path,
        filesystem=fs,
        compression='zstd'
    )
    logger.info(f"  -> Đã đẩy block {arrow_table.num_rows} rows vào: s3://{full_path}")

def write_review_batch(arrow_table, cfg: dict) -> None:
    _write_pyarrow_to_landing(arrow_table, "reviews", cfg)

def write_metadata_batch(arrow_table, cfg: dict) -> None:
    _write_pyarrow_to_landing(arrow_table, "metadata", cfg)