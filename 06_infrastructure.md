# Hạ Tầng và Cấu Hình Hệ Thống

## Tổng Quan Hạ Tầng Docker

TA-RecMind chạy hoàn toàn trong Docker Compose với bốn service chính, được tối ưu cho máy **16 GB RAM**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Network: recsys_net                   │
│                                                                 │
│  ┌────────────────┐  ┌───────────────┐  ┌───────────────────┐  │
│  │     MinIO      │  │ Spark Master  │  │  Spark Worker 1   │  │
│  │  Port 9000     │  │  Port 18080   │  │   Port 18081      │  │
│  │  Port 9001     │  │  Port 17077   │  │                   │  │
│  │  RAM: ~1.5 GB  │  │  RAM: ~1 GB   │  │  RAM: ~8 GB       │  │
│  └────────────────┘  └───────────────┘  └───────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Pipeline Driver                       │    │
│  │  (docker compose run --rm pipeline python ...)          │    │
│  │  RAM: ~5 GB  |  Kết nối: spark://spark-master:7077      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**Phân bổ RAM (16 GB):**

| Service | RAM Limit | Ghi Chú |
|---|---|---|
| MinIO | 1.5 GB | S3-compatible object storage |
| Spark Master | 1.0 GB | Chỉ điều phối, không xử lý data |
| Spark Worker 1 | 8.0 GB | 6 GB JVM + 2 GB overhead/off-heap |
| Pipeline Driver | 5.0 GB | 3 GB driver memory + 2 GB overhead |
| OS + Buffer | ~0.5 GB | Hệ điều hành |
| **Tổng** | **~16 GB** | |

---

## Cấu Hình Chi Tiết Từng Service

### MinIO (Object Storage)

MinIO đóng vai trò S3-compatible object storage — lưu trữ toàn bộ Parquet files qua Bronze, Silver, Gold.

**Cổng:**
- `9000`: S3 API endpoint (Spark dùng `s3a://`)
- `9001`: Web UI → http://localhost:9001

**Bucket Structure:**
```
recsys/
├── landing/    (tạm thời trong Bronze ingestion — xóa sau khi xong)
├── bronze/     (dữ liệu thô đã xử lý)
│   ├── bronze_train.parquet/
│   ├── bronze_val.parquet/
│   ├── bronze_test.parquet/
│   └── bronze_meta.parquet
├── silver/     (dữ liệu đã làm giàu)
│   ├── silver_item_popularity.parquet/
│   ├── silver_item_text_profile.parquet/
│   ├── silver_user_text_profile.parquet/
│   ├── silver_interactions_train.parquet/
│   ├── silver_interactions_val.parquet/
│   └── silver_val_ground_truth.parquet/
└── gold/       (sẵn sàng cho model)
    ├── gold_item_id_map.parquet
    ├── gold_user_id_map.parquet
    ├── gold_edge_index.npy
    ├── gold_item_train_freq.npy
    ├── gold_item_popularity_group.npy
    ├── gold_user_train_freq.npy
    ├── gold_user_activity_group.npy
    ├── gold_negative_sampling_prob.npy
    └── gold_dataset_stats.json
```

**Credentials mặc định:** `minioadmin / minioadmin` (chỉnh trong `.env`)

### Spark Master

Chỉ làm nhiệm vụ điều phối jobs, không xử lý dữ liệu. Cấp 1 GB RAM là đủ.

**UI:** http://localhost:18080 — xem trạng thái jobs, DAG visualization, executor logs, stage details.

### Spark Worker 1

**Cấu hình tối ưu cho 16 GB RAM:**
```yaml
SPARK_WORKER_MEMORY: "6g"   # Bộ nhớ JVM thực tế
SPARK_WORKER_CORES:  "3"    # Giữ lại 1 core cho MinIO/OS
mem_limit: 8g               # Giới hạn Docker (6g JVM + 2g overhead)
cpus: "3.0"
```

**Tại sao 6g JVM nhưng limit 8g Docker?**

JVM heap (6g) là bộ nhớ Java chính. Ngoài ra Spark còn dùng:
- Off-heap memory cho serialization (~1 GB)
- Overhead cho Python worker processes (~0.5 GB)
- Buffer cho shuffle operations

Docker limit 8g đảm bảo không bị OOM killer khi Spark cần burst memory.

### Pipeline Driver

Driver chạy trong container riêng, kết nối đến Spark cluster qua `spark://spark-master:7077`.

**Phân biệt Driver và Executor:**
- **Driver:** Điều phối DAG, thu thập kết quả cuối cùng, chạy Python code
- **Executor (Worker):** Thực sự xử lý data theo lệnh driver

Driver chỉ cần 3g vì hầu hết xử lý xảy ra trên Worker. Tuy nhiên, `collect()` (về driver) hoặc broadcast join lớn cần thêm memory.

---

## Cấu Hình Spark

### Adaptive Query Execution (AQE)

```yaml
spark.sql.adaptive.enabled:                   "true"
spark.sql.adaptive.coalescePartitions.enabled: "true"
spark.sql.adaptive.skewJoin.enabled:           "true"
```

AQE tự động:
- Điều chỉnh số shuffle partitions dựa trên kích thước dữ liệu thực tế
- Gộp partitions nhỏ sau shuffle để tránh overhead của nhiều task nhỏ
- Xử lý skew join khi một partition lớn bất thường

### Shuffle Partitions

```yaml
spark.sql.shuffle.partitions: "100"
```

**Tính toán:** 1 worker × 3 cores = 3 cores. Rule of thumb: 2-4× số cores. Đặt 100 đảm bảo parallelism cho dataset 1.8M rows mà không tạo quá nhiều overhead.

### Broadcast Threshold

```
spark.sql.autoBroadcastJoinThreshold = 104857600  (100 MB)
```

Bảng nhỏ hơn 100 MB được tự động broadcast. `silver_item_popularity` (~50 MB) sẽ tự động broadcast khi join với bảng lớn hơn — không cần `F.broadcast()` tường minh.

### Parquet Compression

- **Bronze:** `zstd` (qua PyArrow) — tỷ lệ nén tốt hơn, decompression nhanh hơn gzip
- **Silver + Gold:** `snappy` (qua Spark) — CPU overhead gần bằng không, phù hợp nhiều lần đọc lại

---

## Biến Môi Trường (`.env`)

```bash
# MinIO credentials
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Spark resources
SPARK_DRIVER_MEMORY=3g
SPARK_EXECUTOR_MEMORY=6g
SPARK_EXECUTOR_CORES=3

# Worker resources
SPARK_WORKER_MEMORY=6g
SPARK_WORKER_CORES=3

# HuggingFace ingestion
HF_STREAM_BATCH_SIZE=100000
HF_TOKEN=hf_xxxxx     # Token cá nhân, không commit lên git

# Giới hạn records để test (null = toàn bộ)
MAX_REVIEW_RECORDS=null
MAX_METADATA_RECORDS=null
```

---

## Cấu Hình Pipeline (`config/config.yaml`)

### Phần `silver`

```yaml
silver:
  write_mode:       "overwrite"
  bronze_hf_repo:   "chuongdo1104/amazon-2023-bronze"
  silver_hf_repo:   "chuongdo1104/amazon-2023-silver"
  min_year:         2010
  max_year:         2025
  short_review_len: 20
  top_k_reviews:    3        # TOP_K cho user text profile
  encode_chunk:     30000    # Chunk size cho LLM embedding
```

### Phần `gold`

```yaml
gold:
  gold_hf_repo:     "chuongdo1104/amazon-2023-gold"
  embed_dim:        128
  neg_sample_beta:  0.75     # P ∝ freq^{-beta}
  user_active_thr:  5        # INACTIVE < 5 ≤ ACTIVE ≤ 20 < SUPER_ACTIVE
  user_super_thr:   20
```

### Phần `reliability_tuning`

```yaml
reliability_tuning:
  selected_config_id: "baseline"
  configs:
    - config_id:      "baseline"
      w_helpful:      1.0      # w(r) = 1 + log(1 + helpful_vote) — chỉ param này có hiệu lực
      # w_verified và w_text đã bỏ — không dùng verified_purchase
```

---

## Quy Trình Vận Hành

### Khởi Động Hạ Tầng

```bash
# Bước 1: Khởi động MinIO
docker compose up -d minio minio-init

# Bước 2: Đợi MinIO healthy (~10s)
docker compose ps    # Status = "healthy"

# Bước 3: Khởi động Spark cluster
docker compose up -d spark-master spark-worker-1

# Bước 4: Đợi Spark Master healthy (~20s)
# Kiểm tra: http://localhost:18080

# Bước 5: Build image nếu chưa có
docker compose build pipeline
```

### Chạy Pipeline

```bash
# Bronze (ste1.py + ste2.py): ~2-4 giờ với full dataset
docker compose run --rm pipeline python src/pipeline_runner.py --step 1_2

# Silver (ste3_silver.py): ~1-2 giờ
docker compose run --rm pipeline python src/pipeline_runner.py --step 3_silver

# Gold (ste4_gold.py): ~30 phút
docker compose run --rm pipeline python src/pipeline_runner.py --step 4_gold

# Toàn bộ pipeline:
docker compose run --rm pipeline python src/pipeline_runner.py --all
```

### Push Lên HuggingFace

```bash
# Bronze
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline \
    python src/bronze/upload-bronze.py

# Silver (xóa cũ + batching 10 file/lô)
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline \
    python src/silver/upload_silver_to_hf.py

# Gold partial (chỉ metadata + edge, không embedding — nhanh)
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline \
    python src/gold/upload_gold_to_hf.py --mode partial

# Gold full (bao gồm embedding arrays)
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline \
    python src/gold/upload_gold_to_hf.py --mode full
```

### Test Nhanh (20k records, ~5-10 phút)

```bash
MAX_REVIEW_RECORDS=20000 MAX_METADATA_RECORDS=5000 \
docker compose run --rm pipeline python src/pipeline_runner.py --step 1_2
```

### Monitoring

```bash
# Logs realtime
docker compose logs -f pipeline

# Spark job progress
open http://localhost:18080

# MinIO objects
open http://localhost:9001

# Disk usage
docker system df
docker volume ls
```

### Dọn Dẹp

```bash
# Xóa landing zone tạm (sau Bronze)
docker compose run --rm pipeline python -c "
import boto3; s3=boto3.client('s3', endpoint_url='http://localhost:9000',
aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin')
# xóa s3://recsys/landing/
"

# Xóa toàn bộ data MinIO (cẩn thận!)
docker volume rm recsys_minio_data
```

---

## Troubleshooting

### OOM Trên Spark Worker

**Triệu chứng:** Job fail với `OutOfMemoryError: GC overhead limit exceeded`

| Nguyên Nhân | Giải Pháp |
|---|---|
| Broadcast join với bảng lớn | Giảm `spark.sql.autoBroadcastJoinThreshold`, dùng `F.broadcast()` tường minh cho bảng nhỏ |
| Shuffle partitions quá thấp | Tăng `shuffle_partitions` từ 100 lên 200 — mỗi partition nhỏ hơn |
| Window function trên dataset lớn | Thay bằng `groupBy + collect_list` như Silver Step 3 |
| Cache tích lũy | Gọi `spark.catalog.clearCache()` + `gc.collect()` giữa các bước |
| DataFrame lineage quá dài | Checkpoint ra MinIO sau mỗi 5+ transforms |

### HuggingFace Download Timeout

**Triệu chứng:** Silver steps fail khi download từ HF

**Giải pháp:**
- Tăng `requests` timeout trong `hf_hub_download`
- Files đã download lưu trong `/tmp/hf_bronze_cache` — lần chạy sau không download lại
- Nếu HF không truy cập được: khôi phục Bronze từ HF repo về MinIO trước khi chạy Silver

### Spark Job Bị Hang

**Triệu chứng:** Job chạy > 30 phút không tiến triển

**Kiểm tra:**
- Spark UI tại `http://localhost:18080`: stage nào đang stuck, task nào chậm nhất
- Nếu một task chạy rất lâu → skew data → cần `spark.sql.adaptive.skewJoin.enabled: true`
- Nếu nhiều task pending → executor OOM → tăng executor memory

### MinIO Connection Refused

**Nguyên nhân:** Spark dùng hostname `minio` (Docker internal) nhưng pipeline chạy ngoài Docker network.

**Giải pháp:** Đảm bảo pipeline container và MinIO cùng network `recsys_net`:
```bash
docker network inspect recsys_recsys_net
```

### KerberosAuthException (Lỗi đã gặp trong pipeline.log)

**Nguyên nhân:** Spark Worker chạy trong container không có user Unix hợp lệ.

**Triệu chứng trong log:**
```
KerberosAuthException: invalid null input: name
at UnixPrincipal.<init>(UnixPrincipal.java:71)
```

**Giải pháp:** Set `SPARK_USER=sparkuser` trong Dockerfile hoặc docker-compose.yml:
```yaml
environment:
  - SPARK_USER=sparkuser
```

Hoặc thêm vào entrypoint script:
```bash
export USER=${USER:-sparkuser}
```

### Py4J Reentrant Call Error

**Triệu chứng trong log:**
```
RuntimeError: reentrant call inside <_io.BufferedReader name=4>
```

**Nguyên nhân:** Python signal handler (Ctrl+C) gọi `cancelAllJobs()` trong khi đang ở giữa một Py4J call — race condition.

**Giải pháp:** Không Ctrl+C job đang chạy. Để job timeout tự nhiên hoặc cancel qua Spark UI.
