# Hạ Tầng và Cấu Hình Hệ Thống

## Tổng Quan Hạ Tầng Docker

TA-RecMind chạy hoàn toàn trong Docker Compose với bốn service chính, được tối ưu cho máy 16 GB RAM.

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network: recsys_net               │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │     MinIO     │  │ Spark Master  │  │  Spark Worker 1 │ │
│  │  Port 9000    │  │  Port 18080   │  │   Port 18081    │ │
│  │  Port 9001    │  │  Port 17077   │  │                 │ │
│  │  RAM: ~1.5 GB │  │  RAM: ~1 GB   │  │  RAM: ~8 GB     │ │
│  └───────────────┘  └───────────────┘  └─────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                Pipeline Driver                        │  │
│  │  (docker compose run --rm pipeline)                   │  │
│  │  RAM: ~5 GB                                           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Phân bổ RAM tổng (16 GB):**

| Service | RAM Limit | Ghi Chú |
|---|---|---|
| MinIO | 1.5 GB | Object storage |
| Spark Master | 1.0 GB | Chỉ điều phối, không xử lý data |
| Spark Worker 1 | 8.0 GB | 6 GB JVM + 2 GB overhead/off-heap |
| Pipeline Driver | 5.0 GB | 3 GB driver memory + 2 GB overhead |
| OS + Buffer | ~0.5 GB | Hệ điều hành |
| **Tổng** | **16 GB** | |

---

## Cấu Hình Chi Tiết Từng Service

### MinIO (Object Storage)

MinIO đóng vai trò S3-compatible object storage, lưu trữ toàn bộ Parquet files qua các tầng Bronze, Silver, Gold.

**Cổng:**
- `9000`: S3 API endpoint (Spark dùng `s3a://`)
- `9001`: Web UI (http://localhost:9001)

**Bucket structure:**
```
recsys/
├── landing/    (tạm thời trong Bronze ingestion)
├── bronze/     (dữ liệu thô đã xử lý)
├── silver/     (dữ liệu đã làm giàu)
├── gold/       (sẵn sàng cho model)
└── splits/     (train/val/test splits)
```

**Credentials mặc định:** `minioadmin / minioadmin` (chỉnh trong `.env`)

### Spark Master

Chỉ làm nhiệm vụ điều phối jobs, không xử lý dữ liệu. Cấp 1 GB RAM là đủ.

**UI:** http://localhost:18080 — xem trạng thái jobs, DAG visualization, executor logs.

### Spark Worker 1

**Cấu hình tối ưu cho 16 GB RAM:**
```
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
- Driver: điều phối DAG, thu thập kết quả cuối cùng
- Executor (trên Worker): thực sự xử lý data

Driver chỉ cần 3g memory vì hầu hết xử lý xảy ra trên Worker. Tuy nhiên, các thao tác như `collect()` (về driver) hoặc broadcast join lớn cần thêm memory trên driver.

---

## Cấu Hình Spark

### Adaptive Query Execution (AQE)

```yaml
aqe_enabled:             "true"
aqe_coalesce_partitions: "true"
aqe_skew_join:           "true"
```

AQE tự động:
- Điều chỉnh số shuffle partitions dựa trên kích thước dữ liệu thực tế
- Gộp partitions nhỏ sau shuffle (coalesce) để tránh overhead của nhiều task nhỏ
- Xử lý skew join tự động khi một partition lớn bất thường

### Shuffle Partitions

```yaml
shuffle_partitions: "100"
```

**Tính toán:** Cluster có 1 worker × 3 cores = 3 cores. Rule of thumb: 2-4 lần số cores. Đặt 100 đảm bảo đủ parallelism cho dataset lớn (1.8M rows) mà không tạo quá nhiều overhead từ nhiều task nhỏ.

### Broadcast Threshold

```
spark.sql.autoBroadcastJoinThreshold = 104857600  (100 MB)
```

Bảng nhỏ hơn 100 MB được tự động broadcast. Bảng `silver_item_popularity` (~50 MB) sẽ tự động được broadcast khi join với bảng lớn hơn.

### Parquet Compression

Bronze dùng `zstd` (qua PyArrow) vì tỷ lệ nén tốt hơn và decompression nhanh hơn gzip.
Silver và Gold dùng `snappy` (qua Spark) — snappy có tỷ lệ nén thấp hơn nhưng CPU overhead gần bằng không, phù hợp cho nhiều lần đọc lại.

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
HF_TOKEN=hf_xxxxx    # Token cá nhân, không commit lên git

# Giới hạn records để test (null = toàn bộ)
MAX_REVIEW_RECORDS=null
MAX_METADATA_RECORDS=null
```

---

## Cấu Hình Pipeline (`config/config.yaml`)

### Phần `silver` (Cần Bổ Sung)

```yaml
silver:
  write_mode:      "overwrite"
  bronze_hf_repo:  "chuongdo1104/amazon-2023-bronze"
  min_year:        2010
  max_year:        2025
  short_review_len: 20
```

### Phần `reliability_tuning`

Cho phép thử nghiệm nhiều bộ hyperparameter trọng số review:

```yaml
reliability_tuning:
  selected_config_id: "baseline"
  configs:
    - config_id: "baseline"
      w_verified:       0.50    # (bỏ qua, không dùng verified_purchase)
      w_text:           0.30
      w_helpful:        0.20
      unverified_value: 0.50
```

Trong TA-RecMind (đã bỏ `verified_purchase`), chỉ `w_helpful` có hiệu lực, ảnh hưởng đến tính toán review weight.

---

## Quy Trình Vận Hành

### Khởi Động Hạ Tầng

```bash
# Bước 1: Khởi động services cơ bản
docker compose up -d minio minio-init

# Bước 2: Đợi MinIO healthy (~10s)
docker compose ps  # Kiểm tra status = "healthy"

# Bước 3: Khởi động Spark cluster
docker compose up -d spark-master spark-worker-1

# Bước 4: Đợi Spark Master healthy (~20s)
# Kiểm tra http://localhost:18080

# Bước 5: Build image nếu chưa có
docker compose build pipeline
```

### Chạy Pipeline

```bash
# Bronze (Bước 1+2): ~2-4 giờ với full dataset
docker compose run --rm pipeline --step 1_2

# Silver (Bước 3): ~1-2 giờ
docker compose run --rm pipeline --step 3_silver

# Toàn bộ (Bronze + Silver + Gold + Model):
docker compose run --rm pipeline --all --config-id baseline
```

### Test Nhanh (20k records)

```bash
docker compose run --rm pipeline-test --step 1_2
# → Chạy trong ~5-10 phút
```

### Monitoring

```bash
# Xem logs realtime
docker compose logs -f pipeline

# Xem Spark job progress
open http://localhost:18080

# Xem MinIO objects
open http://localhost:9001

# Kiểm tra dung lượng disk
docker system df
docker volume ls
```

### Dọn Dẹp

```bash
# Xóa landing zone tạm (sau Bronze)
docker compose run --rm pipeline --cleanup-landing

# Xóa toàn bộ data (cẩn thận!)
docker volume rm recsys_minio_data
```

---

## Troubleshooting

### OOM Trên Spark Worker

**Triệu chứng:** Job fail với `OutOfMemoryError: GC overhead limit exceeded`

**Nguyên nhân và giải pháp:**

1. **Broadcast join với bảng lớn:** Kiểm tra `spark.sql.autoBroadcastJoinThreshold`, giảm xuống nếu cần. Thêm `F.broadcast()` tường minh cho bảng nhỏ.

2. **Shuffle partitions quá thấp:** Tăng `shuffle_partitions` từ 100 lên 200. Mỗi partition lớn hơn → mỗi task cần nhiều memory hơn.

3. **Window function trên dataset lớn:** Thay bằng `groupBy + collect_list` như đã thiết kế trong Silver Step 3.

4. **Cache tích lũy:** Gọi `spark.catalog.clearCache()` và `gc.collect()` giữa các bước.

### HuggingFace Download Timeout

**Triệu chứng:** Silver steps fail khi download từ HF

**Giải pháp:**
- Tăng `requests` timeout trong `hf_hub_download`
- Files đã download lưu trong `/tmp/hf_bronze_cache` — lần chạy sau không download lại
- Nếu HF không truy cập được: khôi phục Bronze từ HF repo về MinIO trước khi chạy Silver

### Spark Job Bị Hang

**Triệu chứng:** Job chạy > 30 phút không tiến triển

**Kiểm tra:**
- Spark UI: stage nào đang stuck, task nào chậm nhất
- Nếu một task chạy rất lâu → skew data, cần `aqe_skew_join: true`
- Nếu nhiều task pending → executor OOM, tăng executor memory

### MinIO Connection Refused

**Nguyên nhân:** Spark dùng hostname `minio` (Docker internal), nhưng pipeline chạy ngoài Docker network.

**Giải pháp:** Đảm bảo pipeline container và MinIO cùng network `recsys_net`. Kiểm tra với:
```bash
docker network inspect recsys_recsys_net
```
