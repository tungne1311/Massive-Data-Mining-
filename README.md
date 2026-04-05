# RecSys Data Engineering Pipeline v3 — PySpark Cluster + MinIO

Dự án này tập trung vào Data Engineering (DE) pipeline, xử lý bộ dữ liệu khổng lồ **Amazon Reviews 2023 (Electronics)** và chuyển đổi nó thành **Feature Store** chuẩn bị cho các mô hình Recommender System (RecSys).

Pipeline chạy trên **Spark standalone cluster** (1 master + 2 workers) thông qua Docker Compose, kết hợp **MinIO** làm storage (S3-compatible) đảm bảo xử lý phân tán với hiệu năng cao.

---

## Kiến trúc Data Engineering Pipeline (Từ 1 đến 6)

Pipeline gồm 6 bước chính (từ Ingestion đến Feature Store), tuân thủ kiến trúc Medallion (Bronze - Silver - Feature Store).

```text
[HuggingFace Stream] 
        │
        ▼  [Single-threaded, Pandas/PyArrow]
  [Bước 1] Ingestion: Stream dữ liệu raw từ HF
  [Bước 2] Bronze Storage: Đẩy dữ liệu thô (chunked Parquet zstd) → s3a://recsys/landing/
        │
        ▼  [Distributed, PySpark Cluster]
  [Bước 3] Silver Cleaning & Computing: Làm sạch, Join Broadcast, Đánh giá độ tin cậy (Reliability Scoring) → s3a://recsys/silver/
        │
        ▼  [Distributed, PySpark Cluster]
  [Bước 4] Labeling: Gán nhãn Interaction, BPR Weights → s3a://recsys/silver/labeled_interactions/
        │
        ▼  [Distributed, PySpark Cluster]
  [Bước 5] Temporal Split: Chia Train/Val/Test theo trục thời gian (Time-based split) → s3a://recsys/splits/
        │
        ▼  [Distributed, PySpark Cluster]
  [Bước 6] Feature Store: Map-Reduce tạo User & Item Features (Wilson Score, Top K) → s3a://recsys/feature_store/
```

### Spark Cluster (Docker Compose)

```
spark-master (recsys_spark_master)  ← Spark UI: http://localhost:8080
    ├── spark-worker-1               ← Worker UI: http://localhost:8081
    └── spark-worker-2               ← Worker UI: http://localhost:8082

pipeline-driver                      ← spark-submit client mode
minio (recsys_minio)                 ← MinIO UI: http://localhost:9001
```

---

## Cài đặt

```bash
# 1. Clone / extract project
cd recsys_pipeline_minio

# 2. Build image (lần đầu — tải S3A JARs ~70MB)
docker compose build

# 3. Khởi động hạ tầng
docker compose up -d minio minio-init spark-master spark-worker-1 spark-worker-2

# 4. Kiểm tra cluster đã ready
curl http://localhost:8080   # Spark Master UI
# → Phải thấy 2 workers "ALIVE"
```

---

## Cách chạy

### A. Chạy bước 3–6 (Bronze đã có sẵn)

```bash
docker compose run --rm pipeline --step 3_6 --config-id baseline
```

### B. Chạy toàn bộ pipeline (1→6)

```bash
docker compose run --rm pipeline --step all --config-id baseline
```

### C. Chạy Grid Search (6 configs × bước 3→6)

```bash
docker compose run --rm pipeline --run-grid-search
```

### D. Chạy bước đơn lẻ

```bash
docker compose run --rm pipeline --step 3 --config-id baseline
docker compose run --rm pipeline --step 4 --config-id baseline
docker compose run --rm pipeline --step 5 --config-id baseline
docker compose run --rm pipeline --step 6 --config-id baseline
```

### E. Test nhanh (20k records)

```bash
docker compose --profile test run --rm pipeline-test --step 1_2
```

---

## Kiểm tra Cluster

| Mục kiểm tra | Cách kiểm tra |
|---|---|
| Spark Master UI | `http://localhost:8080` → Status: ALIVE |
| Workers đã join chưa | Master UI → Workers: 2 workers ALIVE |
| Job đang chạy | Master UI → Running Applications |
| Job distributed | Application UI → Stages → Tasks (nhiều executor) |
| Output vào MinIO | `http://localhost:9001` → Browse recsys bucket |

### Xem logs realtime

```bash
# Logs spark-master
docker compose logs -f spark-master

# Logs worker-1
docker compose logs -f spark-worker-1

# Logs pipeline job
docker compose logs -f   # hoặc xem output của docker compose run
```

---

## Cấu trúc MinIO Output

```
s3a://recsys/
├── bronze/
│   ├── reviews/year_month=YYYY-MM/
│   └── metadata/year_month=YYYY-MM/
├── silver/
│   ├── interactions/config_id=<id>/year_month=YYYY-MM/
│   ├── labeled_interactions/config_id=<id>/year_month=YYYY-MM/
│   └── logs/
│       ├── silver_summary_<config_id>.json
│       └── grid_search_summary_<ts>.json
├── splits/
│   └── config_id=<id>/
│       ├── train/
│       ├── val/
│       └── test/
└── feature_store/
    └── config_id=<id>/train/
        ├── user_features/
        ├── item_features/
        └── user_item_features/
```

---

## Tùy chỉnh tài nguyên

Trong `.env`:
```bash
SPARK_WORKER_MEMORY=6g    # RAM mỗi worker
SPARK_WORKER_CORES=4      # CPU cores mỗi worker
SPARK_EXECUTOR_MEMORY=4g  # RAM mỗi executor
SPARK_EXECUTOR_CORES=2    # CPU cores mỗi executor
```

Trong `config/config.yaml`:
```yaml
spark:
  shuffle_partitions: "200"  # Tăng nếu data lớn
  write_partitions: 0        # 0 = để AQE tự quyết; >0 = repartition về N files
```
