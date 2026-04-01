# RecSys Pipeline v3 — PySpark Cluster + MinIO

Pipeline xử lý Amazon Reviews 2023 (Electronics) → Feature Store cho recommender system.  
Chạy trên **Spark standalone cluster** (1 master + 2 workers) qua Docker Compose.

---

## Kiến trúc tổng thể

```
HuggingFace (stream)
        │
        ▼  [driver-only, single-threaded]
  [Bước 1+2] Ingestion → Bronze  s3a://recsys/bronze/
        │
        ▼  [distributed trên cluster]
  [Bước 3] Silver Cleaning + Signal Scoring   s3a://recsys/silver/interactions/
  [Bước 4] Labeling                            s3a://recsys/silver/labeled_interactions/
  [Bước 5] Temporal Split                      s3a://recsys/splits/
  [Bước 6] Feature Store                       s3a://recsys/feature_store/
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

## Reliability Grid Search

6 configs cho `reliability_score = w_verified × verified_score + w_text × text_quality + w_helpful × helpful_score`:

| config_id | w_verified | w_text | w_helpful | unverified |
|-----------|-----------|--------|-----------|------------|
| baseline  | 0.50      | 0.30   | 0.20      | 0.50       |
| cfg_2     | 0.60      | 0.25   | 0.15      | 0.50       |
| cfg_3     | 0.40      | 0.40   | 0.20      | 0.50       |
| cfg_4     | 0.50      | 0.20   | 0.30      | 0.50       |
| cfg_5     | 0.55      | 0.25   | 0.20      | 0.40       |
| cfg_6     | 0.45      | 0.35   | 0.20      | 0.60       |

Kết quả grid search: `data/logs/grid_search_summary_<ts>.json` + `.csv`  
`downstream_eval_status = "pending_step7_10"` — HR@10/NDCG@10 tính ở bước 7–10.

---

## Temporal Split

Mode **auto**: T = max month trong data  
- train ≤ T-3 | val = T-2..T-1 | test = T

Mode **explicit**: khai báo trong `config/config.yaml`:
```yaml
temporal_split:
  mode: explicit
  train_months: ["2022-01", "2022-02"]
  val_months:   ["2022-03"]
  test_months:  ["2022-04"]
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

---

## Files

```
src/
├── pipeline_runner.py         # CLI entry point (spark-submit target)
├── step1_ingestion.py         # Stream HuggingFace (driver-only)
├── step2_bronze_storage.py    # Ghi + QC Bronze
├── step3_silver_cleaning.py   # Silver + signal scores (distributed)
├── step4_labeling.py          # BPR/ranking labels (distributed)
├── step5_temporal_split.py    # Temporal split (distributed)
├── step6_feature_store.py     # Feature store (distributed)
└── grid_search_reliability.py # Grid search infra

scripts/
└── run_pipeline.sh            # spark-submit wrapper

config/
└── config.yaml

docker-compose.yml
Dockerfile
.env
```
