# TA-RecMindV2: Long-Tail Recommendation Pipeline

TA-RecMindV2 là hệ thống gợi ý sản phẩm cho Amazon Reviews 2023, subset Electronics. Dự án gồm hai phần chính:

- Data pipeline theo kiến trúc Medallion: Bronze -> Silver -> Gold.
- Notebook training trên Colab: hybrid recommender kết hợp LightGCN, text embeddings từ SentenceTransformer và degree-aware intra-layer gate.

Mục tiêu của hệ thống là **warm long-tail recommendation**: cải thiện khả năng xếp hạng các item `TAIL` có ít tương tác trong train nhưng vẫn có graph edge. Cold-start item được giữ trong metadata, item text profile và evaluation ground truth. Protocol training/evaluation chính dùng warm candidate set (`train_freq > 0`) với `IGNORE_COLD_ITEMS = True`.

## Mục Lục

- [Bài Toán](#bài-toán)
- [Nguồn Dữ Liệu](#nguồn-dữ-liệu)
- [Kiến Trúc Data](#kiến-trúc-data)
  - [Bronze](#bronze)
  - [Silver](#silver)
  - [Gold](#gold)
- [Kiến Trúc Mô Hình](#kiến-trúc-mô-hình)
  - [Degree-Aware Intra-Layer Gate](#degree-aware-intra-layer-gate)
  - [Loss](#loss)
  - [Sampling](#sampling)
  - [Evaluation](#evaluation)
- [Orchestration](#orchestration)
- [Hướng Dẫn Chạy Pipeline](#hướng-dẫn-chạy-pipeline)
  - [1. Khởi động hạ tầng](#1-khởi-động-hạ-tầng)
  - [2. Chạy từng tầng](#2-chạy-từng-tầng)
  - [3. Chạy nhanh với sample nhỏ](#3-chạy-nhanh-với-sample-nhỏ)
  - [4. Upload artifacts lên Hugging Face](#4-upload-artifacts-lên-hugging-face)
- [Hướng Dẫn Chạy Training](#hướng-dẫn-chạy-training)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Tài Liệu Chi Tiết](#tài-liệu-chi-tiết)
- [Ghi Chú Kỹ Thuật Quan Trọng](#ghi-chú-kỹ-thuật-quan-trọng)

---

## Bài Toán

Dữ liệu Amazon Reviews có phân phối power-law: một nhóm nhỏ item head nhận phần lớn tương tác, trong khi phần lớn catalog là mid/tail item có rất ít signal cộng tác. Nếu chỉ dùng Matrix Factorization hoặc LightGCN thuần collaborative filtering, model dễ học thiên lệch popularity và bỏ qua tail item.

TA-RecMindV2 xử lý bài toán này bằng ba hướng:

- Dùng `train_freq` để phân nhóm item `HEAD`, `MID`, `TAIL`, tránh leakage từ metadata như `rating_number`.
- Đưa text profile của user/item vào từng layer propagation qua degree-aware gate: item/user ít tương tác dựa vào text nhiều hơn, item/user giàu tương tác dựa vào graph nhiều hơn.
- Huấn luyện và checkpoint theo metric long-tail: tail positive oversampling, warm-only negative sampling, stratified validation và weighted harmonic mean giữa Tail NDCG và Overall NDCG.

---

## Nguồn Dữ Liệu

Nguồn raw:

```text
McAuley-Lab/Amazon-Reviews-2023
├── raw_review_Electronics
└── raw_meta_Electronics
```

Repos Hugging Face sau khi upload artifact:

```text
chuongdo1104/amazon-2023-bronze
chuongdo1104/amazon-2023-silver
chuongdo1104/amazon-2023-gold
```

Các ID chính:

| Khái niệm | Cột |
|---|---|
| User | `reviewer_id` |
| Item | `parent_asin` |
| Implicit interaction | một review hợp lệ, không lọc theo rating |
| Time split | `timestamp` theo từng user |

Bronze giữ mọi rating hợp lệ từ 1 đến 5 như implicit feedback. Rating chỉ được dùng như feature/metadata và dùng để chọn positive/negative review snippets trong Silver text profile, không dùng để loại bỏ interaction.

---

## Kiến Trúc Data

Luồng tổng thể:

```text
Hugging Face raw dataset
        |
        v
Bronze: normalize, dedup, validation, user Core-5, chronological split
        |
        v
Silver: popularity labels, item/user text profiles, enriched interactions, ground truth
        |
        v
Gold: integer ID maps, train edge_index, metadata arrays, negative sampling probability
        |
        v
Colab notebook: encode text, build graph, train/evaluate TA-RecMindV2
```

Storage runtime:

- MinIO/S3 local bucket: `recsys`
- Spark đọc/ghi bằng `s3a://`
- PyArrow/S3FS dùng cho một số artifact single-file và upload
- Hugging Face Hub dùng để backup và để Colab tải artifact trực tiếp

### Bronze

Code:

```text
src/bronze/ste1.py
src/bronze/ste2.py
```

Logic chính:

- Stream raw review/meta từ Hugging Face bằng `datasets`.
- Normalize schema review và metadata bằng PyArrow.
- Review landing zone: `s3://recsys/landing/reviews_temp/`.
- Dedup interaction theo `reviewer_id, parent_asin`.
- Validation: ID không rỗng, timestamp > 0, rating trong `[1, 5]`.
- User Core-5 đang được áp hard-code trong `ste2.py`: chỉ giữ user có ít nhất 5 interactions trước split.
- Chronological split theo từng user:
  - test = interaction có timestamp lớn nhất
  - val = interaction lớn nhất còn lại
  - train = phần còn lại

Bronze artifacts:

```text
s3a://recsys/bronze/
├── bronze_meta.parquet
├── bronze_train.parquet/
├── bronze_val.parquet/
└── bronze_test.parquet/
```

Review schema sau normalize:

```text
reviewer_id, parent_asin, rating, review_title, review_text, timestamp, helpful_vote
```

Metadata schema sau normalize:

```text
parent_asin, title, main_category, store, price, average_rating,
rating_number, categories, features, description, details
```

### Silver

Code:

```text
src/silver/ste3_silver.py
src/silver/silver_step1_popularity.py
src/silver/silver_step2_item_profile.py
src/silver/silver_step3_user_profile.py
src/silver/silver_step4_interactions.py
src/silver/silver_utils.py
```

Orchestration trong `ste3_silver.py`:

```text
Step 1: item popularity
Step 2: item text profile
Step 4: enriched interactions
Step 3: user text profile + val/test ground truth
```

Silver artifacts:

```text
s3a://recsys/silver/
├── silver_item_popularity.parquet/
├── silver_item_text_profile.parquet/
├── silver_interactions_train.parquet/
├── silver_interactions_val.parquet/
├── silver_interactions_test.parquet/
├── silver_user_text_profile.parquet/
├── silver_val_ground_truth.parquet/
└── silver_test_ground_truth.parquet/
```

Popularity:

- Tính `train_freq = count(*)` theo `parent_asin` chỉ từ train.
- Phân nhóm chính: `HEAD`, `MID`, `TAIL`.
- Item có metadata/eval nhưng không có train edge được gán `COLD_START`.
- Không dùng `average_rating` hoặc `rating_number` để phân nhóm popularity.

Text profile:

- `item_text` được tạo từ metadata field tags: title, category, store, features, description, details.
- Có thể thêm positive/negative review snippets từ train-only reviews.
- `user_text` được tạo từ train-only review snippets của user.
- Val/test review text không được dùng để tạo profile, tránh leakage.

Interaction enriched schema:

```text
parent_asin, reviewer_id, rating, timestamp, helpful_vote,
train_freq, popularity_group, year_month
```

Ground truth schema:

```text
parent_asin, reviewer_id, timestamp, rating,
popularity_group, train_freq, is_tail, is_cold_start
```

### Gold

Code:

```text
src/gold/ste4_gold.py
src/gold/gold_step1_id_mapping.py
src/gold/gold_step2_edge_list.py
src/gold/gold_step5_training_meta.py
```

Orchestration trong `ste4_gold.py`:

```text
Step 1: integer ID mapping
Step 2: train edge_index.npy
Step 5: training metadata arrays + negative sampling probability
```

Gold artifacts:

```text
s3a://recsys/gold/
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

`gold_edge_index.npy`:

```text
shape = [2, n_edges]
row 0 = user_idx
row 1 = item_idx
dtype = int64
```

Item group encoding:

```text
HEAD       -> 0
MID        -> 1
TAIL       -> 2
COLD_START -> 3
```

User activity encoding:

```text
INACTIVE     -> 0
ACTIVE       -> 1
SUPER_ACTIVE -> 2
```

Negative sampling probability trong Gold chỉ cấp mass cho warm items (`train_freq > 0`). Cold items có probability bằng 0.

## Orchestration

Entry point:

```text
src/pipeline_runner.py
```

CLI:

```bash
docker compose run --rm pipeline python src/pipeline_runner.py --step 1_2
docker compose run --rm pipeline python src/pipeline_runner.py --step 3_silver
docker compose run --rm pipeline python src/pipeline_runner.py --step 4_gold
docker compose run --rm pipeline python src/pipeline_runner.py --all
```

Các step:

| CLI | Tầng | Mô tả |
|---|---|---|
| `--step 1_2` | Bronze | Hugging Face ingestion, landing zone, dedup, validation, Core-5, chronological split |
| `--step 3_silver` | Silver | Popularity, item/user text profile, enriched interactions, ground truth |
| `--step 4_gold` | Gold | ID maps, edge index, metadata arrays, negative sampling probability |
| `--all` | All | Chạy Bronze -> Silver -> Gold, có clear Spark cache giữa các tầng |

Config:

```text
config/config.yaml
```

Một số override bằng biến môi trường:

```text
SPARK_DRIVER_MEMORY
SPARK_EXECUTOR_MEMORY
SPARK_EXECUTOR_CORES
SPARK_MASTER
MAX_REVIEW_RECORDS
MAX_METADATA_RECORDS
HF_STREAM_BATCH_SIZE
BRONZE_BASE_PATH
MINIO_ENDPOINT
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
MINIO_BUCKET
HF_TOKEN
```

---

## Hướng Dẫn Chạy Pipeline

### 1. Khởi động hạ tầng

```bash
docker compose up -d minio minio-init spark-master spark-worker-1
docker compose ps
```

UI:

```text
MinIO:    http://localhost:9001
Spark UI: http://localhost:18080
```

### 2. Chạy từng tầng

```bash
docker compose run --rm pipeline python src/pipeline_runner.py --step 1_2
docker compose run --rm pipeline python src/pipeline_runner.py --step 3_silver
docker compose run --rm pipeline python src/pipeline_runner.py --step 4_gold
```

Hoặc chạy toàn bộ:

```bash
docker compose run --rm pipeline python src/pipeline_runner.py --all
```

Log:

```text
data/logs/pipeline.log
```

### 3. Chạy nhanh với sample nhỏ

```bash
docker compose run --rm ^
  -e MAX_REVIEW_RECORDS=20000 ^
  -e MAX_METADATA_RECORDS=5000 ^
  pipeline python src/pipeline_runner.py --all
```

Trên bash/Linux:

```bash
docker compose run --rm \
  -e MAX_REVIEW_RECORDS=20000 \
  -e MAX_METADATA_RECORDS=5000 \
  pipeline python src/pipeline_runner.py --all
```

### 4. Upload artifacts lên Hugging Face

Bronze:

```bash
docker compose run --rm -e HF_TOKEN=hf_xxx pipeline \
  python src/bronze/upload-bronze.py
```

Silver:

```bash
docker compose run --rm -e HF_TOKEN=hf_xxx pipeline \
  python src/silver/upload_silver_to_hf.py
```

Gold:

```bash
docker compose run --rm -e HF_TOKEN=hf_xxx pipeline \
  python src/gold/upload_gold_to_hf.py --mode partial
```

`--mode partial` là chế độ đúng với pipeline hiện tại vì text embeddings không được tạo ở Gold local; chúng được encode trong notebook Colab và cache trên Drive.

## Cấu Trúc Dự Án

```text
recsys_pipeline_minio/
├── README.md
├── config/
│   └── config.yaml
├── docs/
│   ├── 01_data_pipeline.md
│   ├── 02_model_architecture.md
│   
├── notebooks/
│   ├── TA_REC_đổi_data.ipynb
│   
│   
├── src/
│   ├── pipeline_runner.py
│   ├── bronze/
│   │   ├── ste1.py
│   │   ├── ste2.py
│   │   └── upload-bronze.py
│   ├── silver/
│   │   ├── ste3_silver.py
│   │   ├── silver_step1_popularity.py
│   │   ├── silver_step2_item_profile.py
│   │   ├── silver_step3_user_profile.py
│   │   ├── silver_step4_interactions.py
│   │   ├── silver_utils.py
│   │   └── upload_silver_to_hf.py
│   └── gold/
│       ├── ste4_gold.py
│       ├── gold_step1_id_mapping.py
│       ├── gold_step2_edge_list.py
│       ├── gold_step5_training_meta.py
│       └── upload_gold_to_hf.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Tài Liệu Chi Tiết

- [Data pipeline](docs/01_data_pipeline.md)
- [Model architecture](docs/02_model_architecture.md)
---

## Ghi Chú Kỹ Thuật Quan Trọng

- Bronze áp user Core-5 trong code; các flag config liên quan Core chỉ đóng vai trò khai báo.
- Silver popularity dùng Pareto item-rank từ `train_freq`.
- Gold tạo artifact cấu trúc và metadata; text embeddings được tạo trong notebook Colab để tận dụng GPU và Drive cache.
- Evaluation chính là warm long-tail. Candidate set gồm warm items khi `IGNORE_COLD_ITEMS = True`.
- Không dùng val/test review text để tạo user/item profile.
