# Kiến Trúc Data Pipeline: Bronze -> Silver -> Gold

Tài liệu này là đặc tả triển khai của data pipeline trong ba thư mục:

- `src/bronze/`
- `src/silver/`
- `src/gold/`

Pipeline xử lý Amazon Reviews 2023 subset Electronics theo mô hình Medallion. Mỗi tầng có trách nhiệm rõ ràng: Bronze chuẩn hóa và split dữ liệu, Silver tạo feature/profile phục vụ training và evaluation, Gold chuyển dữ liệu sang dạng model-ready.

---

## Tổng Quan Luồng Dữ Liệu

```text
HuggingFace
  McAuley-Lab/Amazon-Reviews-2023
  - raw_review_Electronics
  - raw_meta_Electronics
        |
        v
Bronze
  ste1.py: streaming ingestion + PyArrow normalization
  ste2.py: landing zone -> dedup -> validation -> user Core-5 -> chronological split
        |
        v
Silver
  Step 1: item train frequency -> HEAD / MID / TAIL
  Step 2: item text profile từ metadata + train-only item review snippets
  Step 4: train/val/test interactions đã join popularity
  Step 3: user text profile + val/test ground truth
        |
        v
Gold
  Step 1: string ID maps -> integer ID maps
  Step 2: train edge_index.npy
  Step 5: metadata arrays + negative sampling probability + dataset stats
```

Nguyên tắc triển khai:

- Bronze giữ mọi rating hợp lệ 1-5 như implicit interactions, không filter theo rating.
- Bronze áp user Core-5 trước khi split.
- Split train/val/test dựa trên timestamp trong từng user.
- Popularity chỉ tính từ `bronze_train`, không dùng `average_rating` hoặc `rating_number` từ metadata.
- Review text của val/test không được dùng để tạo user profile hoặc item review text.
- Nhãn popularity chính trong pipeline là `HEAD`, `MID`, `TAIL`, `COLD_START`.
- Gold chỉ tạo artifact cấu trúc và metadata cho model. Text embeddings được tạo ở notebook/GPU flow.

---

## Artifact Layout

### Bronze

```text
s3a://recsys/bronze/
├── bronze_meta.parquet
├── bronze_train.parquet/
├── bronze_val.parquet/
└── bronze_test.parquet/
```

`bronze_meta.parquet` là file parquet đơn được ghi bằng PyArrow. Các split review là thư mục parquet do Spark ghi ra.

### Silver

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

`silver_item_text_profile.parquet/` được partition theo `popularity_group`, nên khi đọc bằng Spark/HF cần đọc cả folder artifact.

### Gold

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

---

## Bronze Layer

Bronze gồm ingestion/normalization (`ste1.py`) và storage/split (`ste2.py`). Mục tiêu là biến dữ liệu HuggingFace streaming thành các parquet sạch, có schema ổn định và split theo thời gian.

### `ste1.py`: Streaming Ingestion Và Normalize

`ste1.py` dùng `datasets.load_dataset(..., streaming=True)` và xử lý batch bằng PyArrow. Dataset và batch size lấy từ `config/config.yaml`.

Config liên quan:

```yaml
huggingface:
  review_dataset: "McAuley-Lab/Amazon-Reviews-2023"
  review_subset: "raw_review_Electronics"
  metadata_dataset: "McAuley-Lab/Amazon-Reviews-2023"
  metadata_subset: "raw_meta_Electronics"
  stream_batch_size: 150000
  max_review_records: null
  max_metadata_records: null
```

Review schema sau normalize:

```text
reviewer_id       string
parent_asin       string
rating            int8
review_title      string
review_text       string
timestamp         int64
helpful_vote      int32
```

Review normalization:

- `reviewer_id` lấy từ `user_id`, `reviewer_id`, hoặc `reviewerID`.
- `parent_asin` lấy từ `parent_asin` hoặc `asin`.
- Trim whitespace ở `reviewer_id` và `parent_asin`.
- Bỏ dòng thiếu ID sau trim.
- `timestamp` được cast sang `int64`; nếu giá trị lớn hơn `1_000_000_000_000` thì chia 1000 để đổi millisecond về second.
- `rating` được cast float, clip về `[1, 5]`, round, rồi cast `int8`.
- `review_title` lấy từ field raw `title`; `review_text` lấy từ `text` hoặc `review_text`.
- `helpful_vote` được cast `int32`, default 0 nếu thiếu/null.

Metadata schema sau normalize:

```text
parent_asin       string
title             string
main_category     string
store             string
price             float32
average_rating    float32
rating_number     int32
categories        string
features          string
description       string
details           string
```

Metadata normalization:

- `parent_asin` lấy từ `parent_asin` hoặc `asin`, sau đó trim whitespace.
- Bỏ metadata row thiếu `parent_asin`.
- `categories` dạng list được nối bằng `" > "`.
- `features` dạng list được nối bằng `" | "`.
- `description` dạng list được nối bằng `" "`.
- `details` dạng dict được flatten thành `"key: value | key: value"`, chỉ giữ value dạng string ngắn hơn 200 ký tự.
- `price`, `average_rating`, `rating_number` được cast numeric với default 0.

Các helper chính:

```text
iter_review_batches(cfg)   -> Iterator[pa.Table]
iter_metadata_batches(cfg) -> Iterator[pa.Table]
```

Các iterator này được `ste2.py` gọi để ghi lên MinIO/S3.

### `ste2.py`: Bronze Storage, Dedup, Core-5, Split

`ste2.py` chạy hai nhánh xử lý:

1. Metadata streaming write.
2. Reviews streaming upload vào landing zone rồi Spark reduce/split.

#### Metadata Write

`process_bronze_metadata(cfg)` ghi từng PyArrow table từ `iter_metadata_batches()` vào:

```text
s3://recsys/bronze/bronze_meta.parquet
```

Ghi bằng:

```text
pyarrow.parquet.ParquetWriter(..., compression="zstd")
```

Metadata không được collect toàn bộ vào RAM.

#### Review Landing Zone

Review batches được ghi vào:

```text
s3://recsys/landing/reviews_temp/
```

Tên file batch:

```text
batch_<batch_idx>_<uuid>.parquet
```

Upload song song bằng `ThreadPoolExecutor(max_workers=4)`.

Resume marker:

```text
s3://recsys/landing/reviews_temp/_SUCCESS
```

Marker chứa JSON:

```json
{
  "status": "SUCCESS",
  "batch_count": "<số batch đã upload>",
  "total_rows": "<tổng số review rows>",
  "timestamp": "..."
}
```

Nếu marker tồn tại, Phase 1 được bỏ qua và Spark đi thẳng vào Phase 2. Nếu landing zone có parquet nhưng không có marker, code xóa stale files khi `bronze.cleanup_landing = true` rồi ingest lại từ đầu.

#### Spark Processing

Spark đọc:

```text
s3a://recsys/landing/reviews_temp
```

Thứ tự xử lý:

```text
raw_df
  -> dropDuplicates(["reviewer_id", "parent_asin"])
  -> validation filter
  -> user Core-5
  -> checkpoint core5 to landing/core5_ckpt.parquet
  -> chronological split
  -> write bronze_train/val/test
```

Dedup policy:

```python
df = raw_df.dropDuplicates(["reviewer_id", "parent_asin"])
```

Dedup diễn ra trước validation. Khi một user review cùng một item nhiều lần, Spark giữ một bản ghi trong nhóm trùng lặp; bản ghi được giữ không được chọn theo timestamp.

Validation filter:

```text
reviewer_id is not null and reviewer_id != ""
parent_asin is not null and parent_asin != ""
timestamp is not null and timestamp > 0
rating between 1 and 5
```

User Core-5:

```text
user_counts = df.groupBy("reviewer_id").count()
valid_users = count >= 5
df_core5 = df join valid_users left_semi
```

Core-5 là logic đang chạy trong `ste2.py`; nó không phụ thuộc vào flag `bronze.view` hoặc `bronze.apply_user_core`.

Checkpoint:

```text
s3a://recsys/landing/core5_ckpt.parquet
```

Checkpoint được ghi/đọc lại để cắt Spark lineage trước split.

#### Chronological Split

Split dùng double max join, không dùng window function.

Test:

```text
max_ts_df = max(timestamp) per reviewer_id
test_df = df_core5 where timestamp == max_ts
test_df = test_df.dropDuplicates(["reviewer_id"])
```

Val:

```text
remaining_df = df_core5 left_anti test_df by reviewer_id, parent_asin
max_ts_val = max(timestamp) per reviewer_id trong remaining_df
val_df = remaining_df where timestamp == max_ts_val
val_df = val_df.dropDuplicates(["reviewer_id"])
```

Train:

```text
train_df = remaining_df left_anti val_df by reviewer_id, parent_asin
```

Vì Core-5 được áp trước split, mỗi user có ít nhất 5 interactions trước split và thường còn ít nhất 3 interactions trong train.

#### Bronze Output Write

```text
bronze_test  -> coalesce(5),  sortWithinPartitions("reviewer_id", "timestamp")
bronze_val   -> coalesce(5),  sortWithinPartitions("reviewer_id", "timestamp")
bronze_train -> coalesce(30), sortWithinPartitions("reviewer_id", "timestamp")
```

Tất cả output dùng ZSTD compression và overwrite mode.

### `eda_duckdb.py`

Script EDA đọc `bronze_train`, `bronze_val`, `bronze_test` bằng DuckDB. Nó tạo các view:

```text
train
val
test
interactions_all
user_freq
item_freq
```

Các nhóm phân tích chính:

- Split overview: interactions/users/items/min_ts/max_ts.
- User frequency summary trên cả 3 split.
- Item frequency summary trên cả 3 split.
- Item train frequency summary.
- K-core candidate analysis cho user/item.
- Fixed threshold groups: `HEAD_51_PLUS`, `MID_11_50`, `TAIL_1_10`, `COLD_IN_TRAIN`.
- Quantile groups: `HEAD_GE_P95`, `MID_P80_TO_P95`, `TAIL_LT_P80`.
- Val/test cold-start item vs train.
- Top items/users theo frequency.

Script này phục vụ phân tích, không ghi artifact pipeline bắt buộc.

### `upload-bronze.py`

Upload toàn bộ parquet trong `s3://recsys/bronze` lên HuggingFace dataset repo.

Default:

```text
HF_BRONZE_REPO = chuongdo1104/amazon-2023-bronze
MINIO_BUCKET = recsys
```

Script xóa folder `bronze/` trên repo trước khi upload lại.

---

## Silver Layer

Silver nhận Bronze splits và metadata, tạo popularity labels, text profile, enriched interactions và ground truth.

Orchestrator: `src/silver/ste3_silver.py`

### Silver Orchestration

Inputs:

```text
s3a://recsys/bronze/bronze_train.parquet
s3a://recsys/bronze/bronze_val.parquet
s3a://recsys/bronze/bronze_test.parquet
s3a://recsys/bronze/bronze_meta.parquet
```

Thứ tự chạy:

```text
Step 1: silver_step1_popularity.run()
Step 2: silver_step2_item_profile.run()
Step 4: silver_step4_interactions.run()
Step 3: silver_step3_user_profile.run()
```

`bronze_train` được đọc lần đầu thành:

```text
df_train_raw
df_train_light = reviewer_id, parent_asin, rating, timestamp, helpful_vote
```

`df_train_light` được cache để dùng cho Step 1 và Step 4. Step 3 đọc lại các cột text cần thiết từ Bronze train để giảm RAM.

### Step 1: Item Popularity

File: `silver_step1_popularity.py`

Input:

```text
df_train.select("parent_asin")
```

Tính:

```text
train_freq = count(*) per parent_asin trong train
```

Phân loại theo Pareto item-rank:

```text
HEAD_RATIO = 0.20
MID_RATIO  = 0.10
TAIL_RATIO = 0.70
```

Quy trình:

```text
freq_df = groupBy(parent_asin).count()
freq_rows = freq_df orderBy train_freq desc collect()
head_idx = int(item_count * 0.20)
mid_idx = int(item_count * 0.30)
head_freq_cutoff = freq_rows[head_idx].train_freq
tail_freq_cutoff = freq_rows[mid_idx].train_freq
```

Rule:

```text
HEAD: train_freq >= head_freq_cutoff
MID:  train_freq >= tail_freq_cutoff and train_freq < head_freq_cutoff
TAIL: train_freq < tail_freq_cutoff
```

Vì cutoff dựa trên frequency và có thể có nhiều item bằng nhau tại ngưỡng, số lượng item thực tế của từng group có thể lệch 20/10/70.

Output:

```text
s3a://recsys/silver/silver_item_popularity.parquet/
```

Schema:

```text
parent_asin
train_freq        long
popularity_group  string  # HEAD / MID / TAIL
```

Output được coalesce 5 files, ZSTD, sau đó đọc lại và cache để downstream broadcast join.

### Silver Text Cleaning Utility

File: `silver_utils.py`

`advanced_clean_text(col)` áp dụng các bước:

1. Xóa artifact `" | {}"` ở cuối chuỗi.
2. Xóa `{}`, quote thừa từ chuỗi JSON-like.
3. Xóa HTML tags.
4. Xóa URL.
5. Xóa một số ký tự đặc biệt gây nhiễu: `*_~^\#@%+=|<>``.
6. Lowercase.
7. Chuẩn hóa whitespace và trim.

Emoji và Unicode được giữ lại.

### Step 2: Item Text Profile

File: `silver_step2_item_profile.py`

Inputs:

```text
bronze_meta.parquet
df_popularity
df_train_reviews
```

Metadata columns được đọc:

```text
parent_asin
title
main_category
store
features
categories
description
details
```

`average_rating`, `rating_number`, `price` không được đưa vào item text.

#### Metadata Text

Các field được clean và cắt độ dài:

```text
title             150 chars
main_category     150 chars
store             120 chars
categories        150 chars
features          450 chars
features extended 750 chars nếu description rỗng
description       300 chars
details           150 chars
```

Chuỗi `item_text` dùng field tags:

```text
title: ...
[SEP] main_category: ...
[SEP] store: ...
[SEP] categories: ...
[SEP] features: ...
[SEP] description: ...
[SEP] details: ...
[SEP] positive_reviews: ...
[SEP] negative_reviews: ...
```

Nếu text quá nghèo:

```text
[NO_TEXT] item metadata unavailable
```

#### Train-Only Item Review Text

Mặc định bật:

```yaml
silver:
  use_item_review_text: true
  item_review_top_k_reviews: 2
```

Review filter:

```text
positive: rating >= 4
negative: rating <= 2
rating = 3 không dùng
review_snippet length > 0
```

Snippet:

```text
clean(review_title) max 100 chars
clean(review_text)  max 180 chars
title + " - " + text nếu cả hai tồn tại
```

Scoring:

```text
age_days = max((max_ts - timestamp) / ts_per_day, 0)
recency_weight = exp(-age_days / 365)
helpful_weight = 1 + log(1 + helpful_vote)
review_score = recency_weight * helpful_weight
```

`ts_per_day` tự chọn `86_400_000` nếu timestamp giống millisecond, ngược lại `86_400`.

Aggregation:

```text
groupBy(parent_asin, sentiment)
collect_list(struct(score, timestamp, snippet))
sort_array(desc)
slice top_k
array_join(..., " [SEP] ")
```

Output review columns:

```text
item_review_pos_text
item_review_neg_text
item_review_pos_count
item_review_neg_count
```

Các count này là số review hợp lệ theo sentiment trước khi cắt top-K.

#### Join Popularity Và Cold-Start Label

`df_text` left join `df_popularity`.

Nếu item có metadata nhưng không xuất hiện trong train:

```text
train_freq = 0
popularity_group = COLD_START
```

Output:

```text
s3a://recsys/silver/silver_item_text_profile.parquet/
partitionBy("popularity_group")
```

Schema khi đọc lại bằng Spark:

```text
parent_asin
title
main_category
item_text
item_review_pos_text
item_review_neg_text
item_review_pos_count
item_review_neg_count
text_source_level
token_estimate
train_freq
popularity_group
```

`text_source_level`:

```text
5 = có positive_reviews hoặc negative_reviews
4 = có description hoặc details
3 = có categories
2 = có features
1 = còn lại
```

`token_estimate`:

```text
int(size(split(item_text, whitespace)) * 1.3)
```

### Step 4: Enriched Interactions

File: `silver_step4_interactions.py`

Inputs:

```text
df_train
df_val
df_test
df_popularity
```

Projection:

```text
reviewer_id
parent_asin
rating
timestamp
helpful_vote
```

Join:

```text
left join df_popularity(parent_asin, train_freq, popularity_group)
```

Fallback:

```text
train_freq = 0
popularity_group = COLD_START
```

Derived:

```text
year_month = date_format(from_unixtime(timestamp), "yyyy-MM")
```

Outputs:

```text
silver_interactions_train.parquet/
silver_interactions_val.parquet/
silver_interactions_test.parquet/
```

Schema:

```text
parent_asin
reviewer_id
rating
timestamp
helpful_vote
train_freq
popularity_group
year_month
```

Write layout:

```text
train: coalesce(20), sortWithinPartitions("reviewer_id", "timestamp")
val:   coalesce(5),  sortWithinPartitions("reviewer_id", "timestamp")
test:  coalesce(5),  sortWithinPartitions("reviewer_id", "timestamp")
```

### Step 3: User Text Profile Và Ground Truth

File: `silver_step3_user_profile.py`

Step này chạy sau Step 4 và chỉ dùng train review text để tạo user profile.

#### User Review Weighting

Filter:

```text
rating >= 4 hoặc rating <= 2
review_snippet length > 0
```

Sentiment:

```text
rating >= 4 -> pos
rating <= 2 -> neg
```

Snippet:

```text
clean(review_title) max 120 chars
clean(review_text)  max 220 chars
title + " - " + text nếu cả hai tồn tại
```

Weight:

```text
age_days = max((max_ts - timestamp) / ts_per_day, 0)
recency_weight = exp(-age_days / 365)
helpful_weight = 1 + log(1 + helpful_vote)
review_weight = recency_weight * helpful_weight
```

Top-K:

```yaml
silver:
  user_profile_top_k_reviews: 3
```

Aggregation:

```text
groupBy(reviewer_id, sentiment)
collect_list(struct(review_weight, timestamp, snippet))
sort_array(desc)
slice top_k
array_join(..., " [SEP] ")
```

#### User Profile Output

Output:

```text
s3a://recsys/silver/silver_user_text_profile.parquet/
```

Schema:

```text
reviewer_id
user_text
user_pos_text
user_neg_text
review_count_train
pos_review_count_train
neg_review_count_train
avg_rating
avg_review_weight
```

`user_text`:

```text
positive_preferences: <user_pos_text>
[SEP] negative_preferences: <user_neg_text>
```

Nếu không có text hợp lệ:

```text
[NO_TEXT] User interaction profile
```

Các count/average trong output được tính trên tập review đã qua filter pos/neg và có snippet hợp lệ.

#### Ground Truth Output

Val/test ground truth được build từ `df_val` và `df_test`, sau đó join popularity.

Outputs:

```text
silver_val_ground_truth.parquet/
silver_test_ground_truth.parquet/
```

Schema:

```text
parent_asin
reviewer_id
timestamp
rating
popularity_group
train_freq
is_tail
is_cold_start
```

Flags:

```text
is_tail = 1 nếu popularity_group == TAIL
is_cold_start = 1 nếu popularity_group == COLD_START
```

Step này log distribution theo:

```text
HEAD
MID
TAIL
COLD_START
```

Guardrail log:

```text
COLD_START > 35%
TAIL > 85%
HEAD < 10%
```

Các guardrail chỉ cảnh báo trong log, không dừng job.

### `upload_silver_to_hf.py`

Upload toàn bộ parquet trong `s3://recsys/silver` lên HuggingFace dataset repo.

Default:

```text
HF_SILVER_REPO = chuongdo1104/amazon-2023-silver
MINIO_BUCKET = recsys
```

Script xóa folder `silver/` trên repo trước khi upload lại.

---

## Gold Layer

Gold chuyển Silver artifacts sang dạng training-ready cho PyTorch/PyG.

Orchestrator: `src/gold/ste4_gold.py`

### Gold Orchestration

Inputs:

```text
s3a://recsys/silver/silver_item_popularity.parquet
s3a://recsys/silver/silver_item_text_profile.parquet
s3a://recsys/silver/silver_interactions_train.parquet
s3a://recsys/silver/silver_interactions_val.parquet
s3a://recsys/silver/silver_interactions_test.parquet
```

Thứ tự chạy:

```text
Step 1: gold_step1_id_mapping.run()
Step 2: gold_step2_edge_list.run()
Step 5: gold_step5_training_meta.run()
```

Gold không encode text embedding trong local pipeline; model notebook đọc Silver text và encode offline.

### Step 1: Integer ID Mapping

File: `gold_step1_id_mapping.py`

Item universe:

```text
distinct train items
union distinct val items
union distinct test items
union distinct item_text items
```

User universe:

```text
distinct train users
```

Implementation:

- Collect all item IDs về driver, sort Python, gán `item_idx = 0..N_items-1`.
- Collect user frequency rows về driver, sort Python, gán `user_idx = 0..N_users-1`.
- Enrich item map bằng `title`, `main_category`, `popularity_group`, `train_freq`.
- Ghi map bằng PyArrow + ZSTD.

Item map:

```text
s3://recsys/gold/gold_item_id_map.parquet
```

Schema:

```text
parent_asin       string
item_idx          int32
title             string
main_category     string
popularity_group  string
train_freq        int64
```

Nếu item không có popularity row:

```text
popularity_group = COLD_START
train_freq = 0
```

User map:

```text
s3://recsys/gold/gold_user_id_map.parquet
```

Schema:

```text
reviewer_id          string
user_idx             int32
user_train_freq      int64
user_activity_group  string
```

User activity rule:

```text
INACTIVE     user_train_freq < 5
ACTIVE       user_train_freq <= 20
SUPER_ACTIVE user_train_freq > 20
```

Step 1 trả về:

```text
n_items
n_users
checksum
```

### Step 2: Edge List

File: `gold_step2_edge_list.py`

Input:

```text
silver_interactions_train
gold_item_id_map.parquet
gold_user_id_map.parquet
```

Spark joins:

```text
inter.reviewer_id == user_map.reviewer_id
inter.parent_asin == item_map.parent_asin
```

Intermediate select:

```text
user_idx
item_idx
rating
timestamp
```

Rows with null `user_idx`, `item_idx`, `rating`, or `timestamp` are dropped. Final edge output keeps only:

```text
user_idx
item_idx
```

To avoid large Spark RPC transfer, Step 2 writes temp parquet:

```text
s3a://recsys/gold/temp_edges.parquet
```

Then PyArrow Dataset reads this temp parquet directly from MinIO into driver memory and converts to NumPy.

Output:

```text
s3://recsys/gold/gold_edge_index.npy
```

Shape and dtype:

```text
shape = [2, n_edges]
dtype = int64
edge_index[0] = user_idx
edge_index[1] = item_idx
```

The temp parquet is deleted after `.npy` is saved.

### Step 5: Training Metadata

File: `gold_step5_training_meta.py`

Inputs:

```text
gold_item_id_map.parquet
gold_user_id_map.parquet
step1_info.n_items
step1_info.n_users
edge_info.n_edges
```

Outputs:

```text
gold_item_train_freq.npy
gold_item_popularity_group.npy
gold_user_train_freq.npy
gold_user_activity_group.npy
gold_negative_sampling_prob.npy
gold_dataset_stats.json
```

Item group encoding:

```text
HEAD       -> 0
MID        -> 1
TAIL       -> 2
COLD_START -> 3
```

Accepted aliases in encoder:

```text
WARM_TAIL -> 2
COLD_ITEM -> 3
```

User group encoding:

```text
INACTIVE     -> 0
ACTIVE       -> 1
SUPER_ACTIVE -> 2
```

#### Negative Sampling Probability

Config:

```yaml
gold:
  negative_sampling_strategy: "uniform"
  negative_sampling_beta: 0.75
  neg_sampling_blend_alpha: 0.70
```

Valid strategies:

```text
uniform
popularity
inverse_frequency
blended
```

Warm mask:

```text
warm_item = train_freq > 0
cold_item = train_freq == 0
```

All strategies are normalized over warm items only:

```text
prob[cold_item] = 0
sum(prob[warm_item]) = 1
```

Strategy formulas:

```text
uniform:
  raw_prob[warm] = 1

popularity:
  raw_prob[warm] = train_freq^beta

inverse_frequency:
  raw_prob[warm] = train_freq^(-beta)

blended:
  p_tail = normalize(train_freq^(-beta))
  p_head = normalize(train_freq^(beta * 0.4))
  raw_prob = alpha * p_tail + (1 - alpha) * p_head
```

#### Dataset Stats

`gold_dataset_stats.json` includes:

```text
n_users
n_items
n_edges_train
sparsity
sparsity_pct
n_head_items
n_mid_items
n_tail_items
n_cold_items
n_inactive_users
n_active_users
n_super_active_users
tail_ratio
cold_ratio
inactive_user_ratio
avg_degree_user
avg_degree_item
max_train_freq_item
max_train_freq_user
median_train_freq_item
median_train_freq_user
embedding_dim
negative_sampling_strategy
negative_sampling_beta
neg_sampling_blend_alpha
neg_sampling_strategy
neg_sampling_warm_only
neg_sampling_warm_item_count
neg_sampling_cold_prob_mass
head_ratio_split
mid_ratio_split
tail_ratio_split
```

### `upload_gold_to_hf.py`

Upload Gold artifacts lên HuggingFace.

Default:

```text
HF_GOLD_REPO = chuongdo1104/amazon-2023-gold
MINIO_BUCKET = recsys
mode = partial
```

Files in partial/full mode currently include:

```text
gold/gold_item_id_map.parquet
gold/gold_user_id_map.parquet
gold/gold_edge_index.npy
gold/gold_item_train_freq.npy
gold/gold_item_popularity_group.npy
gold/gold_user_train_freq.npy
gold/gold_user_activity_group.npy
gold/gold_negative_sampling_prob.npy
gold/gold_dataset_stats.json
```

Script xóa folder `gold/` trên repo trước khi upload lại.

---

## Config Và Mức Độ Sử Dụng

### Config Được Dùng Trực Tiếp

```yaml
huggingface:
  review_dataset
  review_subset
  metadata_dataset
  metadata_subset
  stream_batch_size
  max_review_records
  max_metadata_records

minio:
  endpoint
  access_key
  secret_key
  bucket

spark:
  app_name
  master
  driver_memory
  executor_memory
  shuffle_partitions
  parquet_compression

bronze:
  cleanup_landing

silver:
  write_mode
  use_item_review_text
  user_profile_top_k_reviews
  item_review_top_k_reviews

gold:
  embedding_dim
  negative_sampling_strategy
  negative_sampling_beta
  neg_sampling_blend_alpha
  temp_dir
```

### Config Khai Báo Nhưng Chưa Điều Khiển Logic Chạy

```yaml
bronze:
  write_mode
  view
  apply_user_core
  warm_core_user_min_interactions
  upload_workers

silver:
  tail_max_train_freq
  mid_max_train_freq

gold:
  write_mode
  warm_graph_user_min_train_freq
```

Diễn giải:

- Bronze write mode trong split đang hard-code overwrite.
- Bronze user Core-5 đang hard-code `count >= 5`.
- Bronze upload worker đang hard-code `max_workers = 4`.
- Silver popularity đang dùng Pareto rank, không dùng fixed threshold từ config.
- Gold không tạo warm graph artifact từ `warm_graph_user_min_train_freq`.

---

## Evaluation Semantics Của Pipeline

Group definitions:

```text
HEAD       item có train_freq cao theo Pareto cutoff
MID        item ở vùng giữa theo Pareto cutoff
TAIL       item có train_freq thấp nhưng vẫn có train edge
COLD_START item không xuất hiện trong train sau left join popularity
```

Ý nghĩa:

- `HEAD`, `MID`, `TAIL` đều là warm items nếu `train_freq > 0`.
- `TAIL` là warm long-tail: item có graph edge trong train nhưng ít signal.
- `COLD_START` là item có trong metadata/val/test/item universe nhưng không có train edge.
- Negative sampling ở Gold Step 5 chỉ cấp probability cho warm items.
- Notebook model training chính đang dùng warm candidate set (`train_freq > 0`) cho protocol warm long-tail.

---

## Lệnh Chạy Theo Tầng

Bronze:

```bash
docker compose run --rm pipeline python src/pipeline_runner.py --step 1_2
```

Silver:

```bash
docker compose run --rm pipeline python src/pipeline_runner.py --step 3_silver
```

Gold:

```bash
docker compose run --rm pipeline python src/pipeline_runner.py --step 4_gold
```

Full pipeline:

```bash
docker compose run --rm pipeline python src/pipeline_runner.py --all
```
