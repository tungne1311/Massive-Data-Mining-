# Kiến Trúc Data Pipeline: Bronze → Silver → Gold

## Tổng Quan

Pipeline dữ liệu của TA-RecMind theo kiến trúc **Medallion Architecture** gồm ba tầng, mỗi tầng có trách nhiệm rõ ràng và tách biệt hoàn toàn. Nguyên tắc xuyên suốt: **tập test không bao giờ bị động chạm** từ tầng Silver trở đi.

```
HuggingFace (McAuley-Lab/Amazon-Reviews-2023)
     │   raw_review_Electronics (~44M rows)
     │   raw_meta_Electronics   (~161K items)
     ▼
┌──────────────────────────────────────────┐
│              TẦNG BRONZE                 │
│  ste1.py: Streaming ingestion            │
│    → Producer-Consumer threading         │
│    → Batch 100k records → MinIO landing  │
│  ste2.py: PySpark processing             │
│    → rating ≥ 3.0 filter                 │
│    → dropDuplicates                      │
│    → Core-5 filter (users only)          │
│    → Chronological split (Double Max Join)│
│    → train / val / test                  │
│  → MinIO bronze/ + HuggingFace backup    │
└────────────────┬─────────────────────────┘
                 │ bronze_train + bronze_val + bronze_meta
                 ▼
┌──────────────────────────────────────────┐
│              TẦNG SILVER                 │
│  ste3_silver.py: Orchestrator            │
│  Step 1: Item Popularity                 │
│    train_freq CDF → HEAD/MID/TAIL        │
│  Step 2: Item Text Profile               │
│    4-level field-aware token budget      │
│  Step 4: Enrich Interactions             │
│    labels (không edge_weight)            │
│  Step 3: User Text Profile               │
│    top-3 reviews (helpful_vote weight)   │
│  → MinIO silver/ + HuggingFace backup    │
└────────────────┬─────────────────────────┘
                 │ silver_* artifacts
                 ▼
┌──────────────────────────────────────────┐
│              TẦNG GOLD                   │
│  ste4_gold.py: Orchestrator              │
│  Step 1: Integer ID Mapping              │
│  Step 2: Edge List (PyG format)          │
│  Step 3+4: LLM Embedding (→ Colab)       │
│  Step 5: Training Metadata Arrays        │
│  → MinIO gold/ + HuggingFace backup      │
└──────────────────────────────────────────┘
```

---

## Sơ Đồ Lưu Trữ Trên HuggingFace

Toàn bộ pipeline được đồng bộ mapping 1-1 từ MinIO lên HuggingFace theo từng phân lớp độc lập:

**1. Bronze** — Repo: `chuongdo1104/amazon-2023-bronze`
- `bronze/bronze_train.parquet/` — **Thư mục** (PySpark part files, zstd compression)
- `bronze/bronze_val.parquet/`   — **Thư mục**
- `bronze/bronze_test.parquet/`  — **Thư mục** (KHÔNG ĐỘNG VÀO sau bước này)
- `bronze/bronze_meta.parquet`   — **Single file** (PyArrow output)

**2. Silver** — Repo: `chuongdo1104/amazon-2023-silver`
- `silver/silver_item_popularity.parquet/`
- `silver/silver_item_text_profile.parquet/`
- `silver/silver_user_text_profile.parquet/`
- `silver/silver_interactions_train.parquet/`
- `silver/silver_interactions_val.parquet/`
- `silver/silver_val_ground_truth.parquet/`

**3. Gold** — Repo: `chuongdo1104/amazon-2023-gold`
- `gold/gold_item_id_map.parquet`, `gold/gold_user_id_map.parquet` — Single files
- `gold/gold_edge_index.npy` — PyG format `[2, E]`
- `gold/gold_item_train_freq.npy`, `gold/gold_item_popularity_group.npy`
- `gold/gold_user_train_freq.npy`, `gold/gold_user_activity_group.npy`
- `gold/gold_negative_sampling_prob.npy`
- `gold/gold_dataset_stats.json`

> **Lưu ý:** LLM embeddings (`gold_item_embeddings.npy`, `gold_user_embeddings.npy`) **không upload** theo mặc định (`--mode partial`) để tiết kiệm băng thông. Chúng được tính offline trực tiếp trên Colab bằng chunk-based encoding.

---

## Tầng Bronze

### Mục Tiêu

Thu thập dữ liệu thô từ HuggingFace, chuẩn hóa schema, lọc nhiễu cơ bản và phân tách thành ba tập train/val/test với đảm bảo không time leakage.

### Nguồn Dữ Liệu

```
Dataset: McAuley-Lab/Amazon-Reviews-2023
Subset:  raw_review_Electronics   (~44M reviews)
         raw_meta_Electronics      (161,001 items)
```

### Xử Lý Reviews — `ste1.py` + `ste2.py`

**Phase 1 — MAP (HuggingFace → Landing Zone):**

Dữ liệu được kéo theo luồng streaming với batch 100,000 records mỗi lần, xử lý song song qua producer-consumer threading. Mỗi batch được normalize và ghi thẳng ra MinIO landing zone dưới dạng Parquet (compression: zstd).

Hàm `normalize_review` xử lý:
- `reviewer_id` có thể từ `user_id`, `reviewer_id`, hoặc `reviewerID` (tương thích nhiều phiên bản)
- Timestamp: tự động phát hiện millisecond vs second, chuyển về Unix second
- Trường null được thay thế bằng giá trị mặc định hợp lý

**Phase 2 — REDUCE (PySpark Native Processing):**

```
Landing Zone Parquet
        │
        ▼
Lọc Positive Feedback (rating ≥ 3.0)     ← Chặn rò rỉ dữ liệu ngay từ đầu
        │
        ▼
dropDuplicates(["reviewer_id", "parent_asin"])
        │
        ▼
Core-5 Filter (left_semi join)            ← Giữ users có ≥ 5 positive interactions
        │
        ▼
Chronological Split (Double Max Join)
  ├── Test:  interaction mới nhất của mỗi user
  ├── Val:   interaction mới thứ hai
  └── Train: phần còn lại
        │
        ▼
Ghi ra MinIO (repartition=10, sortWithinPartitions)
```

**Tại sao Double Max Join thay vì Window Function?**

`ROW_NUMBER() OVER (PARTITION BY user ORDER BY ts DESC)` tạo ra shuffle stage với single huge partition khi dữ liệu skew (1.8M users phân phối không đều). Double Max Join thực hiện hai lần `groupBy → max(timestamp) → inner join` — mỗi lần chỉ là một shuffle nhỏ, không tập trung dữ liệu vào một partition, tránh OOM.

### Xử Lý Metadata — `ste2.py`

Metadata xử lý bằng PyArrow thuần (không cần Spark) do kích thước nhỏ hơn (161,001 items). Ghi thẳng ra MinIO dưới dạng single Parquet file (`bronze_meta.parquet`).

### Schema Bronze

**Reviews:**
```
reviewer_id       string
parent_asin       string   ← dùng parent_asin, không dùng asin
rating            float32
review_title      string
review_text       string
timestamp         int64    (Unix second)
helpful_vote      int32
verified_purchase bool
```

**Metadata:**
```
parent_asin       string
title             string
main_category     string
store             string
price             float32
average_rating    float32
rating_number     int32    ← KHÔNG dùng để phân loại HEAD/MID/TAIL (data leakage)
categories        string   (ghép chuỗi phân cấp)
features          string   (ghép với "|")
description       string
details           string   (dict → "k: v | k: v")
bought_together   string   (bỏ qua trong pipeline hiện tại)
```

### Output Bronze

```
s3a://recsys/bronze/
├── bronze_train.parquet/    (10 files, sort by reviewer_id + timestamp)
├── bronze_val.parquet/      (10 files)
├── bronze_test.parquet/     (10 files, KHÔNG ĐỘNG VÀO sau bước này)
└── bronze_meta.parquet      (single file)
```

**Backup lên HuggingFace:** Script `upload-bronze.py` dùng `fs.find()` để quét đệ quy mọi file `.parquet` và đẩy lên `chuongdo1104/amazon-2023-bronze`, bảo toàn cấu trúc thư mục part-file.

---

## Tầng Silver

### Nguyên Tắc Cốt Lõi

Tầng Silver thực hiện bốn bước theo thứ tự phụ thuộc. Mọi tính toán chỉ dùng dữ liệu từ tập **train**. `silver_val_ground_truth` chỉ lấy cặp `(user, item)` từ val — không lấy text.

**Thứ tự chạy (ste3_silver.py):** `Step 1 → Step 2 → Step 4 → Step 3`

Chiến lược "Vertical Culling":
- Cache `bronze_train` chỉ với **light columns** (không có Text) cho Step 1+4
- Sau Step 4, `unpersist` ngay lập tức
- Step 3 đọc lại `bronze_train` với **full text columns** độc lập

### Silver Step 1 — Phân Loại Item Popularity

**Vấn đề Data Leakage của `rating_number`:**

```
Item B07H65KP63:
  train_freq      = 1,561   (thực tế trong tập train)
  rating_number   = 710,348 (tổng tích lũy toàn thời gian)
  → Chênh lệch ~455 lần → leakage nghiêm trọng
```

**Biến phân loại đúng:** `train_freq(i)` = số interactions của item `i` CHỈ trong `bronze_train`.

**Xác định ngưỡng HEAD/MID/TAIL:**
```
1. Tính train_freq cho mỗi item (groupBy + count)
2. Sắp xếp giảm dần theo train_freq, tính CDF
3. HEAD: Top 20% items theo train_freq  (chiếm ~80% tương tác — Pareto)
4. MID:  10% items tiếp theo            (vùng đệm)
5. TAIL: Bottom 70% items còn lại       (755,609 items — 72.51%)
```

**Tối ưu bộ nhớ:** Bảng `freq_df` (~1 triệu dòng, 2 cột) được collect về driver (~40MB) để tính CDF bằng Python thuần — nhanh và không cần shuffle. Ngưỡng được broadcast dưới dạng scalar literal.

**Output:** `silver_item_popularity.parquet` — được cache và broadcast cho tất cả bước sau.

### Silver Step 2 — Item Text Profile

**Thứ tự ưu tiên ghép văn bản (Field-Aware Token Budget):**

| Cấp | Trường | Giới Hạn | Lý Do |
|---|---|---|---|
| 1 | `title` | Không cắt | Avg 19 từ, nhận dạng cốt lõi |
| 2 | `features` | 450 chars (~96 tokens) | Thông tin kỹ thuật phân biệt |
| 2* | `features` (extended) | 750 chars nếu thiếu description | Bù đắp khi description rỗng |
| 3 | `categories` | 150 chars (~32 tokens) | Ngữ cảnh danh mục |
| 4a | `description` | 300 chars (~64 tokens) | Mô tả sản phẩm |
| 4b | `details` | 150 chars (~32 tokens) | Brand, material, kích thước |

**Tất cả xử lý bằng Spark SQL built-ins** (`substring`, `coalesce`, `concat_ws`, `regexp_replace`) — không Python UDF — để Spark JVM thực hiện vectorized string operations (nhanh hơn 3-5x so với UDF).

**Separator:** Ghép với ` | ` để LLM phân biệt ranh giới trường rõ ràng.

**Cold-start items:** Items trong metadata nhưng không có trong train → gán nhãn `COLD_START`. Vẫn có item text profile đầy đủ — quan trọng cho LLM alignment loss trong training.

### Silver Step 3 — User Text Profile

**Nguyên tắc tuyệt đối:**
1. Reviews của val/test items **tuyệt đối không** được đưa vào user profile — lọc hoàn toàn từ đầu.
2. Dữ liệu Bronze đã đảm bảo `rating ≥ 3.0` — tránh nhồi nhét sentiment tiêu cực vào vector sở thích.

**Hàm trọng số review:**
```
w(r) = 1 + log(1 + helpful_vote(r))
```

| helpful_vote | w(r) |
|---|---|
| 0 | 1.00 (baseline) |
| 10 | ≈ 3.40 |
| 100 | ≈ 5.61 |
| 3,294 (max EDA) | ≈ 9.40 |

**Chiến lược chống OOM (1.8M users):**

```
Phase 1: compute_review_weights()
  → weighted reviews với snippet (~120 chars title + ~220 chars text)
  → cache + materialize (chặt lineage)

Phase 2: select_topk_reviews()
  → groupBy(reviewer_id) + collect_list(struct(timestamp, weight, snippet))
  → sort_array(desc) theo timestamp → slice(1, TOP_K=3)
  → repartition TRƯỚC groupBy (phân tải đều)

Checkpoint: ghi ra MinIO (chặt lineage Phase 1+2)

Phase 3: aggregate_user_text()
  → groupBy lần 2 (đơn giản, dữ liệu đã nhỏ)
  → concat_ws(" [SEP] ", sorted_snippets)
```

> **Lý do TOP_K=3 thay vì 5:** Giữ token budget dưới 384 (ngưỡng MiniLM), đồng thời giảm GPU memory khi encode.

**`[SEP]` token** phân tách ranh giới review giúp LLM xử lý multi-review input hiệu quả hơn.

**Output bổ sung:** `silver_val_ground_truth.parquet` — chứa `(reviewer_id, parent_asin, popularity_group, is_tail, is_cold_start, train_freq)` từ val, không có text. Phục vụ tính Tail Recall@K, Coverage@K.

### Silver Step 4 — Enrich Interactions

**Edge weight (Loại Bỏ khỏi Silver):**

`edge_weight` bị loại bỏ hoàn toàn ở Silver để chuẩn hóa cấu trúc đối xứng đồ thị. Mặc định Edge Weight = 1.0 implicit.

Temporal decay `exp(-λ × ΔT)` được để lại tầng Gold/Training vì `λ` là hyperparameter cần tune — Silver phải chứa dữ liệu ổn định, không phụ thuộc hyperparameter.

**Layout vật lý tối ưu:**
```python
df.repartition(10, "reviewer_id")
  .sortWithinPartitions("reviewer_id", "timestamp")
  .write.parquet(silver_out)
```

`sortWithinPartitions` khác `orderBy` — sort trong từng partition thay vì sort toàn cục (tránh shuffle thêm). Pattern truy cập phổ biến nhất là "lấy toàn bộ lịch sử của một user" → sort theo `reviewer_id + timestamp` tối ưu I/O.

### Schema Silver

**silver_item_popularity.parquet:**
```
parent_asin       string
train_freq        long
popularity_group  string    (HEAD / MID / TAIL / COLD_START)
```

**silver_item_text_profile.parquet:**
```
parent_asin       string
title             string
main_category     string
item_text         string
text_source_level integer
token_estimate    integer
average_rating    float
rating_number     integer
popularity_group  string    (Partition Column)
train_freq        long
```

**silver_user_text_profile.parquet:**
```
reviewer_id        string
user_text          string
review_count_train long
avg_rating         double
avg_review_weight  double
```

**silver_interactions_[train/val].parquet:**
```
reviewer_id       string
parent_asin       string
rating            float
timestamp         long
helpful_vote      integer
train_freq        long
popularity_group  string
year_month        string
```

**silver_val_ground_truth.parquet:**
```
reviewer_id       string
parent_asin       string
timestamp         long
rating            float
popularity_group  string
train_freq        long
is_tail           integer
is_cold_start     integer
```

### Output Silver

```
s3a://recsys/silver/
├── silver_item_popularity.parquet/     (sort by train_freq desc)
├── silver_item_text_profile.parquet/   (partitioned by popularity_group)
├── silver_user_text_profile.parquet/
├── silver_interactions_train.parquet/  (sort by reviewer_id, timestamp)
├── silver_interactions_val.parquet/
└── silver_val_ground_truth.parquet/
```

**Backup lên HuggingFace:** `upload_silver_to_hf.py` — xóa sạch thư mục `silver/` cũ (`api.delete_folder("silver")`), rồi batching 10 file/lô đẩy lên Hub.

---

## Tầng Gold

> **Trạng thái:** Hoàn thành triển khai (`ste4_gold.py` điều phối toàn bộ).

### Gold Step 1 — ID Mapping

**File:** `gold_step1_id_mapping.py`

LightGCN và PyG yêu cầu node indices là integer liên tục từ 0.

```
user_id_map:  reviewer_id (string) → user_idx  (int, 0 → N_users-1)
item_id_map:  parent_asin (string) → item_idx  (int, 0 → N_items-1)
```

**Quan trọng:** Item map bao gồm **TẤT CẢ** items từ train + val + metadata (bao gồm cold-start). User map được làm giàu thêm `user_train_freq` và nhãn hoạt động (`INACTIVE` / `ACTIVE` / `SUPER_ACTIVE`).

**Ước tính kích thước embedding matrix (d=128, float32):**
```
N_users = 1,847,662
N_items ≈ 1,172,867  (train + cold-start val)
Tổng = ~3,020,529 nodes × 128 × 4 bytes ≈ 1.5 GB
```

### Gold Step 2 — Edge List

**File:** `gold_step2_edge_list.py`

Format chuẩn PyG: `edge_index` shape `[2, E]`.

```python
# Ma trận kề thuần túy Binary Unweighted
# Không có edge_weight để giữ Symmetric Normalization chuẩn
# edge_index chuyển thẳng từ PyArrow sang NumPy
```

### Gold Step 3+4 — LLM Embeddings (Offline trên Colab)

Được di dời lên Google Colab (`TA_RecMind_V2_IntraLayer.ipynb`) để tận dụng GPU miễn phí. Luồng **Chunk-based Encoding & Checkpointing**:

- **Chunking:** Mã hóa băm nhỏ thành lô 30,000 records (`ENCODE_CHUNK=30000`, `BATCH=256`)
- **Checkpoint tự động:** Sau mỗi chunk, serialize thành `ckpt_prefix_i_end.npy` trên Google Drive
- **Auto-resume:** Nếu mất kết nối, tự động bypass các chunk đã xong
- **VStack:** Sau khi hoàn thành, ghép bằng `np.vstack` thành `gold_item_embeddings.npy` và `gold_user_embeddings.npy`
- **I/O 2 trạm:** Copy embedding từ Google Drive sang SSD local (`/content/recsys_cache`) trước khi PyTorch đọc — giảm trễ I/O về micro-second

*(`PYTORCH_ALLOC_CONF="expandable_segments:True"` được set cứng để chống phân mảnh VRAM)*

### Gold Step 5 — Training Metadata

**File:** `gold_step5_training_meta.py`

```
gold_item_train_freq.npy           → [N_items] dùng negative sampling và gate
gold_item_popularity_group.npy     → [N_items] (0=HEAD, 1=MID, 2=TAIL)
gold_user_train_freq.npy           → [N_users] dùng adaptive gate
gold_user_activity_group.npy       → [N_users] (0=INACTIVE, 1=ACTIVE, 2=SUPER_ACTIVE)
gold_negative_sampling_prob.npy    → [N_items] P ∝ freq^{0.75}
gold_dataset_stats.json            → N_users, N_items, sparsity, tail_ratio, ...
```

**Tải từ HuggingFace (chống OOM RAM):**
- Các file `.npy` tải trực tiếp bằng `hf_hub_download` + `np.load()` — không qua Pandas
- `gold_negative_sampling_prob.npy` được normalize lại sau khi tải (`prob / prob.sum()`) để tránh lỗi floating-point trong `multinomial`

**Backup:** `upload_gold_to_hf.py` — hỗ trợ `--mode full` (upload tất cả kể cả embedding) và `--mode partial` (chỉ metadata + edge, không embedding).

### Schema Gold (ID Mappings)

**gold_item_id_map.parquet:**
```
parent_asin       string
item_idx          integer
title             string
main_category     string
popularity_group  string
train_freq        long
```

**gold_user_id_map.parquet:**
```
reviewer_id          string
user_idx             integer
user_train_freq      long
user_activity_group  string
```

---

## Tiêu Thụ Artifact Trong Training Pipeline

### Tầng Bronze → Training
Không được kéo trực tiếp vào VRAM. Dùng như nguồn dự phòng hoặc cho evaluation truyền thống.

### Tầng Silver → Training
- **`silver_item_text_profile` + `silver_user_text_profile`:** HuggingFace `load_dataset()` tự ghép các part files, map với Gold ID Map → chạy qua SentenceTransformer (chunk-based) → ma trận embedding.
- **`silver_item_popularity` + `silver_val_ground_truth`:** Kết hợp để tính Tail Recall@K, NDCG@K phân tầng trong evaluation.
- **`silver_interactions_*`:** Nguồn bổ trợ cho evaluation truyền thống hoặc mở rộng node features.

### Tầng Gold → Training (chính)
- **`gold_dataset_stats.json`:** Parse bằng `json.load()` → cấp phát bộ nhớ GPU cho model
- **`gold_edge_index.npy`:** `np.load()` → `LongTensor` → chuẩn hóa đối xứng → Sparse CSR matrix cho LightGCN
- **`gold_item_train_freq.npy` + `gold_user_train_freq.npy`:** Bơm vào gate `log1p(freq)` và điều chế margin BPR Loss
- **`gold_item_popularity_group.npy` + `gold_user_activity_group.npy`:** Integer flags cho bitwise mask → lọc metrics theo segment nhanh trên GPU
- **`gold_negative_sampling_prob.npy`:** Hiện tại **bị skip** trong XDMH mới nhất — dùng Uniform Sampling + Margin BPR Loss thay thế (tự nhiên hơn)
- **`gold_item_id_map` + `gold_user_id_map`:** Cầu nối map text → row vector indices cho LightGCN

---

## Quản Lý Bộ Nhớ Xuyên Pipeline

### Nguyên Tắc Cache

| Tình Huống | Hành Động |
|---|---|
| DataFrame dùng nhiều lần trong session | `cache()` + `count()` để materialize |
| DataFrame chỉ dùng một lần (write) | Không cache |
| Lineage quá dài (> 5 transform) | Checkpoint ra MinIO, đọc lại |
| Bảng nhỏ join với bảng lớn | `F.broadcast(bảng_nhỏ)` bắt buộc |

### Thứ Tự Unpersist

```python
df_train_slim.cache(); df_train_slim.count()   # materialize
freq_df.cache(); freq_df.count()               # materialize
# ... dùng xong step 1 và 4 ...
df_train_slim.unpersist()   # NGAY SAU KHI không cần nữa — không đợi cuối pipeline
freq_df.unpersist()
```

### Checkpoint Lineage (Silver Step 3)

Sau Phase 2 của Step 3, lineage của `df_topk` bao gồm:
```
HF download → Spark read → filter → withColumn × 3 → repartition → groupBy → sort_array → slice
```

Nếu không checkpoint, bất kỳ action nào trên `df_topk` sẽ trigger recompute toàn bộ chain, bao gồm download lại từ HuggingFace. **Checkpoint sau Phase 2 chặt đứt chain này.**
