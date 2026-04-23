# Kiến Trúc Data Pipeline: Bronze → Silver → Gold

## Tổng Quan

Pipeline dữ liệu của TA-RecMind theo kiến trúc **Medallion Architecture** gồm ba tầng, mỗi tầng có trách nhiệm rõ ràng và tách biệt hoàn toàn. Nguyên tắc xuyên suốt: **tập test không bao giờ bị động chạm** từ tầng Silver trở đi.

```
HuggingFace
(Amazon 2023)
     │
     ▼
┌─────────────────────────────────────┐
│           TẦNG BRONZE               │
│  • Streaming ingestion              │
│  • Schema normalization             │
│  • Core-5 filter (users only)       │
│  • Chronological split              │
│    train / val / test               │
│  → MinIO + HuggingFace backup       │
└─────────────┬───────────────────────┘
              │ bronze_train + bronze_val + bronze_meta
              ▼
┌─────────────────────────────────────┐
│           TẦNG SILVER               │
│  Step 1: Item Popularity            │
│    train_freq → HEAD/MID/TAIL       │
│  Step 2: Item Text Profile          │
│    4-level field-aware text         │
│  Step 3: User Text Profile          │
│    top-5 reviews (train only)       │
│  Step 4: Enrich Interactions        │
│    labels + edge_weight             │
│  → MinIO silver/                    │
└─────────────┬───────────────────────┘
              │ silver_* artifacts
              ▼
┌─────────────────────────────────────┐
│           TẦNG GOLD                 │
│  • Integer ID mapping               │
│  • Edge list (PyG format)           │
│  • LLM embedding (offline)          │
│  • Negative sampling weights        │
│  → MinIO gold/ + numpy files        │
└─────────────────────────────────────┘
```

### Sơ Đồ Lưu Trữ Trên Hugging Face (Hugging Face Topology)

Nhằm đảm bảo tính bền vững, tiết kiệm tài nguyên lưu trữ và dễ dàng triển khai huấn luyện ở mọi môi trường (Docker, Colab, Kaggle), toàn bộ pipeline Data của TA-RecMind được đồng bộ mapping 1-1 từ MinIO local lên hệ sinh thái Datasets của Hugging Face theo từng phân lớp độc lập:

**1. Tầng Bronze (Repo: `chuongdo1104/amazon-2023-bronze`)**
- Chỉ chứa dữ liệu đầu vào đã Normalize và lập lịch tách bộ:
- `bronze/bronze_train.parquet`: Tập dữ liệu huấn luyện.
- `bronze/bronze_val.parquet`: Tập đánh giá (được đẩy lên thông qua script `upload-bronze.py` dùng kỹ thuật xử lý Batch).
- `bronze/bronze_test.parquet`: Tập kiểm thử.
- `bronze/bronze_meta.parquet`: Dữ liệu Metadata gốc.

**2. Tầng Silver (Repo: `chuongdo1104/amazon-2023-silver`)**
- Lưu trữ các artifact đã được làm giàu, clean, và profiling văn bản (thực hiện qua `upload_silver_to_hf.py` với cơ chế delete & batching >10 file/batch):
- `silver/silver_item_popularity.parquet`: Phân loại Head/Mid/Tail.
- `silver/silver_item_text_profile.parquet` & `silver/silver_user_text_profile.parquet`: Các budget text được concatenate theo chuẩn.
- `silver/silver_interactions_train.parquet` & `silver/silver_interactions_val.parquet`.
- `silver/silver_val_ground_truth.parquet`.

**3. Tầng Gold (Repo: `chuongdo1104/amazon-2023-gold`)**
- Được đóng gói bởi `upload_gold_to_hf.py` (Hỗ trợ mode `full` hoặc `partial`), lưu các dạng tensor matrix (Numpy) chuẩn bị cho đồ thị huấn luyện PyTorch:
- `gold/gold_item_id_map.parquet` & `gold/gold_user_id_map.parquet`: Sơ đồ integer indexing.
- `gold/gold_edge_index.npy`: Danh sách cạnh (Dạng PyG).
- `gold/gold_item_train_freq.npy` & `gold/gold_item_popularity_group.npy`: Tần suất và phân nhóm head/mid/tail của Item.
- `gold/gold_user_train_freq.npy` & `gold/gold_user_activity_group.npy`: Tần suất và phân nhóm Inactive/Active của User.
- `gold/gold_negative_sampling_prob.npy`: Dữ liệu phân phối lấy mẫu âm.
- `gold/gold_dataset_stats.json`: Thống kê Graph.

*(Lưu ý: Hệ thống `upload_gold_to_hf.py` đã loại bỏ các mảng embeddings thô khổng lồ ra khỏi tác vụ upload mặc định (bằng file .npy cũ) để giải phóng băng thông. Khâu nhúng LLM Embeddings được thiết kế di dời tối ưu lên thẳng Notebook cấp Colab - XDMH.ipynb)*.

---

## Tầng Bronze

### Mục Tiêu

Thu thập dữ liệu thô từ HuggingFace, chuẩn hóa schema, lọc nhiễu cơ bản và phân tách thành ba tập train/val/test với đảm bảo không time leakage.

### Nguồn Dữ Liệu

```
Dataset: McAuley-Lab/Amazon-Reviews-2023
Subset:  raw_review_Electronics   (reviews)
         raw_meta_Electronics      (metadata)
```

### Xử Lý Reviews (`ste1.py` + `ste2.py`)

**Phase 1 — MAP (HuggingFace → Landing Zone):**

Dữ liệu được kéo theo luồng streaming với batch 100.000 records mỗi lần, xử lý song song qua producer-consumer threading. Mỗi batch được normalize và ghi thẳng ra MinIO landing zone dưới dạng Parquet (compression: zstd).

Hàm normalize_review xử lý các trường hợp:
- `reviewer_id` có thể từ `user_id`, `reviewer_id`, hoặc `reviewerID` (tương thích nhiều phiên bản)
- Timestamp: tự động phát hiện millisecond vs second, chuyển về Unix second
- Trường null được thay thế bằng giá trị mặc định hợp lý

**Phase 2 — REDUCE (PySpark Native Processing):**

```
Landing Zone Parquet
        │
        ▼
Lọc Positive Feedback (rating ≥ 3.0)              # Chặn rò rỉ dữ liệu cực đoan
        │
        ▼
dropDuplicates(["reviewer_id", "parent_asin"])   # Loại trùng lặp
        │
        ▼
Core-5 Filter (left_semi join)                    # Giữ users ≥ 5 interactions (Toàn Positive)
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

Window function `ROW_NUMBER() OVER (PARTITION BY user ORDER BY ts DESC)` tạo ra một shuffle stage với single huge partition khi dữ liệu không đều (skew). Với 1.8M users và phân phối interaction không đều, đây là nguyên nhân chính gây OOM.

Double Max Join thực hiện hai lần `groupBy → max(timestamp) → inner join`, mỗi lần chỉ là một shuffle nhỏ, không tập trung dữ liệu vào một partition.

### Xử Lý Metadata (`ste2.py`)

Metadata không cần Spark — được xử lý bằng PyArrow thuần do kích thước nhỏ hơn (161.001 items). Ghi thẳng ra MinIO dưới dạng file Parquet đơn (`bronze_meta.parquet`).

Các trường quan trọng được thêm vào so với schema gốc:
- `description`: cần cho LLM item embedding
- `details`: dict chứa brand/material, chuyển thành string "key: value | key: value"
- `average_rating`, `rating_number`: giữ để tham chiếu chéo (KHÔNG dùng phân loại)

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
rating_number     int32    ← KHÔNG dùng để phân loại HEAD/MID/TAIL
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

**Backup lên HuggingFace:** Sau khi Bronze hoàn thành, toàn bộ được đẩy lên `chuongdo1104/amazon-2023-bronze` thông qua script `upload-bronze.py` để giải phóng MinIO local.

---

## Tầng Silver

### Nguyên Tắc Cốt Lõi

Tầng Silver thực hiện bốn công việc theo thứ tự phụ thuộc. Mọi tính toán chỉ dùng dữ liệu từ tập train, ngoại trừ `silver_val_ground_truth` chỉ lấy cặp `(user, item)` từ val mà không lấy text.

### Silver Step 1 — Phân Loại Item Popularity

**Vấn đề Data Leakage của `rating_number`:**

Trường `rating_number` trong metadata Amazon 2023 phản ánh tổng lượt đánh giá tích lũy tại thời điểm thu thập dữ liệu — bao gồm cả các interactions xảy ra sau thời điểm phân chia train/test. Từ kết quả EDA:

```
Item B07H65KP63:
  train_freq      = 1,561   (thực tế trong tập train)
  rating_number   = 710,348 (tổng tích lũy toàn thời gian)
  → Chênh lệch ~455 lần → leakage nghiêm trọng
```

**Biến phân loại đúng:** `train_freq(i)` = số interactions của item i CHỈ trong `bronze_train`.

**Xác định ngưỡng phân loại (HEAD/MID/TAIL):**

```
1. Tính train_freq cho mỗi item (groupBy + count)
2. Sắp xếp giảm dần theo train_freq
3. Xác định quy mô phân tử và dùng ranh giới tỷ lệ phần trăm:
4. HEAD: Nhóm Top 20% sản phẩm có tương tác cao nhất. Nắm giữ vai trò trị 'Popularity Bias'.
5. TAIL: Nhóm Bottom 70% sản phẩm dưới cùng. Vừa vặn ôm trọn toàn bộ các sản phẩm thưa thớt, đảm bảo được "giải cứu" bởi Gate Fusion.
6. MID: Nhóm 10% ở giữa làm vùng đệm.
```

**Tối ưu bộ nhớ:** Bảng `freq_df` chỉ có ~1 triệu dòng với 2 cột. Collect về driver (tổng ~40MB) để tính CDF bằng Python thuần — nhanh và không cần shuffle. Kết quả ngưỡng được broadcast dưới dạng scalar literal, không tạo join thêm.

**Tối ưu Implicit Feedback Positive:** Dữ liệu Bronze đã được lọc sạch tương tác nhiễu (`rating >= 3.0`) ngay từ đầu nguồn. Mọi Data chảy vào Silver Layer đã đảm bảo triết lý: "Một tương tác là một phiếu bầu tích cực chắc chắn", chống rác ngữ nghĩa và chặn Data Leakage từ trứng nước.

**Output:** `silver_item_popularity.parquet` — bảng nhỏ, được cache và broadcast cho tất cả bước sau.

### Silver Step 2 — Item Text Profile

**Thứ tự ưu tiên ghép văn bản (Field-Aware Token Budget):**

| Cấp | Trường | Giới Hạn | Lý Do |
|---|---|---|---|
| 1 | `title` | Không cắt | Avg 19 từ, thông tin nhận dạng cốt lõi |
| 2 | `features` | 450 chars (~96 tokens) | Thông tin kỹ thuật, yếu tố phân biệt cốt lõi (Bơm mạnh quota) |
| 2* | `features` (extended) | 750 chars nếu thiếu desc | Bù đắp khi description rỗng (đẩy tất cả dư địa vào feature) |
| 3 | `categories` | 150 chars (~32 tokens) | Ngữ cảnh danh mục |
| 4a | `description` | 300 chars (~64 tokens) | Mô tả sản phẩm (Giảm bớt, nhường token cho features) |
| 4b | `details` | 150 chars (~32 tokens) | Brand, material, kích thước |

**Tất cả xử lý bằng Spark SQL built-ins** (`substring`, `coalesce`, `concat_ws`, `regexp_replace`) — không Python UDF — để Spark JVM thực hiện vectorized string operations, nhanh hơn 3-5x so với UDF.

**Separator:** Ghép với ` | ` để LLM phân biệt ranh giới trường rõ ràng.

**Cold-start items:** Items trong metadata nhưng không có trong train được gán nhãn `COLD_START`. Chúng vẫn có item text profile đầy đủ — quan trọng cho LLM alignment loss trong quá trình training.

### Silver Step 3 — User Text Profile

**Nguyên tắc tuyệt đối:**
1. Reviews của val/test items tuyệt đối không được đưa vào user profile — lọc hoàn toàn từ đầu.
2. Tương tác từ tầng Bronze đã đảm bảo `rating >= 3.0` trước khi tạo text profile để chặn lỗi Semantic Mismatch (tránh nhồi nhét sự thù ghét 1, 2 sao vào vector sở thích của User).

**Hàm trọng số review (đã bỏ verified_purchase):**

```
w(r) = 1 + log(1 + helpful_vote(r))
```

Tính chất:
- `helpful_vote = 0`   → `w = 1.0` (baseline, không loại bỏ)
- `helpful_vote = 10`  → `w ≈ 3.4`
- `helpful_vote = 100` → `w ≈ 5.6`
- `helpful_vote = 3294` (max theo EDA) → `w ≈ 9.4`

Hàm log kiểm soát outlier — review có 3294 helpful_votes không gấp 3294 lần review có 1 helpful_vote.

**Chiến lược chống OOM (1.8M users):**

Window function `ROW_NUMBER() OVER (PARTITION BY reviewer_id ORDER BY w DESC)` sẽ OOM với 1.8M users. Thay bằng:

```
Phase 1: compute_review_weights()
  → weighted reviews với snippet (~120 chars title + ~220 chars text)
  → cache + materialize (chặt lineage)

Phase 2: select_topk_reviews()
  → groupBy(reviewer_id) + collect_list(struct(timestamp, weight, snippet))
  → sort_array(desc) theo Timestamp và slice(1, TOP_K=3) (Fix Temporal Bias)
  → Giữ TOP_K thấp để chặn OOM LLM Tokens.
  → repartition TRƯỚC groupBy (phân tải đều)

Checkpoint: ghi ra MinIO (chặt lineage Phase 1+2)

Phase 3: aggregate_user_text()
  → groupBy lần 2 (đơn giản, dữ liệu đã nhỏ)
  → concat_ws(" [SEP] ", sorted_snippets)
```

**[SEP] token** phân tách ranh giới review giúp LLM xử lý multi-review input hiệu quả hơn.

**Output bổ sung:** `silver_val_ground_truth.parquet` — chứa cặp `(reviewer_id, parent_asin, popularity_group, is_tail, is_cold_start, train_freq)` từ val, không có text. Phục vụ tính điểm Metrics dài hạn (Tail Recall@K, Popularity).

### Silver Step 4 — Enrich Interactions

**Edge weight (Loại Bỏ):**

Hoàn toàn loại bỏ `edge_weight` ở Silver Layer để chuẩn hóa cấu trúc đối xứng Graph. Mặc định gán Edge Weight = 1.0 implicit.

Temporal decay `exp(-λ × ΔT)` để lại tầng Gold qua trực tiếp BPR Loss vì:
- `λ` là hyperparameter cần tune
- Silver nên chứa dữ liệu ổn định, không phụ thuộc hyperparameter

**Layout vật lý tối ưu:**

```python
df.repartition(10, "reviewer_id")          # hash partition
  .sortWithinPartitions("reviewer_id", "timestamp")  # sort local
  .write.parquet(silver_out)
```

`sortWithinPartitions` khác `orderBy` — nó sort trong từng partition thay vì tạo một sort toàn cục (tránh shuffle thêm). Pattern truy cập phổ biến nhất là "lấy toàn bộ lịch sử của một user" → sort theo `reviewer_id` + `timestamp` tối ưu I/O cho pattern này.

### Thứ Tự Chạy Silver (`ste3_silver.py`) và Lý Do

```
Step 1 → Step 2 → Step 4 → Step 3
```

Tất cả được điều phối bởi script `ste3_silver.py`. Điểm tối ưu I/O và Memory:
- Tập `bronze_train` ban đầu được cache riêng biệt chỉ một cụm cột nhẹ (light columns, không tải Text) để xử lý **Step 1** và **Step 4**. Ngay sau khi xong, cache này được giải phóng (`unpersist`).
- Đối với **Step 3** (heavy step), hệ thống đọc lại stream `bronze_train` với các cột Text đầy đủ để xử lý.
- Cấu trúc "Vertical Culling" này triệt tiêu hoàn toàn rủi ro tràn RAM khi shuffle mảng text khổng lồ trên 1.8M users.

### Schema Silver

**silver_item_popularity.parquet:**
```text
parent_asin       string
train_freq        long
popularity_group  string
```

**silver_item_text_profile.parquet:**
```text
parent_asin       string
title             string
main_category     string
item_text         string
text_source_level integer
token_estimate    integer
average_rating    float
rating_number     integer
popularity_group  string   (Partition Column)
train_freq        long
```

**silver_user_text_profile.parquet:**
```text
reviewer_id        string
user_text          string
review_count_train long
avg_rating         double
avg_review_weight  double
```

**silver_interactions_[train/val].parquet:**
```text
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
```text
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

**Backup lên HuggingFace:** Dữ liệu Silver sau khi được xử lý xong có thể đẩy lên Hub thông qua `upload_silver_to_hf.py` (mặc định repo: `chuongdo1104/amazon-2023-silver`). Việc đẩy lên được thực hiện theo cơ chế batching (10 file/batch) bằng `HfApi`.

---

## Tầng Gold

> **Trạng thái:** Hoàn thành triển khai (`ste4_gold.py` điều phối toàn bộ).

### Gold Step 1 — ID Mapping (Integer Indexing)
Tiến hành ở `gold_step1_id_mapping.py`

LightGCN và PyG yêu cầu node indices là integer liên tục bắt đầu từ 0.

```
user_id_map:  reviewer_id (string) → user_idx (int, 0 → N_users-1)
item_id_map:  parent_asin (string) → item_idx (int, 0 → N_items-1)
```

**Quan trọng:** Item map bao gồm TẤT CẢ items từ train + val + metadata. User map được làm giàu qua phép tính `user_train_freq` (với các nhãn `INACTIVE`, `ACTIVE`, `SUPER_ACTIVE`), làm bệ phóng cho **Layer-0 Adaptive Gate** cho phép đối xứng cực hoàn hảo.

**Ước tính kích thước ma trận embedding (d=64, float32):**
```
N_users = 1,847,662
N_items ≈ 1,172,867 (train + cold-start val)
Tổng = 3,020,529 nodes × 64 × 4 bytes ≈ 760 MB
```

### Gold Step 2 — Edge List
Tiến hành ở `gold_step2_edge_list.py`

Format chuẩn PyG: `edge_index` shape `[2, E]` + `edge_weight` shape `[E]`.

```python
# Tại Gold: Ma trận kề thuần túy cho LightGCN (Không trọng số/Edge Weight bị hủy)
# edge_index được chuyển hóa trực tiếp sang NumPy từ PyArrow.
# Lưu ý: Temporal Decay và Rating không qua ma trận kết nối để duy trì Symmetric Normalization.
```

### Gold Step 3+4 — LLM Embeddings (Chunk-based Offline Cache trên Colab)
Bước này đã được di dời trực tiếp lên Google Colab (`XDMH.ipynb`) để tận dụng GPU miễn phí. Để xử lý triệt để rủi ro tràn RAM (OOM) hoặc đứt gãy kết nối, một luồng **Chunk-based Encoding & Checkpointing** đã được định nghĩa rất cụ thể:
- **Chia cấu trúc Lô (Chunking):** Quá trình mã hóa LLM không chạy full dataset mà băm nhỏ thành các lô 30.000 records (`ENCODE_CHUNK=30000`, `BATCH=256`).
- **Checkpoint Lưu Chuyển Tự Động:** Mỗi khi xong một chunk, mảng npy sẽ lập tức serialize thành file dạng `ckpt_prefix_i_end.npy` trên Google Drive. Nếu mất kết nối, bộ đếm tự động bypass các part đã sinh sẵn.
- **Ghép Lô Tự Động (VStack):** Khi mọi tiến trình checkpoint quét toàn bộ tập dữ liệu thành công, chúng được ghép theo chiều dọc bằng `np.vstack` thành artifact gốc `gold_item_embeddings.npy` và `gold_user_embeddings.npy`.

*(Đồng thời, cấu hình `PYTORCH_ALLOC_CONF="expandable_segments:True"` được set cứng ở cấp độ biến môi trường để chống phân mảnh VRAM).* 

### Cơ Chế Lộ Trình I/O Kép (Hệ thống Drive ↔ SSD Local)
Để tránh việc CPU bị thắt cổ chai khi liên tục fetch Tensor lớn trực tiếp mạng từ Google Drive (làm Graph Neural Net bị đói data), kiến trúc sử dụng luồng I/O 2 trạm:
- Tập Embeddings khổng lồ được lệnh hệ thống ảo sao chép cứng (`subprocess.run(["cp", drive_path, local_path])`) sang bộ đệm SSD của máy ảo cục bộ (`/content/recsys_cache`). Sau đó Pytorch mới đọc từ vùng Local SSD này. Giảm trễ I/O về micro-second.

### Gold Step 5 — Training Metadata
Tiến hành ở `gold_step5_training_meta.py`

```
gold_item_train_freq.npy           → [N_items] (dùng negative sampling)
gold_item_popularity_group.npy     → [N_items] (0=HEAD, 1=MID, 2=TAIL)
gold_user_train_freq.npy           → [N_users] (Dùng adaptive gate)
gold_user_activity_group.npy       → [N_users] (0=INACTIVE, 1=ACTIVE, 2=SUPER_ACTIVE)
gold_negative_sampling_prob.npy    → [N_items] (P ∝ freq^{0.75})
gold_dataset_stats.json            → N, sparsity, tail_ratio, ...
```

**Tối ưu nạp dữ liệu từ HuggingFace (Chống OOM RAM):**
- Trong quá trình huấn luyện, các file `.npy` (như `gold_edge_index.npy`, `gold_negative_sampling_prob.npy`) được tải trực tiếp từ HuggingFace Hub vào thư mục cache local và được đọc thẳng vào bộ nhớ bằng `np.load()`.
- Phương pháp này tránh tạo các bản sao không cần thiết (như chuyển từ Parquet sang PyArrow rồi Pandas), giúp tối ưu hóa dung lượng RAM bị chiếm dụng.
- Đặc biệt với `gold_negative_sampling_prob.npy`, ngay sau khi tải, xác suất lấy mẫu âm được chuẩn hóa lại (`prob / prob.sum()`) để đảm bảo các hàm lấy mẫu (như multinomial của PyTorch/Numpy) không báo lỗi rò rỉ độ chính xác dấu phẩy động.

**Backup lên HuggingFace:** Được thực hiện thông qua script `upload_gold_to_hf.py` lên repo HuggingFace (`chuongdo1104/amazon-2023-gold`). Script hỗ trợ hai chế độ `--mode full` (upload toàn bộ metadata lẫn node embeddings) và `--mode partial` (chỉ tải metadata/edge list nhỏ gọn giúp vượt qua giới hạn mạng).

---

### Schema Gold (ID Mappings)

**gold_item_id_map.parquet:**
```text
parent_asin       string
item_idx          integer
title             string
main_category     string
popularity_group  string
train_freq        long
```

**gold_user_id_map.parquet:**
```text
reviewer_id          string
user_idx             integer
user_train_freq      long
user_activity_group  string
```

---

## Quản Lý Bộ Nhớ Xuyên Pipeline

### Nguyên Tắc Cache

| Tình Huống | Hành Động |
|---|---|
| DataFrame dùng nhiều lần trong session | `cache()` + `count()` để materialize |
| DataFrame chỉ dùng một lần (write) | Không cache |
| Lineage quá dài (> 5 transform) | Checkpoint ra MinIO, đọc lại |
| Bảng nhỏ join với bảng lớn | `F.broadcast(bảng_nhỏ)` bắt buộc |

### Checkpoint Lineage

Sau Silver Step 3 Phase 2, lineage của `df_topk` bao gồm:
```
HF download → Spark read → filter → withColumn × 3 → repartition → groupBy → sort_array → slice
```

Nếu không checkpoint, bất kỳ action nào trên `df_topk` sẽ trigger recompute toàn bộ chain, gồm cả việc download lại từ HuggingFace. Checkpoint sau Phase 2 chặt đứt chain này.

### Thứ Tự Unpersist

```python
df_train_slim.cache()
df_train_slim.count()    # materialize
# ... dùng df_train_slim để tính freq_df ...
freq_df.cache()
freq_df.count()          # materialize
df_train_slim.unpersist()  # NGAY SAU KHI không cần nữa
```

Không chờ đến cuối pipeline mới unpersist — bộ nhớ cần được giải phóng ngay để các bước sau không bị OOM.

---

## Chi Tiết Vòng Đời, Cấu Trúc Vật Lý & Cách Hoạt Động Của Từng Tệp Trên Hugging Face

Dựa trên quá trình sinh dữ liệu (ghi file) và các log upload trong script (`upload-bronze.py`, `upload_silver_to_hf.py`, `upload_gold_to_hf.py`), cấu trúc vật lý lưu trên Hugging Face mang đặc trưng rất khác biệt tùy thuộc vào công cụ sinh ra (PySpark hay Pandas/Numpy). 

### 1. Tầng Bronze (`chuongdo1104/amazon-2023-bronze`)
- **Cấu trúc vật lý thực tế trên Hugging Face:**
  - `bronze/bronze_meta.parquet`: **(Single File)** Do được ghi theo stream bằng thư viện `PyArrow` (`pq.ParquetWriter(s3_file, ...)`), đây là một file Parquet đơn nhất.
  - `bronze/bronze_train.parquet/`, `bronze/bronze_val.parquet/`, `bronze/bronze_test.parquet/`: Đây **KHÔNG PHẢI MỘT FILE** mà là các **THƯ MỤC**. Do PySpark ghi (ví dụ df.coalesce(30).write.parquet...), bên trong chúng chứa hàng chục các phân mảnh `.parquet` (vd: `part-00000-xxx.zstd.parquet`).
  - *Cách đẩy lên HF:* Script `upload-bronze.py` dùng vòng lặp `fs.find()` để càn quét đệ quy mọi tệp chứa đuôi `.parquet` và ném lên HF theo đúng đường dẫn thư mục gốc, bảo toàn cấu trúc part-file này.
- **Tiêu thụ ở quá trình Training:**
  - Khởi tạo Dataframe Immutable để dự phòng, mô hình không kéo thẳng mớ file khổng lồ này về VRAM.

### 2. Tầng Silver (`chuongdo1104/amazon-2023-silver`)
- **Cấu trúc vật lý thực tế trên Hugging Face:**
  - Tất cả output Silver như `silver_item_popularity.parquet`, `silver_item_text_profile.parquet`, `silver_user_text_profile.parquet`, `silver_interactions_train.parquet`,... cũng là các **Thư mục**. Do PySpark sử dụng hàm `.write.mode("overwrite").parquet()`, chúng chứa các tệp `part-*.parquet`.
  - *Cách đẩy lên HF:* Tương tự Bronze, lệnh xóa sạch `api.delete_folder("silver")` cạo bỏ rác rưởi của run trước, rồi batching gom cụm 10 file part mỗi lô đẩy lên HF Hub.
- **Tiêu thụ ở quá trình Training:**
  - **`silver_item_text_profile.parquet` & `silver_user_text_profile.parquet`:** Notebook gọi thư viện `datasets` (`load_dataset(...)`). Thư viện này của Hugging Face cực kỳ thông minh ở chỗ che khuất cấu trúc "thư mục", tự ngầm nối các part file lại thành một Table duy nhất. Các text profiles này được map với ID Map của tầng Gold, rồi băm nhỏ đưa qua các chunk `SentenceTransformer` cục bộ (Colab) thành Ma trận Embedding (`384` chiều). 
  - **`silver_interactions_train.parquet` & `silver_interactions_val.parquet`:** Lưu trữ lịch sử tương tác cốt lõi. Trong quy trình huấn luyện đồ thị hiện tại, graph edge được lấy từ Gold (Numpy). Tuy nhiên, file parquet này là Nguồn Bổ Trợ (Auxiliary Source) cực kỳ quan trọng cho các thư viện đánh giá truyền thống hoặc nếu muốn mở rộng Graph Node Features dựa trên Rating/Timestamp.
  - **`silver_item_popularity.parquet` & `silver_val_ground_truth.parquet`:** 
    - `silver_item_popularity.parquet` chứa nhãn bóc tách nhóm HEAD/MID/TAIL theo nguyên lý Pareto.
    - `silver_val_ground_truth.parquet` nhóm (groupby) sẵn danh sách Items thực sự thuộc về user trong tập Validation. Hai tệp này được kết hợp ở khâu Offline Metrics Pipeline cuối cùng để tính toán chính xác Hit Rate (HR) và NDCG, cho phép mổ xẻ phân tích độ chính xác trên nhóm Item Đuôi Dài (Tail-Recall).

### 3. Tầng Gold (`chuongdo1104/amazon-2023-gold`)
- **Cấu trúc vật lý thực tế trên Hugging Face:**
  - Khác hẳn PySpark mỏng nhẹ, Tầng Gold tận dụng `PyArrow` truyền thống và `Numpy` của driver local. Mọi thứ trong `gold/` hoàn toàn là các **file đơn (Single flat files)** cực chuẩn xác, không có thư mục con (No Folders).
  - Khối lượng upload (chế độ partial) rất nhanh vì toàn bộ các mảng Numpy Edge và Metadata đã nén thẳng thành dạng nhị phân 1-1.

- **Tiêu thụ, hoạt động trong Training Pipeline (`XDMH.ipynb`):**
  - **`gold_dataset_stats.json`:**
    - Parse trực tiếp bằng `json.load()` → sinh metadata trung tâm (n_users=1,847,662, n_items=1,610,012). Kích hoạt cấp phát bộ nhớ GPU cho Model.
  - **`gold_edge_index.npy`:**
    - Tải và nạp bằng `np.load()`, biến ngay lập tức thành `LongTensor` và nhồi vào LightGCN. Trải qua chuẩn hóa đối xứng (Degree normalization `torch.bincount(r).pow(-0.5)`) thành ma trận kề siêu thưa (Sparse CSR).
  - **`gold_item_train_freq.npy` & `gold_user_train_freq.npy`:**
    - Các file này nạp thẳng GPU làm tham số "cầm cân nảy mực", điều chế Loss Model:
      1. Bơm thẳng vào `margin_bpr_loss` phân cấp Head/Tail phạt ngặt nghèo lũ Head.
      2. Mở ngưỡng Gamma (Gate Layer) cho mảng Log-Frequency Cold-Start: Cứ Item nào `Frequency` nhỏ (đuôi Tail) sẽ buộc Model nuốt trọn Embedding từ LLM Text.
      3. Xây dựng Tail-Mask (oversampling các tương tác ≤ 5 Freq) trên Loss Alignment đồ thị.
  - **`gold_item_popularity_group.npy` & `gold_user_activity_group.npy`:**
    - Đây là phiên bản biến đổi số học (Integer Flag) của nhãn HEAD/MID/TAIL (cho item) và INACTIVE/ACTIVE/SUPER_ACTIVE (cho user). Chúng cho phép thao tác bitwise hoặc mask array bằng PyTorch cực nhanh trên GPU để lọc Metrics theo từng nhóm Segment (vd: "Liệu model có đang phục vụ nhóm Cold-Start tốt hơn không?").
  - **`gold_item_id_map.parquet` & `gold_user_id_map.parquet`:**
    - File ID trỏ thẳng, làm Cầu nối duy nhất map đống Text hỗn độn ở Tầng Silver trúng phóc theo dòng (Row vector indices) mà LightGCN đang chờ phục vụ.
  - **`gold_negative_sampling_prob.npy`:**
    - Mặc dù vẫn upload đầy đủ, trong phương thức SOTA của XDMH/TA-RecMind GCN mới nhất, file này **được "skip" (bỏ qua)** để quay về kỹ thuật Uniform Random Sampling gốc, vì việc áp margin qua Margin BPR Loss dựa trên Tần suất đã triệt tiêu hoàn hảo Bias một cách tự nhiên hơn là méo mó phân phối xác suất lấy mẫu.

### Nhận Xét Cơ Chế Lưu Trữ & Tiêu Thụ Tối Ưu
Quá trình tạo file tại local Cluster phân rõ vùng "Phân tán (Spark)" vs "Ma trận thuần túy (Numpy)". Khi đưa lên HuggingFace, tất cả được bảo toàn nguyên si cấu trúc nhị phân. Lúc hạ cánh vào máy ảo Training Colab (T4 / L4 GPU), bộ Pipeline không cố gắng nạp Pandas hỗn tạp vào VRAM, mà áp dụng trực diện tư tưởng `Cuda Array Processing`: Edge là list ID 1D, Text Profile là HuggingFace Iterable ảo trên RAM, Embeddings là SSD npy Memory Mapping. Dù đồ thị rất lớn, nó vẫn chạy gọn gàng không nghẽn cổ chai OOM.
