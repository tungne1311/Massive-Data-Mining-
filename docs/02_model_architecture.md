# Kiến Trúc Mô Hình TA-RecMind V2

Tài liệu này mô tả kiến trúc mô hình, dữ liệu đầu vào, quy trình huấn luyện, cơ chế đánh giá và các độ đo đang được sử dụng trong notebook:

```text
notebooks/TA_REC_đổi_data.ipynb
```

TA-RecMind V2 là mô hình gợi ý hybrid graph + text cho bài toán warm long-tail recommendation. Mô hình kết hợp tín hiệu collaborative từ đồ thị user-item với tín hiệu ngữ nghĩa từ `user_text` và `item_text`, sau đó huấn luyện bằng BPR loss kết hợp graph-text alignment.

Protocol chính trong notebook là:

```text
EVAL_PROTOCOL = warm_long_tail_v1
IGNORE_COLD_ITEMS = True
MASK_VALIDATION_IN_TEST = True
```

Với cấu hình này, item cold-start vẫn tồn tại trong metadata/text profile, nhưng không nằm trong candidate set khi train negative sampling và khi evaluate. Candidate set chính chỉ gồm warm items, tức item có `train_freq > 0`.

---

## 1. Cấu Hình Chính

Notebook dùng dict `CFG` làm cấu hình trung tâm.

### Dataset và cache

```text
REPO_ID = chuongdo1104/amazon-2023-gold
SILVER_REPO_ID = chuongdo1104/amazon-2023-silver
DATA_VERSION = amazon_2023_electronics_full_20260525
RUN_ID = amazon_2023_electronics_full_20260525_degree_prior_gate_v2
DRIVE_ROOT = /content/drive/MyDrive/tarecmindV2
DATA_DIR = /content/drive/MyDrive/tarecmindV2/data
SAVE_DIR = /content/drive/MyDrive/tarecmindV2/weights
CACHE_DIR = /content/recsys_cache
SEED = 2026
```

### Model

```text
EMBED_DIM = 128
LLM_DIM = 384
TEXT_ENCODER_NAME = all-MiniLM-L6-v2
TEXT_PROFILE_VERSION = field_tagged_posneg_v2
GCN_LAYERS = 2
TEMPERATURE = 0.2
```

### Degree-aware gate

```text
GATE_PRIOR_ENABLED = True
GATE_GAMMA_MIN = 0.25
GATE_GAMMA_MAX = 0.85
GATE_TYPEWISE_DEGREE_NORM = True
```

### Training

```text
EPOCHS = 50
LR_JOINT = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
CACHE_REFRESH = 1
CHUNK_STEPS = 64
LOSS_TYPE = bpr_graph_text_align_degree_prior_gate_v2
ALIGN_WARMUP_EPOCHS = 3
LAMBDA_U_ALIGN = 0.05
LAMBDA_I_ALIGN = 0.05
```

### Sampling

```text
TAIL_POSITIVE_SAMPLE_RATIO = 0.20
NEGATIVE_POPULARITY_SAMPLE_RATIO = 0.20
```

### Evaluation và checkpoint

```text
EVAL_EVERY = 2
MIN_EPOCHS = 20
MIN_DELTA = 1e-4
REP_VAL_EVERY = 5
REP_VAL_N = 100000
REP_VAL_SEED = 2026
USE_REPRESENTATIVE_FOR_BEST = True
PATIENCE_REP = 4

FAST_SCORE_TYPE = tail_monitor_v2
FAST_SCORE_WEIGHTS = {
  TailNDCG: 0.70,
  TailRecall: 0.20,
  TailCoverage: 0.10
}

CHECKPOINT_SCORE_TYPE = weighted_hmean_warm_tail_overall_ndcg_v2
CHECKPOINT_W_TAIL = 2.0
CHECKPOINT_W_OVERALL = 1.0
CHECKPOINT_BASELINE_OVERALL_NDCG = None
CHECKPOINT_OVERALL_GUARDRAIL_RATIO = 0.95
```

---

## 2. Data Contract

Notebook đọc dữ liệu từ hai tầng Silver và Gold.

### Gold artifacts

```text
gold/gold_dataset_stats.json
gold/gold_edge_index.npy
gold/gold_item_train_freq.npy
gold/gold_item_popularity_group.npy
gold/gold_user_train_freq.npy
gold/gold_user_activity_group.npy
gold/gold_negative_sampling_prob.npy
gold/gold_item_id_map.parquet
gold/gold_user_id_map.parquet
```

Ý nghĩa chính:

| Artifact | Vai trò |
|---|---|
| `gold_dataset_stats.json` | Số user, số item, số edge train, sparsity, average degree |
| `gold_edge_index.npy` | Danh sách cạnh train dạng `[2, n_edges]`, hàng 0 là `user_idx`, hàng 1 là `item_idx` |
| `gold_item_train_freq.npy` | Số interaction train của từng item |
| `gold_item_popularity_group.npy` | Nhãn popularity của item: `0=HEAD`, `1=MID`, `2=TAIL`, `3=COLD_START` |
| `gold_user_train_freq.npy` | Số interaction train của từng user |
| `gold_user_activity_group.npy` | Nhãn activity của user: `0=INACTIVE`, `1=ACTIVE`, `2=SUPER_ACTIVE` |
| `gold_negative_sampling_prob.npy` | Xác suất negative sampling theo chiến lược từ Gold |
| `gold_item_id_map.parquet` | Map `parent_asin -> item_idx` |
| `gold_user_id_map.parquet` | Map `reviewer_id -> user_idx` |

### Silver artifacts

```text
silver/silver_item_text_profile.parquet/
silver/silver_user_text_profile.parquet/
silver/silver_val_ground_truth.parquet/
silver/silver_test_ground_truth.parquet/
```

Text profile dùng để encode:

```text
item_text
user_text
```

Ground truth dùng để validation/test:

```text
reviewer_id
parent_asin
popularity_group
train_freq
is_tail
is_cold_start
```

Silver giữ string IDs. Notebook map string IDs sang integer IDs bằng Gold maps trước khi train/evaluate.

---

## 3. Tổng Quan Kiến Trúc

TA-RecMind V2 gồm ba phần chính:

1. Text encoder tạo semantic embeddings từ profile text.
2. LightGCN trên graph user-item tạo graph embeddings.
3. Degree-aware gated fusion trộn text signal và graph signal.

Luồng tổng quát:

```text
Silver user_text / item_text
        |
        v
SentenceTransformer(all-MiniLM-L6-v2)
        |
        v
Raw text embeddings, dim 384
        |
        v
Linear projection 384 -> 128
        |
        +-----------------------------+
        |                             |
        v                             v
Text view z_L                  ID embeddings E^(0)
                                      |
Gold train edge_index -> A_hat -------+
                                      |
                                      v
              Degree-aware intra-layer gated LightGCN
                                      |
                                      v
                              Graph view z_G
                                      |
                                      v
                    Final fusion alpha*z_G + (1-alpha)*z_L
                                      |
                                      v
                 Dot-product scoring with normalized embeddings
```

Mục tiêu thiết kế:

- Head/high-degree items có nhiều collaborative signal nên được phép dựa nhiều hơn vào graph.
- Tail/low-degree items có ít collaborative signal nên được hỗ trợ mạnh hơn bằng text profile.
- User và item đều có text view, graph view và final representation.
- Long-tail không được xử lý bằng một loss riêng có weight theo group; trọng tâm long-tail nằm ở gate prior, tail positive oversampling, warm-only negative sampling và checkpoint score ưu tiên tail.

---

## 4. Text Embedding

Notebook dùng:

```text
TEXT_ENCODER_NAME = all-MiniLM-L6-v2
LLM_DIM = 384
ENCODE_CHUNK = 30000
FORCE_ENCODE = False
```

### Item text flow

1. Tải `gold_item_id_map.parquet`.
2. Tải `silver_item_text_profile.parquet/`.
3. Merge theo `parent_asin`.
4. Sort theo `item_idx`.
5. Nếu thiếu text, dùng fallback:

```text
[NO_TEXT] Item description
```

6. Encode bằng SentenceTransformer với `normalize_embeddings=True`.

### User text flow

1. Tải `gold_user_id_map.parquet`.
2. Tải `silver_user_text_profile.parquet/`.
3. Merge theo `reviewer_id`.
4. Sort theo `user_idx`.
5. Nếu thiếu text, dùng fallback:

```text
[NO_TEXT] User interaction profile
```

6. Encode bằng SentenceTransformer với `normalize_embeddings=True`.

### Cache

Cache key:

```text
{DATA_VERSION}_{TEXT_PROFILE_VERSION}_{TEXT_ENCODER_NAME}
```

File embedding:

```text
data/gold_item_embeddings_{TEXT_CACHE_KEY}.npy
data/gold_user_embeddings_{TEXT_CACHE_KEY}.npy
/content/recsys_cache/gold_item_embeddings_{TEXT_CACHE_KEY}.npy
/content/recsys_cache/gold_user_embeddings_{TEXT_CACHE_KEY}.npy
```

Mỗi embedding text ban đầu có shape:

```text
item_emb_llm: [num_items, 384]
user_emb_llm: [num_users, 384]
```

Trước training, notebook normalize L2 và đưa về fp16 trên GPU:

```text
item_emb_llm = normalize(item_emb_llm).half()
user_emb_llm = normalize(user_emb_llm).half()
```

Projection từ 384 về 128 được học trong model:

```text
z_L = text_prj(raw_text_embedding)
```

---

## 5. Graph Construction

Train graph lấy từ:

```text
gold/gold_edge_index.npy
```

`gold_edge_index.npy` chứa user-item train edges:

```text
shape = [2, E]
row 0 = user_idx
row 1 = item_idx
```

Notebook dựng graph bipartite đối xứng:

```text
user_idx -> num_users + item_idx
num_users + item_idx -> user_idx
```

Adjacency được chuẩn hóa kiểu LightGCN:

```text
A_hat = D^(-1/2) A D^(-1/2)
edge_weight(src, dst) = deg(src)^(-1/2) * deg(dst)^(-1/2)
```

Sparse adjacency:

```text
shape = [num_users + num_items, num_users + num_items]
dtype = float16
device = cuda nếu có GPU
```

Rating không được dùng làm trọng số graph. Mỗi train edge được xem là một implicit positive interaction.

Warm/cold mask:

```text
warm_item_mask = item_train_freq > 0
cold_item_mask = item_train_freq == 0
```

Khi `IGNORE_COLD_ITEMS = True`, cold items bị loại khỏi negative sampling và candidate set evaluation.

---

## 6. Model Class `TARecMindV2`

### Module chính

```text
user_id_emb = Embedding(num_users, 128)
item_id_emb = Embedding(num_items, 128)
text_prj    = Linear(384, 128)
gate_mlp    = Linear(257, 64) -> ReLU -> Linear(64, 1)
alpha       = learnable scalar, initialized as tensor(0.5)
```

ID embeddings được init:

```text
normal(mean=0, std=0.01)
```

Gate MLP được init rất nhỏ để ban đầu gate đi gần theo degree prior:

```text
linear weight std = 0.001
linear bias = 0
```

Input của `gate_mlp` gồm:

```text
graph_embedding_128
text_embedding_128
degree_feature_1
```

Tổng dimension:

```text
128 + 128 + 1 = 257
```

### Degree feature

Notebook tạo degree feature cho toàn bộ nodes:

```text
freq_all = concat(user_train_freq, item_train_freq)
log_deg_v = log1p(freq_v)
```

Khi `GATE_TYPEWISE_DEGREE_NORM = True`, user và item được normalize riêng:

```text
user_degree_feature = log1p(user_train_freq) / max(log1p(user_train_freq))
item_degree_feature = log1p(item_train_freq) / max(log1p(item_train_freq))
degree_feature = concat(user_degree_feature, item_degree_feature)
```

Kết quả:

```text
d_v in [0, 1]
```

### Degree-aware gate prior

Gate prior:

```text
gamma_prior_v = gamma_min + (gamma_max - gamma_min) * d_v
gamma_min = 0.25
gamma_max = 0.85
```

Learned correction:

```text
delta_v^(l) = gate_mlp([E_v^(l), z_L_v, d_v])
```

Final gate:

```text
gamma_v^(l) = sigmoid(logit(gamma_prior_v) + delta_v^(l))
```

Ý nghĩa:

- `gamma` cao: node tin graph/ID signal nhiều hơn.
- `gamma` thấp: node tin text signal nhiều hơn.
- Head/high-degree nodes có `d_v` cao nên prior đẩy `gamma` lên.
- Tail/low-degree nodes có `d_v` thấp nên prior giữ `gamma` thấp hơn.

Nếu `GATE_PRIOR_ENABLED = False`, gate chỉ là:

```text
gamma_v^(l) = sigmoid(delta_v^(l))
```

### Intra-layer gated LightGCN

Initial graph embedding:

```text
E^(0) = concat(user_id_emb.weight, item_id_emb.weight)
```

Text view:

```text
z_L_all = text_prj(concat(user_emb_llm, item_emb_llm))
```

Mỗi GCN layer:

```text
gamma^(l) = gate(E^(l), z_L, degree_feature)
E_tilde^(l) = gamma^(l) * E^(l) + (1 - gamma^(l)) * z_L
E^(l+1) = A_hat @ E_tilde^(l)
```

Trong code, dòng:

```python
x_fused = torch.lerp(z_llm_all, x, gamma)
```

tương đương:

```text
x_fused = (1 - gamma) * z_L + gamma * E
```

Graph output là trung bình các layer:

```text
z_G = (E^(0) + E^(1) + ... + E^(L)) / (L + 1)
L = GCN_LAYERS = 2
```

### Final representation

Final fusion:

```text
alpha = sigmoid(model.alpha)
h_v = alpha * z_G_v + (1 - alpha) * z_L_v
```

Vì `model.alpha` init bằng `0.5`, giá trị ban đầu của `sigmoid(alpha)` xấp xỉ `0.622`.

Khi scoring, final embeddings được L2-normalize:

```text
h_norm = normalize(h, p=2)
score(u, i) = dot(h_user_norm, h_item_norm)
```

Do đã normalize, dot product tương đương cosine similarity.

---

## 7. Loss Objective

Notebook dùng:

```text
LOSS_TYPE = bpr_graph_text_align_degree_prior_gate_v2
```

Loss tổng:

```text
L = L_BPR + lambda_u * L_user_align + lambda_i * L_item_align
```

`weight_decay` được xử lý bởi AdamW, không cộng thủ công vào tensor loss.

### BPR loss

BPR dùng triplet `(user, positive_item, negative_item)`.

```text
u  = normalize(user_emb)
pi = normalize(pos_emb)
ni = normalize(neg_emb)

s_pos = dot(u, pi)
s_neg = dot(u, ni)

L_BPR = mean(-logsigmoid(s_pos - s_neg))
```

Ý nghĩa:

- Nếu `s_pos > s_neg`, loss giảm.
- Nếu negative item được score cao hơn positive item, loss tăng.
- BPR tối ưu thứ hạng tương đối, phù hợp implicit feedback.

Notebook không dùng group weight trong BPR. Long-tail emphasis đến từ sampling, gate và checkpoint score.

### Graph-text alignment loss

Alignment dùng InfoNCE giữa graph view và text view của cùng node:

```text
z_g = normalize(z_graph)
z_t = normalize(z_text)
sim = z_g @ z_t.T / TEMPERATURE
labels = arange(batch_size)
L_align = cross_entropy(sim, labels)
```

Mục tiêu:

- Graph embedding của một node gần text embedding của chính node đó.
- Graph embedding của node đó xa text embedding của node khác trong batch.

Notebook tính alignment cho:

```text
L_user_align = align(u_G, u_L)
L_item_align = align(pos_G, pos_L)
```

Config:

```text
TEMPERATURE = 0.2
LAMBDA_U_ALIGN = 0.05
LAMBDA_I_ALIGN = 0.05
```

### Alignment warm-up

Trong `ALIGN_WARMUP_EPOCHS = 3` epoch đầu:

```text
loss = alignment_loss
```

Sau warm-up:

```text
loss = bpr_loss + alignment_loss
```

Warm-up giúp text projection và graph-text space ổn định trước khi tối ưu ranking.

---

## 8. Sampling

### Positive sampling

Training chạy theo chunk:

```text
chunk_samples = CHUNK_STEPS * BATCH_SIZE
tail_samples = int(chunk_samples * TAIL_POSITIVE_SAMPLE_RATIO)
std_samples = chunk_samples - tail_samples
```

Với cấu hình hiện tại:

```text
80% random train edges
20% tail train edges
```

Tail train edges được xác định bằng:

```text
item_pop_group[edge_item_idx] == 2
```

Trong đó `2 = TAIL`.

### Negative sampling

Negative sampler:

```text
sample_negatives_uniform_warm(...)
```

Protocol:

```text
80% uniform warm items
20% Gold probability over warm items
```

Gold probability được mask lại:

```text
popularity_neg_prob_warm[cold_item] = 0
popularity_neg_prob_warm /= sum(popularity_neg_prob_warm)
```

False-negative rejection:

```text
train_pair_key = user_idx * num_items + item_idx
```

Notebook sort toàn bộ train pair keys, rồi dùng `torch.searchsorted` để kiểm tra negative item có phải train positive của user đó không. Nếu trùng, negative được resample, tối đa 10 lần.

---

## 9. Training Runtime

### GPU batch auto-tuning

Notebook chọn batch theo VRAM:

| VRAM | `BATCH_SIZE` | `ALIGN_SUBBATCH` | `ACCUM_STEPS` | `ENCODE_BATCH` |
|---|---:|---:|---:|---:|
| `> 70GB` | 32768 | 8192 | 1 | 1024 |
| `> 35GB` | 16384 | 4096 | 2 | 512 |
| `> 16GB` | 4096 | 1024 | 4 | 512 |
| `<= 16GB` | 2048 | 512 | 4 | 256 |

Với GPU lớn, notebook bật:

```text
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### Optimizer và scheduler

Optimizer:

```text
AdamW
weight_decay = 1e-4
```

Param groups:

```text
user_id_emb lr = LR_JOINT
item_id_emb lr = LR_JOINT
text_prj    lr = LR_JOINT * LLM_LR_MULT
gate_mlp    lr = LR_JOINT * LLM_LR_MULT
alpha       lr = LR_JOINT * LLM_LR_MULT
LLM_LR_MULT = 1.0
```

Scheduler:

```text
CosineAnnealingLR(T_max=EPOCHS, eta_min=1e-5)
```

Mixed precision:

```text
torch.amp.autocast("cuda")
torch.amp.GradScaler("cuda")
clip_grad_norm_(model.parameters(), GRAD_CLIP)
```

### Graph reuse

Notebook assert:

```text
CHUNK_STEPS % ACCUM_STEPS == 0
```

Trong mỗi accumulation block:

1. Tính `z_G_all = model.forward_gcn_gated(...)`.
2. Split thành `user_G_all`, `item_G_all`.
3. Reuse graph representations cho nhiều micro-batches.
4. Sau optimizer step thì giải phóng tensor trung gian.

Số step mỗi epoch:

```text
steps_per_epoch = n_train_edges // BATCH_SIZE
```

Phần dư nhỏ hơn `BATCH_SIZE` không được dùng trong epoch đó.

---

## 10. Validation Protocol

### Candidate set

Khi `IGNORE_COLD_ITEMS = True`:

```text
candidate_item_mask = item_train_freq > 0
```

Cold positives bị loại khỏi validation/test warm protocol. Cold items cũng bị set score `-1e9` trong full-ranking.

### Seen-item masking

Khi evaluate, notebook:

1. Tính score user với toàn bộ items.
2. Block item ngoài candidate set.
3. Mask train positives của user.
4. Restore score của ground-truth positive để tính rank.

Với final test, nếu `MASK_VALIDATION_IN_TEST = True`, seen-item mask gồm:

```text
train interactions + validation interactions
```

Điều này phù hợp chronological test: item đã xuất hiện trong train/val của user không được khuyến nghị lại trong test ranking.

### Fast tail-heavy validation

Notebook build stratified eval groups từ validation:

```text
HEAD: 5000
MID:  5000
TAIL: 20000
```

Fast validation chạy mỗi:

```text
EVAL_EVERY = 2 epochs
```

Fast score:

```text
FastScore =
  0.70 * TailNDCG
+ 0.20 * TailRecall
+ 0.10 * TailCoverage
```

Fast score dùng để monitor long-tail progress.

### Representative validation

Representative validation:

```text
REP_VAL_N = 100000
REP_VAL_SEED = 2026
REP_VAL_EVERY = 5
```

Representative sample lấy random warm validation interactions và giữ phân phối tự nhiên của HEAD/MID/TAIL. Khi `USE_REPRESENTATIVE_FOR_BEST = True`, checkpoint tốt nhất được chọn bằng representative validation.

---

## 11. Độ Đo Ranking

Notebook dùng full-ranking single-positive evaluation. Mỗi test/validation row có một ground-truth item. Model xếp hạng ground-truth item trong toàn bộ candidate items sau khi mask seen items.

Với mỗi interaction `(u, i*)`:

```text
rank(u, i*) = 1 + số candidate item có score > score(u, i*)
```

Nếu `rank <= K`, mô hình hit tại K.

### Recall@K

Trong single-positive setting:

```text
Recall@K = hits@K / N
```

Trong đó:

```text
hits@K = số interaction có rank <= K
N = số interaction được evaluate
```

Vì mỗi row chỉ có một positive item, Recall@K tương đương HitRate@K ở cấp interaction.

Ý nghĩa:

- Recall@20 = 0.10 nghĩa là 10% ground-truth positives xuất hiện trong top 20.
- Recall@K chỉ đo có trúng hay không, không phân biệt item đứng rank 1 hay rank K.

### NDCG@K

Notebook dùng:

```text
NDCG@K = 1 / log2(rank + 1) nếu rank <= K
NDCG@K = 0 nếu rank > K
```

Sau đó lấy trung bình trên các interaction:

```text
Mean NDCG@K = sum(NDCG@K per row) / N
```

Ý nghĩa:

- NDCG thưởng cao hơn nếu ground-truth item đứng gần đầu danh sách.
- Nếu item ở rank 1, đóng góp là `1 / log2(2) = 1`.
- Nếu item ở rank 10, đóng góp là `1 / log2(11)`.
- Nếu item ngoài top K, đóng góp là 0.

NDCG@K phù hợp hơn Recall@K khi cần phân biệt chất lượng thứ tự trong top-K.

### Overall metrics

Overall Recall@K và Overall NDCG@K được tính trên toàn bộ warm validation/test interactions:

```text
OVERALL Recall@K = hits_all@K / N_all
OVERALL NDCG@K = sum_ndcg_all@K / N_all
```

Đây là độ đo tổng quát nhất về chất lượng ranking.

### Item group metrics

Notebook report theo item popularity group:

```text
Item_HEAD Recall@K / NDCG@K
Item_MID Recall@K / NDCG@K
Item_TAIL Recall@K / NDCG@K
```

Mỗi group chỉ tính trên những test rows có ground-truth item thuộc group đó:

```text
Item_TAIL Recall@K = tail_hits@K / N_tail_rows
Item_TAIL NDCG@K = tail_ndcg_sum@K / N_tail_rows
```

Ý nghĩa:

- `Item_HEAD` đo khả năng gợi ý item phổ biến.
- `Item_MID` đo vùng trung gian.
- `Item_TAIL` là độ đo quan trọng cho mục tiêu long-tail.

Nếu `IGNORE_COLD_ITEMS = False`, notebook có thể thêm `Item_COLD`. Với cấu hình hiện tại, cold positives bị loại khỏi warm evaluation.

### User group metrics

Notebook report theo user activity:

```text
User_SUPER Recall@K / NDCG@K
User_ACTIVE Recall@K / NDCG@K
User_INACTIVE Recall@K / NDCG@K
```

Group lấy từ `gold_user_activity_group.npy`:

```text
0 = INACTIVE
1 = ACTIVE
2 = SUPER_ACTIVE
```

Ý nghĩa:

- `User_SUPER`: user có nhiều train interactions, collaborative signal mạnh.
- `User_ACTIVE`: user hoạt động vừa phải.
- `User_INACTIVE`: user ít train interactions, thường khó gợi ý hơn.

Các metric này giúp xem model có chỉ tốt với user nhiều lịch sử hay có tổng quát được cho user ít dữ liệu.

---

## 12. Độ Đo Coverage Và Popularity

Các độ đo coverage/popularity đánh giá độ đa dạng và mức độ long-tail của danh sách khuyến nghị, không chỉ đánh giá hit ground truth.

### Coverage@K

Notebook track toàn bộ unique recommended warm candidate items trong top-K của tất cả evaluated users:

```text
Coverage@K =
  số unique recommended candidate items trong top-K
  / số warm candidate items
```

Ý nghĩa:

- Coverage cao: model dùng nhiều item khác nhau trong catalog.
- Coverage thấp: model chỉ lặp lại một nhóm item nhỏ.
- Với long-tail recommendation, coverage giúp phát hiện model có bị collapse về vài head items không.

### TailCoverage@K

```text
TailCoverage@K =
  số unique recommended tail items trong top-K
  / số tail candidate items
```

Chỉ tính trên tail items thuộc candidate set.

Ý nghĩa:

- TailCoverage cao: model có khả năng đưa nhiều tail items khác nhau vào recommendation lists.
- TailCoverage thấp: model có thể vẫn hit một vài tail item nhưng không phủ được long-tail catalog.

### TailShare@K

```text
TailShare@K =
  số recommendation positions là tail items
  / tổng số recommendation positions trong top-K
```

Trong đó tổng số recommendation positions xấp xỉ:

```text
N_users_evaluated * K
```

Ý nghĩa:

- TailShare đo tỷ trọng tail trong danh sách khuyến nghị.
- TailShare cao nhưng Recall/NDCG thấp có thể nghĩa là model đưa nhiều tail item nhưng không đúng.
- TailShare thấp nhưng Overall NDCG cao có thể nghĩa là model thiên về head items.

### ListAvgPopularity@K

```text
ListAvgPopularity@K =
  trung bình train_freq của mọi item xuất hiện ở mọi vị trí top-K
```

Item bị recommend nhiều lần cho nhiều user sẽ được tính nhiều lần.

Ý nghĩa:

- Đo mức phổ biến trung bình của recommendation positions.
- Giá trị cao nghĩa là danh sách khuyến nghị nghiêng về popular/head items.
- Giá trị thấp nghĩa là danh sách có xu hướng đưa nhiều item ít phổ biến hơn.

### ListMedianPopularity@K

```text
ListMedianPopularity@K =
  median train_freq của mọi recommendation positions trong top-K
```

Ý nghĩa:

- Bền hơn average khi có vài item cực kỳ phổ biến.
- Nếu average cao nhưng median thấp, danh sách có thể gồm nhiều tail/mid nhưng bị một số head item kéo trung bình lên.

### UniqueAvgPopularity@K

```text
UniqueAvgPopularity@K =
  trung bình train_freq của unique recommended items
```

Mỗi item chỉ tính một lần dù được recommend cho nhiều user.

Ý nghĩa:

- Đo độ phổ biến của tập item được model phủ tới.
- Khác với `ListAvgPopularity`, metric này ít bị ảnh hưởng bởi item được recommend lặp lại nhiều lần.

### UniqueMedianPopularity@K

```text
UniqueMedianPopularity@K =
  median train_freq của unique recommended items
```

Ý nghĩa:

- Cho biết item điển hình trong tập unique recommended items là head, mid hay tail.
- Hữu ích khi đánh giá coverage long-tail.

---

## 13. Checkpoint Selection

Notebook có hai loại score chính.

### Fast score

Dùng cho fast tail-heavy validation:

```text
FastScore =
  0.70 * TailNDCG
+ 0.20 * TailRecall
+ 0.10 * TailCoverage
```

Ý nghĩa:

- `TailNDCG` có trọng số cao nhất vì model cần đưa tail positive lên vị trí cao.
- `TailRecall` đo tỷ lệ tail positives lọt top-K.
- `TailCoverage` tránh việc model chỉ tập trung vào một số tail items nhỏ.

### Weighted harmonic mean checkpoint score

Checkpoint chính dùng weighted harmonic mean giữa TailNDCG và OverallNDCG:

```text
WHM =
  (w_tail + w_overall)
  / (w_tail / TailNDCG + w_overall / OverallNDCG)
```

Với:

```text
w_tail = 2.0
w_overall = 1.0
```

Nếu một trong hai giá trị bằng 0, score bằng 0.

Ý nghĩa:

- Harmonic mean phạt mạnh trường hợp một metric cao nhưng metric còn lại thấp.
- `w_tail = 2.0` ưu tiên long-tail hơn overall.
- OverallNDCG vẫn được giữ để tránh mô hình cải thiện tail bằng cách phá hỏng chất lượng tổng thể.

### Overall guardrail

Config:

```text
CHECKPOINT_BASELINE_OVERALL_NDCG = None
CHECKPOINT_OVERALL_GUARDRAIL_RATIO = 0.95
```

Khi baseline là `None`, guardrail bị disable. Nếu baseline được set, checkpoint chỉ eligible khi:

```text
OverallNDCG >= 0.95 * baseline_overall_ndcg
```

### Early stopping

```text
MIN_EPOCHS = 20
PATIENCE_REP = 4
```

Patience tăng khi representative checkpoint score không cải thiện sau `MIN_EPOCHS`, hoặc checkpoint bị guardrail block.

---

## 14. Final Test Evaluation

Final test dùng best checkpoint nếu tồn tại:

```text
weights/tarecmind_{RUN_ID}_best.pth
```

Test flow:

1. Load `silver/silver_test_ground_truth.parquet/`.
2. Load Gold user/item maps.
3. Map `reviewer_id -> user_idx`, `parent_asin -> item_idx`.
4. Drop rows không map được.
5. Nếu `IGNORE_COLD_ITEMS = True`, loại cold positives.
6. Candidate set là warm items.
7. Nếu `MASK_VALIDATION_IN_TEST = True`, seen-item mask gồm train + validation interactions.
8. Chạy full-ranking với `Ks = [20, 40]`.

Test segmentation:

```text
Item_HEAD
Item_MID
Item_TAIL
User_INACTIVE
User_ACTIVE
User_SUPER
```

Các metric test chính:

```text
OVERALL Recall@20 / NDCG@20
OVERALL Recall@40 / NDCG@40
Item_HEAD Recall@K / NDCG@K
Item_MID Recall@K / NDCG@K
Item_TAIL Recall@K / NDCG@K
User_SUPER Recall@K / NDCG@K
User_ACTIVE Recall@K / NDCG@K
User_INACTIVE Recall@K / NDCG@K
Coverage@K
TailCoverage@K
TailShare@K
ListAvgPopularity@K
ListMedianPopularity@K
UniqueAvgPopularity@K
UniqueMedianPopularity@K
```

Long-tail final score:

```text
LongTailTestScore@20 =
  0.60 * TailNDCG@20
+ 0.30 * TailRecall@20
+ 0.10 * TailCoverage@20
```

Ý nghĩa:

- `TailNDCG@20` được ưu tiên vì rank trong top 20 rất quan trọng.
- `TailRecall@20` đảm bảo tail positives xuất hiện trong top 20.
- `TailCoverage@20` đảm bảo model không chỉ gợi ý một nhóm tail items hẹp.

Final report được lưu tại:

```text
data/final_evaluation_report.json
```

Report gồm:

```text
protocol
evaluation_type
ignore_cold_items
mask_validation_in_test
test_filtering
full_ranking_overall
full_ranking_tail
coverage
tail_coverage
tail_share
list_avg_popularity
list_median_popularity
unique_avg_popularity
unique_median_popularity
long_tail_test_score
timestamp
```

---

## 15. Checkpoint Và Cache Files

Weights:

```text
weights/tarecmind_{RUN_ID}_best.pth
weights/tarecmind_{RUN_ID}_last.pth
```

Training history:

```text
data/training_history_{RUN_ID}.json
```

Projected text cache:

```text
data/z_llm_projected_{RUN_ID}_{TEXT_PROFILE_VERSION}_{TEXT_ENCODER_NAME}.pt
```

Evaluation cache:

```text
data/eval_sample_groups_{RUN_ID}_{EVAL_PROTOCOL}.pt
data/representative_eval_groups_{EVAL_PROTOCOL}.pt
```

Graph/edge cache:

```text
data/train_edges.pt
data/val_edges.pt
data/val_meta.pt
data/sparse_adj.pt
```

Checkpoint payload:

```text
epoch
model_state_dict
optimizer_state_dict
metrics
config
```

Resume compatibility checks:

```text
LOSS_TYPE
TEXT_PROFILE_VERSION
TEXT_ENCODER_NAME
model state_dict compatibility
```

Nếu loss protocol hoặc text protocol khác, notebook bắt đầu lại từ epoch 1.

---

## 16. Hyperparameter Summary

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Data | `DATA_VERSION` | `amazon_2023_electronics_full_20260525` |
| Data | `RUN_ID` | `amazon_2023_electronics_full_20260525_degree_prior_gate_v2` |
| Text | `TEXT_ENCODER_NAME` | `all-MiniLM-L6-v2` |
| Text | `TEXT_PROFILE_VERSION` | `field_tagged_posneg_v2` |
| Text | `LLM_DIM` | 384 |
| Text | `ENCODE_CHUNK` | 30000 |
| Model | `EMBED_DIM` | 128 |
| Model | `GCN_LAYERS` | 2 |
| Gate | `GATE_PRIOR_ENABLED` | true |
| Gate | `GATE_GAMMA_MIN` | 0.25 |
| Gate | `GATE_GAMMA_MAX` | 0.85 |
| Gate | `GATE_TYPEWISE_DEGREE_NORM` | true |
| Loss | `LOSS_TYPE` | `bpr_graph_text_align_degree_prior_gate_v2` |
| Loss | `TEMPERATURE` | 0.2 |
| Loss | `ALIGN_WARMUP_EPOCHS` | 3 |
| Loss | `LAMBDA_U_ALIGN` | 0.05 |
| Loss | `LAMBDA_I_ALIGN` | 0.05 |
| Sampling | `TAIL_POSITIVE_SAMPLE_RATIO` | 0.20 |
| Sampling | `NEGATIVE_POPULARITY_SAMPLE_RATIO` | 0.20 |
| Training | `EPOCHS` | 50 |
| Training | `LR_JOINT` | 1e-3 |
| Training | `WEIGHT_DECAY` | 1e-4 |
| Training | `GRAD_CLIP` | 1.0 |
| Training | `CACHE_REFRESH` | 1 |
| Training | `CHUNK_STEPS` | 64 |
| Eval | `EVAL_PROTOCOL` | `warm_long_tail_v1` |
| Eval | `IGNORE_COLD_ITEMS` | true |
| Eval | `MASK_VALIDATION_IN_TEST` | true |
| Eval | `EVAL_EVERY` | 2 |
| Eval | `MIN_EPOCHS` | 20 |
| Eval | `REP_VAL_EVERY` | 5 |
| Eval | `REP_VAL_N` | 100000 |
| Eval | `PATIENCE_REP` | 4 |
| Checkpoint | `FAST_SCORE_TYPE` | `tail_monitor_v2` |
| Checkpoint | `CHECKPOINT_SCORE_TYPE` | `weighted_hmean_warm_tail_overall_ndcg_v2` |

---

## 17. Tóm Tắt

TA-RecMind V2 trong notebook là mô hình degree-aware intra-layer gated LightGCN với semantic text projection. Mô hình dùng:

- SentenceTransformer `all-MiniLM-L6-v2` để encode `user_text` và `item_text`.
- LightGCN trên graph user-item train.
- Gate theo degree để trộn graph và text ở từng layer propagation.
- Final fusion giữa graph view và text view.
- BPR loss cho ranking.
- InfoNCE graph-text alignment cho user và positive item.
- Tail positive oversampling để tăng exposure cho tail train edges.
- Warm-only negative sampling để đúng protocol warm long-tail.
- Full-ranking evaluation với Recall@K, NDCG@K, coverage, tail coverage, tail share và popularity diagnostics.
- Checkpoint selection ưu tiên TailNDCG nhưng vẫn giữ OverallNDCG bằng weighted harmonic mean.
