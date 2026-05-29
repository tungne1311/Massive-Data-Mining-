# Kiến Trúc Mô Hình TA-RecMindV2

Tài liệu này mô tả kiến trúc model sử dụng dữ liệu do pipeline Bronze -> Silver -> Gold tạo ra. Ba tầng local trong `src/bronze`, `src/silver`, `src/gold` tạo ra artifact model-ready; phần encode text embedding và huấn luyện TA-RecMindV2 được chạy bằng notebook GPU.

Mục tiêu của mô hình là tối ưu gợi ý sản phẩm long-tail trong warm-item protocol: item tail vẫn có train edge, nhưng tín hiệu collaborative rất ít. Cold-start items vẫn có trong item universe và text profile, nhưng protocol training/evaluation chính đang dùng candidate warm items.

---

## Data Contract Từ Pipeline

### Silver Text Artifacts

Model sử dụng hai artifact text chính:

```text
silver/silver_item_text_profile.parquet/
silver/silver_user_text_profile.parquet/
```

Item text profile cung cấp:

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

User text profile cung cấp:

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

Các text fields được model dùng để encode semantic embeddings:

```text
item_text
user_text
```

`item_text` và `user_text` đã được tạo từ train-safe text sources theo Silver logic. `item_text` có metadata, category, feature, detail và train-only item review snippets. `user_text` có positive/negative preference snippets từ train reviews.

### Gold Graph Artifacts

Model tải các artifact Gold sau:

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

`gold_edge_index.npy`:

```text
shape [2, E]
dtype int64
row 0 = user_idx
row 1 = item_idx
```

Item group encoding:

```text
0 = HEAD
1 = MID
2 = TAIL
3 = COLD_START
```

User group encoding:

```text
0 = INACTIVE
1 = ACTIVE
2 = SUPER_ACTIVE
```

`gold_item_id_map.parquet` và `gold_user_id_map.parquet` được dùng để map string IDs từ Silver val/test ground truth sang integer IDs khi evaluation.

### Validation/Test Labels

Evaluation đọc:

```text
silver/silver_val_ground_truth.parquet/
silver/silver_test_ground_truth.parquet/
```

Ground truth schema:

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

---

## Tổng Quan Kiến Trúc

TA-RecMindV2 là mô hình hybrid graph + text:

```text
Silver item/user text
        |
        v
SentenceTransformer all-MiniLM-L6-v2
        |
        v
Raw semantic embeddings, dim 384
        |
        v
Linear projection 384 -> 128
        |
        +-------------------------------+
        |                               |
        v                               v
Text view z_L                  Graph ID embeddings E^(0)
                                        |
Gold edge_index -> normalized A_hat ----+
                                        |
                                        v
                     Intra-layer gated LightGCN, L=2
                                        |
                                        v
                              Graph view z_G
                                        |
                                        v
                         Final graph/text fusion
                                        |
                                        v
                              h_user, h_item
                                        |
                                        v
                  BPR ranking + graph-text alignment
```

Thành phần chính:

- ID embeddings cho user và item.
- SentenceTransformer semantic embeddings cho user và item text.
- Projection layer đưa text embedding 384 chiều về latent dimension 128.
- Degree-aware gate prior giúp node ít tương tác nghiêng về text, node nhiều tương tác nghiêng về graph.
- Intra-layer gated LightGCN trộn text vào mỗi layer propagation.
- Weighted BPR loss tăng trọng số tail positives và giảm phạt tail negatives.
- Graph-text alignment loss giữ graph view và text view của cùng node gần nhau.

---

## Text Embedding

Config text encoder:

```text
TEXT_ENCODER_NAME = all-MiniLM-L6-v2
TEXT_PROFILE_VERSION = field_tagged_posneg_v2
LLM_DIM = 384
ENCODE_CHUNK = 30000
ENCODE_BATCH = 256 mặc định, tự chỉnh theo VRAM
```

Quy trình:

1. Đọc `item_text` từ `silver_item_text_profile.parquet/`.
2. Đọc `user_text` từ `silver_user_text_profile.parquet/`.
3. Encode bằng SentenceTransformer.
4. Cache raw embeddings trên Drive/local SSD.
5. Load về GPU, normalize L2, cast sang fp16.

Tensor shapes:

```text
item_emb_llm: [num_items, 384]
user_emb_llm: [num_users, 384]
```

Projection trong model:

```text
z_L = text_prj(raw_llm_embedding)
text_prj = Linear(384, 128)
```

Projected text cache:

```text
data/z_llm_projected_<RUN_ID>_<TEXT_PROFILE_VERSION>_<TEXT_ENCODER>.pt
```

---

## Graph Construction

Gold edge list chỉ chứa train edges:

```text
user_idx, item_idx
```

Notebook dựng graph bipartite hai chiều:

```text
user_idx -> num_users + item_idx
num_users + item_idx -> user_idx
```

Adjacency normalization:

```text
A_hat = D^(-1/2) A D^(-1/2)
```

Sparse edge weight trong adjacency:

```text
edge_weight(src, dst) = deg(src)^(-1/2) * deg(dst)^(-1/2)
```

Rating không tham gia adjacency. Mọi train edge là binary observed interaction.

Sparse adjacency được coalesce, chuyển sang CSR và cache:

```text
sparse_adj: float16 CSR on GPU
```

Item node indexing:

```text
global_item_node_idx = num_users + item_idx
```

Frequency tensor:

```text
freq_all = concat(user_train_freq, item_train_freq)
```

---

## Model: TARecMindV2

### Core Hyperparameters

```text
EMBED_DIM = 128
LLM_DIM = 384
GCN_LAYERS = 2
TEMPERATURE = 0.2
```

### Modules

```text
user_id_emb = Embedding(num_users, 128)
item_id_emb = Embedding(num_items, 128)
text_prj    = Linear(384, 128)
gate_mlp    = Linear(257, 64) -> ReLU -> Linear(64, 1)
alpha       = learnable scalar
```

`gate_mlp` nhận:

```text
[graph_embedding_128, text_embedding_128, degree_feature_1]
```

ID embeddings init:

```text
normal(mean=0, std=0.01)
```

Gate MLP init:

```text
linear weights normal(std=0.001)
linear bias = 0
```

Mục đích init nhỏ là để gate ban đầu bám theo degree prior, sau đó learned delta mới điều chỉnh dần.

---

## Degree-Aware Gate

### Degree Feature

Raw frequency:

```text
user_train_freq
item_train_freq
```

Log degree:

```text
log_deg_v = log1p(freq_v)
```

Mặc định normalize riêng user và item:

```text
user_degree_feature = log1p(user_train_freq) / max(log1p(user_train_freq))
item_degree_feature = log1p(item_train_freq) / max(log1p(item_train_freq))
```

Config:

```text
GATE_TYPEWISE_DEGREE_NORM = True
```

Kết quả:

```text
d_v in [0, 1]
```

### Gate Prior

Config:

```text
GATE_PRIOR_ENABLED = True
GATE_GAMMA_MIN = 0.25
GATE_GAMMA_MAX = 0.85
```

Prior:

```text
gamma_prior_v = gamma_min + (gamma_max - gamma_min) * d_v
```

Learned correction:

```text
delta_v^(l) = gate_mlp([E_v^(l), z_L_v, d_v])
```

Final gate:

```text
gamma_v^(l) = sigmoid(logit(gamma_prior_v) + delta_v^(l))
```

Diễn giải:

- `gamma` gần 1: node dùng graph/ID signal nhiều hơn.
- `gamma` gần 0: node dùng text signal nhiều hơn.
- Head items/users có frequency cao, nên `d_v` cao và prior kéo `gamma` lên.
- Tail items/users có frequency thấp, nên prior giữ `gamma` thấp hơn và text đóng vai trò lớn hơn.
- Cold-start items có `train_freq = 0`; trong graph chúng không có train edge nhưng vẫn có text representation nếu nằm trong item universe.

---

## Intra-Layer Gated LightGCN

Initial graph embedding:

```text
E^(0) = concat(user_id_emb.weight, item_id_emb.weight)
```

Text view:

```text
z_L_all = text_prj(concat(user_emb_llm, item_emb_llm))
```

Ở mỗi GCN layer:

```text
gamma^(l) = gate(E^(l), z_L, degree_feature)
E_tilde^(l) = gamma^(l) * E^(l) + (1 - gamma^(l)) * z_L
E^(l+1) = A_hat @ E_tilde^(l)
```

Graph output:

```text
z_G = (E^(0) + E^(1) + ... + E^(L)) / (L + 1)
```

Với:

```text
L = GCN_LAYERS = 2
```

Notebook tính gate theo chunks để tránh VRAM spike:

```text
chunk_size = 500_000 nodes
```

---

## Final Representation Và Scoring

Final fusion:

```text
alpha = sigmoid(model.alpha)
h_v = alpha * z_G_v + (1 - alpha) * z_L_v
```

Training BPR và evaluation đều normalize final embeddings:

```text
h_norm = normalize(h, p=2)
```

Score:

```text
score(u, i) = h_user_norm dot h_item_norm
```

Full-ranking evaluation:

```text
scores = user_final @ item_final.T
```

---

## Training Objective

Loss tổng:

```text
L = L_BPR + lambda_u * L_user_align + lambda_i * L_item_align + weight_decay
```

Config:

```text
LOSS_TYPE = weighted_bpr_graph_text_align_degree_prior_gate_v3
ALIGN_WARMUP_EPOCHS = 3
LAMBDA_U_ALIGN = 0.05
LAMBDA_I_ALIGN = 0.05
WEIGHT_DECAY = 1e-4
```

Training phases:

```text
Epoch 1..3:
  loss = alignment_loss

Epoch 4..EPOCHS:
  loss = weighted_bpr_loss + alignment_loss
```

### Weighted BPR

Scores:

```text
s_pos = normalize(h_u) dot normalize(h_pos)
s_neg = normalize(h_u) dot normalize(h_neg)
raw_loss = -logsigmoid(s_pos - s_neg)
```

Config:

```text
BPR_WEIGHTED = True
BPR_POS_GROUP_WEIGHTS = {HEAD: 1.0, MID: 1.1, TAIL: 1.5}
BPR_NEG_GROUP_WEIGHTS = {HEAD: 1.0, MID: 0.75, TAIL: 0.35}
```

Pair weighting:

```text
pos_w = group_weight(pos_item_group)
neg_w = group_weight(neg_item_group)
pair_w = pos_w * neg_w
pair_w = pair_w / mean(pair_w)
L_BPR = mean(raw_loss * pair_w)
```

Effect:

- Tail positives được nhấn mạnh trong ranking objective.
- Tail negatives bị giảm trọng số để tránh làm model học rằng tail item là negative quá mạnh chỉ vì được sample.

### Graph-Text Alignment

Alignment dùng InfoNCE giữa graph view và text view của cùng node:

```text
z_g = normalize(z_graph)
z_t = normalize(z_text)
sim = z_g @ z_t.T / TEMPERATURE
labels = arange(batch_size)
L_align = cross_entropy(sim, labels)
```

Notebook tính:

```text
L_user_align = align(u_G, u_L)
L_item_align = align(pos_G, pos_L)
```

---

## Sampling

### Positive Edge Sampling

Config:

```text
TAIL_POSITIVE_SAMPLE_RATIO = 0.20
```

Mỗi chunk training gồm:

```text
80% standard random train edges
20% tail train edges
```

Tail edge:

```text
item_pop_group[edge_item_idx] == 2
```

### Negative Item Sampling

Config:

```text
NEGATIVE_POPULARITY_SAMPLE_RATIO = 0.20
```

Warm item mask:

```text
warm_item = item_train_freq > 0
cold_item = item_train_freq == 0
```

Negative sampling:

```text
80% uniform warm items
20% gold_negative_sampling_prob over warm items
```

Trước khi dùng:

```text
popularity_neg_prob_warm = gold_negative_sampling_prob
popularity_neg_prob_warm[cold_item] = 0
popularity_neg_prob_warm /= sum(popularity_neg_prob_warm)
```

False negative rejection:

```text
train_pair_key = user_idx * num_items + item_idx
```

Nếu sampled negative nằm trong train positives của user, sampler resample item đó. Check được vector hóa bằng sorted pair keys và `torch.searchsorted`.

---

## Training Runtime

Config:

```text
EPOCHS = 50
LR_JOINT = 1e-3
GRAD_CLIP = 1.0
CACHE_REFRESH = 1
CHUNK_STEPS = 64
```

Optimizer:

```text
AdamW
weight_decay = 1e-4
```

Scheduler:

```text
CosineAnnealingLR(T_max=EPOCHS, eta_min=1e-5)
```

Mixed precision:

```text
torch.amp.autocast("cuda")
GradScaler
clip_grad_norm_(model.parameters(), GRAD_CLIP)
```

Batch auto-tuning:

```text
VRAM > 70GB:
  BATCH_SIZE = 32768
  ALIGN_SUBBATCH = 8192
  ACCUM_STEPS = 1
  ENCODE_BATCH = 1024

VRAM > 35GB:
  BATCH_SIZE = 16384
  ALIGN_SUBBATCH = 4096
  ACCUM_STEPS = 2
  ENCODE_BATCH = 512

VRAM > 16GB:
  BATCH_SIZE = 4096
  ALIGN_SUBBATCH = 1024
  ACCUM_STEPS = 4
  ENCODE_BATCH = 512

VRAM <= 16GB:
  BATCH_SIZE = 2048
  ALIGN_SUBBATCH = 512
  ACCUM_STEPS = 4
  ENCODE_BATCH = 256
```

Graph reuse:

```text
CHUNK_STEPS % ACCUM_STEPS == 0
```

Notebook tính `z_G_all` ở đầu accumulation block, dùng lại cho các micro-batches trong block, rồi giải phóng sau optimizer step.

---

## Evaluation Protocol

Protocol:

```text
EVAL_PROTOCOL = warm_long_tail_v1
IGNORE_COLD_ITEMS = True
MASK_VALIDATION_IN_TEST = True
```

Candidate set:

```text
candidate_item_mask = item_train_freq > 0
```

Training mask khi evaluate:

```text
score[user_train_items] = -1e9
score[ground_truth_item] được restore để tính rank
```

Fast validation stratified groups:

```text
HEAD: 5000
MID:  5000
TAIL: 20000
```

Representative validation:

```text
REP_VAL_EVERY = 5
REP_VAL_N = 100000
REP_VAL_SEED = 2026
USE_REPRESENTATIVE_FOR_BEST = True
PATIENCE_REP = 4
```

Metrics:

```text
OVERALL Recall@K / NDCG@K
HEAD Recall@K / NDCG@K
MID Recall@K / NDCG@K
TAIL Recall@K / NDCG@K
Coverage@K
TailCoverage@K
TailShare@K
ListAvgPopularity@K
ListMedianPopularity@K
UniqueAvgPopularity@K
UniqueMedianPopularity@K
```

Coverage definitions:

```text
Coverage@K = unique recommended warm candidate items / number of warm candidate items
TailCoverage@K = unique recommended tail items / number of tail candidate items
TailShare@K = count tail recommendations in top-K lists / total top-K recommendations
```

Popularity diagnostics:

```text
ListAvgPopularity@K:
  average train_freq over repeated recommendation list positions

UniqueAvgPopularity@K:
  average train_freq over unique recommended items
```

---

## Checkpoint Selection

Fast monitor score:

```text
FAST_SCORE_TYPE = tail_monitor_v2
FastScore = 0.70 * TailNDCG + 0.20 * TailRecall + 0.10 * TailCoverage
```

Checkpoint score:

```text
CHECKPOINT_SCORE_TYPE = weighted_hmean_warm_tail_overall_ndcg_v2
CHECKPOINT_W_TAIL = 2.0
CHECKPOINT_W_OVERALL = 1.0
```

Weighted harmonic mean:

```text
WHM = (w_tail + w_overall) /
      (w_tail / TailNDCG + w_overall / OverallNDCG)
```

Overall guardrail:

```text
CHECKPOINT_BASELINE_OVERALL_NDCG = None
CHECKPOINT_OVERALL_GUARDRAIL_RATIO = 0.95
```

Khi baseline overall NDCG chưa được set, guardrail không kích hoạt.

---

## Checkpoint Và Cache Files

Checkpoint manager tạo các đường dẫn theo `RUN_ID`.

Main config:

```text
DATA_VERSION = amazon_2023_electronics_full_20260525
RUN_ID = amazon_2023_electronics_full_20260525_degree_prior_gate_weighted_bpr_v3
DRIVE_ROOT = /content/drive/MyDrive/tarecmindV2
```

Files:

```text
weights/tarecmind_<RUN_ID>_best.pth
weights/tarecmind_<RUN_ID>_last.pth
data/training_history_<RUN_ID>.json
data/z_llm_projected_<RUN_ID>_<TEXT_PROFILE_VERSION>_<TEXT_ENCODER>.pt
data/eval_sample_groups_<RUN_ID>_<EVAL_PROTOCOL>.pt
data/representative_eval_groups_<EVAL_PROTOCOL>.pt
```

Checkpoint stores:

```text
model state
optimizer state
epoch
metrics
history
safe config snapshot
```

Compatibility checks use:

```text
TEXT_ENCODER_NAME
EVAL_PROTOCOL
LOSS_TYPE
CHECKPOINT_SCORE_TYPE
selected CFG keys
```

---

## Hyperparameter Summary

| Nhóm | Tham số | Giá trị |
|---|---|---|
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
| Loss | `LOSS_TYPE` | `weighted_bpr_graph_text_align_degree_prior_gate_v3` |
| Loss | `LAMBDA_U_ALIGN` | 0.05 |
| Loss | `LAMBDA_I_ALIGN` | 0.05 |
| Loss | `TEMPERATURE` | 0.2 |
| Loss | `BPR_WEIGHTED` | true |
| Sampling | `TAIL_POSITIVE_SAMPLE_RATIO` | 0.20 |
| Sampling | `NEGATIVE_POPULARITY_SAMPLE_RATIO` | 0.20 |
| Training | `EPOCHS` | 50 |
| Training | `LR_JOINT` | 1e-3 |
| Training | `WEIGHT_DECAY` | 1e-4 |
| Training | `GRAD_CLIP` | 1.0 |
| Training | `CHUNK_STEPS` | 64 |
| Eval | `EVAL_PROTOCOL` | `warm_long_tail_v1` |
| Eval | `IGNORE_COLD_ITEMS` | true |
| Eval | `EVAL_EVERY` | 2 |
| Eval | `MIN_EPOCHS` | 20 |
| Eval | `REP_VAL_EVERY` | 5 |
| Eval | `REP_VAL_N` | 100000 |
| Eval | `PATIENCE_REP` | 4 |

---

## Pipeline Artifact Mapping

| Artifact | Vai trò trong model |
|---|---|
| `silver_item_text_profile.parquet/` | Nguồn `item_text` để encode semantic item embeddings |
| `silver_user_text_profile.parquet/` | Nguồn `user_text` để encode semantic user embeddings |
| `silver_val_ground_truth.parquet/` | Validation labels |
| `silver_test_ground_truth.parquet/` | Test labels |
| `gold_edge_index.npy` | Train graph user-item |
| `gold_item_train_freq.npy` | Item degree, warm/cold mask, popularity diagnostics |
| `gold_item_popularity_group.npy` | Tail edge sampling, weighted BPR, stratified evaluation |
| `gold_user_train_freq.npy` | User degree feature cho gate |
| `gold_user_activity_group.npy` | User activity diagnostics |
| `gold_negative_sampling_prob.npy` | Gold probability component cho negative sampling |
| `gold_item_id_map.parquet` | Map `parent_asin` sang `item_idx` |
| `gold_user_id_map.parquet` | Map `reviewer_id` sang `user_idx` |

TA-RecMindV2 vì vậy là mô hình **degree-aware intra-layer gated LightGCN với semantic text projection**, huấn luyện bằng **weighted BPR + graph-text alignment**, và đánh giá chính theo **warm long-tail stratified ranking protocol**.
