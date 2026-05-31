# Kiến Trúc Mô Hình TA-RecMindV2

Tài liệu này mô tả kiến trúc model và quy trình training/evaluation trong notebook:


Model sử dụng artifact từ pipeline Bronze -> Silver -> Gold. Bronze/Silver/Gold tạo dữ liệu model-ready; notebook GPU/Colab phụ trách tải artifact, encode text bằng SentenceTransformer, dựng graph, train TA-RecMindV2, checkpoint và full-ranking evaluation.

Mục tiêu chính là warm long-tail recommendation: item `TAIL` vẫn có train edge nhưng ít collaborative signal. Cold-start items vẫn tồn tại trong item universe/text profile, nhưng protocol training/evaluation hiện tại đặt `IGNORE_COLD_ITEMS = True`, nên candidate set chính chỉ gồm warm items (`train_freq > 0`).

---

## Central Config Trong Notebook

Notebook dùng một dict `CFG` làm nguồn cấu hình chính.

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

Model config:

```text
EMBED_DIM = 128
LLM_DIM = 384
TEXT_ENCODER_NAME = all-MiniLM-L6-v2
TEXT_PROFILE_VERSION = field_tagged_posneg_v2
GCN_LAYERS = 2
TEMPERATURE = 0.2
```

Gate config:

```text
GATE_PRIOR_ENABLED = True
GATE_GAMMA_MIN = 0.25
GATE_GAMMA_MAX = 0.85
GATE_TYPEWISE_DEGREE_NORM = True
```

Training config:

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

Sampling config:

```text
TAIL_POSITIVE_SAMPLE_RATIO = 0.20
NEGATIVE_POPULARITY_SAMPLE_RATIO = 0.20
```

Evaluation/checkpoint config:

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
CHECKPOINT_SCORE_TYPE = weighted_hmean_warm_tail_overall_ndcg_v2
CHECKPOINT_W_TAIL = 2.0
CHECKPOINT_W_OVERALL = 1.0
CHECKPOINT_BASELINE_OVERALL_NDCG = None
CHECKPOINT_OVERALL_GUARDRAIL_RATIO = 0.95
IGNORE_COLD_ITEMS = True
EVAL_PROTOCOL = warm_long_tail_v1
MASK_VALIDATION_IN_TEST = True
EVAL_STRAT_GROUPS = {HEAD: 5000, MID: 5000, TAIL: 20000}
```

---

## Data Contract Từ Pipeline

### Gold Artifacts

Notebook tải Gold artifact từ `REPO_ID`.

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

`gold_dataset_stats.json` cung cấp:

```text
n_users
n_items
sparsity / sparsity_pct
avg_degree_user
```

`gold_edge_index.npy`:

```text
shape = [2, E]
dtype = int64
row 0 = user_idx
row 1 = item_idx
```

Item popularity encoding:

```text
0 = HEAD
1 = MID
2 = TAIL
3 = COLD_START
```

User activity encoding:

```text
0 = INACTIVE
1 = ACTIVE
2 = SUPER_ACTIVE
```

`gold_negative_sampling_prob.npy` là probability vector `float32` có length bằng `num_items` và tổng xấp xỉ 1. Notebook mask lại vector này để chỉ sample negative từ warm items.

### Silver Artifacts

Notebook tải Silver artifact từ `SILVER_REPO_ID`.

Text profiles:

```text
silver/silver_item_text_profile.parquet/
silver/silver_user_text_profile.parquet/
```

Các text field chính:

```text
item_text
user_text
```

ID mapping khi encode text:

```text
gold/gold_item_id_map.parquet: parent_asin -> item_idx
gold/gold_user_id_map.parquet: reviewer_id -> user_idx
```

Evaluation labels:

```text
silver/silver_val_ground_truth.parquet/
silver/silver_test_ground_truth.parquet/
```

Ground truth schema notebook cần:

```text
reviewer_id
parent_asin
popularity_group
train_freq
is_tail
is_cold_start
```

Val/test ground truth dùng string IDs từ Silver, sau đó map sang integer IDs bằng Gold maps.

---

## Tổng Quan Kiến Trúc

TA-RecMindV2 là hybrid graph + text recommender:

```text
Silver item_text/user_text
        |
        v
SentenceTransformer(all-MiniLM-L6-v2)
        |
        v
Raw semantic embeddings, dim 384
        |
        v
Linear projection 384 -> 128
        |
        +--------------------------------+
        |                                |
        v                                v
Text view z_L                    ID embeddings E^(0)
                                         |
Gold train edge_index -> A_hat ----------+
                                         |
                                         v
                       Degree-aware intra-layer gated LightGCN
                                         |
                                         v
                                Graph view z_G
                                         |
                                         v
                         Final fusion: alpha*z_G + (1-alpha)*z_L
                                         |
                                         v
                            BPR + graph-text alignment
```

Thành phần chính:

- User/item ID embeddings tạo collaborative signal.
- SentenceTransformer tạo semantic embeddings từ `item_text` và `user_text`.
- Linear projection đưa text embedding 384 chiều về latent dimension 128.
- Degree-aware gate prior giúp low-degree/tail nodes dựa nhiều hơn vào text, high-degree/head nodes dựa nhiều hơn vào graph.
- Intra-layer gated LightGCN trộn text vào từng propagation layer.
- Loss chính là standard BPR cộng user/item graph-text alignment.
- Long-tail được xử lý bằng gate, positive tail oversampling, warm-only negative sampling và checkpoint score theo tail NDCG.

---

## Text Embedding

Notebook dùng:

```text
TEXT_ENCODER_NAME = all-MiniLM-L6-v2
LLM_DIM = 384
ENCODE_CHUNK = 30000
ENCODE_BATCH = tự chỉnh theo VRAM
FORCE_ENCODE = False
```

Cache key:

```text
{DATA_VERSION}_{TEXT_PROFILE_VERSION}_{TEXT_ENCODER_NAME}
```

Drive cache:

```text
data/gold_item_embeddings_{TEXT_CACHE_KEY}.npy
data/gold_user_embeddings_{TEXT_CACHE_KEY}.npy
data/ckpt_item_{TEXT_CACHE_KEY}_{start}_{end}.npy
data/ckpt_user_{TEXT_CACHE_KEY}_{start}_{end}.npy
```

Local SSD cache:

```text
/content/recsys_cache/gold_item_embeddings_{TEXT_CACHE_KEY}.npy
/content/recsys_cache/gold_user_embeddings_{TEXT_CACHE_KEY}.npy
```

Item encoding flow:

1. Load `gold_item_id_map.parquet`.
2. Load `silver_item_text_profile.parquet/` bằng `datasets.load_dataset(..., data_dir=...)`.
3. Merge theo `parent_asin`.
4. Sort theo `item_idx`.
5. Fill missing text bằng `[NO_TEXT] Item description`.
6. Encode bằng SentenceTransformer với `normalize_embeddings=True`.

User encoding flow:

1. Load `gold_user_id_map.parquet`.
2. Load `silver_user_text_profile.parquet/`.
3. Merge theo `reviewer_id`.
4. Sort theo `user_idx`.
5. Fill missing text bằng `[NO_TEXT] User interaction profile`.
6. Encode bằng SentenceTransformer với `normalize_embeddings=True`.

Output tensors:

```text
item_emb_llm: [num_items, 384], float32 khi load vào RAM
user_emb_llm: [num_users, 384], float32 khi load vào RAM
```

Trước khi training, notebook normalize L2 trên GPU và cast sang fp16:

```text
item_emb_llm = normalize(item_emb_llm).half()
user_emb_llm = normalize(user_emb_llm).half()
```

Projected text cache:

```text
data/z_llm_projected_{RUN_ID}_{TEXT_PROFILE_VERSION}_{TEXT_ENCODER_NAME}.pt
```

Cache projected text được lưu fp16 nhưng khi load lại được convert về float32. Sau resume, notebook recompute `z_llm_all_cached` để tránh cache stale.

---

## Graph Construction

Train graph lấy từ:

```text
gold/gold_edge_index.npy
```

Notebook dựng graph bipartite đối xứng:

```text
user_idx -> num_users + item_idx
num_users + item_idx -> user_idx
```

Adjacency normalization:

```text
A_hat = D^(-1/2) A D^(-1/2)
edge_weight(src, dst) = deg(src)^(-1/2) * deg(dst)^(-1/2)
```

Sparse adjacency:

```text
shape = [num_users + num_items, num_users + num_items]
format = sparse CSR
dtype = float16
device = cuda nếu có GPU
```

Rating không tham gia graph. Mọi train edge trong `gold_edge_index.npy` được coi là binary observed interaction.

Val edges được build từ:

```text
silver/silver_val_ground_truth.parquet/
gold/gold_item_id_map.parquet
gold/gold_user_id_map.parquet
```

Notebook inner join val ground truth với Gold maps để tạo:

```text
val_edges_t: [2, VAL_SIZE]
val_meta_t:
  is_tail
  is_cold_start
```

Graph/cache files:

```text
data/train_edges.pt
data/val_edges.pt
data/val_meta.pt
data/sparse_adj.pt
```

Warm/cold protocol:

```text
warm_item_mask = item_train_freq > 0
cold_item_mask = item_train_freq == 0
```

Notebook assert train positives không chứa cold items:

```text
cold_item_mask[train_edge_index[1]].sum() == 0
```

Cold items bị loại khỏi negative sampling và candidate set khi `IGNORE_COLD_ITEMS = True`.

---

## Model Class: `TARecMindV2`

### Modules

```text
user_id_emb = Embedding(num_users, 128)
item_id_emb = Embedding(num_items, 128)
text_prj    = Linear(384, 128)
gate_mlp    = Linear(257, 64) -> ReLU -> Linear(64, 1)
alpha       = learnable scalar, initialized as tensor(0.5)
```

ID embedding init:

```text
normal(mean=0, std=0.01)
```

Gate MLP init:

```text
linear weights: normal(std=0.001)
linear bias: 0
```

`gate_mlp` input gồm:

```text
graph_embedding_128
text_embedding_128
degree_feature_1
```

Tổng input dimension:

```text
128 + 128 + 1 = 257
```

### Degree Feature

Notebook tạo:

```text
freq_all = concat(user_train_freq, item_train_freq)
```

Raw degree được log-transform:

```text
log_deg_v = log1p(freq_v)
```

Với `GATE_TYPEWISE_DEGREE_NORM = True`, user và item được normalize riêng:

```text
user_degree_feature = log1p(user_train_freq) / max(log1p(user_train_freq))
item_degree_feature = log1p(item_train_freq) / max(log1p(item_train_freq))
degree_feature = concat(user_degree_feature, item_degree_feature)
```

Kết quả:

```text
d_v in [0, 1]
```

### Degree-Aware Gate Prior

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

Nếu `GATE_PRIOR_ENABLED = False`:

```text
gamma_v^(l) = sigmoid(delta_v^(l))
```

Diễn giải:

- `gamma` cao: node tin graph/ID signal nhiều hơn.
- `gamma` thấp: node tin text signal nhiều hơn.
- Head/high-degree nodes có `d_v` cao, prior đẩy `gamma` lên.
- Tail/low-degree nodes có `d_v` thấp, prior giữ `gamma` thấp hơn.

Notebook tính gate theo chunk:

```text
chunk_size = 500000 nodes
```

### Intra-Layer Gated LightGCN

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

Trong code, fusion dùng:

```python
x_fused = torch.lerp(z_llm_all, x, gamma)
```

Tương đương:

```text
x_fused = (1 - gamma) * z_L + gamma * E
```

Graph output:

```text
z_G = (E^(0) + E^(1) + ... + E^(L)) / (L + 1)
L = GCN_LAYERS = 2
```

### Final Representation

Text projection:

```text
z_L = text_prj(raw_llm_embedding)
```

Final fusion:

```text
alpha = sigmoid(model.alpha)
h_v = alpha * z_G_v + (1 - alpha) * z_L_v
```

Vì `model.alpha` init bằng `0.5`, initial `sigmoid(alpha)` xấp xỉ `0.622`.

Training và evaluation normalize final embeddings trước scoring:

```text
h_norm = normalize(h, p=2)
score(u, i) = dot(h_user_norm, h_item_norm)
```

---

## Loss Objective

Notebook dùng:

```text
LOSS_TYPE = bpr_graph_text_align_degree_prior_gate_v2
```

Loss tổng:

```text
L = L_BPR + lambda_u * L_user_align + lambda_i * L_item_align + weight_decay
```

Trong đó `weight_decay` được xử lý bởi `AdamW`, không cộng thủ công vào tensor loss.

### Standard BPR

Notebook dùng standard BPR:

```text
u = normalize(user_emb)
pi = normalize(pos_emb)
ni = normalize(neg_emb)
s_pos = dot(u, pi)
s_neg = dot(u, ni)
L_BPR = mean(-logsigmoid(s_pos - s_neg))
```

Không có group weight trong BPR loss. Long-tail emphasis nằm ở gate, tail positive sampling, negative sampling và checkpoint/evaluation score.

### Graph-Text Alignment

Alignment dùng InfoNCE giữa graph view và text view của cùng node:

```text
z_g = normalize(z_graph)
z_t = normalize(z_text)
sim = z_g @ z_t.T / TEMPERATURE
labels = arange(batch_size)
L_align = cross_entropy(sim, labels)
```

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
ALIGN_SUBBATCH = tự chỉnh theo VRAM
```

### Alignment Warm-Up

Trong `ALIGN_WARMUP_EPOCHS = 3` epoch đầu:

```text
loss = alignment_loss
```

Sau warm-up:

```text
loss = bpr_loss + alignment_loss
```

---

## Sampling

### Positive Edge Sampling

Notebook train theo chunk:

```text
chunk_samples = CHUNK_STEPS * BATCH_SIZE
tail_samples = int(chunk_samples * TAIL_POSITIVE_SAMPLE_RATIO)
std_samples = chunk_samples - tail_samples
```

Với config hiện tại:

```text
80% random train edges
20% tail train edges
```

Tail train edges được xác định bằng:

```text
item_pop_group[edge_item_idx] == 2
```

Sau khi concat standard edges và tail edges, notebook shuffle lại trong chunk.

### Negative Sampling

Negative sampler:

```text
sample_negatives_uniform_warm(...)
```

Protocol:

```text
80% uniform warm items
20% Gold probability over warm items
```

Trong đó:

```text
warm_item = item_train_freq > 0
cold_item = item_train_freq == 0
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

Notebook sort toàn bộ train pair keys, sau đó dùng `torch.searchsorted` để phát hiện negative item nào thật ra là train positive của user. Conflict được resample bằng uniform warm item, tối đa:

```text
max_retries = 10
```

---

## Training Runtime

### GPU Batch Auto-Tuning

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

### Optimizer Và Scheduler

Optimizer:

```text
AdamW(fused=True nếu cuda)
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

### Graph Reuse Trong Training

Notebook assert:

```text
CHUNK_STEPS % ACCUM_STEPS == 0
```

Mỗi accumulation block:

1. Tính `z_G_all = model.forward_gcn_gated(...)`.
2. Split thành `user_G_all`, `item_G_all`.
3. Dùng lại graph representation cho các micro-batches trong block.
4. Xóa `z_G_all`, `user_G_all`, `item_G_all` sau optimizer step.

Số step mỗi epoch:

```text
steps_per_epoch = n_train_edges // BATCH_SIZE
```

Phần dư nhỏ hơn `BATCH_SIZE` không được dùng trong epoch đó.

`z_llm_all_cached` được refresh mỗi epoch vì:

```text
CACHE_REFRESH = 1
```

---

## Validation Protocol

Protocol:

```text
EVAL_PROTOCOL = warm_long_tail_v1
IGNORE_COLD_ITEMS = True
```

Candidate set:

```text
candidate_item_mask = item_train_freq > 0
```

Cold-start positives/candidates bị loại khỏi validation khi `IGNORE_COLD_ITEMS = True`.

### Fast Tail-Heavy Validation

Notebook tạo eval groups từ val edges theo `EVAL_STRAT_GROUPS`:

```text
HEAD: 5000
MID:  5000
TAIL: 20000
```

Mỗi group chứa:

```text
u
i
is_tail
is_cold
```

Nếu một group không có sample, notebook fallback sang `HEAD`.

Fast validation chạy mỗi:

```text
EVAL_EVERY = 2 epochs
```

Fast validation dùng để monitor, không chọn best checkpoint khi:

```text
USE_REPRESENTATIVE_FOR_BEST = True
```

### Representative Warm Validation

Representative validation sample:

```text
REP_VAL_N = 100000
REP_VAL_SEED = 2026
```

Sample này lấy random warm validation interactions và giữ phân phối HEAD/MID/TAIL tự nhiên. Nó chạy mỗi:

```text
REP_VAL_EVERY = 5 epochs
```

Khi `USE_REPRESENTATIVE_FOR_BEST = True`, representative validation là nguồn chọn best checkpoint.

### Evaluation Masking

Khi evaluate validation:

1. Compute full item scores:

```text
scores = user_final @ item_final.T
```

2. Block item ngoài candidate set:

```text
scores[:, cold_or_blocked_items] = -1e9
```

3. Mask train positives của user:

```text
scores[user_train_items] = -1e9
```

4. Restore score của ground-truth positive item trước khi tính rank:

```text
scores[row, ground_truth_item] = original_positive_score
```

### Validation Metrics

Notebook report:

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
TailCoverage@K = unique recommended tail candidate items / number of tail candidate items
TailShare@K = tail recommendations in top-K lists / total top-K recommendations
```

Popularity diagnostics:

```text
ListAvgPopularity@K:
  average train_freq over repeated recommendation positions

UniqueAvgPopularity@K:
  average train_freq over unique recommended items
```

---

## Checkpoint Selection

Fast monitor score:

```text
FAST_SCORE_TYPE = tail_monitor_v2
FastScore =
  0.70 * TailNDCG
+ 0.20 * TailRecall
+ 0.10 * TailCoverage
```

Checkpoint score:

```text
CHECKPOINT_SCORE_TYPE = weighted_hmean_warm_tail_overall_ndcg_v2
```

Weighted harmonic mean:

```text
WHM = (w_tail + w_overall) /
      (w_tail / TailNDCG + w_overall / OverallNDCG)
```

Weights:

```text
w_tail = 2.0
w_overall = 1.0
```

Guardrail:

```text
CHECKPOINT_BASELINE_OVERALL_NDCG = None
CHECKPOINT_OVERALL_GUARDRAIL_RATIO = 0.95
```

Khi baseline là `None`, overall guardrail bị disable. Nếu baseline được set, checkpoint chỉ eligible khi:

```text
OverallNDCG >= 0.95 * baseline_overall_ndcg
```

Early stopping:

```text
MIN_EPOCHS = 20
PATIENCE_REP = 4
```

Patience chỉ tăng khi representative checkpoint score không cải thiện sau `MIN_EPOCHS`, hoặc bị guardrail block.

---

## Checkpoint Và Cache Files

Checkpoint manager dùng `RUN_ID` để tạo file names.

Weights:

```text
weights/tarecmind_{RUN_ID}_best.pth
weights/tarecmind_{RUN_ID}_last.pth
```

History:

```text
data/training_history_{RUN_ID}.json
```

Projected text cache:

```text
data/z_llm_projected_{RUN_ID}_{TEXT_PROFILE_VERSION}_{TEXT_ENCODER_NAME}.pt
```

Eval group cache:

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

Text embedding cache:

```text
data/gold_item_embeddings_{TEXT_CACHE_KEY}.npy
data/gold_user_embeddings_{TEXT_CACHE_KEY}.npy
/content/recsys_cache/gold_item_embeddings_{TEXT_CACHE_KEY}.npy
/content/recsys_cache/gold_user_embeddings_{TEXT_CACHE_KEY}.npy
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

## Final Test Evaluation

Final test flow:

1. Load best checkpoint nếu `weights/tarecmind_{RUN_ID}_best.pth` tồn tại.
2. Load `silver/silver_test_ground_truth.parquet/`.
3. Load Gold user/item maps.
4. Map `reviewer_id -> user_idx`, `parent_asin -> item_idx`.
5. Drop rows không map được.
6. Giữ rows có `user_idx < num_users` và `item_idx < num_items`.
7. Nếu `IGNORE_COLD_ITEMS = True`, loại cold positives khỏi test.
8. Candidate set là warm items.
9. Nếu `MASK_VALIDATION_IN_TEST = True`, seen-item mask gồm train + validation interactions.

Test segmentation dùng Gold metadata:

```text
Item_HEAD
Item_MID
Item_TAIL
User_INACTIVE
User_ACTIVE
User_SUPER
```

Final representation:

```text
z_llm_all_final = update_llm_cache(...)
z_G_all = model.forward_gcn_gated(...)
item_final_all = normalize(final item repr) trên GPU
user_final_all = normalize(final user repr) trên CPU
```

Full-ranking test chạy:

```text
Ks = [20, 40]
user_batch = 128
```

Test metrics:

```text
OVERALL Recall@K / NDCG@K
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

Long-tail test score:

```text
LongTailTestScore@20 =
  0.60 * TailNDCG@20
+ 0.30 * TailRecall@20
+ 0.10 * TailCoverage@20
```

Final report:

```text
data/final_evaluation_report.json
```

Report fields:

```text
protocol
evaluation_type = full_ranking_only
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

## Hyperparameter Summary

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

## Pipeline Artifact Mapping

| Artifact | Vai trò trong notebook/model |
|---|---|
| `silver_item_text_profile.parquet/` | Nguồn `item_text` để encode item semantic embeddings |
| `silver_user_text_profile.parquet/` | Nguồn `user_text` để encode user semantic embeddings |
| `silver_val_ground_truth.parquet/` | Validation positives, build stratified/representative eval groups |
| `silver_test_ground_truth.parquet/` | Final full-ranking test positives |
| `gold_edge_index.npy` | Train user-item graph |
| `gold_item_train_freq.npy` | Warm/cold mask, item degree feature, popularity diagnostics |
| `gold_item_popularity_group.npy` | Tail edge sampling, stratified validation/test metrics |
| `gold_user_train_freq.npy` | User degree feature cho gate |
| `gold_user_activity_group.npy` | User activity segmentation trong final test |
| `gold_negative_sampling_prob.npy` | 20% Gold-prob component trong negative sampling |
| `gold_item_id_map.parquet` | Map `parent_asin` sang `item_idx`; giữ item order khi encode |
| `gold_user_id_map.parquet` | Map `reviewer_id` sang `user_idx`; giữ user order khi encode |

TA-RecMindV2 trong notebook này là **degree-aware intra-layer gated LightGCN với semantic text projection**, huấn luyện bằng **standard BPR + graph-text alignment**, dùng **tail positive oversampling**, **warm-only negative sampling**, và chọn checkpoint bằng **representative warm validation WHM giữa TailNDCG và OverallNDCG**.
