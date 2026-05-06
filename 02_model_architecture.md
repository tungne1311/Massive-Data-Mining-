# Kiến Trúc Mô Hình TA-RecMind

## Tổng Quan

TA-RecMind xây dựng trên nền tảng **RecMind (Xue et al., 2025)** với sứ mệnh: **"Gợi ý sản phẩm đuôi dài (Long-tail) và Khởi động lạnh (Cold-start) bằng cách dung hợp Tri thức Ngữ nghĩa từ LLM và Cấu trúc Đồ thị"**.

Đây là giải pháp triệt để xử lý nhược điểm cốt lõi của LightGCN: yếu kém khi đối mặt với sparsity và popularity bias. Kiến trúc gồm bốn thành phần hoạt động tuần tự: **LLM Embedding → LightGCN + Gate Fusion → Biểu diễn cuối → Re-ranking**.

```
                    ┌──────────────────────────────────┐
                    │     OFFLINE PRECOMPUTATION        │
                    │                                   │
  Item Text ───────►│  Sentence-Transformer            │──► z^L_i ∈ ℝ^d
  User Text ───────►│  (frozen + LoRA adapters)        │──► z^L_u ∈ ℝ^d
                    │  [cached as .npy arrays]          │
                    └──────────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │       TRAINING LOOP               │
                    │                                   │
  Graph G ─────────►│  LightGCN (L=2 layers)           │
                    │  For l = 0..L-1:                  │
                    │    γ_v^(l) = σ(w_base             │
                    │      + w_sim·cos(z^G_v, z^L_v)   │
                    │      + w_freq·log1p(freq_v))      │
                    │    Ê_v^(l) = γ·z^G + (1-γ)·z^L   │
                    │    E^(l+1) = Â · Ê^(l)  ← FUSED  │
                    │                                   │
                    │  z^G_v = avg(E^0..E^L)            │
                    │  h_v = α·z^G_v + (1-α)·z^L_v     │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │         INFERENCE                 │
                    │  s_model(u,i) = ⟨h_u, h_i⟩       │
                    │  s_adj(u,i) = s_model             │
                    │    - λ_penalty·log(1+train_freq_i)│
                    └──────────────────────────────────┘
```

---

## Thành Phần 1 — LLM Semantic Embedding

### Mục Tiêu

Tạo biểu diễn ngữ nghĩa `z^L_v ∈ ℝ^d` cho mỗi item và user từ văn bản. Đây là **nguồn thông tin duy nhất** cho cold-start items (degree = 0 trong đồ thị).

### Kiến Trúc LLM

| Mô Hình | Tham Số | Chiều | Tầm Quan Trọng | Trạng Thái |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | 384 | Hiệu năng cao, nhẹ, phù hợp pipeline hiện tại | **Hiện tại** |
| `all-mpnet-base-v2` | 109M | 768 | Cải thiện độ chính xác semantic | Mở rộng tương lai |
| `LLaMA-3.2-1B` + LoRA | 1B | 2048 | Khai thác ngữ nghĩa sâu và sentiment | Mở rộng tương lai |

**Công thức mã hóa (RecMind, Eq. 3–4):**
```
z̃^L_v = Pool(F(T_v; adapters))
z^L_v  = W_proj · z̃^L_v ∈ ℝ^d
```

Trong đó `F` là LLM với frozen weights, `adapters` là LoRA layers, `Pool` là mean pooling của non-padding tokens, `W_proj` là ma trận chiếu learnable.

**Ghi chú về LoRA:** LoRA (Hu et al., ICLR 2022) là kỹ thuật dự kiến áp dụng khi nâng cấp lên các mô hình lớn như LLaMA-3.2-1B. Kỹ thuật này phân rã `ΔW = AB` với `r=8`, giúp giảm tham số cần huấn luyện từ hàng tỷ xuống vài triệu, đảm bảo tính khả thi trên GPU 16GB. Trong phiên bản hiện tại dùng MiniLM, hệ thống tập trung vào việc huấn luyện lớp chiếu `W_proj`.

### Item Text Input

Ghép theo 4 cấp với field-aware token budget (xem `01_data_pipeline.md`):
```
[title] | [features_96tokens] | [categories_32tokens] | [description_64tokens] | [details_32tokens]
```

Ví dụ:
```
"Sony WH-1000XM5 Wireless Headphones | Industry-leading noise canceling |
 Electronics > Headphones > Over-Ear | Premium over-ear headphones with
 30-hour battery life | Brand: Sony | Material: Protein leather"
```

### User Text Input

Top-3 review snippets sắp xếp theo timestamp giảm dần (gần nhất trước):
```
"Excellent sound quality, worth every penny [SEP] Battery life is
 amazing, lasted 3 weeks [SEP] Noise canceling works perfectly"
```

**Hàm trọng số review:** `w(r) = 1 + log(1 + helpful_vote(r))`

### Cache Offline & Nạp Dữ Liệu Tối Ưu RAM

- Tất cả embeddings tính trước, lưu thành `.npy` arrays — trong training loop chỉ đọc từ cache, không gọi LLM
- Dữ liệu đồ thị (`gold_edge_index`, `neg_sampling_prob`) tải bằng `hf_hub_download` + `np.load()` — không dùng Pandas
- `neg_sampling_prob` normalize lại sau khi tải (`prob / prob.sum()`) để tránh lỗi `multinomial`
- Validate trước khi feed: `edge_index_raw.shape[1] > 0`, `len(item_train_freq) == num_items`

---

## Thành Phần 2 — Đồ Thị User-Item (Bipartite)

### Bipartite Graph

Biểu diễn bằng ma trận kề thưa chuẩn hóa đối xứng:
```
Â = D^{-1/2} A D^{-1/2}
```

- `A ∈ ℝ^{(N_u + N_i) × (N_u + N_i)}`: ma trận kề bipartite
- `D`: ma trận đường chéo degree
- Chuẩn hóa đối xứng đảm bảo head items không khuếch đại gradient bất cân xứng

**Thiết kế đặc biệt:** Tuyệt đối không đưa Edge Weight vào `Â`. LightGCN giữ nguyên Binary Unweighted Edge Index. Mọi tương tác nhiễu (rating < 3.0) đã bị lọc từ tầng Bronze.

**Temporal Decay & Rating:** Tích hợp vào **BPR Loss** thay vì ma trận kề:
```
w(u,i) = exp(-λ_t × (T_max - timestamp) / T_range)
Loss_BPR(u,i,j) = -w(u,i) · log(σ(ŷ_ui - ŷ_uj))
```

### Thống Kê Đồ Thị (từ EDA)

| Chỉ Số | Giá Trị |
|---|---|
| N_users | 1,847,662 |
| N_items (train) | 1,042,121 |
| N_edges (train) | 1,396,428 |
| Sparsity | 99.9993% |
| Max degree (Item) | 41,183 |
| Median degree (Item) | 2 |
| Max degree (User) | 1,005 |
| Median degree (User) | 5 |

**Nhận xét:** Chênh lệch Median=2 vs Max=41,183 của Item Node chứng minh "head thống trị". Đây là lý do Gate Fusion và Tail-Weighted Loss ra đời.

---

## Thành Phần 3 — LightGCN + Intra-Layer Gate Fusion

### LightGCN Propagation

LightGCN (He et al., 2020) loại bỏ feature transformation và activation function — chỉ giữ neighborhood aggregation:

```
E^(l+1) = Â · E^(l)

Tường minh:
  e^(l+1)_u = Σ_{i ∈ N_u} (1/√|N_u|·√|N_i|) · e^(l)_i
  e^(l+1)_i = Σ_{u ∈ N_i} (1/√|N_i|·√|N_u|) · e^(l)_u

Embedding cuối cùng (trung bình qua các tầng):
  z^G_v = (1/(L+1)) Σ^L_{l=0} E^(l)_v
```

**Số tầng L:** L=2 được khuyến nghị. L=3 có thể gây over-smoothing khi thêm item-item edges.

### Cơ Chế Intra-Layer Gate Fusion (Lõi Đột Phá)

Khác với Late Fusion (chỉ nối vector sau khi tính xong Graph), TA-RecMind sử dụng **Intra-Layer Fusion** — trộn bên trong mỗi layer GNN:

**Bước 1 — Sinh Cổng (Node-wise, Per-Layer):**
$$\gamma_v^{(l)} = \sigma(w_{base} + w_{sim} \cdot \cos(z_v^G, z_v^L) + w_{freq} \cdot \log(1 + freq_v))$$

**Bước 2 — Hòa Trộn Node:**
$$\hat{E}_v^{(l)} = \gamma_v^{(l)} \cdot E_v^{(l)} + (1 - \gamma_v^{(l)}) \cdot z_v^L$$

**Bước 3 — Message Passing với Fused State:**
$$E^{(l+1)} = \hat{A} \cdot \hat{E}^{(l)}$$

Cài đặt trong `TA_RecMind_V2_IntraLayer.ipynb`:
```python
base_gate = w_g + w_t   # w_g: graph weight, w_t: text weight
gamma_v = sigmoid(base_gate + w_sim * cos_sim_v + w_freq * log1p(freq_v))
fused_v = gamma_v * graph_emb_v + (1 - gamma_v) * llm_emb_v
```

**Bipartite Symmetry (Điểm Độc Quyền TA-RecMind):** Gate áp dụng đối xứng cho cả User Node lẫn Item Node — RecMind gốc chỉ cho Item.

**Tại sao xuất sắc?**
- **Head Node** (freq lớn): `log1p(freq)` lớn → `γ` tiến về 1 → tin vào Collaborative Filtering (LightGCN)
- **Tail/Cold-start Node** (freq nhỏ/= 0): `γ` sụt giảm mạnh → Gate Graph bị khóa, Gate LLM mở toang → hút sức mạnh Semantic Embedding
- **Yếu tố `cos_sim`**: Tự động "hòa giải" khi GNN và LLM bất đồng quan điểm

### Biểu Diễn Cuối Cùng

```
h_v = α · z^G_v + (1 - α) · z^L_v      [α = 0.6 gợi ý ban đầu]

Điểm số dự đoán:
s(u, i) = ⟨h_u, h_i⟩   [inner product]
```

### Ablation Study Gates (RecMind Table II)

```
Full model:               Recall@20 = 0.1260
w/o Alignment-Items:      Recall@20 = 0.0974  (-22.7%)
w/o Gate Fusion:          Recall@20 = 0.1087  (-13.7%)
w/o Tail-weighted (ours): TBD — đây là contribution mới của TA-RecMind
```

---

## Thành Phần 4 — Popularity-Penalized Re-ranking

### Mục Tiêu

Điều chỉnh phân phối gợi ý tại inference time để tăng tỷ lệ tail items trong top-K **mà không train lại mô hình**.

### Two-Stage Inference

**Stage 1 — Retrieval (top-200 candidates):**
```
scores = h_u @ H_items.T        [matrix multiply — 1 vector × N_items vectors]
top_200 = topk(scores, 200)     [FAISS IndexFlatIP: < 10ms với 1M items]
```

**Stage 2 — Re-ranking với Popularity Penalty:**
```
s_adjusted(u, i) = s_model(u, i) - λ_penalty · log(1 + train_freq(i))
top_K = topk(s_adjusted, K)
```

**Diễn giải:** `log(1 + train_freq)` lớn với head items → nhân với `λ_penalty > 0` và trừ đi → hạ điểm head items, đẩy tail items lên cao hơn.

**Tune `λ_penalty`:** Tăng dần từ 0 đến 1 trên validation set, quan sát tỷ lệ TAIL items trong top-20. Chọn giá trị cân bằng tốt nhất giữa Tail Recall@20 và Overall Recall@20.

**Lợi thế trong Demo:** Người dùng kéo slider `λ_penalty` → thấy ngay sự thay đổi phân phối Head/Mid/Tail trong recommendations — trực quan hóa tác động của từng thành phần.

---

## Đồ Thị Item-Item Ngữ Nghĩa (Tùy Chọn — Chưa Triển Khai)

Thêm cạnh item-item dựa trên cosine similarity giữa LLM embeddings:
```
sim(i, j) = (z^L_i · z^L_j) / (||z^L_i|| · ||z^L_j||)
```

Thêm cạnh khi `sim(i, j) > θ = 0.7` với K = 5-10 nearest neighbors. Trọng số cạnh `β ∈ [0.3, 0.5]`.

**Mục tiêu:** Tail items ít user neighbors trong đồ thị bipartite → mượn tín hiệu từ head items ngữ nghĩa tương tự qua cạnh item-item.

**Cảnh báo over-smoothing:** Giới hạn K ≤ 10 và θ ≥ 0.7 để tránh embeddings hội tụ về giá trị trung bình.

---

## Hyperparameters Tổng Hợp

| Hyperparameter | Giá Trị | Ghi Chú |
|---|---|---|
| Embedding dim `d` (`EMBED_DIM`) | 128 | Tăng từ 64 để cải thiện sức chứa ngữ nghĩa |
| LightGCN layers `L` | 2 | Cân bằng tín hiệu, tránh over-smoothing |
| Learning rate (Warmup) | 5e-4 | Phase 1: chỉ train LoRA + W_proj + gate |
| Learning rate (Joint) | 1e-3 | Phase 2: train tất cả params |
| Weight Decay | 1e-4 | L2 regularization |
| Batch size | Auto-scaled | A100: 8192 / V100: 6144 / T4: 2048 |
| Gradient Accumulation | 4 | Mô phỏng batch 8192 trên T4 |
| Align Sub-batch | Auto-scaled | A100: 1024 / V100: 768 / T4: 512 |
| λ_1 (`LAMBDA_ALIGN`) | 0.2 | Trọng số alignment loss |
| λ_2 (`LAMBDA_CL`) | 0.1 | Trọng số LAGCL contrastive loss |
| τ (Temperature) | 0.15 | InfoNCE/Contrastive temperature |
| Noise Scale | 0.1 | LAGCL augmentation noise |
| β_ns (negative sampling) | 0.75 | Tune trong {0.5, 0.75, 1.0} |
| λ_penalty (re-ranking) | 0.3 | Tune trên val set |
| λ_t (temporal decay) | 1.0 | Tune trong {0.5, 1.0, 2.0} |
| α (final fusion) | 0.6 | Tune trong [0.4, 0.8] |

### Tự Động Điều Chỉnh Theo Phần Cứng

```python
# Trong TA_RecMind_V2_IntraLayer.ipynb
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
if vram >= 35:     # A100
    BATCH_SIZE, ALIGN_SUBBATCH, GRAD_ACCUM = 8192, 1024, 1
elif vram >= 20:   # V100/L4
    BATCH_SIZE, ALIGN_SUBBATCH, GRAD_ACCUM = 6144, 768, 2
else:              # T4/K80 16GB
    BATCH_SIZE, ALIGN_SUBBATCH, GRAD_ACCUM = 2048, 512, 4
```

### ColabCheckpointManager & Khôi Phục Liên Tục

- **Lưu Kép:** `tarecmind_best.pth` (Best Loss/Metrics) + `tarecmind_last.pth` (auto-resume)
- **History Syncing:** `training_history.json` merge với bản trên Drive (`current_on_drive`), bù trừ epoch thiếu, xếp lại theo thời gian — bảo vệ log kể cả khi browser tắt giữa chừng
