# Chiến Lược Huấn Luyện và Hàm Mất Mát

## Tổng Quan

Hàm mất mát của TA-RecMind là sự kết hợp của bốn thành phần, mỗi thành phần giải quyết một khía cạnh cụ thể của bài toán long-tail recommendation:

$$\mathcal{L}_{total} = \mathcal{L}_{BPR} + \lambda_1 \cdot (\mathcal{L}^U_{align,tw} + \mathcal{L}^I_{align,tw}) + \lambda_2 \cdot \mathcal{L}_{cl} + \beta \cdot \Omega$$

| Thành Phần | Nguồn Gốc | Mục Tiêu |
|---|---|---|
| `Intra-Layer Gate Fusion` | RecMind Eq.8 + TA-RecMind Bipartite Extension | Trộn LLM và Graph *trước* khi Message Pass |
| `L_BPR` | Rendle et al., UAI 2009 | Học preference ranking từ implicit feedback |
| `L^{U,I}_{align,tw}` | RecMind Eq.5 + Novel extension (tail weighting) | Căn chỉnh semantic-collaborative space, ưu tiên tail |
| `L_cl` | LAGCL | Tăng cường representation đa dạng cho tail items |
| `Ω` | L2 norm | Regularization tránh overfitting |

---

## Thành Phần 1 — BPR Loss với Popularity-Debiased Negative Sampling

### BPR Loss

BPR (Bayesian Personalized Ranking, Rendle et al., 2009) là hàm mất mát chuẩn cho implicit feedback:

$$\mathcal{L}_{BPR} = -\sum_{(u,i,j) \in \mathcal{O}} w(u,i) \cdot \log \sigma(s(u,i) - s(u,j))$$

Trong đó:
- `O = {(u, i, j) | i là positive item của u, j là negative item}`
- `s(u, i) = ⟨h_u, h_i⟩`: điểm dự đoán
- `w(u,i) = exp(-λ_t × (T_max - timestamp_ui) / T_range)`: temporal decay weight

**Ý nghĩa:** Gradient phạt mạnh tương tác gần đây (timestamp gần T_max) mà mô hình dự đoán sai — buộc model ưu tiên học từ hành vi mua hàng gần đây của user.

### Popularity-Debiased Negative Sampling

Khi lấy mẫu negative item `j` đồng đều, head items xuất hiện thường xuyên hơn do chúng nhiều hơn — mô hình học đẩy xuống head items nhưng không học phân biệt tail items.

**Giải pháp (RecMind, Sec. IV-E):**

$$P(j \text{ là negative}) \propto \text{train\_freq}(j)^{-\beta_{ns}}$$

Với `β_ns = 0.75`:
- Head items (train_freq cao) → xác suất được chọn làm negative **thấp hơn**
- Tail items (train_freq thấp) → xác suất được chọn làm negative **cao hơn**
- Mô hình buộc phải học phân biệt tail items lẫn nhau

**Hiện trạng trong XDMH mới nhất:** `gold_negative_sampling_prob.npy` đã upload nhưng **bị skip** — dùng Uniform Sampling kết hợp Margin BPR Loss thay thế, vì Margin BPR tự nhiên xử lý bias mà không méo phân phối lấy mẫu.

---

## Thành Phần 2 — Bipartite Intra-Layer Gate Fusion

Đây là **lõi sức mạnh** phân biệt TA-RecMind với mọi phương pháp Late Fusion.

### Node-wise Gate (Mỗi Layer, Mỗi Node)

Tại mỗi layer `l`, với mỗi node `v` (User hoặc Item):

**Bước 1 — Sinh Cổng:**
$$\gamma_v^{(l)} = \sigma\!\left(w_{base} + w_{sim} \cdot \cos(E_v^{(l)}, z_v^L) + w_{freq} \cdot \log(1 + \text{freq}_v)\right)$$

**Bước 2 — Hòa Trộn:**
$$\hat{E}_v^{(l)} = \gamma_v^{(l)} \cdot E_v^{(l)} + (1 - \gamma_v^{(l)}) \cdot z_v^L$$

**Bước 3 — Message Passing với Fused State:**
$$E^{(l+1)} = \hat{A} \cdot \hat{E}^{(l)}$$

### Tại Sao Intra-Layer Fusion Vượt Trội

| Phương pháp | Khi nào trộn | Hạn chế |
|---|---|---|
| Late Fusion | Sau khi GNN xong | Tail items đã bị GNN "bóp chết" trước khi LLM có cơ hội giúp |
| Early Fusion | Trước khi GNN | Mất đi cấu trúc propagation động |
| **Intra-Layer (TA-RecMind)** | Trong từng layer | Tail items mượn thông tin LLM ngay trong message passing |

### Bipartite Symmetry — Điểm Đặc Sắc Độc Quyền

RecMind gốc chỉ áp Gate cho Item. TA-RecMind mở rộng đối xứng cho cả **User Node**:

- User **INACTIVE** (`freq` thấp): Gate Graph đóng, Gate LLM mở → dùng User Text Profile (top-3 reviews)
- User **SUPER_ACTIVE** (`freq` cao): Gate Graph mở → tin vào Collaborative Filtering
- Điều này giải quyết cold-start **user** song song với cold-start **item**

---

## Thành Phần 3 — Tail-Weighted Cross-Modal Alignment Loss (Đóng Góp Mới)

### Alignment Loss Gốc (RecMind, Eq. 5)

InfoNCE loss căn chỉnh hai không gian embedding:

$$\mathcal{L}^U_{align} = -\frac{1}{|B_U|} \sum_{u \in B_U} \log \frac{\exp(\cos(z^G_u, z^L_u) / \tau)}{\sum_{u' \in B_U} \exp(\cos(z^G_u, z^L_{u'}) / \tau)}$$

Mô hình học để `z^G_u` (collaborative) và `z^L_u` (semantic) của cùng một user gần nhau hơn so với `z^L_{u'}` của user khác trong batch.

### Hạn Chế của RecMind

RecMind áp alignment loss đồng đều. Với tail items có degree thấp, `z^G` ít thông tin → gradient alignment nhỏ → mô hình không tập trung căn chỉnh tail items.

### Tail-Weighted Alignment (Đóng Góp Mới)

Thêm trọng số tỷ lệ nghịch với popularity:

$$w_v = \frac{1}{\log(1 + \text{train\_freq}(v))}$$

$$\mathcal{L}^U_{align,tw} = -\frac{1}{|B_U|} \sum_{u \in B_U} w_u \cdot \log \frac{\exp(\cos(z^G_u, z^L_u) / \tau)}{\sum_{u' \in B_U} \exp(\cos(z^G_u, z^L_{u'}) / \tau)}$$

**Tính chất của hàm trọng số:**

| train_freq | w_v |
|---|---|
| 1 (tail cực đoan) | 1/log(2) ≈ **1.44** |
| 5 (tail) | 1/log(6) ≈ 0.56 |
| 100 (mid) | 1/log(101) ≈ 0.22 |
| 1,000 (head) | 1/log(1001) ≈ 0.14 |
| 41,183 (max head) | 1/log(41184) ≈ **0.09** |

Gradient của alignment loss đối với tail items được nhân với `w_v` **lớn hơn 16 lần** so với max head — buộc optimizer tập trung căn chỉnh `z^G` và `z^L` cho tail items.

**Tương tự cho item alignment:**
$$\mathcal{L}^I_{align,tw} = -\frac{1}{|B_I|} \sum_{i \in B_I} w_i \cdot \log \frac{\exp(\cos(z^G_i, z^L_i) / \tau)}{\sum_{i' \in B_I} \exp(\cos(z^G_i, z^L_{i'}) / \tau)}$$

---

## Thành Phần 4 — Long-tail Augmented Contrastive Loss (LAGCL)

### Nguồn Gốc

LAGCL tạo hai augmented views của mỗi node bằng nhiễu Gaussian tỷ lệ nghịch với popularity:

$$h^{(l)'}_i = h^{(l)}_i + \Delta^{(l)'}_i, \quad h^{(l)''}_i = h^{(l)}_i + \Delta^{(l)''}_i$$

Cường độ nhiễu:
$$\sigma_{noise}(i) = \frac{\sigma_0}{1 + \log(1 + \text{train\_freq}(i))}$$

Tail items nhận nhiễu mạnh hơn → hai views khác biệt hơn → mô hình học representation bền vững hơn.

### Contrastive Loss

$$\mathcal{L}_{cl} = \sum_{u \in U} -\log \frac{\exp(s(h'_u, h''_u) / \tau)}{\sum_{v \in U} \exp(s(h'_u, h''_v) / \tau)}$$

### Tại Sao LAGCL Tốt Hơn SimGCL

SimGCL (RecMind Ref. [13]) thêm uniform noise cho tất cả nodes — không giải quyết bất cân xứng head/tail. LAGCL điều chỉnh cường độ nhiễu theo popularity, trực tiếp tăng cường representation đa dạng cho tail items.

---

## Hàm Mất Mát Tổng Hợp

### Công Thức Đầy Đủ

$$\mathcal{L}_{total} = \mathcal{L}_{BPR} + \lambda_1 \cdot (\mathcal{L}^U_{align,tw} + \mathcal{L}^I_{align,tw}) + \lambda_2 \cdot \mathcal{L}_{cl} + \beta \cdot \Omega$$

**Nguồn gốc từng thành phần:**

| Ký Hiệu | Công Thức | Nguồn |
|---|---|---|
| `L_BPR` | BPR + temporal decay + popularity-debiased sampling | Rendle et al., 2009 |
| `L^U_{align,tw}` | InfoNCE với tail weights | RecMind Eq.5 + **Novel** |
| `L^I_{align,tw}` | Tương tự, cho items | RecMind Eq.5 + **Novel** |
| `L_cl` | LAGCL contrastive với noise ∝ 1/freq | LAGCL |
| `Ω` | L2 regularization trên embeddings | Chuẩn |

**So sánh với RecMind gốc:**
```
RecMind:    L = L_CF + λ(L^U_align + L^I_align) + βΩ         [Eq.10]
TA-RecMind: L = L_BPR + λ₁·(L^U_align,tw + L^I_align,tw) + λ₂·L_cl + β·Ω
```

Hai điểm thay đổi: (1) alignment loss được tail-weighted, (2) thêm LAGCL contrastive loss.

---

## Lịch Trình Huấn Luyện Hai Giai Đoạn

### Giai Đoạn 1 — Warm-up (Alignment Only, ~5-10 epochs)

**Mục tiêu:** Đưa `z^G` và `z^L` về cùng không gian trước khi joint training. Nếu bỏ qua, Gate Fusion trong giai đoạn đầu hoạt động với hai không gian embedding không liên kết — gradient không ổn định.

```
Tối ưu:  L_warmup = L^U_{align,tw} + L^I_{align,tw}
Epochs:  5–10
LR:      5e-4
Update:  LoRA adapters, W_proj, gate parameters
Frozen:  LightGCN graph embeddings
```

### Giai Đoạn 2 — Joint Training (~50-100 epochs)

```
Tối ưu:  L_total = L_BPR + λ₁·L_align,tw + λ₂·L_cl + β·Ω
Epochs:  50–100
LR:      1e-3 với cosine decay
Update:  Tất cả parameters
```

**Early stopping:** Theo `NDCG@10` trên validation set, patience = 10 epochs.

### Mini-batch Strategy

Với đồ thị 1.8M users × 1M items, full-batch training không khả thi. Dùng **neighbor sampling** (GraphSAGE style):

1. Sample `B` users làm seed nodes
2. Với mỗi seed user, sample 1-hop neighbors (items đã tương tác)
3. Với mỗi item, sample 1-hop neighbors (users) — chỉ cho L=2
4. Tạo mini-graph từ subgraph này

**Popularity-aware seeding:** Tăng xác suất sample tail items/users làm seed trong mỗi batch để tránh head-dominated batches.

---

## Giao Thức Khởi Tạo

### LightGCN Embeddings

Khởi tạo ngẫu nhiên `E^(0)` từ `N(0, 0.01)`. Với head items, warm-start bằng cách chiếu LLM embedding:

```
E^(0)_i = W_init · z^L_i    [chỉ cho warm-start, fine-tune trong training]
```

### Gate Weights

Gate weight `w` khởi tạo nhỏ `N(0, 0.001)` → gate bắt đầu gần 0.5 (kết hợp đồng đều `z^G` và `z^L`), dần dần học điều chỉnh theo từng node.

---

## Vấn Đề Thực Tiễn và Giải Pháp

### OOM Trong Training

| Nguyên Nhân | Giải Pháp |
|---|---|
| Batch size quá lớn | Auto-scaling theo VRAM thực tế |
| LLM inference trong loop | Offline cache tất cả embeddings trước |
| Gradient tích lũy qua nhiều mini-batch | `zero_grad()` đúng thời điểm |
| LLaMA Scaling (Tương lai) | Dùng LoRA (r=8), gradient checkpointing |

### Training Instability

**Nguyên nhân:** Gradient từ `L_cl` và `L_align` có scale khác nhau so với `L_BPR`.

**Giải pháp:**
- Gradient clipping (`max_norm=1.0`)
- Learning rate schedule riêng từng thành phần loss
- Warm-up phase để ổn định không gian embedding trước

### Đánh Giá Trong Training

Mỗi epoch tính và log (phân tầng theo HEAD/MID/TAIL):
```
Overall:    Recall@20, NDCG@20, Recall@40, NDCG@40
HEAD:       Recall@20, NDCG@20
MID:        Recall@20, NDCG@20
TAIL:       Recall@20, NDCG@20, Coverage@20
Cold-start: Recall@20 (cold-start items only)
INACTIVE:   Recall@20 (users < 5 train interactions)
```

Phát hiện sớm nếu cải thiện tail đến từ chi phí của head — điều không thể chấp nhận.
