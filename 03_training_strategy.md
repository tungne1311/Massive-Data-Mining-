# Chiến Lược Huấn Luyện và Hàm Mất Mát

## Tổng Quan

Hàm mất mát của TA-RecMind là sự kết hợp của bốn thành phần, mỗi thành phần giải quyết một khía cạnh cụ thể của bài toán long-tail recommendation:

```
L_total = L_BPR + λ_1·(L^U_{align,tw} + L^I_{align,tw}) + λ_2·L_cl + β·Ω
```

| Thành Phần | Nguồn Gốc | Mục Tiêu |
|---|---|---|
| `Intra-Layer Fusion` | RecMind Eq.8 + TA-RecMind Bipartite | Trộn LLM Semantic và Graph Topology *trước* khi Message Pass |
| `L_BPR` | Rendle et al., UAI 2009 | Học preference ranking từ implicit feedback |
| `L^{U,I}_{align,tw}` | RecMind Eq.5 + Novel extension | Căn chỉnh semantic-collaborative space (ưu tiên tail) |
| `L_cl` | LAGCL (bài số 8) | Tăng cường đa dạng representation cho tail items |
| `Ω` | Chuẩn L2 | Regularization tránh overfitting |

---

## Thành Phần 1 — BPR Loss với Popularity-Debiased Negative Sampling

### BPR Loss

BPR (Bayesian Personalized Ranking, Rendle et al., 2009) là hàm mất mát chuẩn cho implicit feedback:

```
L_BPR = -Σ_{(u,i,j) ∈ O} log σ(s(u,i) - s(u,j))
```

Trong đó:
- `O = {(u, i, j) | i là positive item của u, j là negative item}`
- `s(u, i) = ⟨h_u, h_i⟩`: điểm dự đoán
- `σ`: hàm sigmoid

BPR tối ưu hóa để `s(u, i) > s(u, j)` — positive item được xếp hạng cao hơn negative item.

### Vấn Đề Với Uniform Negative Sampling

Khi lấy mẫu negative item `j` đồng đều ngẫu nhiên, head items xuất hiện thường xuyên hơn do chúng nhiều hơn trong pool (phân phối không đồng đều). Mô hình học để đẩy xuống head items — điều này không có nghĩa là tail items được đẩy lên.

### Popularity-Debiased Negative Sampling

Điều chỉnh xác suất chọn negative item (RecMind, Sec. IV-E):

```
P(j là negative) ∝ train_freq(j)^{-β_ns}
```

Với `β_ns = 0.75`:
- Head items (train_freq cao) có xác suất được chọn làm negative **thấp hơn**
- Tail items (train_freq thấp) có xác suất được chọn làm negative **cao hơn**
- Mô hình học cách phân biệt tail items lẫn nhau, không chỉ head vs head

**Normalize:** Xác suất được normalize về tổng = 1. Clip trong khoảng [ε, 1-ε] để tránh tail items bị chọn quá nhiều theo hướng ngược lại.

**Tính sẵn tại Gold layer:** `gold_negative_sampling_prob.npy` lưu xác suất đã được tính và normalize, training loop chỉ cần `np.random.choice(N_items, size=batch, p=sampling_prob)`.

**Tính sẵn tại Gold layer:** `gold_negative_sampling_prob.npy` lưu xác suất đã được tính và normalize, training loop chỉ cần `np.random.choice(N_items, size=batch, p=sampling_prob)`.

---

## Thành Phần 2 — Bipartite Intra-Layer Gate Fusion (Lõi Sức Mạnh)

Khác biệt một trời một vực so với phương pháp *Late Fusion* (chỉ nối Vector sau khi tính xong Graph), TA-RecMind và RecMind sử dụng **Intra-Layer Fusion** (Tích hợp sâu bên trong GNN).

**Node-wise Gate (Cổng cho từng Mode):**

Tại **Mỗi Layer $l$**, cho **Mỗi Node $v$** (thuộc User hoặc Item), mô hình tạo ra một cái cổng:
*"Dựa vào mức độ giàu cấu trúc của node $v$, tôi nên tin vào Đồ trúc GNN ($E_v^{(l)}$) hay tin vào Từ vựng LLM ($z_v^L$)?"*.

1. **Sinh Cổng:** Ghép nối vector hiện tại, vector LLM và đặc trưng tần suất (`w_freq_v * log1p(freq_v)`) đưa qua Sigmoid thành $\gamma_v^{(l)} \in [0, 1]$.
2. **Hòa Trộn Node:** $\hat{E}_v^{(l)} = \gamma_v^{(l)} E_v^{(l)} + (1 - \gamma_v^{(l)}) z_v^L$.
3. **Phân Tán (Message Passing):** Đem tensor **đã được trộn** $\hat{E}^{(l)}$ này đi nhân với ma trận kết nối $A$: $E^{(l+1)} = \hat{A} \cdot \hat{E}^{(l)}$.

Đây chính là chìa khóa chống tràn OOM và giải cứu Node Cold-Start vĩ đại nhất của công trình. Thay vì Model dùng Loss Align để "nới lỏng" không gian thì Cổng Gate này chủ động ép GNN phải làm quen với Text Semantic ngay trong quá trình xoay vòng Vector.

## Thành Phần 3 — Tail-Weighted Cross-Modal Alignment Loss (Đóng Góp Mới)

### Alignment Loss Gốc (RecMind, Eq. 5)

RecMind căn chỉnh hai không gian embedding bằng InfoNCE loss:

```
L^U_align = -(1/|B_U|) Σ_{u ∈ B_U} log [
  exp(cos(z^G_u, z^L_u) / τ) /
  Σ_{u' ∈ B_U} exp(cos(z^G_u, z^L_{u'}) / τ)
]
```

Đây là contrastive learning: mô hình học để `z^G_u` (collaborative) và `z^L_u` (semantic) của cùng một user gần nhau hơn so với `z^L_{u'}` của user khác trong batch.

### Hạn Chế của RecMind

RecMind áp dụng alignment loss đồng đều cho tất cả nodes. Với tail items có degree thấp, `z^G` ít thông tin (sparse collaborative signal). Gradient alignment loss cho tail items nhỏ hơn và mô hình không tập trung căn chỉnh chúng.

### Tail-Weighted Alignment (Đóng Góp Mới)

Thêm trọng số tỷ lệ nghịch với popularity vào alignment loss:

```
w_v = 1 / log(1 + train_freq(v))
```

```
L^U_{align,tw} = -(1/|B_U|) Σ_{u ∈ B_U} w_u · log [
  exp(cos(z^G_u, z^L_u) / τ) /
  Σ_{u' ∈ B_U} exp(cos(z^G_u, z^L_{u'}) / τ)
]
```

**Tính chất của hàm trọng số:**

| train_freq | w_v |
|---|---|
| 1 (tail cực đoan) | 1 / log(2) ≈ 1.44 |
| 5 (tail) | 1 / log(6) ≈ 0.56 |
| 100 (mid) | 1 / log(101) ≈ 0.22 |
| 1000 (head) | 1 / log(1001) ≈ 0.14 |
| 41183 (max head) | 1 / log(41184) ≈ 0.09 |

Gradient của alignment loss đối với tail items được nhân với `w_v` lớn hơn, buộc optimizer tập trung nhiều hơn vào việc căn chỉnh `z^G` và `z^L` cho tail items. Điều này trực tiếp giải quyết weakest link của RecMind khi áp dụng vào long-tail focused tasks.

**Tương tự cho item alignment:**

```
L^I_{align,tw} = -(1/|B_I|) Σ_{i ∈ B_I} w_i · log [
  exp(cos(z^G_i, z^L_i) / τ) /
  Σ_{i' ∈ B_I} exp(cos(z^G_i, z^L_{i'}) / τ)
]
```

---

## Thành Phần 4 — Long-tail Augmented Contrastive Loss

### Nguồn Gốc: LAGCL

LAGCL (bài số 8 trong danh sách tài liệu tham khảo) tạo hai augmented views của mỗi node bằng cách thêm nhiễu Gaussian có cường độ tỷ lệ nghịch với popularity:

```
h^{(l)'}_i  = h^{(l)}_i + Δ^{(l)'}_i
h^{(l)''}_i = h^{(l)}_i + Δ^{(l)''}_i
```

Trong đó cường độ của `Δ` tỷ lệ nghịch với `train_freq(i)`:
```
σ_noise(i) = σ_0 / (1 + log(1 + train_freq(i)))
```

Tail items nhận nhiễu mạnh hơn, tạo ra hai views khác biệt hơn, buộc mô hình học representation bền vững hơn.

### Contrastive Loss

```
L_cl = Σ_{u ∈ U} -log [
  exp(s(h'_u, h''_u) / τ) /
  Σ_{v ∈ U} exp(s(h'_u, h''_v) / τ)
]
```

Mô hình học để hai views của cùng một node gần nhau hơn so với views của nodes khác.

### Tại Sao LAGCL Tốt Hơn SimGCL Cho Mục Tiêu Này

SimGCL (được RecMind trích dẫn như Ref. [13]) thêm uniform noise cho tất cả nodes — không giải quyết bất cân xứng head/tail. LAGCL điều chỉnh cường độ nhiễu theo popularity, trực tiếp tăng cường representation đa dạng cho tail items.

---

## Hàm Mất Mát Tổng Hợp

### Công Thức Đầy Đủ

```
L_total = L_BPR + λ_1·(L^U_{align,tw} + L^I_{align,tw}) + λ_2·L_cl + β·Ω
```

**Nguồn gốc từng thành phần:**

| Ký Hiệu | Công Thức | Nguồn |
|---|---|---|
| `L_BPR` | BPR với popularity-debiased sampling | Rendle et al., 2009 |
| `L^U_{align,tw}` | InfoNCE với tail weights | RecMind Eq.5 + Novel |
| `L^I_{align,tw}` | Tương tự, cho items | RecMind Eq.5 + Novel |
| `L_cl` | LAGCL contrastive với noise ∝ 1/freq | LAGCL (bài số 8) |
| `Ω` | L2 regularization trên embedding | Chuẩn |

So sánh với RecMind gốc:
```
RecMind:   L = L_CF + λ(L^U_align + L^I_align) + βΩ        [Eq.10]
TA-RecMind: L = L_BPR + λ_1·(L^U_{align,tw} + L^I_{align,tw}) + λ_2·L_cl + β·Ω
```

Hai điểm thay đổi: (1) alignment loss được tail-weighted, (2) thêm LAGCL contrastive loss.

---

## Lịch Trình Huấn Luyện Hai Giai Đoạn

### Giai Đoạn 1 — Warm-up (Alignment Only)

**Mục tiêu:** Đưa `z^G` và `z^L` về cùng không gian trước khi joint training. Nếu bỏ qua bước này, Gate Fusion trong giai đoạn đầu hoạt động với hai không gian embedding không liên kết, gradient không ổn định.

```
Tối ưu: L_warmup = L^U_{align,tw} + L^I_{align,tw}
Epochs: 5–10 epochs
LR:     5e-4
```

Chỉ update: LoRA adapters, `W_proj`, gate parameters. Frozen: LightGCN embeddings.

### Giai Đoạn 2 — Joint Training

```
Tối ưu: L_total = L_BPR + λ_1·L_align,tw + λ_2·L_cl + β·Ω
Epochs: 50–100 epochs
LR:     1e-3 với cosine decay
```

Update tất cả parameters.

**Early stopping:** Theo `NDCG@10` trên validation set, patience = 10 epochs.

### Mini-batch Strategy

Với đồ thị 1.8M users × 1M items, full-batch training không khả thi. Sử dụng **neighbor sampling** (GraphSAGE style):

1. Sample `B` users làm seed nodes
2. Với mỗi seed user, sample 1-hop neighbors (items đã tương tác)
3. Với mỗi item, sample 1-hop neighbors (users đã tương tác) — chỉ cho L=2
4. Tạo mini-graph từ subgraph này

**Vấn đề với neighbor sampling và tail items:** Sampling ngẫu nhiên ưu tiên head items do phân phối bậc độ lệch. Cần **popularity-aware sampling**: tăng xác suất sample tail items làm seed trong mỗi batch.

---

## Giao Thức Khởi Tạo

### Khởi Tạo LightGCN Embeddings

Khởi tạo ngẫu nhiên `E^(0)` từ `N(0, 0.01)`. Với head items, có thể warm-start bằng cách chiếu LLM embedding về chiều `d`:

```
E^(0)_i = W_init · z^L_i    [chỉ cho L=0, fine-tune trong training]
```

### Khởi Tạo Gate Weights

Gate weight `w` khởi tạo nhỏ (`N(0, 0.001)`) để gate bắt đầu gần 0.5 — tức là ban đầu kết hợp đồng đều `z^G` và `z^L`, dần dần học cách điều chỉnh theo từng node.

---

## Vấn Đề Thực Tiễn và Giải Pháp

### OOM Trong Training

**Nguyên nhân thường gặp:**
- Broadcast bảng lớn vào tất cả workers
- Tích lũy gradient qua nhiều mini-batch trước khi zero_grad

**Giải pháp:**
- Gradient checkpointing trong LLM (nếu dùng LLaMA)
- Mixed precision training (fp16) cho LightGCN
- Giảm batch size và dùng gradient accumulation

### Training Instability

**Nguyên nhân:** Gradient từ L_cl và L_align có scale khác nhau so với L_BPR.

**Giải pháp:** Gradient clipping (`max_norm=1.0`), learning rate schedule riêng cho từng thành phần loss.

### Đánh Giá Trong Training

Tính Recall@20 và NDCG@20 sau mỗi epoch trên validation set. Phân tầng theo HEAD/MID/TAIL để phát hiện sớm nếu cải thiện tail đến từ chi phí của head.
