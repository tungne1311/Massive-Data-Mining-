# Kiến Trúc Mô Hình TA-RecMind

## Tổng Quan

TA-RecMind xây dựng trên nền tảng RecMind (Xue et al., 2025) với sứ mệnh: **"Gợi ý sản phẩm đuôi dài (Long-tail) và Khởi động lạnh (Cold-start) bằng cách dung hợp Tri thức Ngữ nghĩa từ LLM và Cấu trúc Đồ thị"**. 

Đây là giải pháp triệt để xử lý nhược điểm cốt lõi của LightGCN (và GNN nói chung): sự yếu kém khi đối mặt với dữ liệu thưa thớt (sparsity) và thiên lệch phổ biến (popularity bias). Kiến trúc gồm bốn thành phần hoạt động tuần tự: LLM Embedding → LightGCN → Dynamic Gate Fusion → Re-ranking.

```
                    ┌─────────────────────────────────┐
                    │     OFFLINE PRECOMPUTATION       │
                    │                                  │
  Item Text ───────►│  Sentence-Transformer           │──► z^L_i ∈ ℝ^d
  User Text ───────►│  (frozen + LoRA adapters)       │──► z^L_u ∈ ℝ^d
                    │  [cached as numpy arrays]        │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │       TRAINING LOOP              │
                    │                                  │
  Graph G ─────────►│  LightGCN (L=2 layers)          │──► z^G_v ∈ ℝ^d
                    │  E^(l+1) = Â · Ê^(l)            │
                    │                                  │
                    │  Gate Fusion (per node, per layer)│
                    │  γ_v^{(l)} = σ(w_base + w_sim·cos + 
                    │          w_freq_i·log1p(freq_i) +
                    │          w_freq_u·log1p(freq_u))  │
                    │  Ê_v^{(l)} = γ_v·z^G_v + (1-γ_v)·z^L_v│
                    │                                  │
                    │  Final: h_v = α·z^G + (1-α)·z^L │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │         INFERENCE                │
                    │                                  │
                    │  s_model(u,i) = ⟨h_u, h_i⟩      │
                    │  s_adj(u,i) = s_model            │
                    │             - λ·log(1+train_freq)│
                    └─────────────────────────────────┘
```

---

## Thành Phần 1 — LLM Semantic Embedding

### Mục Tiêu

Tạo biểu diễn ngữ nghĩa `z^L_v ∈ ℝ^d` cho mỗi item và user từ văn bản. Đây là nguồn thông tin duy nhất cho **cold-start items** (không có lịch sử tương tác trong train).

### Kiến Trúc LLM

| Mô Hình | Tham Số | Chiều | Môi Trường | Định Vị Lộ Trình |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | 384 | CPU/GPU | **Phase 1 (Thử nghiệm):** Xây dựng pipeline hoàn chỉnh, debug luồng LightGCN $\rightarrow$ Gate Fusion $\rightarrow$ Re-ranking. |
| `all-mpnet-base-v2` | 109M | 768 | GPU 8GB+ | Mở rộng Phase 1. |
| `LLaMA-3.2-1B` + LoRA | 1B (train 5M) | 2048 | GPU 16GB | **Phase 2 (Đột phá/Bảo vệ đề tài):** Nhờ việc không xóa biểu tượng, emoji đánh giá và dấu câu, LLM có thể khai thác cấu trúc tính từ phức tạp và sentiment. |

**Công thức mã hóa (RecMind, Eq. 3–4):**

```
z̃^L_v = Pool(F(T_v; adapters))
z^L_v  = W_proj · z̃^L_v ∈ ℝ^d
```

Trong đó:
- `F`: LLM với frozen weights (chỉ LoRA adapters được train)
- `adapters`: LoRA layers thêm vào attention + MLP blocks
- `Pool`: pooling trên hidden states (thường là mean pooling của non-padding tokens)
- `W_proj`: ma trận chiếu về chiều `d` của GNN (learnable)

**Tại sao LoRA thay vì full fine-tuning?**

LoRA (Hu et al., ICLR 2022) phân rã ma trận cập nhật thành tích hai ma trận hạng thấp: `ΔW = AB` với `A ∈ ℝ^{m×r}`, `B ∈ ℝ^{r×n}`, `r << min(m,n)`. Với `r=8`, số tham số huấn luyện giảm từ hàng tỷ xuống vài triệu — quyết định tính khả thi trên môi trường GPU hạn chế.

### Item Text Input

Ghép theo 4 cấp độ với field-aware token budget (xem chi tiết tại `01_data_pipeline.md`):

```
[title] | [features_64tokens] | [categories_32tokens] | [description_96tokens] | [details_32tokens]
```

Ví dụ:
```
"Sony WH-1000XM5 Wireless Headphones | Industry-leading noise canceling |
 Electronics > Headphones > Over-Ear | Premium over-ear headphones with
 30-hour battery life | Brand: Sony | Material: Protein leather"
```

### User Text Input

Top-5 review snippets, sắp xếp theo trọng số chất lượng giảm dần:

```
"Excellent sound quality, worth every penny [SEP] Battery life is
 amazing, lasted 3 weeks [SEP] Noise canceling works perfectly [SEP] ..."
```

**Hàm trọng số review:**
```
w(r) = 1 + log(1 + helpful_vote(r))
```

### Cache Offline và Nạp Dữ Liệu Tối Ưu RAM

Toàn bộ embeddings được tính trước và lưu thành numpy arrays. Trong training loop, chỉ đọc từ cache — không gọi LLM. Điều này đảm bảo training nhanh và không bị bottleneck bởi LLM inference. (Lưu ý: Trong hệ thống Data Pipeline hiện tại, bước offline embedding này được bỏ qua khi chạy Docker nội bộ để tiết kiệm tài nguyên, và được khuyến nghị chạy trực tiếp trên môi trường có GPU rời như Colab/Kaggle).

**Chiến Lược Nạp Dữ Liệu Đồ Thị (Tối Ưu RAM):**
- Trong mô hình chính thức, dữ liệu đồ thị (như ma trận `gold_edge_index`, mảng `neg_sampling_prob` và tần suất lượt xuất hiện) được nạp trực tiếp qua `hf_hub_download` và `numpy.load()` từ hệ thống cache cục bộ để bảo vệ RAM. Không sử dụng các biến sao chép trung gian.
- **Tính chuẩn xác lấy mẫu âm:** Để đảm bảo ổn định toán học, mảng `neg_sampling_prob` sau khi được nạp từ tệp `.npy` sẽ được chuẩn hóa lại thành mức tổng chính xác bằng 1 (`prob = prob / prob.sum()`) nhằm loại bỏ các lỗi do hàm `multinomial` gây ra đối với chuỗi dấu phẩy động.
- **Tính toàn vẹn Dữ Liệu:** Script nạp dữ liệu chạy xác thực tính toàn vẹn (ví dụ: `edge_index_raw.shape[1] > 0`, `len(item_train_freq) == num_items`) trước khi feed tới graph embeddings.

---

## Thành Phần 2 — Đồ Thị Dị Thể (Heterogeneous Graph)

### Đồ Thị User-Item (Bipartite)

Biểu diễn bằng ma trận kề thưa chuẩn hóa đối xứng (LightGCN, He et al., 2020):

```
Â = D^{-1/2} A D^{-1/2}
```

Trong đó:
- `A ∈ ℝ^{(N_u + N_i) × (N_u + N_i)}`: ma trận kề bipartite
- `D`: ma trận đường chéo của bậc nút (degree matrix)
- Chuẩn hóa đối xứng đảm bảo head items không khuếch đại gradient bất cân xứng

**Tương phản ngôn ngữ - đồ thị (Contrastive Alignment Loss):**
Ngoài BPR Loss chính, sử dụng thêm hàm Loss phụ (InfoNCE) kết hợp $\lambda_1$ và $\lambda_2$ để ép hai biểu diễn (`z^G` và `z^L`) tiến lại gần nhau trong không gian vector.

**Thiết kế đồ thị đặc biệt:** Tuyệt đối không đưa Edge Weight vào ma trận $\hat{A}$. Hệ thống dữ liệu không sinh mảng Weight, LightGCN được giữ nguyên mẫu Binary Unweighted Edge Index để chia sẻ thông điệp Topology chuẩn xác. Mọi tương tác nhiễu (rating < 3.0) đã được trảm triệt để từ tầng Bronze.

**Trọng số thời gian (Temporal Decay) & Rating:**
Thay vì gắn vào ma trận kề, các giá trị này được tích hợp mạnh vào **BPR Loss**. Gradient phạt sẽ lớn khi tương tác gần đây (T_max - timestamp nhỏ) hoặc có rating cao (5 sao) mà mô hình dự đoán sai:
```
Loss_{BPR}(u, i, j) = - w(u,i) \cdot \log(\sigma(\hat{y}_{ui} - \hat{y}_{uj}))
```
Trong đó $w(u,i) = \exp(-\lambda_t \times (T_{max} - timestamp) / T_{range})$

**Chiều Âm/Dương trong Negative Sampling:** Lấy mẫu Âm dựa trên tỷ lệ nghịch với phổ biến (popularity penalty) hoặc tỷ lệ thuận $\propto \text{popularity}(i)^{0.75}$ để xử lý thiên lệch, bắt buộc mô hình phải học để đẩy các Item quá phổ biến xuống.

**Thống kê đồ thị (từ EDA):**

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

**Nhận xét quan trọng từ EDA:** Sự chênh lệch khổng lồ giữa median (2) và max (41,183) degree của Item Node chứng minh hiện tượng "head thống trị". Các nút max degree tạo ra lượng gradient khổng lồ lấn át tail items trong quá trình LightGCN propagation — đây là lý do cần Gate Fusion và Tail-Weighted Loss.

---

## Thành Phần 3 — LightGCN + Gate Fusion

### LightGCN Propagation

LightGCN (He et al., 2020) loại bỏ feature transformation và activation function khỏi GCN chuẩn, chỉ giữ lại neighborhood aggregation:

**Công thức propagation (RecMind, Eq. 1, dẫn từ LightGCN):**
```
E^(l+1) = Â · E^(l)
```

Tường minh hơn, tại tầng l+1:
```
e^(l+1)_u = Σ_{i ∈ N_u} (1 / √|N_u| · √|N_i|) · e^(l)_i    [LightGCN Eq.3]
e^(l+1)_i = Σ_{u ∈ N_i} (1 / √|N_i| · √|N_u|) · e^(l)_u
```

**Embedding cuối cùng (trung bình qua các tầng):**
```
z^G_v = (1 / (L+1)) Σ^L_{l=0} E^(l)_v    [LightGCN; RecMind Eq.2]
```

Trung bình qua các tầng giữ lại thông tin ở nhiều mức độ tổng quát hóa khác nhau và thực nghiệm cho thấy hiệu quả hơn chỉ dùng tầng cuối.

**Số tầng L:** L=2 được khuyến nghị. L=3 (LightGCN gốc) có thể gây over-smoothing khi thêm item-item edges.

### Cơ Chế Cổng Động Dành Cho Nút Dài (Dynamic Gate Fusion Khởi Tiến)

Cấu trúc cốt lõi quyết định tính đột phá của TA-RecMind nằm ở công thức lai ghép (fusion) tinh vi để học động cách bù đắp cấu trúc đồ thị từ văn bản ngữ nghĩa LLM:

$$\gamma_v^{(l)} = \sigma(w_{base} + w_{sim} \cdot \cos(z_v^G, z_v^L) + w_{freq} \cdot \log(1 + freq_v))$$

Với công thức trong `XDMH.ipynb` (Dấu ấn kỹ thuật **độc quyền** của TA-RecMind so với RecMind gốc) tạo sự cân đối Bipartite Symmetry, hỗ trợ tính cho cả User lẫn Item:
```python
# Tần suất của node hiện tại (user hoặc item)
base_gate = w_g + w_t
gamma_v = sigmoid( base_gate + w_sim * cos_sim_v + w_freq * log1p(freq_v) )
```

**Tại sao nó xuất sắc?** 
Dựa vào thuyết kiến trúc Bipartite Gate Symmetry (Đồng nhất cửa kết hợp), mô hình tích hợp tín hiệu động cả User và Item:
- Với các **Head Node** ($freq$ lớn): Nhờ `log1p(freq)`, $\gamma^{(l)}$ tiến về 1. Mô hình dung nạp trọn vẹn sức mạnh Collaborative Filtering của Đồ thị (LightGCN).
- Với các **Tail/Cold-start Node** ($freq$ nhỏ hoặc bằng 0): $\gamma^{(l)}$ sụt giảm mạnh. Cửa Graph bị khóa lại để chặn luồng Noise Topology (do chúng quá ít liên kết), đồng thời Cửa LLM mở toang để hút sức mạnh Semantic Embedding.
- Yếu tố `cos_sim` (với tham số $w_{sim}$): Giúp tự động "hòa giải" và cân đối trọng số khi có sự bất đồng quan điểm giữa thông tin từ mạng GNN hành vi và Profile text.

**Trạng thái fused (Tích hợp lại mạng):**
```
Ê^(l)_v = γ^{(l)} * z^G_v + (1 - γ^{(l)}) * z^L_v
```

**Cập nhật với fused state (RecMind, Eq. 8):**
```
E^(l+1) = Â · Ê^(l)
```

Điểm khác biệt so với LightGCN gốc: thay vì propagate `E^(l)` nguyên bản, propagate `Ê^(l)` đã được làm giàu bằng semantic signal. Điều này cho phép tail items "mượn" thông tin ngữ nghĩa từ neighbor head items thông qua message passing.

**Biểu diễn cuối cùng (RecMind, Eq. 9):**
```
h_v = α · z^G_v + (1 - α) · z^L_v
```

`α` là hyperparameter cần tune (gợi ý khởi đầu: α=0.6).

**Điểm số dự đoán:**
```
s(u, i) = ⟨h_u, h_i⟩   [inner product]
```

### Ablation Study Gates

Kết quả từ RecMind Table II (Xue et al., 2025):
```
Full model:                  Recall@20 = 0.1260
w/o Alignment-Items:         Recall@20 = 0.0974  (-22.7%)
w/o Gate Fusion:             Recall@20 = 0.1087  (-13.7%)
w/o Tail-weighted (ours):    TBD (đây là contribution mới)
```

Kết quả ablation cho thấy Gate Fusion đặc biệt quan trọng với cold/long-tail items.

---

## Thành Phần 4 — Popularity-Penalized Re-ranking

### Mục Tiêu

Điều chỉnh phân phối gợi ý tại inference time để tăng tỷ lệ tail items trong top-K mà không train lại mô hình.

### Two-Stage Inference

**Stage 1 — Retrieval (top-N candidates):**
```
Lấy top-200 items theo s_model(u, i) = ⟨h_u, h_i⟩
```
Sử dụng FAISS approximate nearest neighbor để tìm top-200 nhanh chóng (< 10ms với 1M items).

**Stage 2 — Re-ranking với popularity penalty:**
```
s_adjusted(u, i) = s_model(u, i) - λ_penalty · log(1 + train_freq(i))
```

Trả về top-K theo `s_adjusted`.

**Diễn giải:** log(1 + train_freq) lớn với head items (nhiều tương tác) và nhỏ với tail items (ít tương tác). Nhân với `λ_penalty > 0` và trừ đi sẽ hạ điểm head items và đẩy tail items lên cao hơn trong bảng xếp hạng.

**Tune `λ_penalty`:** Trên validation set, tăng dần `λ_penalty` từ 0 đến 1, quan sát tỷ lệ TAIL items trong top-20 và Tail Recall@20. Chọn `λ_penalty` cân bằng tốt nhất giữa hai chỉ số.

### Lợi Thế của Re-ranking Approach

Không cần train lại mô hình để thay đổi mức độ "tail promotion". Trong web demo, người dùng có thể kéo slider `λ_penalty` và thấy ngay sự thay đổi trong phân phối gợi ý — trực quan hóa tác động của từng thành phần.

---

## Đồ Thị Item-Item Ngữ Nghĩa (Tùy Chọn)

Ngoài đồ thị bipartite user-item, có thể thêm cạnh item-item dựa trên cosine similarity giữa LLM embeddings:

```
sim(i, j) = (z^L_i · z^L_j) / (||z^L_i|| · ||z^L_j||)
```

Thêm cạnh khi `sim(i, j) > θ` với `θ = 0.7` và K từ 5 đến 10 nearest neighbors.

**Mục tiêu:** Tail items không có nhiều user neighbors trong đồ thị bipartite, nhưng có thể có item neighbors ngữ nghĩa (items tương tự về nội dung). Cạnh này cho phép tail items "mượn" tín hiệu từ head items liên quan về mặt ngữ nghĩa.

**Trọng số cạnh item-item:** `β ∈ [0.3, 0.5]`, nhỏ hơn trọng số cạnh user-item để tránh semantic edges ghi đè tín hiệu collaborative.

**Lưu ý về over-smoothing (LGCF):** Khi thêm quá nhiều cạnh, embeddings hội tụ về giá trị trung bình, mất đi tính phân biệt cá thể. Giới hạn K ≤ 10 và `θ ≥ 0.7` để kiểm soát.

---

## Hyperparameters Tổng Hợp (Dựa trên Triển Khai XDMH.ipynb)

| Hyperparameter | Giá Trị Cập Nhật | Ghi Chú |
|---|---|---|
| Embedding dim `d` (`EMBED_DIM`) | 128 | Đã tăng lên 128 để cải thiện sức chứa ngữ nghĩa |
| LightGCN layers `L` | 2 | Khuyến nghị để cân bằng tín hiệu đồ thị |
| Learning rate (Warmup) | 5e-4 | Tốc độ học áp dụng tại Warmup Phase |
| Learning rate (Joint) | 1e-3 | Tốc độ học ở giai đoạn Joint Training chính |
| Weight Decay | 1e-4 | Tránh overfitting |
| Batch size | GPU Auto-scaled | A100: 8192, V100: 6144, T4: 2048 (Kèm Gradient Accumulation = 4) |
| Align Sub-batch (`ALIGN_SUBBATCH`) | (Auto-scaled) | A100: 1024, V100: 768, T4: 512 (Tính Alignment Loss) |
| λ_1 (Alignment weight - `LAMBDA_ALIGN`) | 0.2 | Trọng số cho hàm mất mát căn chỉnh ngôn ngữ-đồ thị |
| λ_2 (Contrastive weight - `LAMBDA_CL`)| 0.1 | Trọng số áp dụng cho LAGCL |
| τ (Temperature) | 0.15 | Nhiệt độ của InfoNCE/Contrastive Loss |
| Noise Scale | 0.1 | Độ nhiễu cho Contrastive Augmentation |
| β_ns (negative sampling) | 0.75 | {0.5, 0.75, 1.0} |
| λ_penalty (re-ranking) | 0.3 | Tune trên val set |
| λ_t (temporal decay) | 1.0 | {0.5, 1.0, 2.0} |

### Tự Động Điều Chỉnh Theo Phần Cứng (Auto-Scaling)

Mô hình tự động điều tiết `Batch Size`, `Align Sub-batch`, `Gradient Accumulation` dựa trên VRAM thực tế để tối đa hóa hiệu suất GPU và phòng chống OOM (Out Of Memory):
- **T4/K80 (16GB VRAM):** Sử dụng `Batch Size` nhỏ (2048) nhưng bù lại bằng Gradient Accumulation Steps lớn (4) để mô phỏng tương đương batch lớn 8192 mà không tràn RAM.
- **A100 (40GB+):** Chạy max capacity với batch thẳng lên 8192.
Điều này giúp pipeline chạy mượt trên cả Free Colab lẫn các hạ tầng compute mạnh.

### Colab Checkpoint Manager & Khôi Phục Liên Tục
Để giải tỏa rủi ro đứt gãy kết nối mạng lưới giữa chừng dưới sức ép timeout của Google Colab Free, module `ColabCheckpointManager` được cắm rễ vào vòng lặp:
- **Lưu Phiên Ảnh Xạ Kép:** Backup song song 2 bản đồ mô hình: `tarecmind_best.pth` (ghi lại trạng thái Loss và Cải thiện Metrics tối đa) và `tarecmind_last.pth` (luôn đè phiên epoch gần nhất để tự auto-resume).
- **History Syncing:** Tập Log đào tạo `training_history.json` liên tục quét lại Log cũ trên Drive (`current_on_drive`), hợp nhất và bù trừ bất kỳ epoch nào còn thiếu (`epoch not in drive_epochs`), rồi xếp lại dọc thời gian thực. Tín hiệu này bảo vệ vô hạn những gì Data đã chạy mà không làm hỏng Metric nếu trình duyệt web tắt.
