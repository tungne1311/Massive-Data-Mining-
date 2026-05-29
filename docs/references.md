# Tài Liệu Tham Khảo

## Bài Báo Chính (Trích Dẫn Trực Tiếp)

### [1] LightGCN — Nền Tảng GNN

**He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).**  
*LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.*  
Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2020), 639–648.

**Đóng góp vào TA-RecMind:**
- Công thức message passing: `E^(l+1) = Â · E^(l)`
- Layer combination (embedding trung bình qua các tầng): `z^G_v = (1/(L+1)) Σ^L_{l=0} E^(l)_v`
- Chuẩn hóa đối xứng: `Â = D^{-1/2} A D^{-1/2}`
- Khuyến nghị số tầng L=2-3 cho Amazon datasets (TA-RecMind chọn L=2 để tránh over-smoothing)

---

### [2] RecMind — Framework Nền Tảng

**Xue, Z., et al. (2025).**  
*RecMind: Large Language Model Powered Agent for Recommendation.*  
arXiv:2509.06286v1.

**Đóng góp vào TA-RecMind:**
- Gate Fusion formula (Eq. 6–8): Ê_v = γ·z^G + (1-γ)·z^L
- Cross-modal Alignment Loss / InfoNCE structure (Eq. 5)
- Final representation (Eq. 9): `h_v = α·z^G + (1-α)·z^L`
- Item/User Text Profile với field-aware token budget (Sec. IV-B)
- Training protocol hai giai đoạn: Warm-up Alignment → Joint Training (Sec. IV-E)
- Chronological leave-one-out split (Sec. V-A)
- Popularity-aware negative sampling (Sec. IV-E)

**Điểm TA-RecMind mở rộng vượt trội:**
- RecMind Gate chỉ cho Item → TA-RecMind áp **đối xứng cho cả User** (Bipartite Symmetry)
- RecMind Alignment Loss đồng đều → TA-RecMind **Tail-Weighted** Alignment
- RecMind không có contrastive augmentation → TA-RecMind thêm **LAGCL**

---

### [3] SGL — InfoNCE Structure

**Wu, J., Wang, X., Feng, F., He, X., Chen, L., Lian, J., & Xie, X. (2021).**  
*Self-supervised Graph Learning for Recommendation.*  
Proceedings of the 44th International ACM SIGIR Conference (SIGIR 2021), 726–735.

**Đóng góp vào TA-RecMind:**
- InfoNCE loss structure được dùng làm nền tảng cho Alignment Loss
- `L_ssl = -Σ_u log[exp(z_u·z'_u/τ) / Σ_{u'} exp(z_u·z'_{u'}/τ)]`
- Lý luận tại sao contrastive learning cải thiện sparse GNN

---

### [4] BPR — Personalized Ranking Loss

**Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009).**  
*BPR: Bayesian Personalized Ranking from Implicit Feedback.*  
Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI 2009), 452–461.

**Đóng góp vào TA-RecMind:**
- `L_BPR = -Σ log σ(s(u,i) - s(u,j))` — nền tảng pairwise learning từ implicit feedback
- Khung lý thuyết cho popularity-debiased negative sampling

---

### [5] LoRA — Efficient Fine-tuning

**Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022).**  
*LoRA: Low-Rank Adaptation of Large Language Models.*  
International Conference on Learning Representations (ICLR 2022).

**Đóng góp vào TA-RecMind:**
- Phân rã `ΔW = AB` với `A ∈ ℝ^{m×r}`, `B ∈ ℝ^{r×n}`, `r << min(m,n)`
- Với `r=8`: giảm tham số LLM từ hàng tỷ xuống vài triệu
- Cho phép fine-tune LLaMA-3.2-1B trên GPU 16GB (T4/Kaggle)

---

### [6] Challenging the Long Tail

**Đóng góp vào TA-RecMind:**
- Nguyên tắc xác định ngưỡng HEAD/MID/TAIL dựa trên CDF power-law
- Nguyên tắc 80/20 (Pareto) làm ranh giới HEAD
- Popularity debiasing trong inference stage (re-ranking)
- Tail Coverage@K metric — đo độ bao phủ của tail items trong recommendations

---

### [7] LAGCL — Long-tail Augmented Contrastive Learning

**Đóng góp vào TA-RecMind:**
- Augmentation noise tỷ lệ nghịch với popularity:
  - `h^{(l)'}_i = h^{(l)}_i + Δ^{(l)'}_i` với `||Δ|| ∝ 1/train_freq`
  - `σ_noise(i) = σ_0 / (1 + log(1 + train_freq(i)))`
- Contrastive loss giữa hai augmented views của cùng node
- Lý luận tại sao LAGCL tốt hơn SimGCL cho long-tail: SimGCL dùng uniform noise (không phân biệt head/tail), LAGCL điều chỉnh theo popularity

---

### [8] Amazon Reviews 2023 Dataset

**Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024).**  
*Bridging Language and Items for Retrieval and Recommendation.*  
arXiv:2403.03952.

Dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

**Đặc điểm được khai thác trong TA-RecMind:**
- Metadata phong phú (title, features, description, details) → Field-Aware Token Budget
- `helpful_vote` field → Review Quality Weighting
- Timestamp chính xác → Chronological Split, Temporal Decay
- `rating_number` trong metadata → phát hiện Data Leakage, chuyển sang dùng `train_freq`

---

## Đóng Góp Kỹ Thuật Mới (Novel Contributions)

Các công thức sau là **đề xuất riêng của TA-RecMind**, không có trong bất kỳ bài báo nào được trích dẫn:

### 1. Adaptive Gate Fusion (Bipartite Symmetry)

$$\gamma_v^{(l)} = \sigma\!\left(w_{base} + w_{sim} \cdot \cos(z_v^G, z_v^L) + w_{freq} \cdot \log(1 + \text{train\_freq}(v))\right)$$

**Tuyên bố nguồn:** RecMind gốc thiết kế Gate tĩnh chỉ cho Item node (Concat). TA-RecMind đề xuất:
1. Gate **động** dựa trên frequency và cosine similarity (không phải concat cứng)
2. Áp **đối xứng cho cả User và Item** (Bipartite Symmetry) — giải quyết cold-start user

### 2. Tail-Weighted Alignment Loss

$$w_v = \frac{1}{\log(1 + \text{train\_freq}(v))}$$

$$\mathcal{L}^{U,I}_{align,tw} = -\frac{1}{|B|} \sum_{v \in B} w_v \cdot \log \frac{\exp(\cos(z^G_v, z^L_v) / \tau)}{\sum_{v' \in B} \exp(\cos(z^G_v, z^L_{v'}) / \tau)}$$

**Tuyên bố nguồn:** Mở rộng từ RecMind Eq.5 (InfoNCE structure) bằng nguyên lý inverse-frequency weighting từ LAGCL. Công thức cụ thể là đề xuất mới — cho phép gradient alignment **lớn hơn 16 lần** với extreme tail items so với max head items.

### 3. Review Quality Weighting

$$w(r) = 1 + \log(1 + \text{helpful\_vote}(r))$$

**Tuyên bố nguồn:** Feature engineering dựa trên đặc điểm Amazon 2023. Dạng `1 + log(1+x)` phổ biến trong TF-IDF và degree normalization, nhưng ứng dụng vào trọng số chất lượng review với baseline = 1 (không loại bỏ review 0 helpful_votes) là đề xuất riêng của đề tài.

### 4. Anti-Leakage Classification

Dùng `train_freq(i)` thay vì `metadata.rating_number` để phân loại HEAD/MID/TAIL.

**Tuyên bố nguồn:** Phát hiện từ kết quả EDA — `rating_number` chênh lệch 137-455× so với `train_freq`. Không có bài báo nào cảnh báo cụ thể về leakage trong `rating_number` của Amazon 2023 dataset.

### 5. Popularity-Penalized Re-ranking

$$s_{adjusted}(u, i) = s_{model}(u, i) - \lambda_{penalty} \cdot \log(1 + \text{train\_freq}(i))$$

**Tuyên bố nguồn:** Dựa trên nguyên lý inverse popularity weighting từ "Challenging the Long Tail" (bài [6]). Công thức cụ thể và tích hợp vào inference pipeline (2-stage: retrieval → re-ranking) là đề xuất kỹ thuật của đề tài.

---

## Hạ Tầng & Công Cụ

| Công Cụ | Phiên Bản | Vai Trò |
|---|---|---|
| Apache Spark | 3.5.2 | Distributed data processing (Bronze, Silver) |
| MinIO | Latest | S3-compatible local object storage |
| HuggingFace Datasets | 2.x | Data hosting và streaming ingestion |
| HuggingFace Hub | Latest | Artifact storage và sharing |
| PyTorch | 2.x | Training, inference |
| PyTorch Geometric (PyG) | 2.x | Graph neural network |
| Sentence-Transformers | 2.x | LLM semantic embedding |
| FAISS | 1.7.x | Approximate nearest neighbor search |
| Streamlit | 1.x | Demo application |

**Links:**
- Apache Spark: https://spark.apache.org/docs/3.5.2/
- MinIO S3: https://min.io/docs/
- HuggingFace Datasets: https://huggingface.co/docs/datasets/
- FAISS: https://github.com/facebookresearch/faiss
- Sentence-Transformers: https://www.sbert.net/

---

## Bảng Nguồn Gốc Công Thức Tổng Hợp

| Công Thức | Ký Hiệu | Nguồn | Loại |
|---|---|---|---|
| Message passing | `E^(l+1) = Â·E^(l)` | LightGCN (He et al., 2020) | Trích dẫn |
| Layer combination | `z^G = avg(E^0..E^L)` | LightGCN Eq.2; RecMind Eq.2 | Trích dẫn |
| LLM encoding | `z^L = W_proj·Pool(F(T))` | RecMind Eq.3-4 | Trích dẫn |
| Symmetric norm | `Â = D^{-1/2}AD^{-1/2}` | LightGCN Eq.1 | Trích dẫn |
| **Gate cổng (adaptive)** | `γ = σ(w_base + w_sim·cos + w_freq·log1p)` | **Novel** | **Đóng góp mới** |
| Gate fusion | `Ê = γ·z^G + (1-γ)·z^L` | RecMind Eq.7 (cấu trúc) | Trích dẫn + mở rộng |
| Propagate fused | `E^(l+1) = Â·Ê^(l)` | RecMind Eq.8 | Trích dẫn |
| Final repr. | `h = α·z^G + (1-α)·z^L` | RecMind Eq.9 | Trích dẫn |
| Alignment loss | InfoNCE structure | RecMind Eq.5; SGL | Trích dẫn |
| **Tail-weighted loss** | `w_v·InfoNCE, w_v = 1/log(1+freq)` | **Novel** | **Đóng góp mới** |
| BPR loss | `L_BPR = -Σ log σ(s_pos - s_neg)` | Rendle et al., 2009 | Trích dẫn |
| Temporal BPR | `w(u,i)·BPR, w=exp(-λ·ΔT)` | Thực tiễn phổ biến | Áp dụng chuẩn |
| Augmentation | `h' = h + Δ, ||Δ|| ∝ 1/freq` | LAGCL | Trích dẫn |
| Contrastive loss | `L_cl` giữa 2 augmented views | LAGCL | Trích dẫn |
| LoRA | `ΔW = AB, r << min(m,n)` | Hu et al., 2022 | Trích dẫn |
| **Review weight** | `1 + log(1 + helpful_vote)` | **Novel** | **Đóng góp mới** |
| **Re-ranking** | `s_adj = s_model - λ·log(1+freq)` | **Novel** | **Đóng góp mới** |
| **Anti-leakage class.** | `train_freq` thay vì `rating_number` | **Novel** | **Đóng góp mới** |
| Neg. sampling prob | `P ∝ freq^{-β}` | RecMind Sec.IV-E | Trích dẫn |
