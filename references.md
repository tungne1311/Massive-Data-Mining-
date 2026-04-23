# Tài Liệu Tham Khảo

## Bài Báo Chính (Trích Dẫn Trực Tiếp)

### [1] LightGCN — Nền Tảng GNN

**He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).**  
*LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.*  
Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2020), 639–648.

**Đóng góp vào TA-RecMind:**
- Công thức message passing: `E^(l+1) = Â · E^(l)`
- Layer combination: `z^G_v = (1/(L+1)) Σ^L_{l=0} E^(l)_v`
- Chuẩn hóa đối xứng: `Â = D^{-1/2} A D^{-1/2}`
- Khuyến nghị số tầng L=3 cho Amazon datasets

---

### [2] RecMind — Framework Nền Tảng

**Xue, Z., et al. (2025).**  
*RecMind: Large Language Model Powered Agent for Recommendation.*  
arXiv:2509.06286v1.

**Đóng góp vào TA-RecMind:**
- Gate Fusion formula (Eq. 6–8)
- Cross-modal Alignment Loss (Eq. 5)
- Final representation (Eq. 9): `h_v = α·z^G + (1-α)·z^L`
- Item/User Text Profile với field-aware token budget (Sec. IV-B)
- Training protocol hai giai đoạn (Sec. IV-E)
- Chronological leave-one-out split (Sec. V-A)
- Popularity-aware negative sampling (Sec. IV-E)

---

### [3] SGL — InfoNCE Structure

**Wu, J., Wang, X., Feng, F., He, X., Chen, L., Lian, J., & Xie, X. (2021).**  
*Self-supervised Graph Learning for Recommendation.*  
Proceedings of the 44th International ACM SIGIR Conference (SIGIR 2021), 726–735.

**Đóng góp vào TA-RecMind:**
- InfoNCE loss structure được dùng làm nền tảng cho Alignment Loss
- `L_ssl = -Σ_u log[exp(z_u·z'_u/τ) / Σ_{u'} exp(z_u·z'_{u'}/τ)]`

---

### [4] BPR — Personalized Ranking Loss

**Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009).**  
*BPR: Bayesian Personalized Ranking from Implicit Feedback.*  
Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI 2009), 452–461.

**Đóng góp vào TA-RecMind:**
- `L_BPR = -Σ log σ(s(u,i) - s(u,j))`
- Nền tảng cho pairwise learning từ implicit feedback

---

### [5] LoRA — Efficient Fine-tuning

**Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., & Chen, W. (2022).**  
*LoRA: Low-Rank Adaptation of Large Language Models.*  
International Conference on Learning Representations (ICLR 2022).

**Đóng góp vào TA-RecMind:**
- Phân rã `ΔW = AB` giảm tham số LLM từ hàng tỷ xuống vài triệu
- Cho phép fine-tune LLM trên GPU 16 GB (Kaggle/Colab)

---

### [6] Challenging the Long Tail

**(Bài số 1 trong danh sách tài liệu gốc)**

**Đóng góp vào TA-RecMind:**
- Nguyên tắc xác định ngưỡng HEAD/MID/TAIL dựa trên CDF power-law
- Nguyên tắc 80/20 (Pareto) làm ranh giới HEAD
- Popularity debiasing trong inference stage
- Tail Coverage@K metric

---

### [7] LAGCL — Long-tail Augmented Contrastive Learning

**(Bài số 8 trong danh sách tài liệu gốc)**

**Đóng góp vào TA-RecMind:**
- Augmentation noise tỷ lệ nghịch với popularity
- `h^{(l)'}_i = h^{(l)}_i + Δ^{(l)'}_i` với `||Δ|| ∝ 1/train_freq`
- Contrastive loss giữa hai augmented views
- Lý luận tại sao LAGCL tốt hơn SimGCL cho long-tail

---

### [8] Graph Neural Network for Amazon Co-purchase

**(Bài số 11 trong danh sách tài liệu gốc)**

**Đóng góp vào TA-RecMind:**
- Ý tưởng item-item semantic edges từ co-purchase data
- Thiết kế đồ thị dị thể (heterogeneous graph)

---

## Đóng Góp Kỹ Thuật Mới (Novel Contributions)

Các công thức sau là **đề xuất riêng của TA-RecMind**, không có trong bất kỳ bài báo nào được trích dẫn:

### Khởi Tiến Cổng Khảo Cứu Động (Adaptive Gate Fusion)

```
γ^{(l)} = σ(w_{base} + w_{sim} \cdot \cos(z^G, z^L) + w_{freq} \cdot \log(1 + train\_freq))
```

**Tuyên bố nguồn:** RecMind gốc thiết kế hàm mảng nối vector tĩnh (Concat). TA-RecMind đề xuất quy trình Cổng thích ứng dựa trên hệ số căn chỉnh Similarity và Logarit tần suất Tương tác để hỗ trợ tập Long-tail tự bù đắp phổ lân cận.

### Tail-Weighted Alignment Loss

```
w_v = 1 / log(1 + train_freq(v))

L^U_{align,tw} = -(1/|B_U|) Σ_{u ∈ B_U} w_u · log [
  exp(cos(z^G_u, z^L_u) / τ) /
  Σ_{u' ∈ B_U} exp(cos(z^G_u, z^L_{u'}) / τ)
]
```

**Tuyên bố nguồn:** Mở rộng từ RecMind Eq.5 (cấu trúc InfoNCE) bằng nguyên lý inverse-frequency weighting từ LAGCL. Công thức cụ thể là đề xuất mới của đề tài.

### Review Quality Weighting

```
w(r) = 1 + log(1 + helpful_vote(r))
```

**Tuyên bố nguồn:** Feature engineering dựa trên đặc điểm dữ liệu Amazon 2023. Dạng `log(1+x)` phổ biến trong TF-IDF và LightGCN degree normalization, nhưng ứng dụng vào trọng số chất lượng review là đề xuất riêng.

### Anti-Leakage Classification

Dùng `train_freq(i)` thay vì `metadata.rating_number` để phân loại HEAD/MID/TAIL — phân tích rủi ro data leakage và giải pháp là phát hiện của đề tài từ kết quả EDA.

### Popularity-Penalized Re-ranking

```
s_adjusted(u, i) = s_model(u, i) - λ_penalty · log(1 + train_freq(i))
```

**Tuyên bố nguồn:** Dựa trên nguyên lý inverse popularity weighting từ "Challenging the Long Tail" (bài 1). Công thức cụ thể là đề xuất kỹ thuật của đề tài.

---

## Tài Liệu Bổ Sung

### Dataset

**Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024).**  
*Bridging Language and Items for Retrieval and Recommendation.*  
arXiv:2403.03952.

Amazon Reviews 2023 dataset, truy cập tại: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

### Infrastructure

**Apache Spark:** https://spark.apache.org/docs/latest/  
**MinIO S3:** https://min.io/docs/  
**HuggingFace Datasets:** https://huggingface.co/docs/datasets/  
**FAISS:** https://github.com/facebookresearch/faiss  
**Sentence-Transformers:** https://www.sbert.net/

---

## Bảng Nguồn Gốc Công Thức Tổng Hợp

| Công Thức | Ký Hiệu | Nguồn | Loại |
|---|---|---|---|
| Message passing | `E^(l+1) = Â·E^(l)` | LightGCN (He et al., 2020) | Trích dẫn trực tiếp |
| Layer combination | `z^G = avg(E^0..E^L)` | LightGCN Eq.2; RecMind Eq.2 | Trích dẫn trực tiếp |
| LLM encoding | `z^L = W_proj·Pool(F(T))` | RecMind Eq.3-4 | Trích dẫn trực tiếp |
| Gate cổng | `γ = σ(w_base + w_sim·cos + w_freq·log(..))` | **Novel** | Đóng góp mới |
| Gate fusion | `Ê = γ·z^G + (1-γ)·z^L` | RecMind Eq.7 | Trích dẫn trực tiếp |
| Final repr. | `h = α·z^G + (1-α)·z^L` | RecMind Eq.9 | Trích dẫn trực tiếp |
| Alignment loss | InfoNCE structure | RecMind Eq.5; SGL | Trích dẫn trực tiếp |
| BPR loss | `L_BPR = -Σ log σ(s_pos - s_neg)` | Rendle et al., 2009 | Trích dẫn trực tiếp |
| Augmentation | `h' = h + Δ, ||Δ|| ∝ 1/freq` | LAGCL | Trích dẫn trực tiếp |
| LoRA | `ΔW = AB, r << min(m,n)` | Hu et al., 2022 | Trích dẫn trực tiếp |
| Tail-weighted loss | `w_v·InfoNCE` | **Novel** | Đóng góp mới |
| Review weight | `1 + log(1 + helpful_vote)` | **Novel** | Đóng góp mới |
| Re-ranking | `s_adj = s_model - λ·log(1+f)` | **Novel** | Đóng góp mới |
| Anti-leakage class. | `train_freq` thay vì `rating_number` | **Novel** | Đóng góp mới |
