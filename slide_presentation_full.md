# NỘI DUNG SLIDE THUYẾT TRÌNH: TA-RecMind
# Tail-Augmented Recommendation with LLM-GNN Alignment

---

## SLIDE 1: TRANG BÌA

**Tiêu đề:** TA-RecMind: Hệ Thống Gợi Ý Sản Phẩm Long-Tail Kết Hợp LLM và Graph Neural Network
**Phụ đề:** Tail-Augmented Recommendation with LLM-GNN Intra-Layer Gate Fusion trên Amazon Electronics 2023
**Thông tin:** Tên sinh viên, GVHD, Trường, Khoa, Năm 2026

---

## SLIDE 2: MỤC LỤC TỔNG QUAN

1. Giới thiệu & Đặt vấn đề (Slides 3-6)
2. Phân tích yêu cầu (Slides 7-8)
3. Phân tích dữ liệu EDA (Slides 9-14)
4. Cơ sở lý thuyết (Slides 15-20)
5. Kiến trúc hệ thống & Data Pipeline (Slides 21-25)
6. Kiến trúc mô hình TA-RecMind (Slides 26-28)
7. Chiến lược huấn luyện (Slides 29-32)
8. Đánh giá & Demo (Slides 33-38)
9. Quản trị hệ thống & Công nghệ (Slides 39-42)
10. Điểm mới & Kết luận (Slides 43-44)

---

## SLIDE 3: BỐI CẢNH — HỆ THỐNG GỢI Ý TRONG THƯƠNG MẠI ĐIỆN TỬ

**Nội dung trình bày:**
- Hệ thống gợi ý (Recommender System) là công nghệ cốt lõi của các nền tảng thương mại điện tử: Amazon, Shopee, Lazada...
- Mục tiêu: Dự đoán sản phẩm mà người dùng có khả năng quan tâm, từ hàng triệu sản phẩm trong catalog
- Các phương pháp phổ biến:
  - Collaborative Filtering (CF): Lọc cộng tác dựa trên hành vi người dùng tương tự
  - Content-Based Filtering: Dựa trên đặc trưng nội dung sản phẩm
  - Hybrid: Kết hợp cả hai
- **Xu hướng hiện đại:** Graph Neural Networks (GNN) + Large Language Models (LLM) — tận dụng cả cấu trúc đồ thị và ngữ nghĩa văn bản

---

## SLIDE 4: VẤN ĐỀ NGHIÊN CỨU — LONG-TAIL & COLD-START

**Nội dung trình bày:**

**Vấn đề Long-Tail (Đuôi Dài):**
- Phân phối tương tác sản phẩm tuân theo Power-Law (Pareto 80/20): Top 20% sản phẩm phổ biến chiếm ~80% tổng tương tác
- 72.51% sản phẩm (755,609 items) chỉ có ≤ 5 tương tác → gọi là "Tail Items"
- Các hệ thống CF truyền thống tối ưu chủ yếu cho Head Items, bỏ qua hàng trăm nghìn sản phẩm chất lượng nhưng ít được biết đến

**Vấn đề Cold-Start (Khởi Động Lạnh):**
- 23.17% items trong tập đánh giá (130,746 items) chưa từng xuất hiện trong tập huấn luyện
- GNN hoàn toàn bị "mù" với các items này (degree = 0 trong đồ thị)
- Sản phẩm mới ra mắt không thể được gợi ý bằng CF thuần túy

**Hệ quả kinh doanh:** Mất cơ hội bán hàng, người dùng chỉ thấy sản phẩm phổ biến, catalog không được khai thác hiệu quả

---

## SLIDE 5: CÂU HỎI NGHIÊN CỨU & MỤC TIÊU ĐỀ TÀI

**Câu hỏi nghiên cứu:**
> Làm thế nào xây dựng hệ thống gợi ý vừa duy trì độ chính xác tổng thể, vừa cải thiện đáng kể khả năng gợi ý Tail Items và Cold-Start Items?

**Mục tiêu cụ thể:**
1. Xây dựng Data Pipeline hoàn chỉnh xử lý ~44M tương tác theo kiến trúc Medallion (Bronze → Silver → Gold)
2. Thiết kế mô hình TA-RecMind kết hợp LightGCN + LLM qua cơ chế Intra-Layer Gate Fusion
3. Đề xuất Tail-Weighted Alignment Loss ưu tiên học biểu diễn cho Tail Items
4. Đánh giá phân tầng theo HEAD/MID/TAIL/COLD-START
5. Demo ứng dụng trực quan với khả năng Zero-Shot Cold-Start Insertion

---

## SLIDE 6: PHẠM VI ĐỀ TÀI & DATASET

**Dataset:** Amazon Reviews 2023 — Subset Electronics (McAuley-Lab)

| Chỉ số | Giá trị |
|---|---|
| Raw interactions | ~44,066,834 reviews |
| Metadata | 161,001 items |
| Positive interactions (rating ≥ 3) | ~36,168,550 |
| Users (sau Core-5 filter) | 1,847,662 |
| Items (train) | 1,042,121 |
| Sparsity đồ thị | 99.9993% |

**Tại sao chọn Amazon Electronics 2023?**

**Ưu điểm (Pros):**
- **Dataset chuẩn quốc tế:** Được sử dụng rộng rãi trong nghiên cứu RecSys (McAuley Lab), dễ dàng so sánh với các nghiên cứu SOTA.
- **Metadata phong phú:** Cung cấp Title, Features, Description, Details... — điều kiện lý tưởng để khai thác sức mạnh của LLM (LLM-GNN alignment).
- **Trường `helpful_vote`:** Cung cấp tín hiệu chất lượng review khách quan, giúp lọc nhiễu tốt hơn so với chỉ dùng rating.
- **Timestamp chính xác:** Đảm bảo chronological split (chia theo thời gian) tuyệt đối, chống rò rỉ dữ liệu (data leakage).
- **Độ thưa (Sparsity) cao:** Phản ánh đúng thực tế thương mại điện tử, là môi trường hoàn hảo để thử nghiệm khả năng xử lý Long-tail của TA-RecMind.

**Nhược điểm (Cons):**
- **Dữ liệu tĩnh (Static Data):** Chỉ bao gồm Review và Rating cuối cùng, thiếu dữ liệu hành vi dạng chuỗi (Session-based) như Click, View, Add-to-cart.
- **Không có dữ liệu tương tác thời gian thực:** Do tính chất là dataset offline, hệ thống không thể thực hiện gợi ý Real-time dựa trên hành vi người dùng đang diễn ra trong session hiện tại.
- **Data Skew cực đoan:** Phân phối Power-law khiến các Head Items chiếm ưu thế tuyệt đối, gây khó khăn cho việc học biểu diễn các Tail Items (nhưng đây cũng là động lực của đề tài).

---

## SLIDE 7: PHÂN TÍCH YÊU CẦU CHỨC NĂNG

**Yêu cầu Data Pipeline:**
- Thu thập dữ liệu từ HuggingFace Hub (streaming ~44M records)
- Xử lý phân tán bằng Apache Spark (PySpark)
- Lọc dữ liệu: rating ≥ 3.0, Core-5 filter, loại bỏ trùng lặp
- Phân tách thời gian: Train/Val/Test không time leakage
- Lưu trữ: MinIO (local S3) + HuggingFace Hub (backup)

**Yêu cầu Mô hình:**
- Xử lý đồ thị Bipartite thưa (sparsity 99.9993%)
- Kết hợp thông tin ngữ nghĩa (LLM) với cấu trúc đồ thị (GNN)
- Giải quyết Cold-Start: gợi ý sản phẩm chưa có tương tác
- Re-ranking giảm Popularity Bias tại inference time

**Yêu cầu Demo:**
- Gợi ý Top-K sản phẩm cho user bất kỳ (< 50ms)
- Zero-Shot Item Insertion: thêm sản phẩm mới, tìm khách hàng tiềm năng (< 200ms)
- Trực quan hóa phân phối HEAD/MID/TAIL trong kết quả gợi ý

---

## SLIDE 8: PHÂN TÍCH YÊU CẦU PHI CHỨC NĂNG

**Hiệu năng:**
- Pipeline xử lý ~44M records trên máy 16GB RAM (Docker)
- Training trên GPU (Google Colab T4 16GB VRAM, A100...)
- Inference < 50ms/query (CPU mode)

**Khả năng mở rộng:**
- Kiến trúc Medallion cho phép thêm tầng xử lý mà không ảnh hưởng tầng trước
- Auto-scaling batch size theo VRAM: T4 (2048) → V100 (6144) → A100 (8192)

**Tính tin cậy:**
- Checkpoint tự động trên Google Drive (ColabCheckpointManager)
- Auto-resume khi mất kết nối Colab
- HuggingFace Hub backup toàn bộ artifacts

**Chống Data Leakage:**
- Tập Test không bị động chạm từ Silver trở đi
- User Text Profile chỉ từ Train reviews (không có Val/Test)
- Phân loại HEAD/MID/TAIL dựa trên train_freq (không dùng rating_number)

---

## SLIDE 9: EDA — TỔNG QUAN DỮ LIỆU

**Quy trình EDA 3 giai đoạn:**
1. **Bronze EDA:** Phân tích dữ liệu thô, cấu trúc đồ thị, kiểm tra time leakage
2. **Silver EDA:** Phân tích Text Pipeline, token length, noise detection
3. **Gold EDA:** Xác nhận ID Mapping, edge list, training metadata

**Thống kê chính — Đồ thị Bipartite (Tập Train):**

| Chỉ số | User Nodes | Item Nodes |
|---|---|---|
| Tổng số | 1,847,662 | 1,042,121 |
| Min degree | 3* | 1 |
| Median degree | 5 | 2 |
| Avg degree | 7.56 | 1.34 |
| Max degree | 1,005 | 41,183 |

*Min User degree = 3 do Core-5 filter (≥5 positive), trừ 1 cho Val và 1 cho Test

---

## SLIDE 10: EDA — PHÂN PHỐI POWER-LAW & PARETO 80/20

**Phân phối Long-Tail (Item Popularity):**
- Biểu đồ Log-Log scale là đường thẳng dốc đứng → xác nhận Power-Law
- CDF cho thấy:
  - **HEAD Items (Top 20%):** Chiếm ~80% tổng tương tác (Pareto principle)
  - **MID Items (10% tiếp theo):** Vùng đệm chuyển tiếp
  - **TAIL Items (Bottom 70%):** 755,609 items (72.51%) — phần lớn chỉ có 1-2 tương tác

**Ngưỡng phân loại (tính từ CDF train_freq):**
```
HEAD: Top 20% items theo train_freq  → chiếm ~80% tương tác
MID:  10% items tiếp theo            → vùng đệm
TAIL: Bottom 70% items còn lại       → 72.51%, mục tiêu chính
```

**Chênh lệch cực đoan:** Median degree Item = 2 vs Max = 41,183 → Head thống trị hoàn toàn. Đây là động lực chính cho Gate Fusion và Tail-Weighted Loss.

---

## SLIDE 11: EDA — PHÁT HIỆN DATA LEAKAGE

**Vấn đề nghiêm trọng — `rating_number` vs `train_freq`:**

```
Item B07H65KP63:
  train_freq    = 1,561   (thực tế trong tập train)
  rating_number = 710,348 (tổng tích lũy toàn thời gian)
  → Chênh lệch ~455 lần

Item khác:
  train_freq    = 4,265
  rating_number = 585,624
  → Chênh lệch ~137 lần
```

**Phân tích:** `rating_number` trong metadata chứa thông tin từ TƯƠNG LAI (bao gồm cả Val/Test). Nếu dùng `rating_number` để phân loại HEAD/MID/TAIL → **Data Leakage nghiêm trọng**.

**Quyết định thiết kế (Anti-Leakage Classification):**
- ✅ Dùng `train_freq(i)` = số tương tác chỉ trong tập Train
- ❌ KHÔNG dùng `rating_number` từ metadata
- Đây là **đóng góp mới** — chặn leakage triệt để

---

## SLIDE 12: EDA — PHÂN PHỐI RATING & REVIEW QUALITY

**Rating Distribution:**

| Rating | Tỷ lệ |
|---|---|
| ⭐⭐⭐⭐⭐ (5 sao) | 65.7% |
| ⭐⭐⭐⭐ (4 sao) | 14.7% |
| ⭐⭐⭐ (3 sao) | 7.0% |
| ⭐⭐ (2 sao) | 4.5% |
| ⭐ (1 sao) | 8.0% |

**Vấn đề:** Rating bị skew nặng — 65.7% là 5 sao → không có khả năng phân biệt chất lượng review.

**Giải pháp — Review Quality Weighting (Đóng góp mới):**
```
w(r) = 1 + log(1 + helpful_vote(r))
```

| helpful_vote | w(r) | Diễn giải |
|---|---|---|
| 0 | 1.00 | Baseline, không loại bỏ |
| 10 | ≈ 3.40 | Review được vote → nặng hơn |
| 100 | ≈ 5.61 | Rất tin cậy |
| 3,294 (max) | ≈ 9.40 | Log kiểm soát outlier |

---

## SLIDE 13: EDA — ĐẶC TRƯNG NGÔN NGỮ (NLP INSIGHTS)

**Mức độ hoàn thiện Metadata:**
- Title: gần 100% items có
- Features: tỷ lệ rỗng tăng mạnh ở Tail Items
- Description: nhiều item không có — đặc biệt Tail Items

**Token Length (sau tokenization):**
- Item Text: đa số 128-256 tokens, rất hiếm > 384
- User Text (top-3 reviews + [SEP]): trung vị ~180 tokens, p95 ≈ 320-350

**Quyết định:** max_length = 384 (MiniLM) bao phủ 99%+ context → TOP_K=3 reviews (không phải 5)

**WordCloud Head vs Tail:**
- HEAD: từ chung (*Cable, Bluetooth, Wireless, USB*)
- TAIL: thông số kỹ thuật ngách, linh kiện chuyên biệt → **cần LLM embedding**

---

## SLIDE 14: EDA — COLD-START & TEMPORAL ANALYSIS

**Cold-Start trong Evaluation:**

| Chỉ số | Giá trị |
|---|---|
| Total eval items (Val/Test) | 564,225 |
| Pure cold-start items (degree = 0) | 130,746 (23.17%) |

→ 23.17% items trong đánh giá chưa từng xuất hiện trong training. GNN hoàn toàn "mù". Gate Fusion + LLM embedding là **linh hồn** của TA-RecMind.

**Kiểm tra Time Leakage (Chronological Split):**
- Valid Time-split Users: **100%**
- Leakage Train-Val: **0**
- Leakage Val-Test: **0**
→ Giao thức chia thời gian chính xác tuyệt đối.

**Temporal Decay:**
- Tương tác trải dài hàng nghìn ngày (nhiều năm)
- Decay formula: `w(u,i) = exp(-λ × (T_max - timestamp) / T_range)`
- Tích hợp vào BPR Loss tại training

---

## SLIDE 15: CƠ SỞ LÝ THUYẾT — LIGHTGCN

**LightGCN (He et al., SIGIR 2020):**
Đơn giản hóa GCN cho Recommendation — loại bỏ feature transformation và activation function, chỉ giữ neighborhood aggregation:

**Message Passing:**
```
E^(l+1) = Â · E^(l)

Tường minh:
  e^(l+1)_u = Σ_{i ∈ N_u} (1/√|N_u|·√|N_i|) · e^(l)_i
  e^(l+1)_i = Σ_{u ∈ N_i} (1/√|N_i|·√|N_u|) · e^(l)_u
```

**Chuẩn hóa đối xứng:** `Â = D^{-1/2} A D^{-1/2}` — đảm bảo Head Items không khuếch đại gradient bất cân xứng

**Layer Combination (trung bình qua các tầng):**
```
z^G_v = (1/(L+1)) Σ^L_{l=0} E^(l)_v
```

**Nhược điểm:** Yếu kém khi degree nhỏ (Tail/Cold-start) — không có thông tin neighbor để truyền.

---

## SLIDE 16: CƠ SỞ LÝ THUYẾT — RECMIND FRAMEWORK

**RecMind (Xue et al., 2025):**
LLM-Powered Agent for Recommendation — kết hợp LLM với GNN.

**Công thức LLM Encoding:**
```
z̃^L_v = Pool(F(T_v; adapters))     ← LLM frozen + LoRA adapters
z^L_v  = W_proj · z̃^L_v ∈ ℝ^d     ← Chiếu về không gian d chiều
```

**Gate Fusion:**
```
Ê_v = γ · z^G_v + (1-γ) · z^L_v   ← Trộn Graph + LLM
E^(l+1) = Â · Ê^(l)               ← Propagate fused state
```

**Final Representation:**
```
h_v = α · z^G_v + (1-α) · z^L_v
```

**Hạn chế của RecMind (mà TA-RecMind giải quyết):**
1. Gate chỉ áp dụng cho Item → bỏ qua cold-start User
2. Alignment Loss đồng đều → không ưu tiên Tail Items
3. Không có contrastive augmentation cho Long-tail

---

## SLIDE 17: CƠ SỞ LÝ THUYẾT — RANKING & CONTRASTIVE LOSS

**1. BPR Loss (Rendle et al., UAI 2009):**
Bayesian Personalized Ranking — chuẩn mực cho Implicit Feedback (chỉ có hành vi, không có rating):
- **Mục tiêu:** Tối ưu hóa thứ tự xếp hạng (Ranking), đảm bảo $score(positive) > score(negative)$.
- **Công thức:** $L_{BPR} = -\sum_{(u,i,j) \in O} \log \sigma(s(u,i) - s(u,j))$

**2. InfoNCE Loss (Oord et al., 2018 / SGL 2021):**
Hàm mất mát cho Contrastive Learning (Học tương phản):
- **Cơ chế:** Kéo các vector "vùng tích cực" (positive views) lại gần nhau và đẩy các vector "vùng tiêu cực" (negatives) ra xa.
- **Công thức:** $L_{ssl} = -\sum \log \frac{\exp(z \cdot z^+ / \tau)}{\sum \exp(z \cdot z^- / \tau)}$
- Giúp biểu diễn (representation) bền vững hơn trước sự thưa thớt dữ liệu.

---

## SLIDE 18: CƠ SỞ LÝ THUYẾT — LLM ENCODING & AUGMENTATION

**1. Sentence-Transformers (Semantic Embedding):**
Biến văn bản thành vector ngữ nghĩa trong không gian đa chiều.

| Model | Tham số | Chiều | Hiệu năng / Tài nguyên |
|---|---|---|---|
| **all-MiniLM-L6-v2** | 22M | 384 | **Nhanh, nhẹ (Chọn cho đồ án)** |
| all-mpnet-base-v2 | 109M | 768 | Chính xác hơn, nặng hơn |
| LLaMA-3.2-1B + LoRA | 1B | 2048 | SOTA, cần GPU chuyên dụng |

**2. LAGCL (Long-tail Augmented Contrastive Learning):**
Cơ chế tăng cường dữ liệu đặc thù cho Long-tail:
- Tạo 2 "góc nhìn" (views) khác nhau cho mỗi node bằng cách thêm nhiễu Gaussian.
- **Lý thuyết:** Noise tỷ lệ nghịch với tần suất: $\sigma_{noise} \propto \frac{1}{\log(freq)}$
- Tail items nhận noise mạnh hơn → Buộc model học biểu diễn bền vững hơn từ dữ liệu ít ỏi.

---

## SLIDE 19: CƠ SỞ LÝ THUYẾT — MEDALLION ARCHITECTURE

**Kiến trúc Medallion (Bronze → Silver → Gold):**
Mô hình xử lý dữ liệu phân tầng, mỗi tầng có trách nhiệm rõ ràng:

**Bronze (Dữ liệu thô đã chuẩn hóa):**
- Thu thập từ nguồn gốc (HuggingFace Hub)
- Chuẩn hóa schema, lọc nhiễu cơ bản
- Phân tách Train/Val/Test theo thời gian

**Silver (Dữ liệu đã làm giàu):**
- Feature engineering: Text Profile, Popularity Classification
- Tất cả tính toán chỉ dùng tập Train (chống leakage)

**Gold (Sẵn sàng cho Model):**
- Integer ID Mapping cho GNN
- Edge List format PyG
- LLM Embeddings (offline)
- Training Metadata Arrays (.npy)

**Nguyên tắc xuyên suốt:** Tập Test không bao giờ bị động chạm từ Silver trở đi.

---

## SLIDE 20: CƠ SỞ LÝ THUYẾT — BIPARTITE GRAPH & SYMMETRIC NORMALIZATION

**Đồ thị User-Item Bipartite:**
- Ma trận kề `A ∈ ℝ^{(N_u + N_i) × (N_u + N_i)}`
- Chuẩn hóa đối xứng: `Â = D^{-1/2} A D^{-1/2}`
- D: ma trận đường chéo degree

**Thiết kế đặc biệt:**
- Binary Unweighted Edge: KHÔNG đưa Edge Weight vào Â
- Mọi tương tác nhiễu (rating < 3.0) đã bị lọc từ Bronze
- Temporal Decay tích hợp vào BPR Loss (không vào ma trận kề)

**Ước tính Embedding Matrix (d=128, float32):**
```
Tổng nodes ≈ 3,020,529 × 128 × 4 bytes ≈ 1.5 GB
```

---

## SLIDE 21: SƠ ĐỒ KIẾN TRÚC TOÀN HỆ THỐNG

*(Sơ đồ Kiến trúc Pipeline)*

```
HuggingFace (McAuley-Lab/Amazon-Reviews-2023)
     │   raw_review_Electronics (~44M rows)
     │   raw_meta_Electronics   (~161K items)
     ▼
┌──────────────────────────────────────────┐
│              TẦNG BRONZE                 │
│  ste1.py: Streaming ingestion            │
│  ste2.py: PySpark processing             │
│    → rating ≥ 3.0 filter, Core-5 filter  │
│    → Chronological split (Double Max Join)│
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│              TẦNG SILVER                 │
│  Step 1: Item Popularity (CDF → H/M/T)  │
│  Step 2: Item Text Profile (4-level)     │
│  Step 4: Enrich Interactions             │
│  Step 3: User Text Profile (top-3)       │
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│              TẦNG GOLD                   │
│  Step 1: Integer ID Mapping              │
│  Step 2: Edge List (PyG format)          │
│  Step 3+4: LLM Embedding (→ Colab)       │
│  Step 5: Training Metadata Arrays        │
└────────────────┬─────────────────────────┘
                 ▼
┌──────────────────────────────────────────┐
│         TRAINING & DEMO (Colab)          │
│  TA_RecMind_V2_IntraLayer.ipynb          │
│  Demo Streamlit: Zero-Shot Cold-Start    │
└──────────────────────────────────────────┘
```

---

## SLIDE 22: TẦNG BRONZE — THU THẬP & CHUẨN HÓA

**Phase 1 — MAP (ste1.py): HuggingFace → Landing Zone**
- Streaming ingestion, batch 100,000 records
- Producer-Consumer threading xử lý song song
- Ghi MinIO landing zone (Parquet, zstd compression)

**Phase 2 — REDUCE (ste2.py): PySpark Processing**
1. Lọc Positive Feedback: `rating ≥ 3.0` → chặn rò rỉ dữ liệu
2. `dropDuplicates(["reviewer_id", "parent_asin"])`
3. Core-5 Filter: giữ users có ≥ 5 positive interactions
4. **Chronological Split (Double Max Join):**
   - Test: interaction mới nhất của mỗi user
   - Val: interaction mới thứ hai
   - Train: phần còn lại

**Tại sao Double Max Join thay vì Window Function?**
- `ROW_NUMBER() OVER(...)` táạo shuffle stage cực lớn khi data skew
- Double Max Join: 2 lần `groupBy → max(ts) → inner join` — chống OOM.

---

## SLIDE 23: TẦNG SILVER — OVERVIEW & STEP 1

**Nguyên tắc cốt lõi:** Mọi tính toán chỉ dùng tập Train. Val ground truth chỉ lấy (user, item) — không text.

**Chiến lược Vertical Culling (chống OOM):**
- Cache bronze_train chỉ với light columns (không text) cho Step 1+4
- Sau Step 4, unpersist ngay
- Step 3 đọc lại với full text columns độc lập

**Silver Step 1 — Item Popularity Classification:**
1. Tính `train_freq` cho mỗi item (groupBy + count)
2. Sắp xếp giảm dần, tính CDF
3. HEAD: Top 20% (chiếm ~80% tương tác)
4. MID: 10% tiếp theo
5. TAIL: Bottom 70% (755,609 items — 72.51%)
6. COLD_START: items trong metadata nhưng không có trong train

---

## SLIDE 24: TẦNG SILVER — STEP 2 & STEP 3

**Step 2 — Item Text Profile (Field-Aware Token Budget):**

| Cấp | Trường | Giới hạn | Lý do |
|---|---|---|---|
| 1 | title | Không cắt | Nhận dạng cốt lõi |
| 2 | features | 450 chars | Thông tin kỹ thuật |
| 2* | features (ext) | 750 chars | Bù đắp thiếu desc |
| 3 | categories | 150 chars | Ngữ cảnh danh mục |
| 4a | description | 300 chars | Mô tả SP |
| 4b | details | 150 chars | Brand, material |

- Separator: ` | ` giúp LLM phân biệt ranh giới trường

**Step 3 — User Text Profile:**
- Top-3 reviews sắp xếp theo timestamp giảm dần
- Trọng số: `w(r) = 1 + log(1 + helpful_vote(r))`
- Ghép bằng `[SEP]` token
- Checkpoint trung gian tại Phase 2 để tránh tải lại từ HF.

---

## SLIDE 25: TẦNG GOLD — ID MAPPING, EDGE LIST, METADATA

**Gold Step 1 — Integer ID Mapping:**
- LightGCN/PyG yêu cầu integer liên tục từ 0
- `user_id_map`: reviewer_id → user_idx (0 → N_users-1)
- `item_id_map`: parent_asin → item_idx (0 → N_items-1)
- Bao gồm TẤT CẢ items (train + val + cold-start)

**Gold Step 2 — Edge List:**
- Format PyG: `edge_index` shape `[2, E]`
- Binary Unweighted — KHÔNG có edge_weight

**Gold Step 3+4 — LLM Embeddings (Offline trên Colab):**
- Chunk-based encoding: 30,000 records/chunk, batch=256
- Checkpoint tự động sau mỗi chunk → Auto-resume
- VStack → `gold_item_embeddings.npy`, `gold_user_embeddings.npy`

**Gold Step 5 — Training Metadata Arrays (.npy files):**
- freq arrays, popularity group, activity group, dataset stats.

---

## SLIDE 26: SƠ ĐỒ MÔ HÌNH TA-RECMIND (TỔNG THỂ)

*(Sơ đồ Kiến trúc Model - Pipeline Hoàn Chỉnh)*

```
┌───────────────────────────────────────────────────────────┐
│                 1. PIPELINE DỮ LIỆU (MEDALLION)           │
│                                                           │
│  [ Bronze ] ─► [ Silver ] ─► [ Gold ]                     │
│  - Thu thập    - Popularity  - ID Mapping                 │
│  - Clean/Split - Text Profile - Edge List & Meta          │
│  - Core-5      - User Profile - Offline LLM Cache         │
└──────────────────────────┬────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────┐
│         2. TIỀN XỬ LÝ & KHỞI TẠO (OFFLINE)                │
│                                                           │
│  [ LLM Encoding ]              [ Graph Init ]             │
│  - all-MiniLM-L6-v2            - Sparse Adj Matrix        │
│  - Chunked Processing          - Node ID Embeddings       │
│  → Vector Semantic (z_L)       → Node Collaborative (z_G) │
└──────────────────────────┬────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────┐
│            3. HUẤN LUYỆN (TA-RECMIND V2)                  │
│                                                           │
│  [ Pha 1: Warm-up Alignment ]                             │
│  - Căn chỉnh z_G & z_L ưu tiên Tail (Tail-Weighted)       │
│                                                           │
│  [ Pha 2: Joint Training với Intra-Layer Gate Fusion ]    │
│     ┌────────────────────────────────────────────────┐    │
│     │ CƠ CHẾ CỔNG ĐỐI XỨNG (BIPARTITE SYMMETRY)      │    │
│     │ - Trộn LLM + GNN bên trong mỗi layer           │    │
│     │ - Áp dụng đồng thời cho cả User và Item        │    │
│     │ - Nonlinear MLP Gate (Freq, Sim, Niche signals)│    │
│     └────────────────────────────────────────────────┘    │
│                                                           │
│  [ Hàm Mất Mát & Tối Ưu (MTL Loss) ]                      │
│  - BPR Loss với 3-Component Negative Sampling             │
│  - LAGCL Contrastive Loss (Noise ∝ 1/freq)                │
└──────────────────────────┬────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────┐
│               4. DỰ ĐOÁN & ỨNG DỤNG                       │
│                                                           │
│  [ Retrieval & Re-ranking ]     [ Zero-Shot Demo ]        │
│  - FAISS Vector Search          - Cold-Item Insertion     │
│  - Popularity-Penalty           - Potential Buyer Search  │
│  → Gợi ý cân bằng Head/Tail     → Zero-shot Recommendation│
└───────────────────────────────────────────────────────────┘
```

---

## SLIDE 27: CƠ CHẾ INTRA-LAYER GATE FUSION — TOÁN HỌC & CƠ CHẾ

**Lõi đột phá — Trộn LLM + GNN BÊN TRONG mỗi layer GNN:**
Thay vì chỉ trộn ở đầu (Early) hoặc cuối (Late), TA-RecMind can thiệp trực tiếp vào quá trình lan truyền (Message Passing), quyết định linh hoạt lượng thông tin cần lấy từ mỗi nguồn ở từng bước nhảy.

**Bước 1 — Sinh cổng (Node-wise, Per-Layer):**
Cổng $\gamma_v^{(l)}$ (giá trị từ 0 đến 1) xác định tỷ lệ giữ lại thông tin GNN vs LLM. Được học tự động qua mạng nơ-ron (MLP) với 4 tín hiệu đầu vào:
$$logit_v^{(l)} = MLP([E_v^{(l)} \parallel z_v^L]) + w_{sim} \cdot \cos(E_v^{(l)}, z_v^L) + w_{freq} \cdot \log(1 + freq_v) - w_{niche} \cdot niche\_ratio_v$$
$$\gamma_v^{(l)} = \sigma(logit_v^{(l)})$$
*(Trong đó: $E_v^{(l)}$ là GNN embedding hiện tại, $z_v^L$ là LLM embedding cố định).*

**Bước 2 — Hòa trộn (Fused State):**
Trạng thái của node được pha trộn trước khi truyền đi:
$$\hat{E}_v^{(l)} = \gamma_v^{(l)} \cdot E_v^{(l)} + (1 - \gamma_v^{(l)}) \cdot z_v^L$$

**Bước 3 — Lan truyền (Message Passing):**
Chỉ gửi trạng thái đã hòa trộn sang các node lân cận (công thức tường minh của LightGCN):
$$E_v^{(l+1)} = \sum_{u \in N_v} \frac{1}{\sqrt{|N_v| |N_u|}} \hat{E}_u^{(l)}$$

**Bước 4 — Layer Readout (Tổng hợp biểu diễn cuối):**
Trung bình cộng các tầng đã hòa trộn kết hợp với tín hiệu LLM ban đầu:
$$h_v = \alpha \cdot \left( \frac{1}{L+1} \sum_{l=0}^{L} \hat{E}_v^{(l)} \right) + (1-\alpha) \cdot z_v^L$$

**Hành vi tự động (Smart Logic):**
- **Hard Masking (Pure Cold-Start):** Khi $freq_v = 0$, tín hiệu $\log(1+0)$ triệt tiêu, hệ thống đẩy $\gamma \approx 0 \rightarrow$ **Sử dụng 100% LLM Embedding**. Giải quyết hoàn hảo bài toán vật phẩm mới chưa ai tương tác.
- **Niche Lovers (Người dùng ngách):** Định nghĩa qua tỷ lệ $niche\_ratio_u = \frac{\sum_{i \in N_u} \mathbb{I}(freq_i \le \tau_{tail})}{|N_u|}$. Người dùng có chỉ số này cao sẽ làm giảm $\gamma$, buộc hệ thống tin vào LLM để hiểu sở thích "dị" của họ, thay vì bị GNN kéo về xu hướng số đông (Head).
- **Nonlinear MLP:** Tự động điều tiết mối quan hệ phi tuyến tính phức tạp giữa Collaborative (Hành vi) và Semantic (Ngữ nghĩa).

---

## SLIDE 28: TÍNH MỚI CỦA CƠ CHẾ CỔNG ĐỐI XỨNG (BIPARTITE SYMMETRY)

**1. Tính mới vượt trội so với RecMind (Xue et al., 2025):**
- **RecMind gốc:** Thiết kế bất đối xứng — Chỉ có cổng cho Item, dùng GNN thuần cho User $\rightarrow$ Bất lực với Inactive/Cold-start User.
- **TA-RecMind (Ours):** Thiết kế Gate đối xứng (Bipartite Symmetry) — Áp dụng chung cơ chế cổng linh hoạt cho **cả User và Item**. 

**2. Khắc phục vấn đề Cold-start User:**
- **User INACTIVE** (freq thấp): Cổng Graph đóng chặn nhiễu $\rightarrow$ Trích xuất trực tiếp sở thích từ **User Text Profile** (tổng hợp từ review cũ).
- **User SUPER_ACTIVE** (freq cao): Cổng Graph mở rộng $\rightarrow$ Tận dụng tối đa sức mạnh lan truyền của Collaborative Filtering.
- **Kết quả:** Xóa bỏ ranh giới cứng nhắc giữa người dùng mới và cũ.

---

## SLIDE 28B: PHÂN TÍCH CHUYÊN SÂU — TẠI SAO INTRA-LAYER ĐÁNH BẠI CÁC KIẾN TRÚC KHÁC?

**1. Sự thất bại của LATE FUSION đối với Tail Items:**
- **Công thức:** $E^{(L)} = \hat{A}^L E^{(0)}$ (Chạy xong GNN) $\rightarrow$ $h_v = \alpha E_v^{(L)} + (1-\alpha) z_v^L$ (Mới trộn LLM)
- **Bản chất vấn đề:** Trong quá trình truyền tin qua $L$ tầng, tín hiệu của Tail Items (vốn rất yếu, chỉ nối với 1-2 Head Users) sẽ bị **"nuốt chửng" (Over-smoothing)** bởi đặc trưng của các Head lân cận. Khi trả về $E_v^{(L)}$, biểu diễn của Tail Item đã bị méo mó hoàn toàn. Việc cộng LLM vào lúc này là **quá muộn** để cứu vãn cấu trúc đã hỏng.

**2. Điểm yếu của EARLY FUSION:**
- **Công thức:** $E_v^{(0)} = \gamma E_v^{init} + (1-\gamma) z_v^L$ (Trộn 1 lần duy nhất ở đầu) $\rightarrow$ Chạy GNN L tầng.
- **Bản chất vấn đề:** Trọng số $\gamma$ bị tĩnh, không thích ứng theo từng layer. Khi đi sâu vào các tầng ($l=1, 2...$), do bản chất thưa thớt của đồ thị đuôi dài, tín hiệu nguyên bản của LLM dần **bị phai nhạt và pha loãng**. GNN dần "quên" mất thông tin ngữ nghĩa ban đầu.

**3. Sự vượt trội của INTRA-LAYER FUSION (TA-RecMind):**
- **Công thức:** $\hat{E}_v^{(l)} = \gamma_v^{(l)} E_v^{(l)} + (1 - \gamma_v^{(l)}) z_v^L$ (Trộn trước mỗi bước nhảy)
- **Giải pháp:** 
  - Tại **bất kỳ** bước lan truyền nào, cổng động $\gamma$ sẽ đánh giá xem biểu diễn hiện tại có đang bị nhiễu hay phai nhạt không. Nếu có, nó lập tức mở ra để **bơm thêm một liều "nhiên liệu ngữ nghĩa" thuần khiết** từ $z_v^L$ vào node trước khi truyền đi tiếp.
  - **💡 Dòng chảy ngữ nghĩa (Semantic Flow) - Insight cốt lõi:** Nhờ cơ chế này, tín hiệu LLM cực nét của một Cold-Item không bị giam giữ tại chỗ. Nó trực tiếp "bơi" ngược lên đồ thị, truyền sang User đầu tiên tương tác với nó ($E_u^{(1)} \leftarrow \hat{A}_{ui} \hat{E}_i^{(0)}$), giúp tự động cập nhật *real-time* sở thích của User bằng thông tin ngữ nghĩa của Item.

---

## SLIDE 29: CHIẾN LƯỢC HÀM MẤT MÁT TRONG TA-RECMIND (ÁP DỤNG)

**Hàm mất mát tổng thể (Multi-Task Learning):**
$$L_{total} = L_{BPR} + \lambda_1 \cdot (L^U_{align,tw} + L^I_{align,tw}) + \lambda_2 \cdot L_{cl} + \beta \cdot \Omega$$

| Thành phần | Vai trò trong dự án |
|---|---|
| **$L_{BPR}$** | Học xếp hạng sản phẩm (Positive > Negative) dựa trên tương tác thực. |
| **$L_{align,tw}$** | **Đóng góp mới:** Căn chỉnh semantic (LLM) và collaborative (GNN), ưu tiên Tail. |
| **$L_{cl}$** | Contrastive loss với LAGCL noise, tăng cường biểu diễn cho sản phẩm ngách. |
| **$\Omega$** | L2 Regularization, kiểm soát độ lớn tham số, chống overfitting. |

**So sánh sự nâng cấp:**
- **RecMind:** Chỉ căn chỉnh đồng đều cho mọi Items.
- **TA-RecMind:** Căn chỉnh đối xứng (User & Item) + Trọng số hóa theo tần suất (Tail-weighted) + Tăng cường tương phản (LAGCL).

---

## SLIDE 30: ĐỘT PHÁ VỀ CĂN CHỈNH & LẤY MẪU (NOVELTY)

**1. Tail-Weighted Alignment Loss:**
- **Vấn đề:** Tail items có gradient alignment rất nhỏ, model thường bỏ qua.
- **Giải pháp:** Nhân trọng số tỷ lệ nghịch popularity: $w_v = 1 / \log(1 + freq_v)$
- **Kết quả:** Gradient căn chỉnh của Tail Items **gấp ~16 lần** so với Head Items cực đoan.

**2. 3-Component Negative Sampling (Chiến lược lấy mẫu tiêu cực):**
Để giải quyết Popularity Bias, ta không lấy mẫu đồng đều (Uniform) mà chia làm 3 phần:
- **Thành phần 1 (40%):** Uniform Sampling (Giữ phân phối tự nhiên).
- **Thành phần 2 (40%):** Popularity-biased ($freq^{0.75}$) — Buộc model phân biệt các Head Items.
- **Thành phần 3 (20%):** Tail-edge oversampling — Đảm bảo gradient từ các vùng ngách (Tail).

$\rightarrow$ Giúp mô hình không chỉ học từ các sản phẩm phổ biến mà còn học sâu về các sản phẩm ít tương tác.

---

## SLIDE 31: CHIẾN LƯỢC HUẤN LUYỆN 2 GIAI ĐOẠN

**Giai đoạn 1 — Warm-up Alignment (5-10 epochs):**
- **Mục tiêu:** Đưa không gian ngữ nghĩa (LLM) và không gian cộng tác (Graph) về cùng một hệ quy chiếu trước khi trộn.
- **Loss:** $L_{warmup} = L^U_{align,tw} + L^I_{align,tw}$
- **Learning Rate:** 5e-4
- **Cập nhật (Update):** Lớp chiếu ($W_{proj}$), trọng số cổng (Gate MLP, $w_{freq}, w_{sim}$).
- **Đóng băng (Frozen):** Mô hình LLM (đã cache offline) và Graph Embeddings (để giữ cấu trúc CF ban đầu không bị méo).

**Giai đoạn 2 — Joint Training (50-100 epochs):**
- **Loss:** $L_{total}$ đầy đủ (BPR + Alignment + Contrastive).
- **Learning Rate:** 1e-3 với Cosine Decay.
- **Cập nhật:** Toàn bộ tham số mạng (Graph Embeddings, Gates, Projection).
- **Early stopping:** Dựa trên **NDCG@20** trên tập Validation, patience = 10 epochs.

**Auto-scaling Mini-batch theo GPU (Chống OOM):**

| Loại GPU | VRAM | Batch Size | Gradient Accumulation |
|---|---|---|---|
| T4/K80 (Colab Free) | 16GB | 2048 | 4 bước |
| V100/L4 (Kaggle/Pro) | 20-24GB | 6144 | 2 bước |
| A100 (Colab Pro+) | 40-80GB | 8192 | 1 bước |

---

## SLIDE 32: POPULARITY-PENALIZED RE-RANKING

**Two-Stage Inference:**

**Stage 1 — Retrieval (top-200):**
```
scores = h_u @ H_items.T       [1 vector × N_items]
top_200 = topk(scores, 200)    [FAISS: < 10ms với 1M items]
```

**Stage 2 — Re-ranking (Đóng góp mới):**
```
s_adj(u,i) = s_model(u,i) - λ_penalty · log(1 + train_freq(i))
top_K = topk(s_adj, K)
```

- Head items (train_freq cao): bị trừ nhiều → hạ điểm
- Tail items: bị trừ ít → đẩy lên cao hơn
- **Không cần retrain** — chỉ tune λ_penalty trên val set

---

## SLIDE 33: GIAO THỨC ĐÁNH GIÁ LONG-TAIL

**Chronological Leave-One-Out:**
- Test: interaction mới nhất mỗi user (1 item/user)
- Val: interaction mới thứ hai
- Train: phần còn lại (≥ 3 items/user do Core-5)

**Metrics bắt buộc:**

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| Recall@K | hit/1 (leave-one-out) | Item đúng có trong top-K? |
| NDCG@K | 1/log₂(rank+1) | Penalize vị trí thấp |
| Tail Recall@K | Recall chỉ cho TAIL items | **Metric chính** |
| Tail Coverage@K | distinct tail trong top-K / |TAIL| | Độ đa dạng |
| Cold Recall@K | Recall chỉ cho cold-start | Gate Fusion hiệu quả? |

**Phân tầng bắt buộc:** HEAD / MID / TAIL / COLD-START / INACTIVE users

---

## SLIDE 34: BASELINES SO SÁNH & ABLATION STUDY

**Baselines chính:**

| Model | Loại | Lý do chọn |
|---|---|---|
| MF (Matrix Factorization) | CF truyền thống | Baseline đơn giản nhất |
| LightGCN | GNN | Base model của TA-RecMind |
| SGL | GNN + Contrastive | So sánh contrastive learning |
| RecMind | LLM + GNN | Framework nền tảng |
| **TA-RecMind (ours)** | LLM + GNN + Long-tail | **Đề xuất đề tài** |

**Ablation Study bắt buộc:**

| Variant | Thay đổi | Mục đích đo |
|---|---|---|
| Full | TA-RecMind đầy đủ | Baseline đề tài |
| w/o Tail-Weight | Bỏ w_v khỏi alignment | Đóng góp tail weighting |
| w/o LAGCL | Bỏ L_cl | Đóng góp contrastive |
| w/o Re-ranking | Bỏ popularity penalty | Đóng góp re-ranking |
| w/o LLM | Chỉ LightGCN | Đóng góp LLM embedding |
| w/o Gate | Late Fusion thay Intra-Layer | Đóng góp Gate |
| w/o Bipartite | Gate chỉ cho Item | Đóng góp Bipartite Symmetry |

---

## SLIDE 35: HYPERPARAMETERS TỔNG HỢP

| Hyperparameter | Giá trị | Ghi chú |
|---|---|---|
| Embedding dim d | 128 | Tăng từ 64, cải thiện sức chứa |
| LightGCN layers L | 2 | Tránh over-smoothing |
| LR (Warmup) | 5e-4 | Chỉ LoRA + W_proj + gate |
| LR (Joint) | 1e-3 | Tất cả params |
| Weight Decay | 1e-4 | L2 regularization |
| λ₁ (Alignment) | 0.2 | Trọng số alignment loss |
| λ₂ (Contrastive) | 0.1 | Trọng số LAGCL |
| τ (Temperature) | 0.15 | InfoNCE temperature |
| λ_penalty | 0.3 | Re-ranking strength |
| λ_t (temporal) | 1.0 | Temporal decay |
| α (final fusion) | 0.6 | z^G vs z^L balance |

---

## SLIDE 36: DEMO — TAB 1: KHÁM PHÁ NHÓM NGƯỜI DÙNG

**Platform:** Streamlit trên Google Colab

**Sidebar Controls:**
- Dropdown: Inactive (<5) / Active (5-20) / Super Active (>20)
- Random User button
- Top-K slider (20/40)
- λ_penalty slider (0.0 → 1.0)
- Diversity Control: tỷ lệ Head/Mid/Tail mục tiêu

**Layout 2 cột:**
- **Cột Trái — Lịch sử:** Sản phẩm user đã tương tác, badge HEAD/MID/TAIL
- **Cột Phải — Gợi ý Top-K:** Grid sản phẩm với badge màu nổi bật:
  - 🔴 HEAD (>20 tương tác), 🟣 MID, 🟡 TAIL (≤5), 🔵 COLD (0)

**Progress Bar phân phối:** HEAD █████ 40% | MID ██ 15% | TAIL ████ 35% | COLD █ 10%

---

## SLIDE 37: DEMO — TAB 2: ZERO-SHOT COLD-START

**Mục đích:** Chứng minh LLM Fusion — bán hàng mới ngay từ giây đầu tiên.

**Input:** Các trường thông tin chuẩn (Title, Features, Category, Description, Details)

**Luồng xử lý:**
1. Tiền xử lý văn bản (Clean, Lowercase, Remove HTML/URLs)
2. Ghép chuỗi theo hierarchy: `Title | Features | Category | Desc | Details`
3. Cắt chuỗi theo budget (384 tokens ~ MiniLM limit)
4. SentenceTransformer encode → vector 128 chiều (sau projection)
5. Cosine similarity với toàn bộ item matrix → **Similar Items**
6. Cosine similarity với toàn bộ user matrix → **Potential Buyers**

**Output 2 panel:**
- Panel 1: Top-5 sản phẩm tương đồng trong catalog
- Panel 2: Top-5 khách hàng tiềm năng + lịch sử mua hàng

**Hiệu năng:** < 200ms (bao gồm encoding ~150ms)

---

## SLIDE 38: DEMO — KỊCH BẢN THUYẾT PHỤC

**Kịch bản 1 — Inactive User:**
1. Chọn "Inactive" → Random User → chỉ 3 interactions (chuột, bàn phím, headset gaming)
2. Gợi ý: laptop gaming accessories, RGB keyboard, gaming chair
3. "Gu" hoàn toàn khớp dù chỉ 3 tương tác → Gate LLM mở, đọc review text

**Kịch bản 2 — Tăng λ_penalty:**
1. Kéo λ_penalty 0 → 1 → TAIL progress bar tăng 20% → 50%
2. Chứng minh re-ranking điều chỉnh phân phối **không cần retrain**

**Kịch bản 3 — Cold-Start Insertion:**
1. Nhập sản phẩm mới (vd: Tên: Tai nghe chạy bộ, Features: Chống nước IPX7, pin 12h...)
2. Hệ thống xây dựng "Item Text Profile" chuẩn như lúc training
3. Similar items: chính xác về ngữ nghĩa (các loại tai nghe thể thao)
4. Potential buyers: lịch sử mua khớp hoàn hảo → WOW moment

---

## SLIDE 39: LƯU TRỮ & HUGGINGFACE HUB

**3 Repositories trên HuggingFace:**

**1. chuongdo1104/amazon-2023-bronze:**
- bronze_train.parquet/, bronze_val.parquet/, bronze_test.parquet/
- bronze_meta.parquet

**2. chuongdo1104/amazon-2023-silver:**
- silver_item_popularity.parquet/
- silver_item_text_profile.parquet/
- silver_user_text_profile.parquet/
- silver_interactions_*.parquet, silver_val_ground_truth.parquet/

**3. chuongdo1104/amazon-2023-gold:**
- gold_item_id_map.parquet, gold_user_id_map.parquet
- gold_edge_index.npy ([2, E] PyG format)
- Các file `.npy` metadata (train_freq, popularity_group...)
- gold_dataset_stats.json

---

## SLIDE 40: QUẢN LÝ BỘ NHỚ & TỐI ƯU

**Nguyên tắc Cache Spark:**

| Tình huống | Hành động |
|---|---|
| DataFrame dùng nhiều lần | cache() + count() materialize |
| DataFrame dùng 1 lần | Không cache |
| Lineage > 5 transform | Checkpoint ra MinIO |
| Bảng nhỏ join bảng lớn | F.broadcast() bắt buộc |

**Chống OOM Training:**

| Nguyên nhân | Giải pháp |
|---|---|
| Batch size quá lớn | Auto-scaling theo VRAM |
| LLM inference trong loop | Offline cache .npy |
| Gradient tích lũy | zero_grad() đúng lúc |
| LLaMA Scaling (Tương lai) | LoRA r=8, gradient checkpointing |

**PYTORCH_ALLOC_CONF="expandable_segments:True"** — chống phân mảnh VRAM

---

## SLIDE 41: CÔNG NGHỆ SỬ DỤNG

| Công nghệ | Phiên bản | Vai trò |
|---|---|---|
| Apache Spark | 3.5.2 | Xử lý dữ liệu phân tán (Bronze, Silver) |
| MinIO | Latest | S3-compatible local object storage |
| Docker Compose | 24.x | Container orchestration |
| HuggingFace Hub | Latest | Artifact storage & sharing |
| PyTorch | 2.x | Training & inference |
| PyTorch Geometric | 2.x | Graph Neural Network |
| Sentence-Transformers | 2.x | LLM semantic embedding |
| FAISS | 1.7.x | Approximate nearest neighbor |
| Streamlit | 1.x | Demo application |
| Google Colab | GPU T4/L4 | Training environment |

---

## SLIDE 42: BẢNG NGUỒN GỐC CÔNG THỨC

| Công thức | Nguồn | Loại |
|---|---|---|
| E^(l+1) = Â·E^(l) | LightGCN (He 2020) | Trích dẫn |
| z^L = W_proj·Pool(F(T)) | RecMind Eq.3-4 | Trích dẫn |
| **γ = σ(w_base + w_sim·cos + w_freq·log1p)** | **Novel** | **Đóng góp mới** |
| Ê = γ·z^G + (1-γ)·z^L | RecMind Eq.7 | Trích dẫn + mở rộng |
| h = α·z^G + (1-α)·z^L | RecMind Eq.9 | Trích dẫn |
| **w_v·InfoNCE (tail-weighted)** | **Novel** | **Đóng góp mới** |
| L_BPR = -Σ log σ(s_pos - s_neg) | Rendle 2009 | Trích dẫn |
| σ_noise ∝ 1/freq | LAGCL | Trích dẫn |
| **w(r) = 1 + log(1 + helpful_vote)** | **Novel** | **Đóng góp mới** |
| **s_adj = s - λ·log(1+freq)** | **Novel** | **Đóng góp mới** |
| **Anti-leakage: train_freq** | **Novel** | **Đóng góp mới** |

---

## SLIDE 43: TỔNG HỢP 5 ĐIỂM MỚI (NOVEL CONTRIBUTIONS)

**1. Adaptive Gate Fusion (Bipartite Symmetry):**
- Gate động dựa trên frequency + cosine similarity
- Áp đối xứng cho CẢ User và Item (RecMind chỉ cho Item)
- Giải quyết cold-start User song song với cold-start Item

**2. Tail-Weighted Alignment Loss:**
- w_v = 1/log(1+train_freq) nhân vào InfoNCE
- Gradient tail gấp 16× so với max head
- Buộc optimizer ưu tiên căn chỉnh z^G ↔ z^L cho tail

**3. Anti-Leakage Classification:**
- Phát hiện rating_number chênh 137-455× so với train_freq
- Dùng train_freq thay vì rating_number → chặn data leakage

**4. Review Quality Weighting:**
- w(r) = 1 + log(1 + helpful_vote)
- Giải quyết rating skew (65.7% = 5 sao)

**5. Popularity-Penalized Re-ranking:**
- s_adj = s_model - λ·log(1+train_freq)
- Điều chỉnh phân phối tại inference, không cần retrain

---

## SLIDE 44: KẾT LUẬN & HƯỚNG PHÁT TRIỂN

**Kết luận:**
- Xây dựng thành công Data Pipeline Medallion xử lý ~44M tương tác
- Thiết kế mô hình TA-RecMind: LightGCN + LLM + Intra-Layer Gate Fusion
- 5 đóng góp kỹ thuật mới giải quyết Long-tail và Cold-start
- Demo trực quan: gợi ý Top-K + Zero-Shot Cold-Start Insertion
- Đánh giá phân tầng toàn diện (HEAD/MID/TAIL/COLD/INACTIVE)

**Hướng phát triển tương lai:**
1. **Multi-category expansion:** Mở rộng sang nhiều danh mục (không chỉ Electronics)
2. **Advanced LLM Scaling:** Nâng cấp từ MiniLM lên **LLaMA-3.2-1B + LoRA** để khai thác ngữ nghĩa sâu hơn
3. **Item-Item semantic edges:** Thêm cạnh item-item dựa trên cosine similarity LLM
4. **Real-time inference:** Tích hợp FAISS IndexIVFFlat cho approximate top-K (O(√N))
5. **A/B Testing:** Triển khai thử nghiệm thực tế đo lường CTR và conversion rate

---
**Trân trọng cảm ơn Thầy Cô và các bạn đã lắng nghe!**
