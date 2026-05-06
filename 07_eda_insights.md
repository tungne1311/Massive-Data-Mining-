# Phân Tích EDA và Insights Dữ Liệu

## Tổng Quan

Quá trình Phân Tích Dữ Liệu Khám Phá (EDA) của TA-RecMind được thực hiện qua ba giai đoạn:

1. **Bronze EDA** (`src/EDA_Bronze_V2.ipynb`): Phân tích dữ liệu thô sau Ingestion và Chronological Split — định hình vấn đề cấu trúc đồ thị, xu hướng tương tác, rủi ro Data Leakage.
2. **Silver EDA** (`src/EDA_Silver_V2.ipynb`): Phân tích chuyên sâu Text Pipeline (Word length, Token length, Noise detection) và kỹ thuật tính toán trọng số cạnh.
3. **Gold EDA** (`src/EDA_gold (1).ipynb`): Xác nhận cấu trúc ID Mapping, edge list, và training metadata arrays.

Các insights dưới đây **trực tiếp quyết định** kiến trúc và cấu hình của toàn bộ AI Pipeline.

---

## 1. Phân Tích Ưu Nhược Điểm Dataset (Amazon Electronics 2023)

Việc lựa chọn Amazon Electronics 2023 không chỉ dựa trên quy mô mà còn dựa trên các đặc tính kỹ thuật phù hợp với mục tiêu giải quyết Long-tail và Cold-start.

### Ưu Điểm (Pros)
- **Chuẩn Academic Quốc Tế:** Là tập dữ liệu kế thừa từ chuỗi dataset Amazon danh tiếng của McAuley Lab. Việc sử dụng dataset này giúp đề tài có tính kế thừa và dễ dàng đối sánh (benchmark) với các nghiên cứu SOTA (State-of-the-Art) trong lĩnh vực RecSys.
- **Hệ Sinh Thái Metadata Phong Phú:** Khác với các dataset chỉ có ID (như MovieLens), Amazon cung cấp Title, Features, Description, và Details. Đây là "nguyên liệu" vàng để thực hiện **LLM-GNN Alignment**, cho phép hệ thống hiểu được bản chất sản phẩm ngay cả khi chưa có tương tác (Cold-start).
- **Tín Hiệu Chất Lượng Khách Quan (`helpful_vote`):** Cung cấp một chiều dữ liệu về sự tin cậy của review. Điều này cho phép TA-RecMind xây dựng cơ chế **Review Quality Weighting**, vượt qua hạn chế của việc chỉ dựa vào Rating (vốn bị skew nặng).
- **Tính Chính Xác Về Thời Gian:** Timestamp đi kèm từng interaction cho phép thực hiện **Chronological Split** chuẩn xác. Điều này cực kỳ quan trọng để đảm bảo không có rò rỉ dữ liệu từ tương lai vào quá khứ, giúp kết quả đánh giá phản ánh đúng thực tế triển khai.

### Nhược Điểm (Cons)
- **Tính Tĩnh Của Dữ Liệu (Static Data):** Dataset chỉ lưu trữ kết quả cuối cùng (Review/Rating), thiếu vắng các dữ liệu hành vi dạng chuỗi (Session-based) như: click-stream, thời gian dừng (dwell time), hoặc các hành động "thêm vào giỏ hàng" nhưng không mua.
- **Hạn Chế Gợi Ý Real-time:** Do không có dữ liệu tương tác người dùng theo thời gian thực (real-time user activity), hệ thống TA-RecMind tập trung vào tối ưu hóa gợi ý **Personalized Offline** thay vì gợi ý dựa trên session hiện hành. Đây là một giới hạn cần lưu ý khi so sánh với các hệ thống thương mại thực tế.
- **Độ Lệch Phân Phối (Data Skew):** Phân phối Power-law cực đoan (80/20) tạo ra thách thức lớn cho việc học. Tuy nhiên, trong khuôn khổ đề tài này, đây lại được xem là "môi trường thử nghiệm lý tưởng" để chứng minh tính hiệu quả của các cơ chế Gate Fusion và Tail-Weighted Loss.

---

## 2. Hệ Sinh Thái Dữ Liệu (Macro Statistics)

### Quy Mô Đồ Thị (Tập Train)

| Chỉ Số | Giá Trị | Nhận Xét Kiến Trúc |
|---|---|---|
| Raw interactions (Silver) | ~44,066,834 | Bao gồm mọi rating |
| Positive interactions (rating ≥ 3) | ~36,168,550 | Filtered tại Bronze |
| Tổng Users | 1,847,662 | Kích thước không gian Users |
| Tổng Items (Train) | 1,042,121 | Kích thước không gian Items |
| Tổng Interactions (Train) | 1,396,428 | Số cạnh thực tế của đồ thị Bipartite |
| **Sparsity** | **99.9993%** | Mức thưa thớt cực đoan |

**Insight:** Sparsity 99.9993% giải thích vì sao Collaborative Filtering truyền thống thất bại trên Long-tail. Hệ thống bắt buộc phải dựa vào **LightGCN** (lan truyền item-item similarity gián tiếp) kết hợp **LLM semantic embedding** để làm giàu vector cho nodes ít tương tác.

### Thống Kê Bậc Nút (Node Degree Skewness)

- **Item Node:** Min=1, Median=2, Avg=1.34, **Max=41,183**
- **User Node:** Min=3\*, Median=5, Avg=7.56, **Max=1,005**

\*(Min User degree = 3 do Core-5 filter tại Landing Zone, sau đó trừ 1 cho Val và 1 cho Test)

**Hệ Quả (Bottleneck):** Chênh lệch khổng lồ Median=2 vs Max=41,183 của Item Node chứng minh "Head thống trị". Các nút max degree tạo gradient lớn lấn át Tail items trong LightGCN propagation. Đây là lý do **Intra-Layer Gate Fusion** và **Tail-Weighted Alignment Loss** ra đời.

### Kiểm Tra Time Leakage (Chronological Split)

Qua truy vấn Double Max Join kiểm chứng tập Train/Val/Test:
- Valid Time-split Users: **100%**
- Leakage Train-Val (Train time > Val time): **0**
- Leakage Val-Test (Val time > Test time): **0**

→ Giao thức chia thời gian hoạt động chính xác tuyệt đối.

---

## 2. Phân Phối Vùng Đuôi Dài (Power-Law & Data Leakage)

### Phân Phối Power-Law (Pareto 80/20)

Biểu đồ Item Popularity trên Log-Log scale là đường thẳng dốc đứng. CDF cho thấy:

- **Head Items (Top 20%):** Đóng góp ~80% tổng tương tác (Pareto principle — chuẩn trên Amazon Dataset)
- **Tail Items (≤ 5 tương tác):** **755,609 items** (72.51% items trong train) — hàng trăm nghìn item chỉ có đúng 1 tương tác

**Implication cho thiết kế ngưỡng:**
```
HEAD:  Top 20% items theo train_freq  → chiếm ~80% tổng interactions
MID:   10% items tiếp theo            → vùng đệm
TAIL:  Bottom 70% items còn lại       → 72.51%, mục tiêu chính
```

### Rủi Ro Data Leakage (rating_number vs train_freq)

Chỉ có 161,001 items có metadata đầy đủ. Khi so sánh `metadata.rating_number` và `train_freq`:

```
Item B07H65KP63:
  train_freq    = 1,561   (trong tập train)
  rating_number = 710,348 (tổng lịch sử)
  → Chênh lệch ~455 lần → leakage nghiêm trọng

Trường hợp khác:
  train_freq    = 4,265
  rating_number = 585,624
  → Chênh lệch ~137 lần
```

**Quyết định thiết kế:** `rating_number` chỉ dùng để hiển thị. Phân loại HEAD/MID/TAIL **duy nhất** dựa vào `train_freq` (quá khứ) để chặn hoàn toàn Data Leakage.

---

## 3. Khám Phá Đặc Trưng Ngôn Ngữ (NLP Insights)

### Mức Độ Hoàn Thiện Metadata (Completeness)

Không phải Amazon item nào cũng đầy đủ 4 trường văn bản:
- **Has Title:** Gần như 100%
- **Has Features:** Tỷ lệ rỗng tăng mạnh tại nhánh đuôi
- **Has Description:** Nhiều item không có — đặc biệt tail items

**Quyết định:** Áp dụng cơ chế **Field-Aware Fallback**:
```
Title (không cắt) > Features (750 chars nếu thiếu desc) > 
Categories (150) > Description (300) > Details (150)
```

Khi description rỗng, `features` được mở rộng từ 450 lên 750 chars để bù đắp.

### Token Length & Budget (MiniLM max_length=384)

Sau HuggingFace Tokenization:
- **Item Text:** Đa số tập trung 128–256 tokens, rất hiếm > 384 tokens
- **User Text** (ghép top-3 reviews với `[SEP]`): Trung vị ~180 tokens, p95 ≈ 320–350 tokens

**Quyết định:** `max_length = 384` của `all-MiniLM-L6-v2` bao phủ 99%+ context, tiết kiệm hàng chục GB RAM so với `max_length = 768`. Đây là lý do chọn TOP_K=3 reviews thay vì 5.

### Noise Detection

Khoảng 2-5% chuỗi Text chứa:
- Thẻ HTML (`<br>`, `<li>`, `<p>`) hoặc raw Markup
- URL links bên thứ ba
- Multi-space liên tục, ký tự đặc biệt

**Giải pháp trong Silver Step 2:** Dùng `regexp_replace` Spark SQL:
```python
F.regexp_replace(col, r'<[^>]+>', '')    # Remove HTML tags
F.regexp_replace(col, r'http\S+', ' ')   # Remove URLs  
F.regexp_replace(col, r'\s+', ' ')       # Normalize whitespace
```

### WordCloud (Head vs Tail Vocabulary)

Sự đối lập từ vựng:
- **HEAD Items:** Từ chức năng chung: *Cable, Bluetooth, Wireless, Case, Charger, Screen, USB*
- **TAIL Items:** Thông số kỹ thuật lõi, tên mã linh kiện ngách, vật liệu thiết kế chuyên biệt

**Implication:** Tail items cần LLM embedding để nắm bắt ngữ nghĩa chuyên biệt — Collaborative Filtering không thể làm điều này với degree = 1-2.

---

## 4. Quản Lý Trọng Số Cạnh (Edge Dynamics & Decay)

### Rating Distribution & Helpfulness

**Rating:** Thiên kiến cực thịnh tại 5 Sao — 65.7% reviews đạt 5 sao:

| Rating | Tỷ Lệ |
|---|---|
| 5 sao | 65.7% |
| 4 sao | 14.7% |
| 3 sao | 7.0% |
| 2 sao | 4.5% |
| 1 sao | 8.0% |

Việc dùng rating thuần để phân loại chất lượng bị nhiễu cực kỳ nặng. Hầu hết "positive" reviews đều là 5 sao — không có độ phân biệt.

**Helpful Votes:** Chỉ một tỷ lệ nhỏ reviews được vote — đây là tín hiệu chất lượng đáng tin cậy hơn.

**Quyết định:** `w(r) = 1 + log(1 + helpful_vote(r))`:

| helpful_vote | w(r) | Diễn giải |
|---|---|---|
| 0 | 1.00 | Baseline, không loại bỏ |
| 10 | ≈ 3.40 | Review được vote nhiều → nặng hơn |
| 100 | ≈ 5.61 | Review rất được tin cậy |
| 3,294 (max) | ≈ 9.40 | Không bị vô hạn hóa nhờ log |

### Temporal Decay (Độ Lệch Thời Gian)

Khoảng ngày giữa interaction cũ và T_max:
- Đa phần tương tác trải dài hàng nghìn ngày (nhiều năm trước)
- Công thức: `decay = exp(-λ × (T_max - timestamp) / T_range)`
- Với λ=1.0: Trọng số trượt từ ~0.2 (rất cũ) đến 1.0 (gần T_max)

**Quyết định:** Không đưa decay vào Silver (hyperparameter phụ thuộc). Tích hợp trực tiếp vào BPR Loss tại training time:
```python
w_ui = exp(-lambda_t * (T_max - timestamp_ui) / T_range)
bpr_loss = -sum(w_ui * log(sigmoid(s_pos - s_neg)))
```

---

## 5. Môi Trường Đánh Giá & Cold-Start

### Rào Cản Evaluator: Pure Cold-Start

Thống kê từ tập Evaluation (Val + Test):

| Chỉ Số | Giá Trị |
|---|---|
| Total eval items | 564,225 |
| Pure cold-start items | **130,746 (23.17%)** |

**Đánh giá kiến trúc:** 23.17% items trong tập kiểm thử chưa từng xuất hiện ở tập luyện. GNN (LightGCN) hoàn toàn "Mù" với nhóm này (degree = 0). Hệ thống **chỉ** xử lý được nhóm này nếu có LLM Text Embedding được tiêm vào qua Gate Fusion.

**So sánh LightGCN vs TA-RecMind với cold-start:**
```
LightGCN:   z^G_cold = embedding khởi tạo ngẫu nhiên → vô nghĩa
TA-RecMind: γ_cold = 0 (freq=0 → log1p=0 → sigmoid nhỏ)
            → h_cold ≈ z^L_cold (100% từ LLM)
            → Sản phẩm mới được gợi ý dựa hoàn toàn vào Item Text Profile
```

### Ground Truth Grouping

Tại Val/Test, phân bổ popularity group **khác hẳn** Train:
- Train: 28% HEAD, 10% MID, 62% TAIL (theo tương tác)
- Val/Test: phần lớn là TAIL và COLD-START

**Implication cho Evaluation:** Dùng trung bình gộp (Mean Recall) sẽ bị HEAD items che khuất hiệu suất thực của hệ thống. **Bắt buộc phải tách lớp Metrics** (Tail_NDCG@20, Cold_Recall@20) thay vì Overall metrics đơn thuần.

---

## 6. Insights Từ Pipeline Log (Runtime Data)

Dữ liệu từ `data/logs/pipeline.log` xác nhận các thống kê thực tế:

### Silver Step 3 (Silver Cleaning) — Run Thành Công

```
Thời gian xử lý Silver:   ~1.5 giờ (08:40 → 10:13)
Output rows:               44,066,834
avg_reliability:           0.5883
```

### Silver Step 4 (Labeling) — Phân Phối interaction_type

```
strong_positive:  31,214,972  (70.8%) — rating 5 sao + helpful_vote cao
weak_positive:     4,953,578  (11.2%) — rating 4-5 sao, ít helpful_vote
neutral:           1,045,028   (2.4%) — rating 3 sao
medium_negative:   2,035,213   (4.6%) — rating 2 sao
hard_negative:     4,818,043  (10.9%) — rating 1 sao
```

→ Xác nhận pipeline Bronze đã filter đúng (rating ≥ 3.0) — mọi negative vào Silver đều bị loại.

### Silver Step 5 (Temporal Split) — Phân Tách Tập Dữ Liệu

```
Train: 43,925,339 rows | 18,200,233 users | 1,602,487 items
Val:      137,715 rows |    106,094 users |    52,239 items
Test:       3,780 rows |      2,993 users |     3,101 items
```

*(Lưu ý: Đây là số liệu của run cũ Silver trước khi tái cấu trúc pipeline hiện tại — số liệu Bronze EDA phản ánh pipeline mới hơn)*

---

## 7. Tóm Tắt Quyết Định Kiến Trúc Từ EDA

| Insight EDA | Quyết Định Kiến Trúc |
|---|---|
| Sparsity 99.9993% | Dùng GNN (LightGCN) thay vì MF/CF truyền thống |
| Head Median=2, Max=41,183 | Intra-Layer Gate Fusion + Tail-Weighted Loss |
| `rating_number` chênh 455× vs `train_freq` | Anti-Leakage Classification: dùng `train_freq` |
| Rating skew 65.7% = 5 sao | Review weight dựa trên `helpful_vote`, không phải rating |
| 23.17% cold-start trong Val/Test | Gate fusion với γ→0 cho cold-start (100% LLM) |
| Item text: 128-256 tokens | max_length=384 cho MiniLM — bao phủ 99%+ |
| User text: top-3 reviews ≈ 180 tokens median | TOP_K=3 thay vì 5 — tránh OOM |
| Tương tác trải dài nhiều năm | Temporal decay trong BPR Loss (không ở Silver) |
| HEAD items vocabulary: generic | TAIL/COLD items cần LLM semantic embedding |
