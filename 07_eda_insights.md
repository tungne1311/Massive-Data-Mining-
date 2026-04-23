# Phân Tích EDA và Insights Dữ Liệu

## Tổng Quan

Quá trình Phân Tích Dữ Liệu Khám Phá (EDA) của TA-RecMind được thực hiện qua hai giai đoạn:
1. **Bronze EDA**: Phân tích trên dữ liệu thô ngay sau khi Ingestion và Chronological Split, nhằm định hình các vấn đề cấu trúc đồ thị, xu hướng tương tác, và rủi ro Data Leakage.
2. **Silver EDA**: Phân tích chuyên sâu về Text Pipeline (Word length, Token length, Noise detection) và kỹ thuật tính toán trọng số cạnh (Temporal Decay) nhằm chuẩn bị cấu hình tốt nhất cho LLM Embeddings và LightGCN.

Các insights dưới đây trực tiếp quyết định kiến trúc và cấu hình (config) của AI Pipeline.

---

## 1. Hệ Sinh Thái Dữ Liệu (Macro Statistics)

### Quy Mô Đồ Thị (Tập Train)

| Chỉ Số | Giá Trị | Nhận Xét Kiến Trúc |
|---|---|---|
| Tổng Users | 1,847,662 | Kích thước không gian Users |
| Tổng Items (Train) | 1,042,121 | Kích thước không gian Items |
| Tổng Interactions | 1,396,428 | Số lượng cạnh thực tế của đồ thị Bipartite |
| **Sparsity** | **99.9993%** | Mức độ thưa thớt cực đoan đặc trưng của đồ thị Amazon. |

**Insight**: Sparsity 99.9993% giải thích vì sao Collaborative Filtering truyền thống hoặc Matrix Factorization thất bại trên Long-tail. Hệ thống bắt buộc phải dựa vào cơ chế message passing của **LightGCN** (để lan truyền item-item similarity gián tiếp) kết hợp với nội dung văn bản (LLM semantic) để làm giàu vector.

### Thống Kê Bậc Nút (Node Degree Skewness)

Sự thống trị của các Head Items thể hiện rõ rệt qua phân phối bậc (degree) lệch:
- **Item Node**: `Min = 1`, `Median = 2`, `Avg = 1.34`, **`Max = 41,183`**
- **User Node**: `Min = 3`*, `Median = 5`, `Avg = 7.56`, **`Max = 1,005`**

*(Ghi chú: Min User degree = 3 do áp dụng Core-5 filter từ Landing Zone, sau đó trừ đi 1 interaction cho Val và 1 cho Test).*

**Hệ quả (Vấn đề Bottleneck)**: Sự chênh lệch khổng lồ giữa Median và Max degree của Item Node chứng minh hiện tượng "Head thống trị". Các nút max degree sẽ tạo ra một lượng gradient khổng lồ lấn át các Tail items trong quá trình truyền trọng số của LightGCN. Đây là lý do **Gate Fusion** và **Tail-Weighted Alignment Loss** ra đời nhằm ép mô hình chú ý đến Tail.

### Kiểm Tra Time Leakage (Chronological Split)
Thông qua truy vấn Double Max Join kiểm chứng tập Train/Val/Test:
- Valid Time-split Users: **100%**
- Leakage Train-Val (Train time > Val time): **0**
- Leakage Val-Test (Val time > Test time): **0**
-> Giao thức chia thời gian hoạt động chính xác tuyệt đối.

---

## 2. Phân Phối Vùng Đuôi Dài (Power-Law & Data Leakage)

### Phân Phối Power-Law (Pareto 80/20)
Kết quả từ biểu đồ Item Popularity trên đồ thị Log-Log là một đường thẳng dốc đứng. Đồ thị CDF (Tích lũy) cho thấy:
- **Head Items (Top 20%)**: Đóng góp 80% tổng cộng lượt tương tác (nguyên tắc Pareto chuẩn trên Amazon Dataset).
- **Tail Items (≤ 5 tương tác)**: Có tới **755,609 items** (tương đương **72.51%** lượng Items trong tập đào tạo), nhưng đóng góp một lượng tương tác vô cùng nhỏ bé. Đáng chú ý, hàng trăm nghìn item chỉ có đúng 1 tương tác.

### Rủi Ro Data Leakage (Rating Number vs Train Freq)
Chỉ có **161,001 items** có metadata đầy đủ. Khi so sánh biến số \`metadata_rating_number\` (tổng lượt review lấy từ bộ dữ liệu) và \`train_freq\`:
- Có những item \`train_freq = 4,265\` nhưng \`rating_number = 585,624\`. Độ lệch tương lai lọt vào hàng trăm ngàn tương tác.
- **Quyết định**: Metadata `rating_number` chỉ dùng hiển thị. Việc xếp loại Head/Mid/Tail **duy nhất** dựa vào `train_freq` (quá khứ) để chặn hoàn toàn Data Leakage tương lai.

---

## 3. Khám Phá Đặc Trưng Ngôn Ngữ (NLP Insights)

Tầng Silver chuẩn bị đầu vào văn bản cho LLM (Sentence-Transformers). Đánh giá text giúp quy hoạch kích thước vector embedding hiệu quả:

### Mức Độ Hoàn Thiện (Metadata Completeness)
Không phải Amazon item nào cũng đầy đủ 4 trường văn bản:
- **Has Title**: Gần như 100%.
- **Has Features & Description**: Tỷ lệ rỗng rớt mạng khá nhiều tại nhánh đuôi.
**Quyết định**: Áp dụng cơ chế **Cấp bậc ưu tiên (Fallback)**: `Title > Features / Details > Description > Category` để đảm bảo model text không gặp NULL.

### Tokens Length & Budget (Sentence-Transformers MiniLM)
Sau quá trình HuggingFace Tokenization:
- **Item Text**: Độ dài mã phần lớn tập trung từ **128 - 256 tokens**, cực kỳ hiếm mẫu > 384 tokens.
- **User Text** (ghép top 5 reviews bằng \`[SEP]\`): Đạt trung vị khoảng trung bình **~256 tokens**, percentise thứ 95 đạt ngưỡng **~350 - 380 tokens**. 
**Quyết định**: Ngưỡng `max_length = 384` của `all-MiniLM-L6-v2` là cấu hình tối ưu vừa đẹp để bao bọc 99% lượng context mà tiết kiệm hàng chục GB RAM so với `max_length = 768`.

### Noise Detection (Yếu Tố Nhiễu Văn Bản)
Một lượng tỷ lệ các chuỗi Text (chiếm khoảng 2 - 5%) chứa:
- Thẻ HTML (`<br>`, `<li>`) hoặc raw Markup.
- URL links bên thứ ba.
- Multi-space liên tục.
**Đề xuất Pipeline**: Cần Regex Cleaner loại bỏ HTML và URL trong tầng Bronze sang Silver để đảm bảo Embedding thuần "Semantic".

### WordCloud (Head vs Tail Vocab)
Sự đối lập giữa các từ khoá:
- **HEAD Items**: Chiếm sóng bởi từ vựng chức năng chung: *Cable, Bluetooth, Wireless, Case, Charger, Screen...*
- **TAIL Items**: Hướng sang thông số kỹ thuật lõi, tên mã linh kiện ngách, vật liệu thiết kế...

---

## 4. Quản Lý Trọng Số Cạnh (Edge Dynamics & Decay)

### Rating Distribution & Helpfulness
- **Rating**: Thiên kiến cực thịnh tại `5 Sao`. Khoảng > 60% reviews của Amazon đạt 5 sao, khiến việc dùng Rating thuần để phân loại chất lượng bị nhiễu tích cực.
- **Helpful Votes (Review Quality)**: Chỉ một lượng reviews nhất định có độ tin cậy được vote.
**Quyết định thiết kế**: Tích hợp Logarit. `w(r) = 1 + log(1 + helpful_vote(r))` để trọng số tương tác được phóng đại với những đánh giá chất lượng cao, khử nhiễu từ các spam 5sao.

### Temporal Decay (Độ Lệch Thời Gian)
Khoảng ngày giữa tương tác cũ (Timestamp) và mốc giới hạn mới nhất ($T_{max}$):
- Đa phần tương tác rơi vào dải trải dài hàng nghìn ngày (nhiều năm trước).
- Công thức: `decay = exp(-λ × (T_max - timestamp) / T_range)`
- **Kết quả mô phỏng**: Với $\lambda = 1.0$, biểu đồ trọng số cạnh trượt dài từ 0.2 đến 1.0, làm chìm đi tương tác quá cũ, tăng độ nặng cho các tương tác gần $T_{max}$.

---

## 5. Môi Trường Đánh Giá & Cold-Start

### Rào Cản Evaluator: Pure Cold-Start
Thống kê từ tập Evaluation (Val + Test):
- `total_eval_items` = 564,225 items
- `pure_cold_start_items` = **130,746 items (tương đương 23.17%)**

**Đánh giá kiến trúc**: Có tới 23.17% items trong tập Kiểm thử chưa từng xuất hiện ở phần Luyện. Đồ thị Collaborative (LightGCN) hoàn toàn "Mù" (degree = 0) với nhánh này. Hệ thống chỉ xử lý được 23.17% này nếu có biểu diễn Text Embedding (từ LLM) tiêm vào qua Gate Fusion.

### Ground Truth Grouping
- Tại tập Test/Val, phân bổ độ nhóm (Group) khác biệt hẳn so với Train (Tỷ trọng Tail/Cold-Start chiếm thế chủ đạo). Giao thức đánh giá (Evaluation Protocol) bắt buộc phải tách lớp Metrics (Tail_NDCG@20, Head_NDCG@20) thay vì tính trung bình gộp (Mean) truyền thống. Dùng trung bình gộp sẽ bị Head Items che mắt hiệu suất thực của nền tảng.

