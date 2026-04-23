# Giao Thức Đánh Giá Long-tail

## Tổng Quan

Hệ thống gợi ý truyền thống chỉ báo cáo Recall@K và NDCG@K tổng thể — các metrics này ẩn đi việc hệ thống đang bỏ qua tail items. TA-RecMind yêu cầu bộ metrics **phân tầng theo popularity** để đánh giá toàn diện.

---

## Phân Tách Dữ Liệu Đánh Giá

### Chronological Leave-One-Out (RecMind, Sec. V-A)

```
Với mỗi user u, sắp xếp interactions theo timestamp tăng dần:
  → Interaction mới nhất    → Test set
  → Interaction mới thứ hai → Validation set
  → Phần còn lại            → Training set
```

**Đảm bảo không time leakage:**
- Reviews dùng để xây dựng user text profile chỉ từ Training set interactions
- Reviews thuộc validation và test interactions bị loại hoàn toàn trước khi feed vào LLM

**Thống kê phân tách (từ EDA):**

| Tập | Số Users | Số Items | Số Interactions |
|---|---|---|---|
| Train | 1,847,662 | 1,042,121 | 1,396,428 |
| Val | 1,847,662 | 564,225 | 1,847,662 |
| Test | 1,847,662 | ~564,225 | 1,847,662 |

**Cold-start trong Val/Test:**

```
total_eval_items     = 564,225
pure_cold_start      = 130,746  (23.17%)
```

23.17% items trong val/test chưa từng xuất hiện trong train. GNN (LightGCN) hoàn toàn bị mù với nhóm này (degree = 0). Đây là lý do Gate Fusion + LLM embedding là linh hồn của TA-RecMind.

---

## Metrics Chuẩn (Bắt Buộc)

### Recall@K

```
Recall@K(u) = |{positive items của u trong top-K recommendations}| / |{positive items của u}|
```

Với leave-one-out protocol, mỗi user có đúng 1 positive item trong val/test:
```
Recall@K(u) = 1 nếu item đúng trong top-K, 0 nếu không
Recall@K    = (1/|U|) Σ_u Recall@K(u)
```

### NDCG@K (Normalized Discounted Cumulative Gain)

```
NDCG@K(u) = DCG@K(u) / IDCG@K(u)
DCG@K(u)  = Σ^K_{k=1} rel_k / log_2(k+1)
```

Với leave-one-out, `rel_k = 1` nếu item tại vị trí k là positive item, 0 nếu không. NDCG penalize việc positive item xuất hiện ở vị trí thấp trong danh sách.

**Giá trị K:** K = 20 và K = 40 (theo RecMind, Sec. V-C).

**All-item ranking với negative sampling:** Với mỗi user trong evaluation, rank positive item của họ so với 100 negative items được sample ngẫu nhiên (không trùng với positive). Điều này tương đương với all-item ranking nhưng tiết kiệm tính toán hơn.

---

## Metrics Long-tail Đặc Thù (Bắt Buộc Bổ Sung)

### Tail Recall@K

Chỉ tính Recall@K trên subset test interactions có positive item là tail item:

```
Tail Recall@K = (1/|U_tail|) Σ_{u: pos_item(u) ∈ TAIL} Recall@K(u)
```

Trong đó `U_tail = {u | positive item của u trong eval set là TAIL item}`.

Đây là metric trực tiếp nhất đo hiệu quả của hệ thống với mục tiêu chính. Một hệ thống có thể có Recall@20 cao nhờ gợi ý tốt với head items nhưng Tail Recall@20 thấp — và đây là thất bại đối với mục tiêu của đề tài.

### Tail NDCG@K

Tương tự, tính NDCG@K chỉ cho tail interactions.

### Tail Coverage@K

Tỷ lệ distinct tail items xuất hiện trong toàn bộ recommendations trên tất cả users:

```
Coverage_tail(K) = |{i ∈ TAIL : ∃u, i ∈ Top-K(u)}| / |TAIL|
```

**Ý nghĩa:** Hệ thống có thể có Tail Recall@K cao nhờ gợi ý đúng cho một số user, nhưng coverage thấp nếu luôn gợi ý cùng vài tail items phổ biến. Coverage đo độ đa dạng của tail items được gợi ý.

**Từ EDA:** `|TAIL| = 755,609 items` (72.51% tổng số items trong train). Coverage@20 = 1.0 nghĩa là tất cả 755,609 tail items đều được gợi ý ít nhất một lần trong toàn bộ users — đây là mục tiêu lý tưởng.

### Popularity Distribution Analysis

Báo cáo phân phối `train_freq` của recommended items trên toàn bộ users, so sánh với baseline. Cụ thể:

```
Median train_freq của recommended items:
  LightGCN baseline: X (cao → thiên về head)
  RecMind:           Y
  TA-RecMind:        Z (thấp hơn → thiên về tail)
```

Vẽ histogram hoặc boxplot để trực quan hóa: TA-RecMind nên có phân phối `train_freq` dịch về phía trái (tail) so với baseline.

---

## Phân Tầng Kết Quả

### Phân Tầng Theo Item Popularity

Báo cáo Recall@K và NDCG@K riêng cho ba nhóm:

```
Nhóm HEAD: positive item ∈ HEAD items (top 20% theo train_freq)
Nhóm MID:  positive item ∈ MID items
Nhóm TAIL: positive item ∈ TAIL items (bottom ~72%)
```

**Điều cần kiểm tra:** Cải thiện TAIL không đến từ chi phí giảm HEAD. Bảng kết quả lý tưởng:

| Model | HEAD Recall@20 | MID Recall@20 | TAIL Recall@20 | Overall |
|---|---|---|---|---|
| LightGCN | High | Medium | Low | Medium |
| RecMind | High | Medium-High | Low-Medium | Medium-High |
| **TA-RecMind** | High | **High** | **High** | **High** |

Nếu TA-RecMind tăng TAIL nhưng giảm HEAD đáng kể — đây là trade-off không chấp nhận được vì ảnh hưởng đến trải nghiệm tổng thể.

### Phân Tầng Theo User Activity (Bipartite Insight)

Báo cáo tương tự nhưng phân nhóm theo cấp độ hoạt động của User (Tận dụng tệp `gold_user_activity_group.npy`):

```
SUPER_ACTIVE (> 20 acts): Tương đương sức mạnh Collaborative (Thiên vị GNN mạnh)
ACTIVE (5 - 20 acts): Hoạt động ổn định
INACTIVE/COLD (< 5 acts): Users cực kỳ lười. Kỳ vọng kiến trúc Node-wise Gate Layer-0 sẽ cứu vãn nhóm này bằng cách ép mở cổng lấy tin từ Profile Review Text.
```

Việc chứng minh sự cải thiện NDCG/Recall tại nhóm **INACTIVE** chính là mũi nhọn tối thượng để bảo vệ tư tưởng của đề tài trước giáo sư rằn ri.

---

## Baselines So Sánh

### Baselines Chính

| Model | Loại | Lý Do Chọn |
|---|---|---|
| **MF** (Matrix Factorization) | CF truyền thống | Baseline đơn giản nhất |
| **LightGCN** | GNN | Base model của TA-RecMind |
| **SGL** | GNN + Contrastive | So sánh contrastive learning |
| **RecMind** | LLM + GNN | Base framework được mở rộng |
| **TA-RecMind (ours)** | LLM + GNN + Long-tail | Đề xuất của đề tài |

### Ablation Study Bắt Buộc

| Variant | Thay Đổi | Mục Đích |
|---|---|---|
| **Full** | TA-RecMind đầy đủ | Baseline của đề tài |
| **w/o Tail-Weight** | Bỏ w_v khỏi alignment loss | Đo đóng góp của tail weighting |
| **w/o LAGCL** | Bỏ L_cl | Đo đóng góp của contrastive loss |
| **w/o Re-ranking** | Bỏ popularity penalty | Đo đóng góp của re-ranking |
| **w/o LLM** | Chỉ dùng LightGCN | Đo đóng góp của LLM embedding |
| **w/o Gate** | Bỏ Gate Fusion | Đo đóng góp của cơ chế gate |

---

## Implementation Đánh Giá

### All-item Ranking

Với mỗi user trong evaluation:
1. Lấy embedding `h_u`
2. Tính điểm với toàn bộ `N_items` items: `scores = h_u @ H_items.T` (matrix multiply)
3. Loại bỏ items đã tương tác trong train (training items)
4. Áp re-ranking nếu evaluating full model
5. Lấy top-K theo score sau re-ranking
6. So sánh với positive item trong eval set

**Tối ưu tốc độ:** FAISS IndexFlatIP cho approximate top-K, kết quả xấp xỉ nhưng nhanh hơn brute-force O(N) nhiều lần. Với 1M items và 1.8M users, brute-force cần ~1.8 triệu phép matrix multiply — không khả thi. FAISS với IndexIVFFlat giảm xuống còn O(sqrt(N)) mỗi query.

### Logging Phân Tầng

Mỗi epoch, tính và log:
```
Overall:   Recall@20, NDCG@20, Recall@40, NDCG@40
HEAD:      Recall@20, NDCG@20
MID:       Recall@20, NDCG@20
TAIL:      Recall@20, NDCG@20, Coverage@20
Cold-start: Recall@20 (cold-start items only)
```

Lưu vào JSON log để vẽ learning curves sau training.

---

## Tiêu Chí Thành Công

Đề tài được coi là thành công nếu TA-RecMind đạt:

1. **TAIL Recall@20 tăng ≥ 20%** so với LightGCN baseline
2. **Overall Recall@20 không giảm quá 5%** so với LightGCN baseline  
3. **Tail Coverage@20 ≥ 30%** (ít nhất 30% tail items được gợi ý cho ít nhất một user)
4. **Cold-start Recall@20 ≥ 50%** của warm items recall (LLM embedding phải hữu ích cho cold-start)

Ngưỡng cụ thể sẽ được điều chỉnh dựa trên kết quả thực nghiệm với tập Electronics.
