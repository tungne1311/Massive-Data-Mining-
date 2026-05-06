# Giao Thức Đánh Giá Long-tail

## Tổng Quan

Hệ thống gợi ý truyền thống chỉ báo cáo Recall@K và NDCG@K tổng thể — các metrics này **ẩn đi** việc hệ thống đang bỏ qua tail items. TA-RecMind yêu cầu bộ metrics **phân tầng theo popularity** để đánh giá toàn diện khả năng phục vụ long-tail và cold-start.

---

## Phân Tách Dữ Liệu Đánh Giá

### Chronological Leave-One-Out (RecMind, Sec. V-A)

```
Với mỗi user u, sắp xếp interactions theo timestamp tăng dần:
  → Interaction mới nhất    → Test set    (1 item/user)
  → Interaction mới thứ hai → Val set     (1 item/user)
  → Phần còn lại            → Train set   (≥ 3 items/user, do Core-5 filter)
```

**Đảm bảo không time leakage:**
- Reviews dùng để xây dựng user text profile chỉ từ Training set interactions
- Reviews thuộc val và test interactions bị loại hoàn toàn trước khi feed vào LLM
- Chronological split kiểm chứng: Leakage Train-Val = 0, Leakage Val-Test = 0 (EDA Bronze)

### Thống Kê Phân Tách (từ EDA)

| Tập | Số Users | Số Items | Số Interactions |
|---|---|---|---|
| Train | 1,847,662 | 1,042,121 | 1,396,428 |
| Val | 1,847,662 | 564,225 | 1,847,662 |
| Test | 1,847,662 | ~564,225 | ~1,847,662 |

**Cold-start trong Val/Test:**
```
total_eval_items    = 564,225
pure_cold_start     = 130,746  (23.17%)
```

23.17% items trong val/test chưa từng xuất hiện trong train. GNN hoàn toàn bị mù (degree = 0). Đây là lý do Gate Fusion + LLM embedding là **linh hồn** của TA-RecMind.

---

## Metrics Chuẩn (Bắt Buộc)

### Recall@K

$$\text{Recall@K}(u) = \frac{|\{\text{positive items của } u \text{ trong top-K}\}|}{|\{\text{positive items của } u\}|}$$

Với leave-one-out protocol (mỗi user có đúng 1 positive item trong val/test):
$$\text{Recall@K}(u) = \begin{cases} 1 & \text{nếu item đúng trong top-K} \\ 0 & \text{nếu không} \end{cases}$$
$$\text{Recall@K} = \frac{1}{|U|} \sum_u \text{Recall@K}(u)$$

### NDCG@K (Normalized Discounted Cumulative Gain)

$$\text{NDCG@K}(u) = \frac{\text{DCG@K}(u)}{\text{IDCG@K}(u)}, \quad \text{DCG@K}(u) = \sum_{k=1}^K \frac{rel_k}{\log_2(k+1)}$$

Với leave-one-out: `rel_k = 1` nếu item tại vị trí k là positive item, 0 nếu không. NDCG penalize việc positive item xuất hiện ở vị trí thấp.

**Giá trị K:** K = 20 và K = 40 (theo RecMind, Sec. V-C).

**Evaluation protocol:** Rank positive item so với 100 negative items sample ngẫu nhiên (tương đương all-item ranking nhưng tiết kiệm tính toán).

---

## Metrics Long-tail Đặc Thù (Bắt Buộc Bổ Sung)

### Tail Recall@K

Chỉ tính Recall@K trên subset test interactions có positive item là tail item:

$$\text{Tail Recall@K} = \frac{1}{|U_{tail}|} \sum_{u: pos\_item(u) \in TAIL} \text{Recall@K}(u)$$

Đây là metric **trực tiếp nhất** đo hiệu quả của hệ thống với mục tiêu chính. Một hệ thống có Recall@20 cao nhờ head items nhưng Tail Recall@20 thấp là **thất bại** đối với mục tiêu đề tài.

### Tail NDCG@K

Tương tự, tính NDCG@K chỉ cho tail interactions.

### Tail Coverage@K

Tỷ lệ distinct tail items xuất hiện trong toàn bộ recommendations trên tất cả users:

$$\text{Coverage}_{tail}(K) = \frac{|\{i \in TAIL : \exists u, i \in \text{Top-K}(u)\}|}{|TAIL|}$$

**Ý nghĩa:** Hệ thống có thể có Tail Recall@K cao nhưng coverage thấp nếu luôn gợi ý cùng vài tail items phổ biến. Coverage đo **độ đa dạng** của tail items được gợi ý.

**Mục tiêu lý tưởng:** Coverage@20 cao nhất có thể — mọi tail item trong catalog đều được gợi ý ít nhất một lần. `|TAIL| = 755,609 items` (72.51% train items).

### Cold-start Recall@K

Chỉ tính Recall@K trên subset test interactions có positive item là cold-start item (train_freq = 0):

$$\text{Cold Recall@K} = \frac{1}{|U_{cold}|} \sum_{u: pos\_item(u) \in COLD} \text{Recall@K}(u)$$

**Ý nghĩa:** Metric trực tiếp đánh giá khả năng Gate Fusion tận dụng LLM embedding cho items chưa từng xuất hiện trong train.

### Popularity Distribution Analysis

Báo cáo phân phối `train_freq` của recommended items trên toàn bộ users:

```
Median train_freq của recommended items:
  LightGCN baseline: X  (cao → thiên về head)
  RecMind:           Y
  TA-RecMind:        Z  (thấp hơn → thiên về tail — mục tiêu)
```

Vẽ histogram hoặc boxplot: TA-RecMind nên có phân phối `train_freq` dịch về phía trái (tail) so với baseline.

---

## Phân Tầng Kết Quả

### Phân Tầng Theo Item Popularity

Báo cáo Recall@K và NDCG@K riêng cho ba nhóm:

```
Nhóm HEAD: positive item ∈ HEAD items  (top 20% theo train_freq)
Nhóm MID:  positive item ∈ MID items   (10% tiếp theo)
Nhóm TAIL: positive item ∈ TAIL items  (bottom 70%, ≈ 755,609 items)
```

**Bảng kết quả lý tưởng:**

| Model | HEAD Recall@20 | MID Recall@20 | TAIL Recall@20 | Overall |
|---|---|---|---|---|
| LightGCN | High | Medium | **Low** | Medium |
| RecMind | High | Medium-High | Low-Medium | Medium-High |
| **TA-RecMind** | High | **High** | **High** | **High** |

**Điều kiện bắt buộc:** Cải thiện TAIL không đến từ chi phí giảm HEAD. Nếu TA-RecMind tăng TAIL nhưng giảm HEAD đáng kể → đây là trade-off **không chấp nhận được**.

### Phân Tầng Theo User Activity

Báo cáo tương tự phân nhóm theo cấp độ hoạt động User (từ `gold_user_activity_group.npy`):

| Nhóm | Định Nghĩa | Kỳ Vọng |
|---|---|---|
| SUPER_ACTIVE (> 20 acts) | Sức mạnh Collaborative mạnh | GNN hoạt động tốt |
| ACTIVE (5-20 acts) | Hoạt động ổn định | GNN + LLM cân bằng |
| INACTIVE/COLD (< 5 acts) | Users cực kỳ thưa | Gate phải mở LLM, cứu bằng User Text Profile |

**Chứng minh INACTIVE users cải thiện** là mũi nhọn tối thượng để bảo vệ tư tưởng đề tài: Bipartite Gate Symmetry giải quyết cold-start không chỉ ở phía Item mà còn ở phía User.

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

| Variant | Thay Đổi | Mục Đích Đo |
|---|---|---|
| **Full** | TA-RecMind đầy đủ | Baseline của đề tài |
| **w/o Tail-Weight** | Bỏ `w_v` khỏi alignment loss | Đóng góp của tail weighting |
| **w/o LAGCL** | Bỏ `L_cl` | Đóng góp của contrastive loss |
| **w/o Re-ranking** | Bỏ popularity penalty | Đóng góp của re-ranking |
| **w/o LLM** | Chỉ dùng LightGCN | Đóng góp của LLM embedding |
| **w/o Gate** | Bỏ Gate Fusion → Late Fusion | Đóng góp của Intra-Layer Gate |
| **w/o Bipartite** | Gate chỉ cho Item, không User | Đóng góp của Bipartite Symmetry |

---

## Implementation Đánh Giá

### All-item Ranking

Với mỗi user trong evaluation:
1. Lấy embedding `h_u`
2. Tính điểm với toàn bộ N_items: `scores = h_u @ H_items.T` (matrix multiply)
3. Loại bỏ items đã tương tác trong train
4. Áp re-ranking nếu evaluating full model
5. Lấy top-K theo score sau re-ranking
6. So sánh với positive item trong eval set

**Tối ưu tốc độ:** FAISS `IndexIVFFlat` cho approximate top-K — giảm từ O(N) brute-force xuống O(√N) mỗi query. Với 1M items và 1.8M users, FAISS là bắt buộc.

### Logging Phân Tầng (Mỗi Epoch)

```python
metrics = {
    "epoch": epoch,
    "overall": {"recall@20": ..., "ndcg@20": ..., "recall@40": ..., "ndcg@40": ...},
    "head":    {"recall@20": ..., "ndcg@20": ...},
    "mid":     {"recall@20": ..., "ndcg@20": ...},
    "tail":    {"recall@20": ..., "ndcg@20": ..., "coverage@20": ...},
    "cold":    {"recall@20": ...},
    "inactive":{"recall@20": ...},
}
```

Lưu vào `training_history.json` để vẽ learning curves sau training.

---

## Tiêu Chí Thành Công

Đề tài được coi là thành công nếu TA-RecMind đạt:

| Tiêu Chí | Mục Tiêu | Lý Do |
|---|---|---|
| **TAIL Recall@20** | Tăng ≥ 20% so với LightGCN | Cải thiện long-tail đáng kể |
| **Overall Recall@20** | Không giảm > 5% so với LightGCN | Không sacrifice accuracy tổng thể |
| **Tail Coverage@20** | ≥ 30% | Ít nhất 30% tail items được gợi ý |
| **Cold-start Recall@20** | ≥ 50% của warm items recall | LLM embedding hữu ích cho cold-start |
| **INACTIVE Recall@20** | Tăng ≥ 15% so với LightGCN | Bipartite Gate giải quyết cold-start user |

> Ngưỡng cụ thể sẽ được điều chỉnh dựa trên kết quả thực nghiệm với tập Electronics. Mục tiêu chính là chứng minh **xu hướng cải thiện** nhất quán trên tail và cold-start, không chỉ về số tuyệt đối.
