# 🎨 Thiết Kế Ứng Dụng Demo: TA-RecMind V2

## Tổng Quan

Demo TA-RecMind không chỉ in ra danh sách sản phẩm — nó phải **kể câu chuyện**:
> *"Hệ thống hiểu người dùng như thế nào, đối xử công bằng với sản phẩm vô danh ra sao, và gán hàng mới cho đúng khách hàng tiềm năng ngay từ giây đầu tiên?"*

**Platform:** Streamlit (Python) — deploy trên Google Colab qua ngrok hoặc localhost.

**File:** `src/demo_recmind_final.ipynb` (notebook), có thể export thành `recmind_demo.py`.

**Dữ liệu tĩnh nạp vào RAM server (một lần duy nhất khi khởi động):**
- `tarecmind_demo_embeddings.pth`: `user_final_all [N_users, 128]` + `item_final_all [N_items, 128]`
- `silver_item_text_profile.parquet`: Tên, mô tả, category để hiển thị
- `gold_item_id_map.parquet` + `gold_user_id_map.parquet`: Nhãn HEAD/MID/TAIL/COLD và INACTIVE/ACTIVE/SUPER_ACTIVE

---

## Cấu Trúc Giao Diện (2 Tab Chính)

---

### 📺 Tab 1: Khám Phá Nhóm Người Dùng

**Mục đích:** Chứng minh hệ thống không bị Popularity Bias và phục vụ tốt nhóm Inactive.

#### Sidebar Controls

```
Dropdown: [Inactive (< 5 tương tác)] / [Active (5-20)] / [Super Active (> 20)]
Button:   🎲 Random User  → Bốc ngẫu nhiên một user trong phân khúc đó
Slider:   Top-K = 20 / 40
Sliders:  Diversity Control — tỷ lệ mục tiêu Head/Mid/Tail
          HEAD: [0% ──── 50% ──────────── 100%]
          MID:  [0% ─── 20% ──────────── 100%]
          TAIL: [0% ──────── 30% ──────── 100%]
          → Hệ thống rerank để thỏa mãn tỷ lệ này
Slider:   λ_penalty (Re-ranking strength) = [0.0 ──── 0.3 ──── 1.0]
```

#### Layout Chính (2 Cột)

**Cột Trái — Lịch Sử Người Dùng:**
- Hiển thị sản phẩm user đã tương tác trong quá khứ (từ train set)
- Mỗi card: Tên sản phẩm, Thể loại, Số sao, Badge HEAD/MID/TAIL
- Mục đích: Người xem demo hiểu "gu" của khách hàng → so sánh với gợi ý

**Cột Phải — Gợi Ý Top-K:**
- Hiển thị grid sản phẩm mà hệ thống dự đoán
- Mỗi card phải có Badge màu nổi bật:
  - 🔴 `[🔥 HEAD]` — Sản phẩm hot, >20 tương tác trong train
  - 🟣 `[⚡ MID]` — Sản phẩm phổ biến vừa phải
  - 🟡 `[🌟 TAIL]` — Sản phẩm ngách, ≤5 tương tác
  - 🔵 `[❄️ COLD]` — Sản phẩm mới tinh, chưa từng có tương tác

**Progress Bar Phân Phối Thực Tế:**
```
HEAD ████████░░░░░░░░  40%
MID  ███░░░░░░░░░░░░░  15%
TAIL █████████░░░░░░░  35%
COLD ██░░░░░░░░░░░░░░  10%
```

**Ngưỡng phân loại được tính động từ quantile thực tế của dữ liệu:**
```python
# Tính từ gold_item_train_freq.npy
head_threshold = np.percentile(item_freq[item_freq > 0], 80)  # top 20%
mid_threshold  = np.percentile(item_freq[item_freq > 0], 70)  # top 30%
# TAIL: còn lại; COLD: train_freq == 0
```

---

### 📺 Tab 2: Phép Màu Cold-Start (Zero-Shot Item Insertion)

**Mục đích:** Demo sức mạnh LLM Fusion. Khẳng định hệ thống có thể **bán được hàng mới ngay từ giây đầu tiên ra mắt**.

#### Khu Vực Nhập Liệu

```
Text Input: 1. Tên sản phẩm (Title)
Text Area:  2. Tính năng (Features)
Text Input: 3. Danh mục (Categories)
Text Area:  4. Mô tả chi tiết (Description)
Text Input: 5. Chi tiết kỹ thuật (Details)

Button: 🚀 Thêm Sản Phẩm & Phân Tích Mạng Graph
```

**Ví dụ thú vị để demo:** Tai nghe Bluetooth chuyên dùng chạy bộ dưới mưa, hoặc cáp sạc USB-C cho xe hơi cổ.

#### Luồng Xử Lý (Inference)

```python
def inject_and_find_users(title, features, category, description, details, top_k=5):
    # 1. Tiền xử lý & Ghép chuỗi theo hierarchy chuẩn Silver Step 2
    #    Hierarchy: Title | Features | Category | Description | Details
    #    Budget: Title (150), Features (450/750), Category (150), Desc (300), Details (150)
    text_profile = build_standard_profile(title, features, category, description, details)
    
    # 2. Encode bằng SentenceTransformer (384 chiều)
    raw_vec = model.encode(text_profile)
    
    # 3. Chiếu về không gian embedding 128 chiều (W_proj) và normalize
    new_item_vec = W_proj @ raw_vec
    new_item_vec = F.normalize(new_item_vec, dim=0)

    # 4. Sản phẩm mới chưa có trong Graph → train_freq = 0
    #    Gate tự động: γ → 0, dùng 100% LLM Vector
    
    # 3a. Tìm Item tương đồng (semantic similarity)
    item_scores = item_matrix @ new_item_vec      # [N_items]
    top_similar_items = topk(item_scores, top_k)
    
    # 3b. Tìm User tiềm năng (user với vector gần nhất)
    user_scores = user_matrix @ new_item_vec      # [N_users]
    top_potential_users = topk(user_scores, top_k)
    
    return top_similar_items, top_potential_users
```

#### Kết Quả Trả Về (Output 2 Chiều)

**Panel 1 — Sản Phẩm Tương Đồng:**
- Top-5 items trong catalog có vector gần nhất với sản phẩm mới
- Mục đích: Chứng minh mô hình "hiểu" ngữ nghĩa — tai nghe chạy bộ → tìm ra các tai nghe sport/outdoor khác

**Panel 2 — Khách Hàng Tiềm Năng:**
- `🏆 Top 5 Khách Hàng Tiềm Năng Nhất Sẽ Mua Sản Phẩm Này`
- Hiển thị lịch sử mua hàng của 5 người đó
- **WOW moment:** Họ toàn mua đồ thể thao, tai nghe thể thao, phụ kiện running — semantic match hoàn hảo!

---

## Kiến Trúc Kỹ Thuật Demo

### Thuật Toán Cốt Lõi

```python
import torch
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_demo_data():
    """Nạp toàn bộ một lần, cache vĩnh viễn trong session."""
    embs = torch.load('tarecmind_demo_embeddings.pth', map_location='cpu')
    user_matrix = embs['user_embeddings'].float()  # [N_users, 128]
    item_matrix = embs['item_embeddings'].float()  # [N_items, 128]
    
    item_meta = pd.read_parquet('silver_item_text_profile.parquet',
                                 columns=['parent_asin', 'title', 'main_category',
                                          'popularity_group', 'train_freq'])
    user_meta = pd.read_parquet('gold_user_id_map.parquet')
    item_map  = pd.read_parquet('gold_item_id_map.parquet')
    
    # Tính ngưỡng HEAD/MID/TAIL từ quantile thực tế
    freqs = item_map['train_freq'].values
    nonzero_freqs = freqs[freqs > 0]
    head_thr = np.percentile(nonzero_freqs, 80)
    mid_thr  = np.percentile(nonzero_freqs, 70)
    
    return user_matrix, item_matrix, item_meta, user_meta, item_map, head_thr, mid_thr

def get_recommendations(user_idx, top_k=20, lambda_penalty=0.3):
    """Tab 1: Lấy gợi ý cho 1 User."""
    user_vec = user_matrix[user_idx]              # [128]
    scores = item_matrix @ user_vec               # [N_items] — siêu tốc
    
    # Loại bỏ items đã tương tác trong train
    train_items = get_user_train_items(user_idx)
    scores[train_items] = -float('inf')
    
    # Re-ranking với popularity penalty
    freq_arr = torch.tensor(item_map['train_freq'].values, dtype=torch.float)
    scores_adj = scores - lambda_penalty * torch.log1p(freq_arr)
    
    top_scores, top_item_indices = torch.topk(scores_adj, top_k)
    return top_item_indices.tolist()

def inject_cold_start(text_desc, top_k_items=5, top_k_users=5):
    """Tab 2: Zero-shot insertion."""
    minilm = SentenceTransformer('all-MiniLM-L6-v2')
    vec = torch.tensor(minilm.encode(text_desc))
    vec = torch.nn.functional.normalize(vec, p=2, dim=0)
    # Chiếu về dim 128 nếu cần (W_proj)
    
    item_sims = item_matrix @ vec
    user_sims = user_matrix @ vec
    
    top_items = torch.topk(item_sims, top_k_items).indices.tolist()
    top_users = torch.topk(user_sims, top_k_users).indices.tolist()
    return top_items, top_users
```

### Hiệu Năng

- **Phản hồi Tab 1:** < 50ms (phép nhân 1 vector × 1.6M vector trên CPU với float32)
- **Phản hồi Tab 2:** < 200ms (bao gồm SentenceTransformer encoding ~150ms)
- **RAM server:** ~2.5 GB (user_matrix + item_matrix + metadata)

---

## Lộ Trình Phát Triển

| Bước | Công Việc | Thời Gian Ước Tính |
|---|---|---|
| 1 | Load embedding → implement get_recommendations() | 2 giờ |
| 2 | Build Tab 1 UI: Sidebar + 2-column layout + Cards with Badges | 3 giờ |
| 3 | Diversity Control: rerank theo tỷ lệ HEAD/MID/TAIL mục tiêu | 2 giờ |
| 4 | Build Tab 2 UI: Form nhập liệu + Output 2 panels | 2 giờ |
| 5 | Tích hợp SentenceTransformer cho cold-start injection | 1 giờ |
| 6 | Polish: màu sắc badge, animation, progress bars | 1 giờ |

**Tổng: ~1 ngày làm việc**

**Công cụ:** Streamlit + PyTorch (CPU mode) + SentenceTransformer. Không cần viết HTML/CSS nếu dùng `st.columns`, `st.metric`, `st.progress`.

---

## Gợi Ý Kịch Bản Demo Thuyết Phục

### Kịch Bản 1 — Inactive User
1. Chọn "Inactive" → Random User → User chỉ có 3 interactions (vd: 1 con chuột gaming, 1 bàn phím, 1 headset)
2. Hệ thống gợi ý: laptop gaming accessories, RGB keyboard, gaming chair → **"Gu" hoàn toàn khớp dù chỉ 3 tương tác**
3. Highlight: 4/20 items được gợi ý là TAIL — sản phẩm ngách, đúng nhu cầu cụ thể

### Kịch Bản 2 — Tăng λ_penalty
1. Kéo λ_penalty từ 0 → 1 → Progress Bar TAIL tăng từ 20% → 50%
2. Chứng minh: Re-ranking điều chỉnh được phân phối mà **không cần retrain**

### Kịch Bản 3 — Cold-start Insertion
1. Nhập: "Waterproof Running Headphones, IPX7, Bone Conduction, Bluetooth 5.3"
2. Similar items: Aftershokz Aeropex, Shokz OpenRun → **chính xác về ngữ nghĩa**
3. Top users: Toàn người hay mua đồ thể thao, tai nghe outdoor → **WOW moment**
