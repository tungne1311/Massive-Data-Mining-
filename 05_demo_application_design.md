# 🎨 Thiết Kế Giao Diện Demo: TA-RecMind V2

Một bản Demo xuất sắc cho hệ thống Recommender System (đặc biệt là hệ thống mạnh về Long-tail/Cold-start như của bạn) không chỉ là việc in ra danh sách sản phẩm. Nó phải kể được câu chuyện: **"Hệ thống hiểu người dùng như thế nào, và đối xử công bằng với các sản phẩm vô danh ra sao?"**.

Dưới đây là kịch bản thiết kế chi tiết (UI/UX và Logic) dùng để báo cáo hoặc thuyết trình.

---

## 1. Cấu Trúc Giao Diện (Streamlit hoặc FastAPI + React)

Giao diện sẽ được chia làm 2 Tab (hoặc 2 Màn hình chính):

### 📺 Tab 1: Khám Phá Nhóm Người Dùng (User Segments Showcase)
Mục đích: Chứng minh hệ thống không bị "Popularity Bias" và phục vụ tốt nhóm Inactive.

1. **Thanh điều khiển (Sidebar):**
   * Một Dropdown chọn Phân Khúc: `[Inactive (Mua < 5)]`, `[Active (Mua 5-20)]`, `[Super Active (Mua > 20)]`.
   * Nút: `🎲 Random User` (Bốc ngẫu nhiên một người trong phân khúc đó).
   * **Thanh trượt cấu hình:** Tùy chỉnh số lượng gợi ý Top-K (VD: 20 hoặc 40).
   * **Tùy chỉnh phân bổ (Diversity Control):** Cho phép nhập tỷ lệ mục tiêu Head/Mid/Tail (Ví dụ: 50% Head, 20% Mid, 30% Tail). Hệ thống sẽ rerank danh sách để thỏa mãn tỷ lệ này.
2. **Khu vực Lịch sử & Gợi Ý (Chia cột hiển thị):**
   * **Cột Lịch Sử (Bên trái):** Hiển thị các sản phẩm người dùng đã mua/tương tác trong quá khứ. (Hiển thị Tựa đề, Thể loại, và số sao). Giúp người xem Demo đối chiếu được "Gu" của khách hàng.
   * **Cột Gợi Ý Top-K (Bên phải):** Hiển thị dạng lưới (Grid) các sản phẩm mà hệ thống dự đoán.
   * **ĐIỂM NHẤN:** Góc phải của mỗi thẻ (Card) sản phẩm phải có một Badge (Nhãn dán) màu sắc nổi bật:
     * 🔴 `[🔥 HEAD]` (Sản phẩm hot, ai cũng mua)
     * 🟣 `[⚡ MID]` (Sản phẩm phổ biến vừa phải)
     * 🟡 `[🌟 TAIL]` (Sản phẩm ngách, ít người mua)
     * 🔵 `[❄️ COLD]` (Sản phẩm mới tinh, chưa từng có lượt tương tác)
   * Phía trên Grid, có một thanh Progress Bar thống kê tỷ lệ thực tế đạt được.

---

### 📺 Tab 2: Phép Màu Cold-Start (Zero-Shot Item Insertion)
Mục đích: Demo sức mạnh của Dual-View (LLM Fusion). Khẳng định hệ thống có thể bán được hàng mới ngay từ giây đầu tiên ra mắt.

1. **Khu vực Nhập Liệu (Thêm Sản Phẩm Mới):**
   * Form cho phép người Demo nhập vào: Tên sản phẩm, và Mô tả chi tiết (Review/Description).
   * Vd: Nhập mô tả một cái tai nghe Bluetooth chuyên dùng để chạy bộ dưới mưa.
   * Nút: `🚀 Thêm Sản Phẩm & Phân Tích`
2. **Luồng Xử Lý Dưới Ngầm (Inference):**
   * Demo gọi Text Encoder (MiniLM) biến đoạn mô tả trên thành 1 vector 128 chiều.
   * Vì sản phẩm mới chưa có trong Graph (Không có cạnh GNN), hệ thống sẽ gán trọng số Graph = 0, và dùng 100% LLM Vector.
   * Sinh ra `new_item_vec` và chuẩn hóa (L2-Norm).
3. **Kết Quả Trả Về (Output 2 Chiều):**
   * **Sản phẩm tương đồng:** App quét tìm trong danh mục các Item hiện có xem những sản phẩm nào có vector gần nhất với món đồ vừa thêm để chứng minh mô hình "hiểu" ngữ nghĩa chính xác.
   * **Gợi ý cho Khách hàng tiềm năng:** App tìm trong 1.4 triệu Users, ai là người có điểm Dot-Product cao nhất với Vector tai nghe chạy bộ này. In ra danh sách: `🏆 Top 5 Khách Hàng Tiềm Năng Nhất Sẽ Mua Sản Phẩm Này`. 
   * Hiển thị lịch sử của 5 người đó (sẽ thấy họ toàn mua đồ thể thao, đồ chạy bộ). Khách mời xem Demo sẽ phải "Wow" vì độ chính xác về mặt ngữ nghĩa!

---

## 2. Kiến Trúc Kỹ Thuật (Cách Hiện Thực)

Để làm được Demo này cực mượt (phản hồi dưới 50ms), bạn không cần bê nguyên mô hình PyTorch nặng nề lên Web.

### A. Dữ liệu tĩnh cần nạp vào RAM máy chủ (Web Server):
1. **`tarecmind_demo_embeddings.pth`**: File chứa `user_final_all` và `item_final_all` vừa xuất ra ở bước trước.
2. **`silver_step2_item_profile.parquet`** (hoặc bảng metadata tương đương): Dùng để tra cứu Tên sản phẩm, nội dung khi biết ID.
3. **`gold_user_id_map.parquet` / `gold_item_id_map.parquet`**: Chứa thông tin nhãn Head/Tail/Cold và Inactive/Active.

### B. Thuật toán Xử Lý Cốt Lõi:
```python
import torch
import numpy as np

# Khởi động Server: Load duy nhất 1 lần
embs = torch.load('tarecmind_demo_embeddings.pth')
user_matrix = embs['user_embeddings'] # [1483920, 128]
item_matrix = embs['item_embeddings'] # [1610012, 128]

# API: Lấy gợi ý cho 1 User (Tab 1)
def get_recommendations(user_idx, top_k=20):
    user_vec = user_matrix[user_idx] # [128]
    # Phép nhân siêu tốc 1 Vector với 1.6 Triệu Vector
    scores = torch.matmul(item_matrix, user_vec.T) 
    top_scores, top_items = torch.topk(scores, top_k)
    return top_items.tolist() # Gửi mảng ID này ra Frontend để map thành Tên sản phẩm

# API: Thêm sản phẩm Cold-start (Tab 2)
def inject_and_find_users(text_description, top_k=5):
    # 1. Gọi MiniLM biến Text thành Vector
    new_item_vec = minilm_encode(text_description) # [128]
    # 2. Chuẩn hóa vector
    new_item_vec = torch.nn.functional.normalize(new_item_vec, p=2, dim=0)
    
    # 3a. Tìm Item tương đồng
    item_scores = torch.matmul(item_matrix, new_item_vec.T)
    top_item_scores, top_similar_items = torch.topk(item_scores, top_k)
    
    # 3b. Tìm User tiềm năng
    user_scores = torch.matmul(user_matrix, new_item_vec.T)
    top_user_scores, top_users = torch.topk(user_scores, top_k)
    return top_similar_items.tolist(), top_users.tolist()
```

## 3. Lộ Trình Phát Triển Nhanh (1 Ngày)
1. **Công cụ khuyên dùng:** Sử dụng thư viện **`Streamlit`** (Python). Streamlit có sẵn các Component để vẽ Dropdown, Grid ảnh, và Progress Bar mà không cần viết 1 dòng HTML/CSS nào.
2. Lấy dữ liệu Text (Tên sản phẩm) từ bảng Silver ghép với kết quả Tensor để hiển thị lên UI.
3. Định nghĩa các màu sắc cho Badge thật bắt mắt.
