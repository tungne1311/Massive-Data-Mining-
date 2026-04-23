# TA-RecMind: Tail-Augmented Recommendation with LLM-GNN Alignment

> **Hệ thống gợi ý sản phẩm long-tail trên Amazon Electronics**  
> Kết hợp LightGCN, Sentence-Transformers và Gate Fusion để giải quyết popularity bias

---

## Mục Lục

- [Tổng Quan](#tổng-quan)
- [Vấn Đề Nghiên Cứu](#vấn-đề-nghiên-cứu)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt](#hướng-dẫn-cài-đặt)
- [Hướng Dẫn Chạy Pipeline](#hướng-dẫn-chạy-pipeline)
- [Kết Quả EDA Chính](#kết-quả-eda-chính)
- [Đóng Góp Kỹ Thuật](#đóng-góp-kỹ-thuật)
- [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

---

## Tổng Quan

TA-RecMind (**T**ail-**A**ugmented **Rec**ommendation **Mind**) là hệ thống gợi ý học sâu được thiết kế để giải quyết thách thức cốt lõi của bài toán **Long-tail Recommendation** trên tập dữ liệu Amazon Reviews 2023 (Electronics).

Hệ thống tích hợp ba thành phần chính:

| Thành Phần | Vai Trò |
|---|---|
| **LightGCN** | Học biểu diễn cộng tác (collaborative embeddings) qua message passing trên đồ thị |
| **Sentence-Transformers** | Học biểu diễn ngữ nghĩa (semantic embeddings) từ văn bản item và user |
| **Gate Fusion** | Căn chỉnh động hai không gian embedding, đặc biệt hiệu quả với tail items |

**Đóng góp so với RecMind gốc (Xue et al., 2025):**
- Tail-Weighted Alignment Loss — tăng cường gradient cho tail items trong quá trình huấn luyện
- User Review Weighting theo `helpful_vote` — lọc nhiễu từ reviews chất lượng thấp
- Popularity-Penalized Re-ranking — điều chỉnh phân phối gợi ý tại inference time
- Phân loại HEAD/MID/TAIL dựa trên CDF Pareto từ training set (chống Data Leakage)

---

## Vấn Đề Nghiên Cứu

### Bối Cảnh

Tập dữ liệu Amazon Electronics 2023 thể hiện phân phối **power-law** cực đoan:

```
99.9993% sparsity — đồ thị user-item gần như rỗng hoàn toàn
72.51%  items là Tail (≤ 5 tương tác trong train)
23.17%  items trong val/test chưa từng xuất hiện trong train (cold-start)
```

Vấn đề: Các hệ thống gợi ý truyền thống (LightGCN, MF) tối ưu hóa chủ yếu cho **head items** do chúng chiếm 80% tổng tương tác (nguyên tắc Pareto), bỏ qua phần lớn catalog sản phẩm.

### Câu Hỏi Nghiên Cứu

> Làm thế nào để xây dựng hệ thống gợi ý vừa duy trì độ chính xác tổng thể vừa cải thiện đáng kể khả năng gợi ý **tail items** chất lượng cao nhưng chưa được khám phá?

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
│                                                             │
│  HuggingFace ──► Bronze ──► Silver ──► Gold ──► Model      │
│  (Amazon 2023)   (Raw)     (Clean)   (Ready)               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURE                       │
│                                                             │
│  Item/User Text ──► Sentence-Transformer ──► z^L_v         │
│                                                  │          │
│  Interaction Graph ──► LightGCN ──────────────► │          │
│                         (L layers)    z^G_v      │          │
│                                         │        │          │
│                                    Gate Fusion   │          │
│                                    γ = σ(w[z^G ∥ z^L ∥ d̃]) │
│                                         │                   │
│                                    h_v = α·z^G + (1-α)·z^L │
│                                         │                   │
│                                    s(u,i) = ⟨h_u, h_i⟩    │
│                                         │                   │
│                              Popularity-Penalized Re-ranking│
└─────────────────────────────────────────────────────────────┘
```

Xem chi tiết tại [`docs/02_model_architecture.md`](./docs/02_model_architecture.md).

---

## Cấu Trúc Dự Án

```
ta-recmind/
├── README.md                        # File này
├── docs/
│   ├── 01_data_pipeline.md          # Kiến trúc pipeline Bronze-Silver-Gold
│   ├── 02_model_architecture.md     # Chi tiết kiến trúc mô hình
│   ├── 03_training_strategy.md      # Hàm mất mát và chiến lược huấn luyện
│   ├── 04_evaluation_protocol.md    # Giao thức đánh giá Long-tail
│   └── 05_web_demo.md               # Kiến trúc web demo
├── config/
│   └── config.yaml                  # Cấu hình toàn pipeline
├── src/
│   ├── pipeline_runner.py           # Main entry point điều phối toàn bộ
│   ├── bronze/
│   │   ├── ste1.py                  # Bronze: HuggingFace ingestion
│   │   ├── ste2.py                  # Bronze: Spark processing & split
│   │   └── upload-bronze.py         # Push Bronze schema (nếu có)
│   ├── silver/
│   │   ├── ste3_silver.py               # Silver: Orchestrator
│   │   ├── silver_step1_popularity.py   # Silver: HEAD/MID/TAIL classification
│   │   ├── silver_step2_item_profile.py # Silver: Item text profile
│   │   ├── silver_step3_user_profile.py # Silver: User text profile
│   │   ├── silver_step4_interactions.py # Silver: Enrich interactions
│   │   └── upload_silver_to_hf.py       # Tải Silver artifacts lên HF
│   └── gold/
│       ├── ste4_gold.py                 # Gold: ID Mapping & Embeddings Orchestrator
│       └── upload_gold_to_hf.py         # Tải Gold artifacts lên HF
├── docker-compose.yml               # Hạ tầng Docker (16GB RAM)
├── Dockerfile                       # Image với Spark + Python deps
└── .env                             # Biến môi trường (không commit)
```

---

## Yêu Cầu Hệ Thống

| Tài Nguyên | Tối Thiểu | Khuyến Nghị |
|---|---|---|
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8 cores |
| Disk | 50 GB | 100 GB |
| GPU | Không bắt buộc (Bronze/Silver) | 16 GB VRAM (training) |
| Docker | 24.x+ | 24.x+ |
| Python | 3.10+ | 3.11 |

**Phân bổ RAM (16 GB):**
```
MinIO:         ~1.5 GB
Spark Master:  ~1.0 GB
Spark Worker:  ~8.0 GB (6 GB JVM + 2 GB overhead)
Pipeline Driver: ~5.0 GB
OS + buffer:   ~0.5 GB
```

---

## Hướng Dẫn Cài Đặt

### 1. Clone và Cấu Hình

```bash
git clone https://github.com/<your-repo>/ta-recmind.git
cd ta-recmind
cp .env.example .env
# Chỉnh sửa .env theo máy thực tế
```

### 2. Khởi Động Hạ Tầng

```bash
# Khởi động MinIO + Spark cluster
docker compose up -d minio minio-init spark-master spark-worker-1

# Kiểm tra trạng thái
docker compose ps

# Spark UI: http://localhost:18080
# MinIO UI: http://localhost:9001
```

### 3. Kiểm Tra Kết Nối

```bash
# Test pipeline với 20k records
docker compose run --rm pipeline-test --step 1_2
```

---

## Hướng Dẫn Chạy Pipeline

### Chạy Toàn Bộ Pipeline (Từ đầu tới cuối)

```bash
docker compose run --rm pipeline python src/pipeline_runner.py --all
```

### Chạy Tách Rời Từng Lớp Data (Bronze / Silver / Gold)

Hệ thống cho phép chạy độc lập từng bước để debug hoặc phân bổ tài nguyên.

```bash
# Bước 1+2: Ingestion & Tạo Bronze schemas (Lọc Core-5, Time-split theo timestamp)
docker compose run --rm pipeline python src/pipeline_runner.py --step 1_2

# Bước 3: Tạo Silver schema (Phân loại Popularity, Tokenize texts, Enrich profiles)
docker compose run --rm pipeline python src/pipeline_runner.py --step 3_silver

# Bước 4: Tạo Gold schemas (ID Mapping, Re-mapping nodes, Dense Embeddings)
docker compose run --rm pipeline python src/pipeline_runner.py --step 4_gold
```

### Push Artifacts lên Hugging Face

Sau khi chạy xong các layer ở MinIO, bạn có thể đẩy Data Lake lên Cloud Dataset repo. Đừng quên cung cấp `HF_TOKEN` nếu nó chưa ở trong `.env`.

```bash
# Push Silver Dataset
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline python src/silver/upload_silver_to_hf.py

# Push Gold Dataset (Ghi đè tham số repo thông qua command)
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline python src/gold/upload_gold_to_hf.py --mode full --repo-id <your-repo>/amazon-2023-gold
```

### Cấu Hình Thử Nghiệm Nhanh

```bash
# Chỉ lấy 20k reviews để test pipeline
MAX_REVIEW_RECORDS=20000 MAX_METADATA_RECORDS=5000 \
docker compose run --rm pipeline --step 1_2
```

---

## Kết Quả EDA Chính

### Thống Kê Dataset (Amazon Electronics 2023)

| Chỉ Số | Giá Trị |
|---|---|
| Tổng users (sau Core-5 filter) | 1,847,662 |
| Tổng items (train) | 1,042,121 |
| Tổng interactions (train) | 1,396,428 |
| Sparsity | 99.9993% |
| Tail items (≤ 5 tương tác) | 72.51% |
| Cold-start items trong Val/Test | 23.17% |

### Phân Phối Rating

| Rating | Tỷ Lệ |
|---|---|
| ⭐⭐⭐⭐⭐ (5 sao) | 65.7% |
| ⭐⭐⭐⭐ (4 sao) | 14.7% |
| ⭐⭐⭐ (3 sao) | 7.0% |
| ⭐⭐ (2 sao) | 4.5% |
| ⭐ (1 sao) | 8.0% |

> **Nhận xét:** Rating skew nặng về 5 sao (65.7%) — đây là lý do hàm trọng số `w(r)` dựa trên `helpful_vote` quan trọng để phân biệt chất lượng review thực sự.

---

## Đóng Góp Kỹ Thuật

| Đóng Góp | Mô Tả | Nguồn Gốc |
|---|---|---|
| **Tail-Weighted Alignment Loss** | `w_v = 1/log(1 + train_freq)` nhân vào alignment loss | Novel (mở rộng từ RecMind Eq.5 + LAGCL) |
| **Review Quality Weighting** | `w(r) = 1 + log(1 + helpful_vote)` | Novel (feature engineering Amazon 2023) |
| **Anti-Leakage Classification** | Dùng `train_freq` thay vì `rating_number` từ metadata | Novel (phân tích leakage) |
| **Popularity-Penalized Re-ranking** | `s_adj = s_model - λ·log(1 + train_freq)` | Novel (dựa trên Challenging Long Tail) |

---

## Tài Liệu Tham Khảo

1. He, X. et al. (2020). **LightGCN: Simplifying and Powering GCN**. SIGIR 2020.
2. Xue, et al. (2025). **RecMind: LLM-Powered Recommendation**. arXiv:2509.06286v1.
3. Wu, J. et al. (2021). **Self-supervised Graph Learning for Recommendation** (SGL). SIGIR 2021.
4. Hu, E. et al. (2022). **LoRA: Low-Rank Adaptation of LLMs**. ICLR 2022.
5. Rendle, S. et al. (2009). **BPR: Bayesian Personalized Ranking**. UAI 2009.

Xem đầy đủ tại [`docs/references.md`](./docs/references.md).
