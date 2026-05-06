# TA-RecMind: Tail-Augmented Recommendation with LLM-GNN Alignment

> **Hệ thống gợi ý sản phẩm long-tail trên Amazon Electronics 2023**  
> Kết hợp LightGCN, Sentence-Transformers và Intra-Layer Gate Fusion để triệt tiêu popularity bias và giải quyết cold-start

---

## Mục Lục

- [Tổng Quan](#tổng-quan)
- [Vấn Đề Nghiên Cứu](#vấn-đề-nghiên-cứu)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt & Chạy Pipeline](#hướng-dẫn-cài-đặt--chạy-pipeline)
- [Kết Quả EDA Chính](#kết-quả-eda-chính)
- [Đóng Góp Kỹ Thuật (Novel Contributions)](#đóng-góp-kỹ-thuật-novel-contributions)
- [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

---

## Tổng Quan

**TA-RecMind** (**T**ail-**A**ugmented **Rec**ommendation **Mind**) là hệ thống gợi ý học sâu được thiết kế để giải quyết thách thức cốt lõi của bài toán **Long-tail Recommendation** và **Cold-start** trên tập dữ liệu Amazon Reviews 2023 — subset Electronics (~44M interactions).

Hệ thống xây dựng trên nền tảng **RecMind (Xue et al., 2025)** với ba đóng góp kỹ thuật cốt lõi: Bipartite Intra-Layer Gate Fusion, Tail-Weighted Alignment Loss, và Popularity-Penalized Re-ranking.

### Ba Thành Phần Chính

| Thành Phần | Vai Trò |
|---|---|
| **LightGCN (L=2)** | Học biểu diễn cộng tác (collaborative embeddings) qua message passing trên đồ thị user-item bipartite |
| **Sentence-Transformers** | Học biểu diễn ngữ nghĩa (semantic embeddings) từ văn bản item và user (offline cache) |
| **Intra-Layer Gate Fusion** | Trộn động hai không gian embedding *bên trong* mỗi layer GNN — đặc biệt hiệu quả với tail/cold-start |

### Đóng Góp So Với RecMind Gốc (Xue et al., 2025)

| Đóng Góp | Mô Tả | File Liên Quan |
|---|---|---|
| **Bipartite Gate Symmetry** | Gate fusion áp dụng đối xứng cho cả User và Item node (RecMind chỉ cho Item) | `02_model_architecture.md` |
| **Tail-Weighted Alignment Loss** | `w_v = 1/log(1+train_freq)` nhân vào InfoNCE loss, ép optimizer tập trung tail | `03_training_strategy.md` |
| **Anti-Leakage Classification** | Phân loại HEAD/MID/TAIL dựa trên `train_freq` chứ không phải `rating_number` (chống data leakage) | `01_data_pipeline.md` |
| **Popularity-Penalized Re-ranking** | `s_adj = s_model - λ·log(1+train_freq)` tại inference time | `02_model_architecture.md` |
| **Review Quality Weighting** | `w(r) = 1 + log(1 + helpful_vote)` lọc nhiễu spam 5-sao | `01_data_pipeline.md` |

---

## Vấn Đề Nghiên Cứu

### Bối Cảnh

Tập dữ liệu Amazon Electronics 2023 thể hiện phân phối **power-law** cực đoan:

```
Tổng raw interactions:    ~44,066,834 rows (sau Silver cleaning)
Positive interactions:     ~36M (rating ≥ 3.0)
Users (sau Core-5 filter): 1,847,662
Items (train):             1,042,121
Sparsity:                  99.9993%
Tail items (≤ 5 tương tác): 72.51% (755,609 items)
Cold-start trong Val/Test:  23.17% (130,746 items)
```

**Vấn đề:** Các hệ thống gợi ý truyền thống (LightGCN, MF) tối ưu hóa chủ yếu cho **head items** do chúng chiếm ~80% tổng tương tác (Pareto principle), bỏ qua phần lớn catalog sản phẩm. GNN hoàn toàn bị mù với cold-start items vì chúng có degree = 0 trong đồ thị.

### Câu Hỏi Nghiên Cứu

> Làm thế nào để xây dựng hệ thống gợi ý vừa duy trì độ chính xác tổng thể vừa cải thiện đáng kể khả năng gợi ý **tail items** và **cold-start items** — những sản phẩm chất lượng nhưng chưa được khám phá?

---

## Kiến Trúc Hệ Thống

```
┌──────────────────────────────────────────────────────────────────┐
│                    MEDALLION DATA PIPELINE                       │
│                                                                  │
│  HuggingFace ──► BRONZE ──► SILVER ──► GOLD ──► Training        │
│  (Amazon 2023)   (Raw)      (Clean)   (Ready)   (Colab/Kaggle)   │
│                                                                  │
│  Lưu trữ: MinIO (local) + HuggingFace Hub (backup)              │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    MODEL ARCHITECTURE                            │
│                                                                  │
│  Item Text ──► Sentence-Transformer ──► z^L_i  (offline cache)  │
│  User Text ──► Sentence-Transformer ──► z^L_u  (offline cache)  │
│                                            │                     │
│  Graph G ──► LightGCN (L=2 layers) ──► z^G_v                    │
│                                            │                     │
│  ┌─────── INTRA-LAYER GATE FUSION ─────────┤                     │
│  │  γ_v = σ(w_base + w_sim·cos(z^G,z^L)   │                     │
│  │           + w_freq·log1p(freq_v))       │                     │
│  │  Ê_v = γ_v·z^G_v + (1-γ_v)·z^L_v      │                     │
│  │  E^(l+1) = Â · Ê^(l)  ← fused!        │                     │
│  └─────────────────────────────────────────┘                     │
│                                            │                     │
│  h_v = α·z^G_v + (1-α)·z^L_v             │                     │
│  s(u,i) = ⟨h_u, h_i⟩                     │                     │
│                                            │                     │
│  Re-ranking: s_adj = s - λ·log(1+freq_i)  ▼                     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    LOSS FUNCTION                                  │
│                                                                  │
│  L = L_BPR + λ₁·(L^U_align,tw + L^I_align,tw) + λ₂·L_cl + β·Ω │
│                                                                  │
│  L_BPR:         Popularity-debiased negative sampling            │
│  L_align,tw:    InfoNCE + tail weighting (Novel)                 │
│  L_cl:          LAGCL contrastive (noise ∝ 1/freq)               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Cấu Trúc Dự Án

```
recsys_pipeline_minio/
├── README.md                            # File này — tổng quan dự án
├── 01_data_pipeline.md                  # Kiến trúc pipeline Bronze→Silver→Gold (chi tiết)
├── 02_model_architecture.md             # Kiến trúc mô hình TA-RecMind (chi tiết)
├── 03_training_strategy.md              # Hàm mất mát & chiến lược huấn luyện (chi tiết)
├── 04_evaluation_protocol.md            # Giao thức đánh giá Long-tail (chi tiết)
├── 05_demo_application_design.md        # Thiết kế Streamlit Demo App (chi tiết)
├── 06_infrastructure.md                 # Hạ tầng Docker + MinIO + Spark (chi tiết)
├── 07_eda_insights.md                   # EDA insights & quyết định kiến trúc (chi tiết)
├── references.md                        # Tài liệu tham khảo & nguồn gốc công thức
│
├── config/
│   └── config.yaml                      # Cấu hình toàn bộ pipeline
│
├── src/
│   ├── pipeline_runner.py               # Main entry point — điều phối toàn bộ pipeline
│   ├── EDA_Bronze_V2.ipynb              # EDA tầng Bronze
│   ├── EDA_Silver_V2.ipynb              # EDA tầng Silver
│   ├── EDA_gold (1).ipynb               # EDA tầng Gold
│   ├── TA_RecMind_V2_IntraLayer.ipynb   # Training notebook (phiên bản IntraLayer Gate)
│   ├── demo_recmind_final.ipynb         # Demo notebook hoàn chỉnh
│   │
│   ├── bronze/
│   │   ├── ste1.py                      # HuggingFace streaming ingestion (Producer-Consumer)
│   │   ├── ste2.py                      # PySpark Bronze processing & Chronological split
│   │   └── upload-bronze.py             # Push Bronze artifacts lên HuggingFace Hub
│   │
│   ├── silver/
│   │   ├── ste3_silver.py               # Silver orchestrator (điều phối 4 bước)
│   │   ├── silver_step1_popularity.py   # HEAD/MID/TAIL classification (train_freq CDF)
│   │   ├── silver_step2_item_profile.py # Item text profile (Field-Aware Token Budget)
│   │   ├── silver_step3_user_profile.py # User text profile (Top-K reviews, helpfulness)
│   │   ├── silver_step4_interactions.py # Enrich interactions (labels, edge_weight)
│   │   ├── upload_silver_to_hf.py       # Push Silver artifacts lên HuggingFace Hub
│   │   └── patch_xdmh.py                # Patch/fix script cho XDMH notebook
│   │
│   └── gold/
│       ├── ste4_gold.py                 # Gold orchestrator (ID mapping → Edge → Meta)
│       ├── gold_step1_id_mapping.py     # Integer indexing cho users & items
│       ├── gold_step2_edge_list.py      # PyG-format edge_index
│       ├── gold_step5_training_meta.py  # Training metadata arrays (npy)
│       └── upload_gold_to_hf.py         # Push Gold artifacts lên HuggingFace (full/partial)
│
├── data/
│   └── logs/
│       └── pipeline.log                 # Runtime logs của pipeline
│
├── docker-compose.yml                   # Hạ tầng: MinIO + Spark + Pipeline Driver
├── Dockerfile                           # Image: Spark 3.5.2 + Python 3.12 + deps
├── .env                                 # Biến môi trường (không commit lên git)
├── .dockerignore                        # Loại trừ file khi build image
├── .gitignore                           # Loại trừ file khỏi git
└── requirements.txt                     # Python dependencies
```

---

## Yêu Cầu Hệ Thống

### Môi Trường Data Pipeline (Docker)

| Tài Nguyên | Tối Thiểu | Khuyến Nghị |
|---|---|---|
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8 cores |
| Disk | 50 GB | 100 GB |
| Docker | 24.x+ | 24.x+ |
| Python | 3.10+ | 3.12 |

**Phân bổ RAM (16 GB):**
```
MinIO:            ~1.5 GB
Spark Master:     ~1.0 GB
Spark Worker:     ~8.0 GB  (6 GB JVM + 2 GB overhead/off-heap)
Pipeline Driver:  ~5.0 GB  (3 GB driver memory + 2 GB overhead)
OS + Buffer:      ~0.5 GB
```

### Môi Trường Training (Google Colab / Kaggle)

| GPU | VRAM | Batch Size | Ghi Chú |
|---|---|---|---|
| T4 / K80 | 15-16 GB | 2048 | Free Colab — dùng Gradient Accumulation × 4 |
| V100 | 16-32 GB | 6144 | Kaggle GPU |
| A100 | 40-80 GB | 8192 | Colab Pro+ |
| L4 | 24 GB | 4096 | Colab Pro |

---

## Hướng Dẫn Cài Đặt & Chạy Pipeline

### 1. Clone và Cấu Hình

```bash
git clone https://github.com/<your-repo>/ta-recmind.git
cd ta-recmind
cp .env.example .env
# Chỉnh sửa .env: HF_TOKEN, MINIO credentials, Spark resources
```

### 2. Khởi Động Hạ Tầng Docker

```bash
# Khởi động MinIO + Spark cluster
docker compose up -d minio minio-init spark-master spark-worker-1

# Kiểm tra trạng thái (đợi healthy)
docker compose ps

# Spark UI: http://localhost:18080
# MinIO UI: http://localhost:9001 (minioadmin/minioadmin)
```

### 3. Chạy Pipeline Theo Từng Tầng

```bash
# Tầng Bronze: Ingestion (HuggingFace streaming) + Spark processing
# Thời gian: ~2-4 giờ với full dataset (44M rows)
docker compose run --rm pipeline python src/pipeline_runner.py --step 1_2

# Tầng Silver: Popularity, Text Profiles, Interactions
# Thời gian: ~1-2 giờ
docker compose run --rm pipeline python src/pipeline_runner.py --step 3_silver

# Tầng Gold: ID Mapping, Edge List, Training Metadata
# Thời gian: ~30 phút
docker compose run --rm pipeline python src/pipeline_runner.py --step 4_gold
```

### 4. Push Artifacts Lên HuggingFace Hub

```bash
# Đẩy Bronze (backup toàn bộ)
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline \
    python src/bronze/upload-bronze.py

# Đẩy Silver (batching 10 files/lô, xóa cũ trước)
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline \
    python src/silver/upload_silver_to_hf.py

# Đẩy Gold (mode partial: chỉ metadata + edge, không embedding)
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline \
    python src/gold/upload_gold_to_hf.py --mode partial

# Đẩy Gold (mode full: bao gồm cả numpy embeddings)
docker compose run --rm -e HF_TOKEN=hf_xxxx pipeline \
    python src/gold/upload_gold_to_hf.py --mode full
```

### 5. Test Nhanh (20k records)

```bash
# Kiểm tra pipeline end-to-end với dataset nhỏ (~5-10 phút)
MAX_REVIEW_RECORDS=20000 MAX_METADATA_RECORDS=5000 \
docker compose run --rm pipeline python src/pipeline_runner.py --step 1_2
```

### 6. Training (Google Colab)

Mở `src/TA_RecMind_V2_IntraLayer.ipynb` trên Google Colab:
- Notebook tự động tải artifacts từ HuggingFace Hub
- LLM embedding tính offline theo chunk (30k records/chunk, checkpoint tự động)
- ColabCheckpointManager tự động resume nếu runtime bị ngắt

---

## Kết Quả EDA Chính

### Thống Kê Dataset (Amazon Electronics 2023)

| Chỉ Số | Giá Trị | Nguồn |
|---|---|---|
| Raw interactions (sau Silver) | ~44,066,834 | `pipeline.log` |
| Positive interactions (rating ≥ 3) | ~36,168,550 | EDA Bronze |
| Users (sau Core-5 filter) | 1,847,662 | EDA Bronze |
| Items (train) | 1,042,121 | EDA Bronze |
| Items (train + cold val) | ~1,172,867 | EDA Gold |
| Interactions (train) | 1,396,428 | EDA Bronze |
| Sparsity đồ thị | 99.9993% | EDA Bronze |
| Tail items (≤ 5 tương tác) | 755,609 (72.51%) | EDA Silver |
| Cold-start items trong Val/Test | 130,746 (23.17%) | EDA Silver |

### Phân Phối Rating (Lý Do Dùng helpful_vote)

| Rating | Tỷ Lệ | Nhận Xét |
|---|---|---|
| ⭐⭐⭐⭐⭐ (5 sao) | 65.7% | Skew nặng → không dùng rating để phân loại chất lượng |
| ⭐⭐⭐⭐ (4 sao) | 14.7% | |
| ⭐⭐⭐ (3 sao) | 7.0% | Ngưỡng "positive" tối thiểu |
| ⭐⭐ (2 sao) | 4.5% | Bị lọc tại Bronze |
| ⭐ (1 sao) | 8.0% | Bị lọc tại Bronze |

> **Quyết định thiết kế:** Thay vì dùng rating, dùng `w(r) = 1 + log(1 + helpful_vote)` để xếp hạng chất lượng review. Hàm log kiểm soát outlier (review có 3,294 helpful_votes không gấp 3,294 lần review có 1 helpful_vote).

### Bằng Chứng Data Leakage (Lý Do Không Dùng `rating_number`)

```
Item B07H65KP63:
  train_freq      = 1,561   (thực tế trong tập train)
  rating_number   = 710,348 (tổng tích lũy toàn thời gian)
  → Chênh lệch ~455 lần → leakage nghiêm trọng nếu dùng rating_number
```

---

## Đóng Góp Kỹ Thuật (Novel Contributions)

| Đóng Góp | Công Thức | Nguồn Gốc |
|---|---|---|
| **Adaptive Gate Fusion (Bipartite)** | `γ = σ(w_base + w_sim·cos + w_freq·log1p(freq))` | Mở rộng RecMind Eq.6-8, đối xứng User+Item |
| **Tail-Weighted Alignment** | `w_v = 1/log(1+train_freq)` × InfoNCE | Mở rộng RecMind Eq.5 + LAGCL inverse-freq |
| **Anti-Leakage Classification** | Dùng `train_freq` thay vì `rating_number` | Phát hiện từ EDA (chênh lệch 455 lần) |
| **Review Quality Weighting** | `w(r) = 1 + log(1 + helpful_vote)` | Feature engineering Amazon 2023 |
| **Popularity-Penalized Re-ranking** | `s_adj = s_model - λ·log(1 + train_freq)` | Dựa trên "Challenging the Long Tail" |

---

## Luồng Dữ Liệu Tổng Thể (HuggingFace Repos)

```
chuongdo1104/amazon-2023-bronze
├── bronze/bronze_train.parquet/   (thư mục, nhiều part files — PySpark output)
├── bronze/bronze_val.parquet/     (thư mục)
├── bronze/bronze_test.parquet/    (thư mục — KHÔNG ĐỘNG VÀO sau Bronze)
└── bronze/bronze_meta.parquet     (single file — PyArrow output)

chuongdo1104/amazon-2023-silver
├── silver/silver_item_popularity.parquet/
├── silver/silver_item_text_profile.parquet/
├── silver/silver_user_text_profile.parquet/
├── silver/silver_interactions_train.parquet/
├── silver/silver_interactions_val.parquet/
└── silver/silver_val_ground_truth.parquet/

chuongdo1104/amazon-2023-gold
├── gold/gold_item_id_map.parquet      (single file)
├── gold/gold_user_id_map.parquet      (single file)
├── gold/gold_edge_index.npy           (single file — PyG format [2, E])
├── gold/gold_item_train_freq.npy
├── gold/gold_item_popularity_group.npy
├── gold/gold_user_train_freq.npy
├── gold/gold_user_activity_group.npy
├── gold/gold_negative_sampling_prob.npy
└── gold/gold_dataset_stats.json
```

---

## Tài Liệu Tham Khảo

1. **He et al. (2020).** LightGCN: Simplifying and Powering GCN. *SIGIR 2020.*
2. **Xue et al. (2025).** RecMind: LLM-Powered Recommendation. *arXiv:2509.06286v1.*
3. **Wu et al. (2021).** Self-supervised Graph Learning for Recommendation (SGL). *SIGIR 2021.*
4. **Hu et al. (2022).** LoRA: Low-Rank Adaptation of LLMs. *ICLR 2022.*
5. **Rendle et al. (2009).** BPR: Bayesian Personalized Ranking. *UAI 2009.*
6. **LAGCL.** Long-tail Augmented Graph Contrastive Learning.
7. **Hou et al. (2024).** Amazon Reviews 2023 Dataset. *arXiv:2403.03952.*

Xem đầy đủ tại [`references.md`](./references.md).
