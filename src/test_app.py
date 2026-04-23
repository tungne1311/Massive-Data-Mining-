
# ═══════════════════════════════════════════════════════════════════════════════
#  TA-RecMind V2 — Interactive Demo App
#  Streamlit · Dark Neon Theme · Dual-View Recommender
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TA-RecMind V2 · Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e2d45;
    --accent-blue: #00d4ff;
    --accent-purple: #a855f7;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --text: #e2e8f0;
    --text-dim: #64748b;
}

.stApp { background: var(--bg); font-family: 'Syne', sans-serif; color: var(--text); }
.stApp > header { display: none; }

.hero-header {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a0e1a 50%, #120d1e 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #00d4ff, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem; color: var(--text-dim); margin-top: 0.4rem;
    letter-spacing: 0.12em; text-transform: uppercase;
}

.pill {
    display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
    font-family: 'Space Mono', monospace; font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; vertical-align: middle;
}
.pill-head  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.pill-mid   { background: rgba(168,85,247,0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }
.pill-tail  { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.pill-cold  { background: rgba(0,212,255,0.12);  color: #00d4ff; border: 1px solid rgba(0,212,255,0.25); }

.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.1rem 0.9rem; margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(0,212,255,0.3); }
.card-history { border-color: rgba(168,85,247,0.3); background: rgba(26,34,53,0.5); }
.card-title {
    font-size: 0.88rem; font-weight: 600; color: var(--text);
    margin-bottom: 0.3rem; line-height: 1.35;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.card-meta { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--text-dim); }
.card-score { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: var(--accent-blue); font-weight: 700; }

.dist-bar {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.8rem 1.2rem; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;
}
.dist-item { font-size: 0.78rem; display: flex; align-items: center; gap: 0.3rem; }
.dist-dot  { width:10px; height:10px; border-radius:50%; display:inline-block; }

.section-label {
    font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--text-dim); margin-bottom: 0.8rem;
    padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
}
</style>
""",unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
DEMO_CONFIG = json.loads(os.environ.get("RECMIND_DEMO_CONFIG", "{}"))
EMBEDDINGS_PATH = DEMO_CONFIG.get("EMBEDDINGS_PATH", "")
ENCODER_MODEL   = DEMO_CONFIG.get("ENCODER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Data Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading embeddings...")
def load_embeddings():
    # Added explicit check for empty path and file existence
    if not EMBEDDINGS_PATH or not Path(EMBEDDINGS_PATH).is_file():
        return None, None
    data = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=False)
    uk = next((k for k in ["user_final_all", "user_embeddings"] if k in data), None)
    ik = next((k for k in ["item_final_all", "item_embeddings"] if k in data), None)
    if uk is None or ik is None: return None, None
    return data[uk].float(), data[ik].float()

@st.cache_resource(show_spinner="📦 Loading metadata...")
def load_item_meta():
    if os.path.exists('/content/item_meta_cache.parquet'):
        return pd.read_parquet('/content/item_meta_cache.parquet')
    return None

@st.cache_resource(show_spinner="🗺 Loading ID maps...")
def load_item_map():
    if os.path.exists('/content/item_map_cache.parquet'):
        return pd.read_parquet('/content/item_map_cache.parquet')
    return None

@st.cache_resource(show_spinner="🗺 Loading User maps...")
def load_user_map():
    if os.path.exists('/content/user_map_cache.parquet'):
        return pd.read_parquet('/content/user_map_cache.parquet')
    return None

@st.cache_resource(show_spinner="📝 Loading user reviews...")
def load_user_reviews():
    if os.path.exists('/content/user_reviews_cache.parquet'):
        return pd.read_parquet('/content/user_reviews_cache.parquet')
    return None

@st.cache_resource(show_spinner="🔤 Loading text encoder...")
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ENCODER_MODEL, device="cpu")

# ── Utility Functions ─────────────────────────────────────────────────────────
def item_type_label(rating_number, head_t=1000, mid_t=50):
    if rating_number is None or rating_number == 0: return "cold"
    elif rating_number >= head_t: return "head"
    elif rating_number >= mid_t: return "mid"
    else: return "tail"

def render_type_badge(t):
    badges = {
        "head": '<span class="pill pill-head">🔥 HEAD</span>',
        "mid":  '<span class="pill pill-mid">⚡ MID</span>',
        "tail": '<span class="pill pill-tail">🌟 TAIL</span>',
        "cold": '<span class="pill pill-cold">❄️ COLD</span>',
    }
    return badges.get(t, "")

def get_recommendations(user_idx, user_emb, item_emb, top_k=20):
    u_vec = user_emb[user_idx]
    scores = torch.matmul(item_emb, u_vec)
    top_scores, top_idx = torch.topk(scores, top_k)
    return top_idx.tolist(), top_scores.tolist()

def get_user_history(user_idx, user_map, item_emb, user_emb, n_hist=4):
    """Lấy lịch sử user từ user_map nếu có, ngược lại giả lập để demo."""
    if user_map is not None:
        idx_col = "mapped_id" if "mapped_id" in user_map.columns else "user_idx"
        if idx_col in user_map.columns:
            row = user_map[user_map[idx_col] == user_idx]
            if len(row) > 0:
                row_data = row.iloc[0]
                for col in ['history', 'interacted_items', 'item_list', 'item_id_list', 'history_item_ids']:
                    if col in row_data:
                        hist = row_data[col]
                        if isinstance(hist, (list, np.ndarray)) and len(hist) > 0:
                            return list(hist)[:n_hist]
                        elif isinstance(hist, str):
                            return [int(x) for x in str(hist).replace('[','').replace(']','').split(',') if x.strip().isdigit()][:n_hist]

    # Giả lập nếu không có cột lịch sử thật
    u_vec = user_emb[user_idx]
    scores = torch.matmul(item_emb, u_vec)
    top_scores, top_idx = torch.topk(scores, 100 + n_hist)
    return top_idx.tolist()[100:]

def rerank_by_ratio(pool_idxs, pool_scores, item_meta, top_k, target_head, target_mid, target_tail):
    # Tính số lượng mục tiêu cho từng nhóm
    total = target_head + target_mid + target_tail
    if total == 0: total = 1
    k_head = int(top_k * (target_head / total))
    k_mid = int(top_k * (target_mid / total))
    k_tail = top_k - k_head - k_mid

    selected_idxs = []
    selected_scores = []
    quota = {"head": k_head, "mid": k_mid, "tail": k_tail, "cold": 0}

    # Duyệt pool và pick greedily
    for idx, score in zip(pool_idxs, pool_scores):
        if len(selected_idxs) >= top_k: break

        itype = "tail"
        if item_meta is not None:
            row = item_meta[item_meta["item_idx"] == idx]
            if len(row) > 0:
                itype = row.iloc[0].get("item_type", "tail")
            else:
                itype = "cold"
        else:
            itype = "cold"

        if quota.get(itype, 0) > 0:
            selected_idxs.append(idx)
            selected_scores.append(score)
            quota[itype] -= 1

    # Lấp đầy nếu thiếu
    if len(selected_idxs) < top_k:
        for idx, score in zip(pool_idxs, pool_scores):
            if len(selected_idxs) >= top_k: break
            if idx not in selected_idxs:
                selected_idxs.append(idx)
                selected_scores.append(score)

    return selected_idxs, selected_scores

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""<div class="hero-header">
  <p class="hero-sub">Dual-View Recommender · LightGCN + Semantic Fusion</p>
  <h1 class="hero-title">TA-RecMind V2</h1>
</div>""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
user_emb, item_emb = load_embeddings()
item_meta          = load_item_meta()
user_map           = load_user_map()
user_reviews       = load_user_reviews()

head_thresh = 1000
mid_thresh = 50
if item_meta is not None and "rating_number" in item_meta.columns:
    head_thresh = max(8, int(item_meta["rating_number"].quantile(0.80)))
    mid_thresh = max(3, int(item_meta["rating_number"].quantile(0.70)))

if item_meta is not None and "item_idx" not in item_meta.columns:
    item_meta["item_idx"] = item_meta.index

if item_meta is not None and "item_type" not in item_meta.columns:
    rn_col = "rating_number" if "rating_number" in item_meta.columns else None
    if rn_col: item_meta["item_type"] = item_meta[rn_col].apply(lambda x: item_type_label(x, head_thresh, mid_thresh))
    else: item_meta["item_type"] = "tail"

n_users = user_emb.shape[0] if user_emb is not None else 0

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📺 Khám Phá Gợi Ý", "🚀 Zero-Shot Cold-Start"])

with tab1:
    col_ctrl, col_history, col_main = st.columns([1.2, 1.2, 2.8], gap="large")
    with col_ctrl:
        st.markdown('<p class="section-label">1. Điều khiển</p>', unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:0.75rem; color:#64748b; margin-bottom: 1rem; padding: 0.5rem; background: var(--surface2); border-radius: 8px;">
        <b>Ngưỡng Data Thực Tế:</b><br/>
        🔥 Head: >= {head_thresh} ratings (Top 20%)<br/>
        ⚡ Mid: >= {mid_thresh} ratings (Top 30%)<br/>
        🌟 Tail: < {mid_thresh} ratings<br/>
        </div>""", unsafe_allow_html=True)
        top_k = st.slider("Số lượng (Top-K)", 6, 60, 24, step=6)

        st.markdown('<p class="section-label" style="margin-top:1.5rem">Tỷ lệ Head/Mid/Tail (%)</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        target_head = c1.number_input("Head", min_value=0, max_value=100, value=50, step=5)
        target_mid = c2.number_input("Mid", min_value=0, max_value=100, value=20, step=5)
        target_tail = c3.number_input("Tail", min_value=0, max_value=100, value=30, step=5)

        random_btn = st.button("🎲 Random User", use_container_width=True, type="primary")
        if "current_user_idx" not in st.session_state or random_btn:
            st.session_state.current_user_idx = int(np.random.randint(0, max(1, n_users)))
        user_idx = st.session_state.current_user_idx
        st.info(f"Đang hiển thị cho User ID: {user_idx}")

    with col_history:
        st.markdown('<p class="section-label">2. Đã mua (Lịch sử)</p>', unsafe_allow_html=True)
        if user_emb is not None:
            hist_idxs = get_user_history(user_idx, user_map, item_emb, user_emb)
            for hidx in hist_idxs:
                htitle = f"Item #{hidx}"
                hratings = 0
                hbadge = "tail"
                if item_meta is not None:
                    row = item_meta[item_meta["item_idx"] == hidx]
                    if len(row) > 0:
                        htitle = str(row.iloc[0].get("title", htitle))[:60]
                        hratings = int(row.iloc[0].get("rating_number", 0))
                        hbadge = row.iloc[0].get("item_type", "tail")
                badge_html = render_type_badge(hbadge)

                st.markdown(f"""<div class="card card-history">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;">
                    <span style="font-family:Space Mono,monospace;font-size:0.6rem;color:#a855f7;">LỊCH SỬ</span>
                    {badge_html}
                  </div>
                  <div class="card-title" style="font-size:0.8rem;">{htitle}</div>
                  <div style="margin-top:0.5rem;">
                    <span class="card-meta">{hratings:,} ratings</span>
                  </div>
                </div>""", unsafe_allow_html=True)
                with st.expander("🔍 Chi tiết & Review của bạn"):
                    if item_meta is not None and len(row) > 0:
                        st.markdown(f"**Tên:** {row.iloc[0].get('title', 'N/A')}")
                        st.markdown(f"**Giá:** ${row.iloc[0].get('price', 'N/A')}")
                    else:
                        st.markdown("Không có thông tin chi tiết sản phẩm.")
                        
                    # Hiển thị Review nếu có data
                    st.markdown("---")
                    if user_reviews is not None:
                        # Lọc review của user này cho item này
                        u_rev = user_reviews[(user_reviews['user_idx'] == user_idx) & (user_reviews['item_idx'] == hidx)]
                        if len(u_rev) > 0:
                            u_rating = u_rev.iloc[0].get('rating', 0)
                            u_text = u_rev.iloc[0].get('reviewText', 'Không có nội dung.')
                            st.markdown(f"**Bạn đã đánh giá:** {'⭐' * int(u_rating)}")
                            st.markdown(f'> *"{u_text}"*')
                        else:
                            st.markdown("*(Không tìm thấy review chi tiết)*")
                    else:
                        st.markdown("*(Chưa tải dữ liệu Review)*")

    with col_main:
        st.markdown('<p class="section-label">3. Gợi ý cá nhân hóa</p>', unsafe_allow_html=True)
        if user_emb is None: st.warning("Chưa có embeddings.")
        else:
            t0 = time.time()
            pool_idxs, pool_scores = get_recommendations(user_idx, user_emb, item_emb, top_k=top_k * 4)
            top_item_idxs, top_scores = rerank_by_ratio(pool_idxs, pool_scores, item_meta, top_k, target_head, target_mid, target_tail)
            elapsed_ms = (time.time() - t0) * 1000

            rec_rows = []
            for iidx, score in zip(top_item_idxs, top_scores):
                if item_meta is not None:
                    row = item_meta[item_meta["item_idx"] == iidx]
                    if len(row) > 0:
                        d = row.iloc[0].to_dict()
                        d["_score"] = score
                        rec_rows.append(d)
                        continue
                rec_rows.append({"title": f"Item #{iidx}", "item_type": "cold", "_score": score})

            counts = {"head":0, "mid":0, "tail":0, "cold":0}
            for r in rec_rows:
                t = r.get("item_type", "tail")
                counts[t] = counts.get(t, 0) + 1
            total_rec = len(rec_rows) or 1

            st.markdown(f"""<div class="dist-bar">
              <span style="font-size:0.72rem;color:#64748b;font-family:Space Mono,monospace;">TOP-{top_k} MIX</span>
              <span class="dist-item"><span class="dist-dot" style="background:#ef4444"></span><span style="color:#ef4444">{counts['head']} HEAD ({(counts['head']/total_rec)*100:.0f}%)</span></span>
              <span class="dist-item"><span class="dist-dot" style="background:#a855f7"></span><span style="color:#a855f7">{counts['mid']} MID ({(counts['mid']/total_rec)*100:.0f}%)</span></span>
              <span class="dist-item"><span class="dist-dot" style="background:#f59e0b"></span><span style="color:#f59e0b">{counts['tail']} TAIL ({(counts['tail']/total_rec)*100:.0f}%)</span></span>
              <span class="dist-item"><span class="dist-dot" style="background:#00d4ff"></span><span style="color:#00d4ff">{counts['cold']} COLD</span></span>
              <span style="margin-left:auto;font-family:Space Mono,monospace;font-size:0.65rem;">⚡ {elapsed_ms:.1f}ms</span>
            </div>""", unsafe_allow_html=True)

            grid_cols = st.columns(3)
            for i, row in enumerate(rec_rows):
                with grid_cols[i % 3]:
                    badge = render_type_badge(row.get("item_type", "tail"))
                    title = str(row.get("title"))[:80]
                    score = row.get("_score", 0)
                    if hasattr(score, 'item'): score = score.item()

                    st.markdown(f"""<div class="card">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;">
                        <span style="font-family:Space Mono,monospace;font-size:0.6rem;color:#475569;">#{i+1}</span>
                        {badge}
                      </div>
                      <div class="card-title">{title}</div>
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;">
                        <span class="card-meta">{int(row.get("rating_number", 0)):,} ratings</span>
                        <span class="card-score">s={score:.3f}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)
                    with st.expander("🔍 Xem chi tiết"):
                        st.markdown(f"**Tên:** {row.get('title', 'N/A')}")
                        cats = row.get('categories', 'N/A')
                        st.markdown(f"**Categories:** {cats}")
                        price = row.get('price', 'N/A')
                        st.markdown(f"**Giá:** ${price}")
                        desc = str(row.get('description', 'Không có mô tả'))
                        if len(desc) > 200: desc = desc[:200] + '...'
                        st.markdown(f"**Mô tả:** {desc}")

with tab2:
    st.markdown('<p class="section-label">Thêm Sản Phẩm Mới & Xem Zero-Shot Graph Response</p>', unsafe_allow_html=True)
    c_in, c_sim, c_usr = st.columns([1.5, 1.5, 1.5], gap="large")

    with c_in:
        new_title = st.text_input("Tên sản phẩm mới", "Tai nghe thể thao chống nước IPX7")
        new_desc = st.text_area("Mô tả chi tiết", "Thiết kế không dây nhỏ gọn, bass mạnh mẽ, pin 12h, chống ồn chủ động. Rất phù hợp cho chạy bộ, tập gym hoặc đi mưa.")
        do_zero_shot = st.button("🚀 Thêm & Phân Tích Mạng Graph", type="primary", use_container_width=True)

    if do_zero_shot:
        encoder = load_encoder()
        if not encoder:
            st.error("Chưa load được encoder (SentenceTransformers).")
        elif item_emb is None or user_emb is None:
            st.error("Chưa load được embeddings.")
        else:
            with st.spinner("Đang trích xuất Vector LLM & Tính toán độ tương đồng..."):
                t0 = time.time()
                text_feat = f"{new_title} {new_desc}"
                vec = encoder.encode(text_feat)
                vec_tensor = torch.tensor(vec, dtype=torch.float32)
                vec_tensor = F.normalize(vec_tensor, p=2, dim=0)

                # 1. Tìm item tương đồng (Semantic similarity in item space)
                item_scores = torch.matmul(item_emb, vec_tensor)
                sim_item_scores, sim_item_idxs = torch.topk(item_scores, 4)

                # 2. Tìm user tiềm năng (User-Item matching)
                user_scores = torch.matmul(user_emb, vec_tensor)
                u_scores, u_idxs = torch.topk(user_scores, 4)
                el_ms = (time.time() - t0) * 1000

            st.success(f"Hoàn tất trong {el_ms:.1f}ms! Vector Embedding đã sẵn sàng tương tác ngay lập tức.")

            with c_sim:
                st.markdown('<p style="color:#00d4ff;font-weight:700;">Sản Phẩm Tương Đồng</p>', unsafe_allow_html=True)
                for s_score, s_idx in zip(sim_item_scores.tolist(), sim_item_idxs.tolist()):
                    htitle = f"Item #{s_idx}"
                    hratings = 0
                    if item_meta is not None:
                        row = item_meta[item_meta["item_idx"] == s_idx]
                        if len(row) > 0:
                            htitle = str(row.iloc[0].get("title", htitle))[:70]
                            hratings = int(row.iloc[0].get("rating_number", 0))

                    st.markdown(f"""<div class="card" style="border-color:rgba(0,212,255,0.4)">
                      <div class="card-title">{htitle}</div>
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;">
                        <span class="card-meta">{hratings:,} ratings</span>
                        <span class="card-score">s={s_score:.3f}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with c_usr:
                st.markdown('<p style="color:#a855f7;font-weight:700;">Gợi ý cho (Top KH Tiềm Năng)</p>', unsafe_allow_html=True)
                for u_score, u_idx in zip(u_scores.tolist(), u_idxs.tolist()):
                    st.markdown(f"""<div class="card" style="border-color:rgba(168,85,247,0.4)">
                      <div class="card-title">User ID: {u_idx}</div>
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;">
                        <span class="card-meta">VIP Customer</span>
                        <span class="card-score">s={u_score:.3f}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)
