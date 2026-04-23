import json

path = 'd:\\recsys_pipeline_minio\\recsys_pipeline_minio\\src\\demo_recmind_final.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "app_code = '''" in source:
            # 1. Add load_user_reviews function
            old_loader = """@st.cache_resource(show_spinner="🔤 Loading text encoder...")
def load_encoder():"""
            new_loader = """@st.cache_resource(show_spinner="📝 Loading user reviews...")
def load_user_reviews():
    if os.path.exists('/content/user_reviews_cache.parquet'):
        return pd.read_parquet('/content/user_reviews_cache.parquet')
    return None

@st.cache_resource(show_spinner="🔤 Loading text encoder...")
def load_encoder():"""
            source = source.replace(old_loader, new_loader)
            
            # 2. Call load_user_reviews in main
            old_data_load = """user_emb, item_emb = load_embeddings()
item_meta          = load_item_meta()
user_map           = load_user_map()"""
            new_data_load = """user_emb, item_emb = load_embeddings()
item_meta          = load_item_meta()
user_map           = load_user_map()
user_reviews       = load_user_reviews()"""
            source = source.replace(old_data_load, new_data_load)

            # 3. Modify history card expander to show user review
            old_history_expander = """                with st.expander("🔍 Chi tiết"):
                    if item_meta is not None and len(row) > 0:
                        st.markdown(f"**Tên:** {row.iloc[0].get('title', 'N/A')}")
                        st.markdown(f"**Giá:** ${row.iloc[0].get('price', 'N/A')}")
                    else:
                        st.markdown("Không có thông tin chi tiết")"""
            
            new_history_expander = """                with st.expander("🔍 Chi tiết & Review của bạn"):
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
                            st.markdown(f"> *\"{u_text}\"*")
                        else:
                            st.markdown("*(Không tìm thấy review chi tiết)*")
                    else:
                        st.markdown("*(Chưa tải dữ liệu Review)*")"""
            
            source = source.replace(old_history_expander, new_history_expander)
            
            # Split lines back to list with newlines
            lines = source.splitlines(keepends=True)
            cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print("Patched review feature")
