import json

path = 'd:\\recsys_pipeline_minio\\recsys_pipeline_minio\\src\\demo_recmind_final.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "app_code = '''" in source:
            # Replace the rendering part
            old_code = """                    st.markdown(f\"\"\"<div class=\"card\">
                      <div style=\"display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;\">
                        <span style=\"font-family:Space Mono,monospace;font-size:0.6rem;color:#475569;\">#{i+1}</span>
                        {badge}
                      </div>
                      <div class=\"card-title\">{title}</div>
                      <div style=\"display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;\">
                        <span class=\"card-meta\">{int(row.get("rating_number", 0)):,} ratings</span>
                        <span class=\"card-score\">s={score:.3f}</span>
                      </div>
                    </div>\"\"\", unsafe_allow_html=True)"""
                    
            new_code = """                    st.markdown(f\"\"\"<div class=\"card\">
                      <div style=\"display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;\">
                        <span style=\"font-family:Space Mono,monospace;font-size:0.6rem;color:#475569;\">#{i+1}</span>
                        {badge}
                      </div>
                      <div class=\"card-title\">{title}</div>
                      <div style=\"display:flex;justify-content:space-between;align-items:center;margin-top:0.5rem;\">
                        <span class=\"card-meta\">{int(row.get("rating_number", 0)):,} ratings</span>
                        <span class=\"card-score\">s={score:.3f}</span>
                      </div>
                    </div>\"\"\", unsafe_allow_html=True)
                    with st.expander("🔍 Xem chi tiết"):
                        st.markdown(f"**Tên:** {row.get('title', 'N/A')}")
                        cats = row.get('categories', 'N/A')
                        st.markdown(f"**Categories:** {cats}")
                        price = row.get('price', 'N/A')
                        st.markdown(f"**Giá:** ${price}")
                        desc = str(row.get('description', 'Không có mô tả'))
                        if len(desc) > 200: desc = desc[:200] + '...'
                        st.markdown(f"**Mô tả:** {desc}")"""
            
            source = source.replace(old_code, new_code)
            
            # Update history too
            old_history = """                st.markdown(f\"\"\"<div class=\"card card-history\">
                  <div style=\"display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;\">
                    <span style=\"font-family:Space Mono,monospace;font-size:0.6rem;color:#a855f7;\">LỊCH SỬ</span>
                    {badge_html}
                  </div>
                  <div class=\"card-title\" style=\"font-size:0.8rem;\">{htitle}</div>
                  <div style=\"margin-top:0.5rem;\">
                    <span class=\"card-meta\">{hratings:,} ratings</span>
                  </div>
                </div>\"\"\", unsafe_allow_html=True)"""

            new_history = """                st.markdown(f\"\"\"<div class=\"card card-history\">
                  <div style=\"display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;\">
                    <span style=\"font-family:Space Mono,monospace;font-size:0.6rem;color:#a855f7;\">LỊCH SỬ</span>
                    {badge_html}
                  </div>
                  <div class=\"card-title\" style=\"font-size:0.8rem;\">{htitle}</div>
                  <div style=\"margin-top:0.5rem;\">
                    <span class=\"card-meta\">{hratings:,} ratings</span>
                  </div>
                </div>\"\"\", unsafe_allow_html=True)
                with st.expander("🔍 Chi tiết"):
                    if item_meta is not None and len(row) > 0:
                        st.markdown(f"**Tên:** {row.iloc[0].get('title', 'N/A')}")
                        st.markdown(f"**Giá:** ${row.iloc[0].get('price', 'N/A')}")
                    else:
                        st.markdown("Không có thông tin chi tiết")"""
            
            source = source.replace(old_history, new_history)
            
            # Split lines back to list with newlines
            lines = source.splitlines(keepends=True)
            cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print("Patched successfully")
