import json
import re

path = 'd:\\recsys_pipeline_minio\\recsys_pipeline_minio\\src\\demo_recmind_final.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "app_code = '''" in source:
            # Change item_type_label to use dynamic thresholds
            old_func = """def item_type_label(rating_number):
    if rating_number is None or rating_number == 0: return "cold"
    elif rating_number >= 1000: return "head"
    elif rating_number >= 50: return "mid"
    else: return "tail\""""
            
            new_func = """# Tính toán ngưỡng Pareto thực tế từ Data (vd: Top 20% là Head)
head_thresh = 1000
mid_thresh = 50
if item_meta is not None and "rating_number" in item_meta.columns:
    head_thresh = max(8, int(item_meta["rating_number"].quantile(0.80)))
    mid_thresh = max(3, int(item_meta["rating_number"].quantile(0.50)))

def item_type_label(rating_number):
    if rating_number is None or rating_number == 0: return "cold"
    elif rating_number >= head_thresh: return "head"
    elif rating_number >= mid_thresh: return "mid"
    else: return "tail\""""
            
            source = source.replace(old_func, new_func)
            
            # Change the metrics display to show the thresholds
            old_metrics = """        st.markdown('<p class="section-label">1. Điều khiển</p>', unsafe_allow_html=True)"""
            new_metrics = """        st.markdown('<p class="section-label">1. Điều khiển</p>', unsafe_allow_html=True)
        st.markdown(f\"\"\"<div style="font-size:0.75rem; color:#64748b; margin-bottom: 1rem; padding: 0.5rem; background: var(--surface2); border-radius: 8px;">
        <b>Ngưỡng Data Thực Tế:</b><br/>
        🔥 Head: >= {head_thresh} ratings (Top 20%)<br/>
        ⚡ Mid: >= {mid_thresh} ratings (Top 50%)<br/>
        🌟 Tail: < {mid_thresh} ratings<br/>
        </div>\"\"\", unsafe_allow_html=True)"""
            
            source = source.replace(old_metrics, new_metrics)

            # Split lines back to list with newlines
            lines = source.splitlines(keepends=True)
            cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print("Patched successfully")
