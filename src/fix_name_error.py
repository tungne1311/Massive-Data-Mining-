import json

path = 'd:\\recsys_pipeline_minio\\recsys_pipeline_minio\\src\\demo_recmind_final.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # 1. Clean up the global threshold code
        old_global = """# Tính toán ngưỡng Pareto thực tế từ Data (vd: Top 20% là Head)
head_thresh = 1000
mid_thresh = 50
if item_meta is not None and "rating_number" in item_meta.columns:
    head_thresh = max(8, int(item_meta["rating_number"].quantile(0.80)))
    mid_thresh = max(3, int(item_meta["rating_number"].quantile(0.70)))  # Top 30% (vì Head=20%, Mid=10%)

def item_type_label(rating_number):
    if rating_number is None or rating_number == 0: return "cold"
    elif rating_number >= head_thresh: return "head"
    elif rating_number >= mid_thresh: return "mid"
    else: return "tail\""""
        
        new_global = """def item_type_label(rating_number, head_t=1000, mid_t=50):
    if rating_number is None or rating_number == 0: return "cold"
    elif rating_number >= head_t: return "head"
    elif rating_number >= mid_t: return "mid"
    else: return "tail\""""
        
        source = source.replace(old_global, new_global)
        
        # 2. Add the threshold calculation after load_item_meta()
        old_load = """# ── Load Data ─────────────────────────────────────────────────────────────────
user_emb, item_emb = load_embeddings()
item_meta          = load_item_meta()
user_map           = load_user_map()
user_reviews       = load_user_reviews()

if item_meta is not None and "item_idx" not in item_meta.columns:"""
        
        new_load = """# ── Load Data ─────────────────────────────────────────────────────────────────
user_emb, item_emb = load_embeddings()
item_meta          = load_item_meta()
user_map           = load_user_map()
user_reviews       = load_user_reviews()

head_thresh = 1000
mid_thresh = 50
if item_meta is not None and "rating_number" in item_meta.columns:
    head_thresh = max(8, int(item_meta["rating_number"].quantile(0.80)))
    mid_thresh = max(3, int(item_meta["rating_number"].quantile(0.70)))

if item_meta is not None and "item_idx" not in item_meta.columns:"""
        
        source = source.replace(old_load, new_load)
        
        # 3. Update the apply function for item_type_label
        old_apply = """if item_meta is not None and "item_type" not in item_meta.columns:
    rn_col = "rating_number" if "rating_number" in item_meta.columns else None
    if rn_col: item_meta["item_type"] = item_meta[rn_col].apply(item_type_label)"""
        
        new_apply = """if item_meta is not None and "item_type" not in item_meta.columns:
    rn_col = "rating_number" if "rating_number" in item_meta.columns else None
    if rn_col: item_meta["item_type"] = item_meta[rn_col].apply(lambda x: item_type_label(x, head_thresh, mid_thresh))"""
        
        source = source.replace(old_apply, new_apply)

        # Write lines back
        if source != ''.join(cell['source']):
            cell['source'] = source.splitlines(keepends=True)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print("Fixed NameError")
