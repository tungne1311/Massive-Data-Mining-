import json

path = 'd:\\recsys_pipeline_minio\\recsys_pipeline_minio\\src\\demo_recmind_final.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "app_code = '''" in source:
            old_quantile = """mid_thresh = max(3, int(item_meta["rating_number"].quantile(0.50)))"""
            new_quantile = """mid_thresh = max(3, int(item_meta["rating_number"].quantile(0.70)))  # Top 30% (vì Head=20%, Mid=10%)"""
            source = source.replace(old_quantile, new_quantile)
            
            old_metrics = """⚡ Mid: >= {mid_thresh} ratings (Top 50%)"""
            new_metrics = """⚡ Mid: >= {mid_thresh} ratings (Top 30%)"""
            source = source.replace(old_metrics, new_metrics)

            # Split lines back to list with newlines
            lines = source.splitlines(keepends=True)
            cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print("Patched correctly to 70th percentile")
