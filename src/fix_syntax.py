import json

path = 'd:\\recsys_pipeline_minio\\recsys_pipeline_minio\\src\\demo_recmind_final.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Sửa lỗi string format bị double quotes lồng nhau
        if 'st.markdown(f"> *"{u_text}"*")' in source:
            source = source.replace('st.markdown(f"> *"{u_text}"*")', 'st.markdown(f\'> *"{u_text}"*\')')
            
            # Tách lại thành list
            lines = source.splitlines(keepends=True)
            cell['source'] = lines

with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print("Fixed syntax error")
