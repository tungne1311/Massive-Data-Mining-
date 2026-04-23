import json

path = 'd:\\recsys_pipeline_minio\\recsys_pipeline_minio\\src\\demo_recmind_final.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "app_code = '''" in source:
            app_code = source.split("app_code = '''")[1].split("'''")[0]
            with open('d:\\recsys_pipeline_minio\\recsys_pipeline_minio\\src\\test_app.py', 'w', encoding='utf-8') as out:
                out.write(app_code)
            print("Extracted to test_app.py")
            break
