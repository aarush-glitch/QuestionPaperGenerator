import sys, os, importlib, json
out = {}
root = os.path.abspath(os.path.dirname(__file__) + '/..')
out['cwd'] = os.path.abspath('.')
out['csv_exists'] = os.path.exists('final_database_coe.csv')
out['etl_script'] = os.path.exists('scripts/etl_data_structures.py')
out['precompute_script'] = os.path.exists('scripts/precompute_course_assets.py')
modules = ['langchain_community.vectorstores','langchain_ollama','ollama','fitz','faiss','langchain','pytesseract','cv2']
mods = {}
for m in modules:
    try:
        importlib.import_module(m)
        mods[m] = True
    except Exception as e:
        mods[m] = str(e)
out['imports'] = mods
out['keyword_json'] = os.path.exists('questions_generation/keyword.json')
out['question_gen'] = os.path.exists('questions_generation/question_gen.py')
print(json.dumps(out))
