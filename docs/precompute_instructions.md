ETL & Precompute instructions

1) Create a Python virtual environment and install dependencies

On Windows (cmd.exe):

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install langchain_community langchain-ollama ollama
```

2) Run the ETL for Data Structures CSV

```
python scripts/etl_data_structures.py
```

This will write:
- `data/courses/data_structures/cleaned_questions.json`
- `data/courses/data_structures/manifest.json`
- optionally `data/courses/data_structures/faiss_index/` if FAISS + Ollama embeddings are available

3) Precompute assets for other courses (example)

```
python scripts/precompute_course_assets.py --course "Algorithms" --out-slug algorithms
```

This creates `data/courses/algorithms/cleaned_questions.json`, `manifest.json`, and optional `faiss_index/`.

4) Verify in Streamlit UI

Start the streamlit app (if using `streamlit run streamlit_app.py` or your project entry):

```
streamlit run streamlit_app.py
```

Open the wizard page (`pages/2_User_Pipeline.py`), select the course (e.g. "Data Structures"). The page will automatically load precomputed cleaned questions and populate the cleaned data preview. If a prebuilt FAISS index exists for the course, it will be copied into `new_faiss_index/` and search will be available immediately.
