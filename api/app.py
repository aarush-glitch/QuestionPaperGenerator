
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import shutil
import json

# Import backend logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Pytesseract import refinement
from questions_generation import question_gen
from Clean_subtopics import subtopics as subtopic_cleaner
from Store_and_embed import ollama_store, ollama_search


app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
EXTRACTED_TEXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'questions_generation', 'extracted_output.txt')
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), '..', 'questions_generation', 'output_questions.json')
CLEANED_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), '..', 'questions_generation', 'questions.json')
KEYWORDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'questions_generation', 'keyword.json')
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'new_faiss_index')

@app.route('/')
def index():
    return 'API is running.'

# Placeholder endpoints for each pipeline step

# 1. OCR & Refinement
@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Run OCR and refinement
    try:
        if filename.lower().endswith('.pdf'):
            result = refinement.handle_pdf(file_path)
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            result = refinement.handle_image(file_path)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        # Save extracted text
        with open(EXTRACTED_TEXT_PATH, 'w', encoding='utf-8') as f:
            f.write(result)
        return jsonify({'message': 'File uploaded and OCR complete.', 'extracted_text': result[:500]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 2. Question Generation
@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        with open(EXTRACTED_TEXT_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
        with open(KEYWORDS_PATH, 'r', encoding='utf-8') as kf:
            topic_keywords = json.load(kf)
        output = question_gen.generate_questions(text, topic_keywords)
        question_gen.save_questions(output, filename=QUESTIONS_PATH)
        return jsonify({'message': 'Questions generated.', 'counts': {k: len(v) for k, v in output.items()}}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 3. Subtopic Cleaning
@app.route('/api/clean-subtopics', methods=['POST'])
def clean_subtopics():
    try:
        subtopic_cleaner.clean_subtopics(QUESTIONS_PATH, CLEANED_QUESTIONS_PATH)
        return jsonify({'message': 'Subtopics cleaned.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 4. Embedding & FAISS
@app.route('/api/embed', methods=['POST'])
def embed():
    try:
        docs = ollama_store.load_questions(CLEANED_QUESTIONS_PATH)
        ollama_store.store_in_faiss(docs, index_path=FAISS_INDEX_PATH)
        return jsonify({'message': 'Embedding and FAISS index creation complete.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 5. Semantic Search & Filtering
@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    marks = data.get('marks')
    difficulty = data.get('difficulty')
    cognitive = data.get('cognitive')
    try:
        # Load FAISS and perform search
        embeddings = ollama_search.embeddings
        db = ollama_search.FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        results = db.similarity_search(query, k=20)
        if marks and difficulty and cognitive:
            filtered = ollama_search.smart_filter(results, marks, difficulty, cognitive)
        else:
            filtered = results
        # Format for frontend
        formatted = [
            {
                'question': d.page_content,
                **d.metadata
            } for d in filtered
        ]
        return jsonify({'results': formatted}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
