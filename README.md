## Data Preprocessing steps
### Overview of My Contribution

### 1. OCR Text Extraction
- Built a script (`without flask.py`) to extract text from both **PDFs** and **images**.
- Used `PyMuPDF` to convert PDF pages to images and applied `pytesseract` for OCR.
- Applied basic checks for diagrams, mathematical symbols, and shape content.
- Output saved as: `extracted_output.txt`

#### 2. Text Refinement
- Implemented `refinement.py` to:
  - Clean and normalize OCR text (remove junk, fix spacing, allow math symbols).
  - Analyze content for math presence or diagram indicators.
- Result used for question generation.

#### 3. Question Generation
- Developed `questions_generator.py`:
  - Split the cleaned text into meaningful chunks.
  - Generated questions using the model `iarfmoose/t5-base-question-generator`.
  - Used `Ollama (llama3)` via LangChain to predict **subtopics** for each question.
  - Labeled each question with topic, subtopic, marks, difficulty, time, and cognitive level.
- Output saved as: `output_questions.json`

#### 4. Subtopic Cleaning
- Created `subtopics.py` to:
  - Clean noisy subtopic outputs using regex.
  - Standardize them into clear, short academic labels.
- Output saved as: `questions.json`

#### 5. Embedding & Vector Store Creation
- Built `ollama_store.py` to:
  - Load final questions from `questions.json`.
  - Generate embeddings using `OllamaEmbeddings` (model: llama3).
  - Store them in a **FAISS** vector index for retrieval.
- Output: `faiss_index_ollama/`

#### 6. Search & Filtering (Validation)
- Implemented and tested `ollama_search.py` to:
  - Query the FAISS index semantically.
  - Apply filters like marks, difficulty, and cognitive level.
  - Print matching questions with their metadata.

### Output
- `extracted_output.txt`: OCR + refined content
- `output_questions.json`: Raw question generation output
- `questions.json`: Cleaned and finalized questions with proper subtopics
- `faiss_index_ollama/`: Vector store containing embedded questions


### Tools & Models Used
- `pytesseract`, `PyMuPDF`, `OpenCV`
- `transformers` (T5 question generator)
- `LangChain` + `Ollama` with `llama3`
- `FAISS` for vector storage and retrieval
