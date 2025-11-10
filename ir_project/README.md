# Information Retrieval System

A complete semantic search system for educational questions using vector embeddings and FAISS for similarity search. This project demonstrates core Information Retrieval concepts including document embedding, semantic similarity search, and intelligent filtering strategies.

## 🎯 Project Overview

This Information Retrieval system provides:
- **Semantic Search**: Natural language queries using vector embeddings
- **Multi-criteria Filtering**: Filter by marks, difficulty, cognitive level
- **Intelligent Fallback**: Hierarchical filter relaxation for optimal results
- **REST API**: FastAPI-based web service
- **Interactive Demo**: Streamlit web interface
- **Evaluation Metrics**: Standard IR metrics (Precision@K, Recall@K, MAP, NDCG)

## 📁 Project Structure

```
ir_project/
├── data/
│   ├── questions.json          # Question dataset with metadata
│   └── sample_queries.txt      # Example test queries
├── embedding/
│   └── create_index.py         # Vector store creation and management
├── search/
│   └── semantic_search.py      # Core search engine with filtering
├── evaluation/
│   └── metrics.py              # IR evaluation metrics
├── demo/
│   ├── streamlit_demo.py       # Interactive web interface
│   └── api_demo.py             # REST API service
├── indexes/
│   └── faiss_index/            # FAISS vector store (created after setup)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Ollama (embedding model)
# Download from: https://ollama.ai/
# Then pull the embedding model:
ollama pull nomic-embed-text:latest
```

### 2. Create Vector Index

```bash
# Navigate to the embedding directory
cd embedding

# Create the FAISS vector index from questions
python create_index.py
```

### 3. Run Demos

**Streamlit Web Interface:**
```bash
cd demo
streamlit run streamlit_demo.py
```
Access at: http://localhost:8501

**REST API:**
```bash
cd demo
python api_demo.py
```
Access at: http://localhost:8000/docs

### 4. Test Search Functionality

```bash
# Test the search engine directly
cd search
python semantic_search.py
```

## 🔍 Core Features

### Semantic Search Engine

The system uses Ollama's `nomic-embed-text` model to create vector embeddings of questions, enabling semantic similarity search that goes beyond keyword matching.

**Key Components:**
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: Dense vector representations of questions
- **Metadata**: Rich question attributes (topic, difficulty, cognitive level)

### Intelligent Filtering

The search engine implements a hierarchical filtering strategy:

1. **Exact Match**: All specified criteria
2. **Relaxed Difficulty**: Keep marks and cognitive level
3. **Relaxed Cognitive**: Keep marks and difficulty  
4. **Marks Only**: Just the marks criterion
5. **Similarity Fallback**: Top similarity results

### Example Usage

```python
from search.semantic_search import SemanticSearchEngine

# Initialize search engine
engine = SemanticSearchEngine("../indexes/faiss_index")

# Search with filtering
results = engine.search(
    query="Natural language processing techniques",
    target_marks=3,
    target_difficulty="medium",
    target_cognitive="applying",
    max_results=10
)

# Display results
for result in results:
    print(f"Question: {result['question']}")
    print(f"Topic: {result['topic']} → {result['subtopic']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
    print("-" * 50)
```

## 📊 Dataset

The system works with a structured question dataset containing:

- **Questions**: Natural language educational questions
- **Topics**: Subject areas (e.g., "Artificial Intelligence")
- **Subtopics**: Specific areas (e.g., "Natural Language Processing")
- **Marks**: Point values (1-5)
- **Difficulty**: easy, medium, hard
- **Cognitive Level**: remembering, understanding, applying, analyzing, evaluating, creating
- **Question Type**: mcq, short, long, practical

## 🛠️ API Endpoints

### POST /search
Search for questions with full filtering options.

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "target_marks": 3,
    "target_difficulty": "medium",
    "max_results": 5
  }'
```

### GET /search
Simple search via query parameters.

```bash
curl "http://localhost:8000/search?q=deep learning&marks=2&difficulty=easy"
```

### GET /statistics
Get dataset statistics and distributions.

```bash
curl "http://localhost:8000/statistics"
```

## 📈 Evaluation Metrics

The system includes standard IR evaluation metrics:

- **Precision@K**: Relevant results in top-K
- **Recall@K**: Coverage of relevant results
- **F1@K**: Harmonic mean of precision and recall
- **Average Precision**: Area under precision-recall curve
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision across queries

```python
from evaluation.metrics import evaluate_search_results

# Evaluate search results
evaluation = evaluate_search_results(
    query="machine learning",
    results=search_results
)

print(f"Precision@5: {evaluation['metrics']['precision_at_5']:.3f}")
print(f"Recall@5: {evaluation['metrics']['recall_at_5']:.3f}")
```

## 🎓 Educational Value

This project demonstrates key IR concepts:

1. **Vector Space Model**: Documents as points in high-dimensional space
2. **Semantic Similarity**: Meaning-based rather than keyword-based search
3. **Query Processing**: Natural language to vector transformation
4. **Result Ranking**: Similarity scores and metadata filtering
5. **Evaluation**: Standard metrics for assessing retrieval quality

## 🔧 Configuration

### Environment Variables (Optional)

Create a `.env` file:

```bash
OLLAMA_MODEL=nomic-embed-text:latest
FAISS_INDEX_PATH=./indexes/faiss_index
MAX_SEARCH_RESULTS=50
SIMILARITY_THRESHOLD=0.0
```

### Customization

- **Embedding Model**: Change in `semantic_search.py` and `create_index.py`
- **Filtering Logic**: Modify `_smart_filter()` in `semantic_search.py`
- **Evaluation Metrics**: Add custom metrics in `evaluation/metrics.py`

## 🚀 Advanced Usage

### Batch Processing

```python
# Process multiple queries
queries = [
    "machine learning algorithms",
    "computer vision techniques", 
    "natural language processing"
]

for query in queries:
    results = engine.search(query, max_results=5)
    print(f"Query: {query} → {len(results)} results")
```

### Custom Relevance Scoring

```python
# Implement custom relevance scoring
def custom_relevance_score(result, query_context):
    base_score = result['similarity_score']
    
    # Boost recent or popular topics
    topic_boost = 1.1 if result['topic'] == 'AI' else 1.0
    
    # Penalize very short questions
    length_penalty = 0.9 if len(result['question']) < 50 else 1.0
    
    return base_score * topic_boost * length_penalty
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📝 License

This project is for educational purposes. Please ensure you have proper licenses for any datasets used.

## 🔗 References

- [FAISS Documentation](https://faiss.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [Information Retrieval Evaluation](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

---

**Built for Information Retrieval and Semantic Web coursework** 📚🔍