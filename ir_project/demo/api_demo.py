"""
Information Retrieval System - REST API Demo
===========================================

FastAPI-based REST API for the semantic search functionality.
Provides endpoints for searching questions and getting dataset statistics.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from search.semantic_search import SemanticSearchEngine

# Initialize FastAPI app
app = FastAPI(
    title="Information Retrieval API",
    description="Semantic search API for educational questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize search engine
search_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup."""
    global search_engine
    try:
        index_path = str(Path(__file__).parent.parent / "indexes" / "faiss_index")
        search_engine = SemanticSearchEngine(index_path)
        print("✅ Search engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")

# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    target_marks: Optional[int] = Field(None, description="Target question marks (1-5)")
    target_difficulty: Optional[str] = Field(None, description="Target difficulty level")
    target_cognitive: Optional[str] = Field(None, description="Target cognitive level")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")

class SearchResult(BaseModel):
    question: str
    similarity_score: float
    topic: str
    subtopic: str
    marks: int
    difficulty_level: str
    cognitive_level: str
    question_type: str
    time: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    filters: Dict[str, Any]
    total_results: int
    results: List[SearchResult]
    execution_time_ms: float

class StatisticsResponse(BaseModel):
    total_documents: int
    topics: Dict[str, int]
    difficulty_levels: Dict[str, int]
    cognitive_levels: Dict[str, int]
    marks_distribution: Dict[str, int]
    question_types: Dict[str, int]

# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Information Retrieval API",
        "version": "1.0.0",
        "status": "running" if search_engine else "search_engine_not_loaded",
        "endpoints": {
            "search": "/search",
            "statistics": "/statistics",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if search_engine else "unhealthy",
        "search_engine_loaded": search_engine is not None
    }

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_questions(request: SearchRequest):
    """
    Search for questions using semantic similarity with optional filtering.
    
    - **query**: Natural language search query
    - **target_marks**: Filter by question marks (1-5)
    - **target_difficulty**: Filter by difficulty level (easy, medium, hard)
    - **target_cognitive**: Filter by cognitive level (remembering, understanding, etc.)
    - **max_results**: Maximum number of results to return
    - **min_similarity**: Minimum similarity score threshold
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not available")
    
    import time
    start_time = time.time()
    
    try:
        results = search_engine.search(
            query=request.query,
            target_marks=request.target_marks,
            target_difficulty=request.target_difficulty,
            target_cognitive=request.target_cognitive,
            max_results=request.max_results,
            min_similarity=request.min_similarity
        )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SearchResponse(
            query=request.query,
            filters={
                "target_marks": request.target_marks,
                "target_difficulty": request.target_difficulty,
                "target_cognitive": request.target_cognitive,
                "min_similarity": request.min_similarity
            },
            total_results=len(results),
            results=[SearchResult(**result) for result in results],
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_questions_get(
    q: str = Query(..., description="Search query"),
    marks: Optional[int] = Query(None, description="Target marks"),
    difficulty: Optional[str] = Query(None, description="Target difficulty"),
    cognitive: Optional[str] = Query(None, description="Target cognitive level"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum results"),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity")
):
    """
    Search for questions using GET method (alternative to POST).
    """
    request = SearchRequest(
        query=q,
        target_marks=marks,
        target_difficulty=difficulty,
        target_cognitive=cognitive,
        max_results=max_results,
        min_similarity=min_similarity
    )
    return await search_questions(request)

@app.get("/statistics", response_model=StatisticsResponse, tags=["Statistics"])
async def get_statistics():
    """Get dataset statistics and distributions."""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not available")
    
    try:
        stats = search_engine.get_statistics()
        return StatisticsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.get("/similar/{question_id}", tags=["Search"])
async def find_similar_questions(
    question_id: int,
    k: int = Query(5, ge=1, le=20, description="Number of similar questions")
):
    """Find questions similar to a specific question by ID."""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not available")
    
    # This would require implementing question ID mapping
    # For now, return a placeholder response
    raise HTTPException(status_code=501, detail="Similar questions by ID not implemented")

@app.get("/topics", tags=["Statistics"])
async def get_topics():
    """Get all available topics and subtopics."""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not available")
    
    try:
        stats = search_engine.get_statistics()
        return {
            "topics": list(stats["topics"].keys()),
            "topic_counts": stats["topics"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get topics: {str(e)}")

# Example usage and testing endpoints

@app.get("/examples", tags=["Examples"])
async def get_example_queries():
    """Get example queries for testing the API."""
    return {
        "example_queries": [
            {
                "query": "Natural language processing techniques for text classification",
                "description": "Search for NLP-related questions"
            },
            {
                "query": "Machine learning algorithms",
                "target_marks": 3,
                "target_difficulty": "medium",
                "description": "Search for medium-difficulty 3-mark ML questions"
            },
            {
                "query": "Computer vision applications",
                "target_cognitive": "applying",
                "description": "Search for application-level computer vision questions"
            }
        ],
        "curl_examples": [
            'curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d \'{"query": "machine learning", "max_results": 5}\'',
            'curl "http://localhost:8000/search?q=deep learning&marks=2&difficulty=easy"',
            'curl "http://localhost:8000/statistics"'
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)