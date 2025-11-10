"""
Information Retrieval System - Semantic Search Module
=====================================================

This module provides semantic search capabilities for questions using FAISS
vector similarity search with intelligent filtering and ranking.

Features:
- Semantic similarity search using vector embeddings
- Multi-criteria filtering (marks, difficulty, cognitive level)
- Hierarchical filter relaxation for optimal results
- Configurable result ranking and scoring
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Advanced semantic search engine for question retrieval with intelligent filtering."""
    
    def __init__(self, index_path: str, model_name: str = "nomic-embed-text:latest"):
        """
        Initialize the SemanticSearchEngine.
        
        Args:
            index_path: Path to the FAISS vector store
            model_name: Name of the Ollama embedding model
        """
        self.index_path = index_path
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = None
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load the FAISS vector store."""
        try:
            if not Path(self.index_path).exists():
                raise FileNotFoundError(f"FAISS index not found at: {self.index_path}")
            
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            logger.info(f"✅ Loaded FAISS index from: {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def search(self, 
               query: str, 
               target_marks: Optional[int] = None,
               target_difficulty: Optional[str] = None,
               target_cognitive: Optional[str] = None,
               max_results: int = 20,
               min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional filtering.
        
        Args:
            query: Natural language search query
            target_marks: Target question marks (1, 2, 3, etc.)
            target_difficulty: Target difficulty level ('easy', 'medium', 'hard')
            target_cognitive: Target cognitive level ('remembering', 'understanding', etc.)
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            List of search results with metadata and similarity scores
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not loaded")
        
        logger.info(f"Searching for: '{query}'")
        logger.info(f"Filters - Marks: {target_marks}, Difficulty: {target_difficulty}, Cognitive: {target_cognitive}")
        
        # Get initial similarity search results
        similar_docs = self.vector_store.similarity_search_with_score(query, k=max_results * 2)
        
        # Filter by minimum similarity if specified
        if min_similarity > 0:
            similar_docs = [(doc, score) for doc, score in similar_docs if score >= min_similarity]
        
        # Apply intelligent filtering
        filtered_docs = self._smart_filter(
            similar_docs, target_marks, target_difficulty, target_cognitive
        )
        
        # Convert to result format
        results = []
        for doc, score in filtered_docs[:max_results]:
            result = {
                'question': doc.page_content,
                'similarity_score': float(score),
                'metadata': dict(doc.metadata),
                'topic': doc.metadata.get('topic', 'unknown'),
                'subtopic': doc.metadata.get('subtopic', 'unknown'),
                'marks': doc.metadata.get('marks', 0),
                'difficulty_level': doc.metadata.get('difficulty_level', 'unknown'),
                'cognitive_level': doc.metadata.get('cognitive_level', 'unknown'),
                'question_type': doc.metadata.get('question_type', 'unknown'),
                'time': doc.metadata.get('time', 'unknown')
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} results after filtering")
        return results
    
    def _smart_filter(self, 
                     docs_with_scores: List[Tuple[Document, float]], 
                     marks: Optional[int], 
                     difficulty: Optional[str], 
                     cognitive: Optional[str]) -> List[Tuple[Document, float]]:
        """
        Apply intelligent hierarchical filtering with fallback strategies.
        
        The filtering strategy follows this hierarchy:
        1. Exact match on all criteria
        2. Relax difficulty, keep marks and cognitive level
        3. Relax cognitive level, keep marks and difficulty
        4. Keep only marks criteria
        5. Return top similarity results if no criteria match
        """
        if not any([marks, difficulty, cognitive]):
            return docs_with_scores
        
        # Convert to lowercase for case-insensitive comparison
        target_difficulty = difficulty.lower() if difficulty else None
        target_cognitive = cognitive.lower() if cognitive else None
        
        # Strategy 1: Exact match on all specified criteria
        exact_matches = []
        for doc, score in docs_with_scores:
            metadata = doc.metadata
            
            marks_match = marks is None or str(metadata.get("marks")) == str(marks)
            difficulty_match = (target_difficulty is None or 
                              metadata.get("difficulty_level", "").lower() == target_difficulty)
            cognitive_match = (target_cognitive is None or 
                             metadata.get("cognitive_level", "").lower() == target_cognitive)
            
            if marks_match and difficulty_match and cognitive_match:
                exact_matches.append((doc, score))
        
        if exact_matches:
            logger.info(f"Found {len(exact_matches)} exact matches")
            return exact_matches
        
        # Strategy 2: Relax difficulty, keep marks and cognitive
        if marks is not None and target_cognitive is not None:
            relaxed_diff = []
            for doc, score in docs_with_scores:
                metadata = doc.metadata
                if (str(metadata.get("marks")) == str(marks) and 
                    metadata.get("cognitive_level", "").lower() == target_cognitive):
                    relaxed_diff.append((doc, score))
            
            if relaxed_diff:
                logger.info(f"Found {len(relaxed_diff)} matches (relaxed difficulty)")
                return relaxed_diff
        
        # Strategy 3: Relax cognitive, keep marks and difficulty
        if marks is not None and target_difficulty is not None:
            relaxed_cog = []
            for doc, score in docs_with_scores:
                metadata = doc.metadata
                if (str(metadata.get("marks")) == str(marks) and 
                    metadata.get("difficulty_level", "").lower() == target_difficulty):
                    relaxed_cog.append((doc, score))
            
            if relaxed_cog:
                logger.info(f"Found {len(relaxed_cog)} matches (relaxed cognitive level)")
                return relaxed_cog
        
        # Strategy 4: Keep only marks criteria
        if marks is not None:
            marks_only = []
            for doc, score in docs_with_scores:
                if str(doc.metadata.get("marks")) == str(marks):
                    marks_only.append((doc, score))
            
            if marks_only:
                logger.info(f"Found {len(marks_only)} matches (marks only)")
                return marks_only
        
        # Strategy 5: Fallback to similarity-only results
        logger.info(f"No filter matches found, returning top {min(3, len(docs_with_scores))} similarity results")
        return docs_with_scores[:3]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.vector_store:
            return {"error": "Vector store not loaded"}
        
        # Get all documents to analyze
        sample_search = self.vector_store.similarity_search("sample", k=1000)
        
        stats = {
            "total_documents": len(sample_search),
            "topics": {},
            "difficulty_levels": {},
            "cognitive_levels": {},
            "marks_distribution": {},
            "question_types": {}
        }
        
        for doc in sample_search:
            metadata = doc.metadata
            
            # Count topics
            topic = metadata.get("topic", "unknown")
            stats["topics"][topic] = stats["topics"].get(topic, 0) + 1
            
            # Count difficulty levels
            difficulty = metadata.get("difficulty_level", "unknown")
            stats["difficulty_levels"][difficulty] = stats["difficulty_levels"].get(difficulty, 0) + 1
            
            # Count cognitive levels
            cognitive = metadata.get("cognitive_level", "unknown")
            stats["cognitive_levels"][cognitive] = stats["cognitive_levels"].get(cognitive, 0) + 1
            
            # Count marks distribution
            marks = metadata.get("marks", "unknown")
            stats["marks_distribution"][str(marks)] = stats["marks_distribution"].get(str(marks), 0) + 1
            
            # Count question types
            q_type = metadata.get("question_type", "unknown")
            stats["question_types"][q_type] = stats["question_types"].get(q_type, 0) + 1
        
        return stats
    
    def find_similar_questions(self, question_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find questions similar to a given question text."""
        return self.search(question_text, max_results=k)

def main():
    """Example usage of the SemanticSearchEngine."""
    try:
        # Initialize search engine
        project_root = Path(__file__).parent.parent
        index_path = str(project_root / "indexes" / "faiss_index")
        search_engine = SemanticSearchEngine(index_path)
        
        # Example searches
        test_queries = [
            {
                "query": "Natural language processing techniques for text classification",
                "target_marks": 3,
                "target_difficulty": "medium",
                "target_cognitive": "applying"
            },
            {
                "query": "Machine learning algorithms",
                "target_marks": 2,
                "target_difficulty": "easy"
            },
            {
                "query": "Computer vision applications",
                "target_marks": 1
            }
        ]
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"TEST SEARCH #{i}")
            print(f"{'='*60}")
            
            results = search_engine.search(**test)
            
            print(f"Query: {test['query']}")
            print(f"Filters: Marks={test.get('target_marks')}, "
                  f"Difficulty={test.get('target_difficulty')}, "
                  f"Cognitive={test.get('target_cognitive')}")
            print(f"Found {len(results)} results:\n")
            
            for j, result in enumerate(results[:3], 1):
                print(f"{j}. Question: {result['question']}")
                print(f"   Topic: {result['topic']} → {result['subtopic']}")
                print(f"   Marks: {result['marks']}, Difficulty: {result['difficulty_level']}")
                print(f"   Cognitive: {result['cognitive_level']}")
                print(f"   Similarity: {result['similarity_score']:.4f}")
                print()
        
        # Show statistics
        print(f"\n{'='*60}")
        print("VECTOR STORE STATISTICS")
        print(f"{'='*60}")
        
        stats = search_engine.get_statistics()
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Topics: {len(stats['topics'])}")
        print(f"Difficulty Levels: {list(stats['difficulty_levels'].keys())}")
        print(f"Cognitive Levels: {list(stats['cognitive_levels'].keys())}")
        print(f"Marks Distribution: {stats['marks_distribution']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()