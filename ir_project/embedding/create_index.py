"""
Information Retrieval System - Vector Store Creation
===================================================

This module creates and manages FAISS vector stores for question embeddings
using Ollama's nomic-embed-text model.

Features:
- Load question datasets from JSON
- Create document embeddings with metadata
- Store/load FAISS vector indexes
- Support for incremental index updates
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages creation and updates of FAISS vector stores for question embeddings."""
    
    def __init__(self, model_name: str = "nomic-embed-text:latest"):
        """
        Initialize the VectorStoreManager.
        
        Args:
            model_name: Name of the Ollama embedding model to use
        """
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.model_name = model_name
        
    def load_questions(self, file_path: str) -> List[Document]:
        """
        Load questions from JSON file and convert to Document objects.
        
        Args:
            file_path: Path to the questions JSON file
            
        Returns:
            List of Document objects with question text and metadata
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            docs = []
            total_questions = 0
            
            for mark_group, questions in data.items():
                logger.info(f"Processing {len(questions)} questions from {mark_group}")
                
                for q in questions:
                    doc = Document(
                        page_content=q["question"],
                        metadata={
                            "topic": q.get("topic", "unknown"),
                            "subtopic": q.get("subtopic", "unknown"),
                            "marks": q.get("marks", 0),
                            "question_type": q.get("question_type", "unknown"),
                            "difficulty_level": q.get("difficulty_level", "unknown"),
                            "cognitive_level": q.get("cognitive_level", "unknown"),
                            "time": q.get("time", "unknown"),
                            "image": q.get("image", None),
                            "mark_group": mark_group
                        }
                    )
                    docs.append(doc)
                    total_questions += 1
            
            logger.info(f"Successfully loaded {total_questions} questions")
            return docs
            
        except FileNotFoundError:
            logger.error(f"Questions file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in questions file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            raise
    
    def create_vector_store(self, docs: List[Document], index_path: str) -> FAISS:
        """
        Create a new FAISS vector store from documents.
        
        Args:
            docs: List of Document objects to embed
            index_path: Path where the FAISS index will be saved
            
        Returns:
            FAISS vector store instance
        """
        if os.path.exists(index_path):
            logger.warning(f"Index already exists at {index_path}. Use update_vector_store() to add documents.")
            return self.load_vector_store(index_path)
        
        logger.info(f"Creating FAISS index with {len(docs)} documents...")
        db = FAISS.from_documents(docs, self.embeddings)
        
        # Create directory if it doesn't exist
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        
        db.save_local(index_path)
        logger.info(f"✅ FAISS index saved to: {index_path}")
        
        # Save metadata
        self._save_metadata(index_path, len(docs))
        
        return db
    
    def load_vector_store(self, index_path: str) -> FAISS:
        """
        Load an existing FAISS vector store.
        
        Args:
            index_path: Path to the FAISS index
            
        Returns:
            FAISS vector store instance
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        
        logger.info(f"Loading FAISS index from: {index_path}")
        db = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        
        # Load metadata if available
        metadata = self._load_metadata(index_path)
        if metadata:
            logger.info(f"Index contains {metadata.get('document_count', 'unknown')} documents")
        
        return db
    
    def update_vector_store(self, index_path: str, new_docs: List[Document]) -> FAISS:
        """
        Add new documents to an existing vector store.
        
        Args:
            index_path: Path to the existing FAISS index
            new_docs: List of new Document objects to add
            
        Returns:
            Updated FAISS vector store instance
        """
        db = self.load_vector_store(index_path)
        
        logger.info(f"Adding {len(new_docs)} new documents to existing index...")
        db.add_documents(new_docs)
        
        db.save_local(index_path)
        logger.info(f"✅ Updated FAISS index saved to: {index_path}")
        
        # Update metadata
        metadata = self._load_metadata(index_path) or {}
        old_count = metadata.get('document_count', 0)
        self._save_metadata(index_path, old_count + len(new_docs))
        
        return db
    
    def _save_metadata(self, index_path: str, document_count: int):
        """Save index metadata."""
        metadata = {
            "document_count": document_count,
            "model_name": self.model_name,
            "created_at": str(Path(index_path).stat().st_ctime if os.path.exists(index_path) else "unknown")
        }
        
        metadata_path = Path(index_path) / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self, index_path: str) -> Optional[Dict[str, Any]]:
        """Load index metadata."""
        metadata_path = Path(index_path) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

def main():
    """Main function to create vector store from questions.json"""
    # Initialize the manager
    manager = VectorStoreManager()
    
    # Define paths relative to the project root
    project_root = Path(__file__).parent.parent
    questions_file = str(project_root / "data" / "questions.json")
    index_path = str(project_root / "indexes" / "faiss_index")
    
    try:
        # Load questions
        documents = manager.load_questions(questions_file)
        
        # Create vector store
        vector_store = manager.create_vector_store(documents, index_path)
        
        # Test the created index
        test_query = "Natural language processing techniques"
        results = vector_store.similarity_search(test_query, k=3)
        
        print(f"\n🔍 Test search for: '{test_query}'")
        print(f"Found {len(results)} similar questions:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc.page_content}")
            print(f"   Topic: {doc.metadata.get('topic')}")
            print(f"   Marks: {doc.metadata.get('marks')}")
            print(f"   Difficulty: {doc.metadata.get('difficulty_level')}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()