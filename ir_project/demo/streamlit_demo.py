"""
Information Retrieval System - Streamlit Demo
============================================

Interactive web interface for testing the semantic search functionality.
This demo allows users to search questions using natural language queries
with optional filtering by marks, difficulty, and cognitive level.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from search.semantic_search import SemanticSearchEngine
from evaluation.metrics import evaluate_search_results

# Page configuration
st.set_page_config(
    page_title="Information Retrieval Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_search_engine():
    """Load the search engine (cached for performance)."""
    try:
        index_path = str(Path(__file__).parent.parent / "indexes" / "faiss_index")
        return SemanticSearchEngine(index_path)
    except Exception as e:
        st.error(f"Failed to load search engine: {e}")
        return None

def main():
    st.title("🔍 Information Retrieval System Demo")
    st.markdown("---")
    
    # Load search engine
    search_engine = load_search_engine()
    if not search_engine:
        st.error("❌ Search engine not available. Please ensure the FAISS index is created.")
        st.info("Run `python embedding/create_index.py` to create the index first.")
        return
    
    # Sidebar for filters
    st.sidebar.header("🎛️ Search Filters")
    
    # Query input
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., Natural language processing techniques for text classification",
        help="Enter a natural language query to search for similar questions"
    )
    
    # Filter options
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        marks_options = ["Any", "1", "2", "3", "4", "5"]
        selected_marks = st.selectbox("📊 Marks:", marks_options)
        target_marks = None if selected_marks == "Any" else int(selected_marks)
        
        difficulty_options = ["Any", "easy", "medium", "hard"]
        selected_difficulty = st.selectbox("📈 Difficulty:", difficulty_options)
        target_difficulty = None if selected_difficulty == "Any" else selected_difficulty
    
    with col2:
        cognitive_options = ["Any", "remembering", "understanding", "applying", "analyzing", "evaluating", "creating"]
        selected_cognitive = st.selectbox("🧠 Cognitive Level:", cognitive_options)
        target_cognitive = None if selected_cognitive == "Any" else selected_cognitive
        
        max_results = st.slider("📋 Max Results:", 1, 20, 10)
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        min_similarity = st.slider("Minimum Similarity Score:", 0.0, 1.0, 0.0, 0.1)
        show_metadata = st.checkbox("Show Full Metadata", False)
        show_scores = st.checkbox("Show Similarity Scores", True)
    
    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True) or query:
        if not query.strip():
            st.warning("Please enter a search query.")
            return
        
        # Perform search
        with st.spinner("Searching..."):
            results = search_engine.search(
                query=query,
                target_marks=target_marks,
                target_difficulty=target_difficulty,
                target_cognitive=target_cognitive,
                max_results=max_results,
                min_similarity=min_similarity
            )
        
        # Display results
        if not results:
            st.warning("No results found. Try adjusting your query or filters.")
            return
        
        st.success(f"Found {len(results)} results")
        
        # Results display
        for i, result in enumerate(results, 1):
            with st.expander(f"Result {i}: {result['question'][:100]}...", expanded=i <= 3):
                # Question text
                st.markdown(f"**Question:** {result['question']}")
                
                # Basic metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Marks", result['marks'])
                with col2:
                    st.metric("Difficulty", result['difficulty_level'].title())
                with col3:
                    st.metric("Cognitive Level", result['cognitive_level'].title())
                with col4:
                    if show_scores:
                        st.metric("Similarity", f"{result['similarity_score']:.3f}")
                
                # Topic information
                st.markdown(f"**Topic:** {result['topic']} → {result['subtopic']}")
                st.markdown(f"**Type:** {result['question_type']} | **Time:** {result['time']}")
                
                # Full metadata if requested
                if show_metadata:
                    st.json(result['metadata'])
    
    # Statistics section
    st.markdown("---")
    st.header("📊 Dataset Statistics")
    
    if st.button("Show Statistics"):
        with st.spinner("Calculating statistics..."):
            stats = search_engine.get_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("General Stats")
            st.metric("Total Questions", stats['total_documents'])
            st.metric("Topics", len(stats['topics']))
            
        with col2:
            st.subheader("Marks Distribution")
            for marks, count in stats['marks_distribution'].items():
                st.metric(f"{marks} Mark(s)", count)
        
        with col3:
            st.subheader("Difficulty Levels")
            for diff, count in stats['difficulty_levels'].items():
                st.metric(diff.title(), count)
        
        # Detailed breakdowns
        st.subheader("Detailed Breakdowns")
        
        tab1, tab2, tab3 = st.tabs(["Topics", "Cognitive Levels", "Question Types"])
        
        with tab1:
            st.bar_chart(stats['topics'])
        
        with tab2:
            st.bar_chart(stats['cognitive_levels'])
        
        with tab3:
            st.bar_chart(stats['question_types'])
    
    # Sample queries section
    st.markdown("---")
    st.header("💡 Sample Queries")
    
    sample_queries = [
        "Natural language processing techniques for text classification",
        "Machine learning algorithms for pattern recognition",
        "Computer vision applications in real-world scenarios",
        "Deep learning neural network architectures",
        "Artificial intelligence ethics and bias detection"
    ]
    
    cols = st.columns(len(sample_queries))
    for i, sample_query in enumerate(sample_queries):
        with cols[i]:
            if st.button(f"Try: {sample_query[:30]}...", key=f"sample_{i}"):
                st.experimental_set_query_params(query=sample_query)
                st.experimental_rerun()

if __name__ == "__main__":
    main()