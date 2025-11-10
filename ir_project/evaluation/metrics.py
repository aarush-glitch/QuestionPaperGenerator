"""
Information Retrieval System - Evaluation Metrics
================================================

This module provides evaluation metrics for assessing the quality of
information retrieval results, including precision, recall, F1-score,
and other standard IR metrics.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IRMetrics:
    """Class for computing standard Information Retrieval evaluation metrics."""
    
    @staticmethod
    def precision_at_k(relevant_items: List[str], retrieved_items: List[str], k: int) -> float:
        """
        Calculate Precision@K metric.
        
        Args:
            relevant_items: List of relevant item IDs
            retrieved_items: List of retrieved item IDs (ordered by relevance)
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k <= 0 or not retrieved_items:
            return 0.0
        
        retrieved_at_k = retrieved_items[:k]
        relevant_set = set(relevant_items)
        relevant_retrieved = sum(1 for item in retrieved_at_k if item in relevant_set)
        
        return relevant_retrieved / min(k, len(retrieved_at_k))
    
    @staticmethod
    def recall_at_k(relevant_items: List[str], retrieved_items: List[str], k: int) -> float:
        """
        Calculate Recall@K metric.
        
        Args:
            relevant_items: List of relevant item IDs
            retrieved_items: List of retrieved item IDs (ordered by relevance)
            k: Number of top results to consider
            
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_items or k <= 0:
            return 0.0
        
        retrieved_at_k = retrieved_items[:k]
        relevant_set = set(relevant_items)
        relevant_retrieved = sum(1 for item in retrieved_at_k if item in relevant_set)
        
        return relevant_retrieved / len(relevant_items)
    
    @staticmethod
    def f1_at_k(relevant_items: List[str], retrieved_items: List[str], k: int) -> float:
        """
        Calculate F1@K metric.
        
        Args:
            relevant_items: List of relevant item IDs
            retrieved_items: List of retrieved item IDs (ordered by relevance)
            k: Number of top results to consider
            
        Returns:
            F1@K score (0.0 to 1.0)
        """
        precision = IRMetrics.precision_at_k(relevant_items, retrieved_items, k)
        recall = IRMetrics.recall_at_k(relevant_items, retrieved_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def average_precision(relevant_items: List[str], retrieved_items: List[str]) -> float:
        """
        Calculate Average Precision (AP) metric.
        
        Args:
            relevant_items: List of relevant item IDs
            retrieved_items: List of retrieved item IDs (ordered by relevance)
            
        Returns:
            Average Precision score (0.0 to 1.0)
        """
        if not relevant_items:
            return 0.0
        
        relevant_set = set(relevant_items)
        precision_sum = 0.0
        relevant_found = 0
        
        for i, item in enumerate(retrieved_items, 1):
            if item in relevant_set:
                relevant_found += 1
                precision_at_i = relevant_found / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items) if relevant_items else 0.0
    
    @staticmethod
    def mean_average_precision(queries_results: List[Tuple[List[str], List[str]]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries.
        
        Args:
            queries_results: List of (relevant_items, retrieved_items) tuples
            
        Returns:
            MAP score (0.0 to 1.0)
        """
        if not queries_results:
            return 0.0
        
        ap_scores = [
            IRMetrics.average_precision(relevant, retrieved)
            for relevant, retrieved in queries_results
        ]
        
        return np.mean(ap_scores)
    
    @staticmethod
    def ndcg_at_k(relevant_items: List[str], retrieved_items: List[str], k: int, 
                  relevance_scores: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            relevant_items: List of relevant item IDs
            retrieved_items: List of retrieved item IDs (ordered by relevance)
            k: Number of top results to consider
            relevance_scores: Optional dict mapping item IDs to relevance scores (0-3)
            
        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if k <= 0 or not retrieved_items:
            return 0.0
        
        # Default relevance scores (binary: 1 if relevant, 0 if not)
        if relevance_scores is None:
            relevant_set = set(relevant_items)
            relevance_scores = {item: 1.0 if item in relevant_set else 0.0 
                              for item in retrieved_items}
        
        # Calculate DCG@K
        dcg = 0.0
        for i, item in enumerate(retrieved_items[:k], 1):
            rel_score = relevance_scores.get(item, 0.0)
            dcg += rel_score / np.log2(i + 1)
        
        # Calculate IDCG@K (Ideal DCG)
        ideal_scores = sorted([relevance_scores.get(item, 0.0) for item in relevant_items], 
                             reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores[:k], 1):
            idcg += score / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0

class SearchEvaluator:
    """Evaluator for semantic search results with domain-specific metrics."""
    
    def __init__(self):
        self.metrics = IRMetrics()
    
    def evaluate_search_results(self, query: str, results: List[Dict[str, Any]], 
                              ground_truth: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate search results for a single query.
        
        Args:
            query: The search query
            results: List of search result dictionaries
            ground_truth: Optional list of relevant question IDs
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not results:
            return {
                "precision_at_1": 0.0,
                "precision_at_3": 0.0,
                "precision_at_5": 0.0,
                "recall_at_5": 0.0,
                "f1_at_5": 0.0,
                "average_precision": 0.0,
                "ndcg_at_5": 0.0
            }
        
        # Extract question IDs/text from results
        retrieved_items = [result['question'] for result in results]
        
        # If no ground truth provided, use heuristic relevance
        if ground_truth is None:
            ground_truth = self._generate_heuristic_relevance(query, results)
        
        # Calculate metrics
        metrics = {}
        for k in [1, 3, 5]:
            metrics[f"precision_at_{k}"] = self.metrics.precision_at_k(ground_truth, retrieved_items, k)
            if k == 5:
                metrics[f"recall_at_{k}"] = self.metrics.recall_at_k(ground_truth, retrieved_items, k)
                metrics[f"f1_at_{k}"] = self.metrics.f1_at_k(ground_truth, retrieved_items, k)
                metrics[f"ndcg_at_{k}"] = self.metrics.ndcg_at_k(ground_truth, retrieved_items, k)
        
        metrics["average_precision"] = self.metrics.average_precision(ground_truth, retrieved_items)
        
        return metrics
    
    def _generate_heuristic_relevance(self, query: str, results: List[Dict[str, Any]]) -> List[str]:
        """
        Generate heuristic relevance judgments based on similarity scores and metadata.
        
        This is a simplified approach for demonstration. In practice, you would have
        human-annotated relevance judgments or a more sophisticated relevance function.
        """
        # Consider top results with high similarity scores as relevant
        relevant_threshold = 0.7  # Adjust based on your similarity score distribution
        
        relevant_items = []
        for result in results:
            similarity_score = result.get('similarity_score', 0.0)
            if similarity_score >= relevant_threshold:
                relevant_items.append(result['question'])
        
        # If no high-similarity results, consider top 3 as relevant
        if not relevant_items and results:
            relevant_items = [result['question'] for result in results[:3]]
        
        return relevant_items
    
    def evaluate_filtering_accuracy(self, results: List[Dict[str, Any]], 
                                  target_marks: Optional[int] = None,
                                  target_difficulty: Optional[str] = None,
                                  target_cognitive: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate the accuracy of metadata filtering.
        
        Args:
            results: List of search result dictionaries
            target_marks: Expected marks value
            target_difficulty: Expected difficulty level
            target_cognitive: Expected cognitive level
            
        Returns:
            Dictionary of filtering accuracy metrics
        """
        if not results:
            return {"filter_accuracy": 0.0, "marks_accuracy": 0.0, 
                   "difficulty_accuracy": 0.0, "cognitive_accuracy": 0.0}
        
        total_results = len(results)
        marks_correct = 0
        difficulty_correct = 0
        cognitive_correct = 0
        all_correct = 0
        
        for result in results:
            marks_match = (target_marks is None or 
                          result.get('marks') == target_marks)
            difficulty_match = (target_difficulty is None or 
                               result.get('difficulty_level', '').lower() == target_difficulty.lower())
            cognitive_match = (target_cognitive is None or 
                              result.get('cognitive_level', '').lower() == target_cognitive.lower())
            
            if marks_match:
                marks_correct += 1
            if difficulty_match:
                difficulty_correct += 1
            if cognitive_match:
                cognitive_correct += 1
            if marks_match and difficulty_match and cognitive_match:
                all_correct += 1
        
        return {
            "filter_accuracy": all_correct / total_results,
            "marks_accuracy": marks_correct / total_results if target_marks else 1.0,
            "difficulty_accuracy": difficulty_correct / total_results if target_difficulty else 1.0,
            "cognitive_accuracy": cognitive_correct / total_results if target_cognitive else 1.0
        }

def evaluate_search_results(query: str, results: List[Dict[str, Any]], 
                          ground_truth: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate search results.
    
    Args:
        query: The search query
        results: List of search result dictionaries
        ground_truth: Optional list of relevant question IDs
        
    Returns:
        Dictionary containing evaluation metrics and analysis
    """
    evaluator = SearchEvaluator()
    
    # Basic retrieval metrics
    retrieval_metrics = evaluator.evaluate_search_results(query, results, ground_truth)
    
    # Result analysis
    analysis = {
        "total_results": len(results),
        "topics_covered": len(set(r.get('topic', 'unknown') for r in results)),
        "difficulty_distribution": defaultdict(int),
        "cognitive_distribution": defaultdict(int),
        "marks_distribution": defaultdict(int),
        "avg_similarity_score": np.mean([r.get('similarity_score', 0.0) for r in results]) if results else 0.0,
        "similarity_score_std": np.std([r.get('similarity_score', 0.0) for r in results]) if results else 0.0
    }
    
    # Calculate distributions
    for result in results:
        analysis["difficulty_distribution"][result.get('difficulty_level', 'unknown')] += 1
        analysis["cognitive_distribution"][result.get('cognitive_level', 'unknown')] += 1
        analysis["marks_distribution"][str(result.get('marks', 0))] += 1
    
    return {
        "query": query,
        "metrics": retrieval_metrics,
        "analysis": analysis
    }

def main():
    """Example usage of the evaluation metrics."""
    # Example data for testing
    sample_results = [
        {
            "question": "What is machine learning?",
            "similarity_score": 0.95,
            "topic": "AI",
            "marks": 2,
            "difficulty_level": "easy",
            "cognitive_level": "remembering"
        },
        {
            "question": "Explain neural networks",
            "similarity_score": 0.87,
            "topic": "AI",
            "marks": 3,
            "difficulty_level": "medium",
            "cognitive_level": "understanding"
        },
        {
            "question": "Implement a CNN",
            "similarity_score": 0.75,
            "topic": "AI",
            "marks": 5,
            "difficulty_level": "hard",
            "cognitive_level": "applying"
        }
    ]
    
    # Evaluate the results
    evaluation = evaluate_search_results(
        query="machine learning concepts",
        results=sample_results
    )
    
    print("Evaluation Results:")
    print(json.dumps(evaluation, indent=2, default=str))

if __name__ == "__main__":
    main()