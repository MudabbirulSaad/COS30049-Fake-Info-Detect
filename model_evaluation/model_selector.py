"""
Model selection module for misinformation detection research.
Implements model comparison and selection procedures.
"""

import pandas as pd
import joblib
import json
import logging
from typing import Dict, Any, Tuple
from pathlib import Path
from .config import (
    MODEL_OUTPUT_DIR, BEST_MODEL_FILE, VECTORIZER_FILE, 
    RESULTS_FILE, EVALUATION_METRICS
)

logger = logging.getLogger(__name__)


class ModelSelector:
    """Handles model comparison and selection procedures."""
    
    def __init__(self):
        """Initialize the model selector."""
        self.comparison_results = None
        self.best_model_name = None
        self.best_model = None
        
    def create_comparison_table(self, evaluation_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a comparison table of model performance metrics.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        logger.info("Creating model comparison table")
        
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            }
            comparison_data.append(row)
        
        self.comparison_results = pd.DataFrame(comparison_data)
        
        # Sort by F1-Score in descending order
        self.comparison_results = self.comparison_results.sort_values(
            'F1-Score', ascending=False
        ).reset_index(drop=True)
        
        logger.info("Model comparison table created")
        return self.comparison_results
    
    def print_comparison_table(self) -> None:
        """Print the model comparison table."""
        if self.comparison_results is None:
            raise ValueError("Comparison table not created. Run create_comparison_table first.")
        
        print(f"\n{'='*80}")
        print("MODEL PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        # Format the table for better display
        formatted_df = self.comparison_results.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
        
        print(formatted_df.to_string(index=False))
        print(f"{'='*80}")
    
    def select_best_model(self, trained_models: Dict[str, Any], 
                         evaluation_results: Dict[str, Dict[str, Any]],
                         selection_metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Select the best performing model based on specified metric.
        
        Args:
            trained_models: Dictionary of trained models
            evaluation_results: Dictionary of evaluation results
            selection_metric: Metric to use for selection
            
        Returns:
            Tuple: (best_model_name, best_model_object)
        """
        if selection_metric not in EVALUATION_METRICS:
            raise ValueError(f"Invalid selection metric: {selection_metric}")
        
        logger.info(f"Selecting best model based on {selection_metric}")
        
        best_score = -1
        best_model_name = None
        
        for model_name, results in evaluation_results.items():
            score = results[selection_metric]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError("No valid model found for selection")
        
        self.best_model_name = best_model_name
        self.best_model = trained_models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best {selection_metric}: {best_score:.4f}")
        
        return self.best_model_name, self.best_model
    
    def print_best_model_recommendation(self) -> None:
        """Print recommendation for the best model."""
        if self.best_model_name is None:
            raise ValueError("Best model not selected. Run select_best_model first.")
        
        print(f"\n{'='*80}")
        print("MODEL SELECTION RECOMMENDATION")
        print(f"{'='*80}")
        print(f"Recommended Model: {self.best_model_name.replace('_', ' ').title()}")
        print(f"Selection Criteria: Highest F1-Score")
        
        if self.comparison_results is not None:
            best_row = self.comparison_results.iloc[0]
            print(f"Performance Metrics:")
            print(f"  Accuracy:  {best_row['Accuracy']:.4f}")
            print(f"  Precision: {best_row['Precision']:.4f}")
            print(f"  Recall:    {best_row['Recall']:.4f}")
            print(f"  F1-Score:  {best_row['F1-Score']:.4f}")
        
        print(f"{'='*80}")
    
    def save_best_model(self, vectorizer: Any) -> None:
        """
        Save the best model and vectorizer to disk.
        
        Args:
            vectorizer: Fitted TF-IDF vectorizer
        """
        if self.best_model is None:
            raise ValueError("Best model not selected. Run select_best_model first.")
        
        # Create output directory if it doesn't exist
        MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Save best model
        model_path = MODEL_OUTPUT_DIR / BEST_MODEL_FILE
        joblib.dump(self.best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        # Save vectorizer
        vectorizer_path = MODEL_OUTPUT_DIR / VECTORIZER_FILE
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"TF-IDF vectorizer saved to {vectorizer_path}")
        
        print(f"\nModel artifacts saved:")
        print(f"  Best model: {model_path}")
        print(f"  Vectorizer: {vectorizer_path}")
    
    def save_evaluation_results(self, evaluation_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            evaluation_results: Dictionary of evaluation results
        """
        MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for model_name, results in evaluation_results.items():
            serializable_results[model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
        
        results_path = MODEL_OUTPUT_DIR / RESULTS_FILE
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
    
    def get_best_model_info(self) -> Dict[str, Any]:
        """
        Get information about the selected best model.
        
        Returns:
            Dict: Best model information
        """
        if self.best_model_name is None:
            raise ValueError("Best model not selected. Run select_best_model first.")
        
        return {
            'model_name': self.best_model_name,
            'model_object': self.best_model,
            'comparison_results': self.comparison_results
        }
