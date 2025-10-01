"""
Model evaluation module for misinformation detection research.
Implements comprehensive evaluation procedures and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from .config import FIGURE_SIZE, CONFUSION_MATRIX_CMAP

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles comprehensive evaluation of trained models."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        
    def evaluate_model(self, model: Any, model_name: str, 
                      X_test: Any, y_test: Any) -> Dict[str, Any]:
        """
        Evaluate a single model and return comprehensive metrics.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict: Evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"Evaluation completed for {model_name}:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return results
    
    def print_model_metrics(self, model_name: str) -> None:
        """
        Print detailed metrics for a specific model.
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {model_name}")
        
        results = self.evaluation_results[model_name]
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS: {model_name.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        
        print(f"\nClassification Report:")
        print("-" * 40)
        # Convert classification report to readable format
        report = results['classification_report']
        for class_label in ['0', '1']:
            if class_label in report:
                print(f"Class {class_label}:")
                print(f"  Precision: {report[class_label]['precision']:.4f}")
                print(f"  Recall:    {report[class_label]['recall']:.4f}")
                print(f"  F1-Score:  {report[class_label]['f1-score']:.4f}")
        
        if 'macro avg' in report:
            print(f"Macro Average:")
            print(f"  Precision: {report['macro avg']['precision']:.4f}")
            print(f"  Recall:    {report['macro avg']['recall']:.4f}")
            print(f"  F1-Score:  {report['macro avg']['f1-score']:.4f}")
    
    def plot_confusion_matrix(self, model_name: str, save_path: str = None) -> None:
        """
        Plot confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for {model_name}")
        
        conf_matrix = self.evaluation_results[model_name]['confusion_matrix']
        
        plt.figure(figsize=FIGURE_SIZE)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=CONFUSION_MATRIX_CMAP,
                   xticklabels=['Reliable', 'Unreliable'],
                   yticklabels=['Reliable', 'Unreliable'])
        plt.title(f'Confusion Matrix: {model_name.replace("_", " ").title()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def evaluate_all_models(self, trained_models: Dict[str, Any], 
                           X_test: Any, y_test: Any) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models.
        
        Args:
            trained_models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict: Complete evaluation results for all models
        """
        logger.info("Starting evaluation for all models")
        
        for model_name, model in trained_models.items():
            try:
                self.evaluate_model(model, model_name, X_test, y_test)
            except Exception as e:
                logger.error(f"Evaluation failed for {model_name}: {str(e)}")
                continue
        
        logger.info(f"Evaluation completed for {len(self.evaluation_results)} models")
        return self.evaluation_results
    
    def get_evaluation_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all evaluation results.
        
        Returns:
            Dict: Complete evaluation results
        """
        return self.evaluation_results
