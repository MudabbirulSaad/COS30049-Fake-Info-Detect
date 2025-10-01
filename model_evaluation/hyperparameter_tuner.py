"""
Hyperparameter tuning module for Random Forest optimization.
Implements systematic hyperparameter optimization using GridSearchCV.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from .config import RANDOM_STATE, FIGURE_SIZE, CONFUSION_MATRIX_CMAP

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Implements hyperparameter tuning for Random Forest classifier."""
    
    def __init__(self):
        """Initialize the hyperparameter tuner."""
        self.param_grid = None
        self.grid_search = None
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.tuning_results = None
        
    def define_parameter_grid(self) -> Dict[str, list]:
        """
        Define the hyperparameter grid for Random Forest tuning.
        
        Returns:
            Dict: Parameter grid for GridSearchCV
        """
        logger.info("Defining hyperparameter grid for Random Forest")
        
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        total_combinations = 1
        for param, values in self.param_grid.items():
            total_combinations *= len(values)
        
        logger.info(f"Parameter grid defined with {total_combinations} combinations:")
        for param, values in self.param_grid.items():
            logger.info(f"  {param}: {values}")
        
        return self.param_grid
    
    def run_grid_search(self, X_train: Any, y_train: Any, 
                       cv_folds: int = 5, scoring: str = 'f1_macro',
                       n_jobs: int = -1) -> GridSearchCV:
        """
        Execute grid search for hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
            
        Returns:
            GridSearchCV: Fitted grid search object
        """
        if self.param_grid is None:
            self.define_parameter_grid()
        
        logger.info("Starting hyperparameter tuning with GridSearchCV")
        logger.info(f"Cross-validation folds: {cv_folds}")
        logger.info(f"Scoring metric: {scoring}")
        logger.info(f"Parallel jobs: {n_jobs}")
        
        # Initialize base Random Forest classifier
        rf_classifier = RandomForestClassifier(random_state=RANDOM_STATE)
        
        # Initialize GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=rf_classifier,
            param_grid=self.param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        # Record start time
        start_time = time.time()
        
        # Fit grid search
        logger.info("Executing grid search (this may take several minutes)")
        self.grid_search.fit(X_train, y_train)
        
        # Record end time
        end_time = time.time()
        tuning_duration = end_time - start_time
        
        # Extract results
        self.best_model = self.grid_search.best_estimator_
        self.best_params = self.grid_search.best_params_
        self.best_score = self.grid_search.best_score_
        
        logger.info(f"Grid search completed in {tuning_duration:.2f} seconds")
        logger.info(f"Best cross-validation score: {self.best_score:.4f}")
        
        return self.grid_search
    
    def print_tuning_results(self) -> None:
        """Print detailed results from hyperparameter tuning."""
        if self.grid_search is None:
            raise ValueError("Grid search not completed. Run run_grid_search first.")
        
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING RESULTS")
        print(f"{'='*80}")
        
        print(f"Best Cross-Validation F1-Score: {self.best_score:.4f}")
        print(f"\nBest Hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\nTop 5 Parameter Combinations:")
        results_df = pd.DataFrame(self.grid_search.cv_results_)
        top_results = results_df.nlargest(5, 'mean_test_score')[
            ['mean_test_score', 'std_test_score', 'params']
        ]
        
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"  {idx}. Score: {row['mean_test_score']:.4f} "
                  f"(±{row['std_test_score']:.4f}) - {row['params']}")
        
        print(f"{'='*80}")
    
    def evaluate_tuned_model(self, X_test: Any, y_test: Any) -> Dict[str, Any]:
        """
        Evaluate the tuned model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict: Evaluation results
        """
        if self.best_model is None:
            raise ValueError("Best model not available. Run run_grid_search first.")
        
        logger.info("Evaluating tuned Random Forest model")
        
        # Generate predictions
        y_pred = self.best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        self.tuning_results = {
            'model_name': 'tuned_random_forest',
            'best_params': self.best_params,
            'best_cv_score': self.best_score,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
        
        logger.info(f"Tuned model evaluation completed:")
        logger.info(f"  Test Accuracy: {accuracy:.4f}")
        logger.info(f"  Test Precision: {precision:.4f}")
        logger.info(f"  Test Recall: {recall:.4f}")
        logger.info(f"  Test F1-Score: {f1:.4f}")
        
        return self.tuning_results
    
    def print_evaluation_results(self) -> None:
        """Print detailed evaluation results for the tuned model."""
        if self.tuning_results is None:
            raise ValueError("Tuned model not evaluated. Run evaluate_tuned_model first.")
        
        results = self.tuning_results
        
        print(f"\n{'='*80}")
        print("TUNED MODEL EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Test Accuracy:  {results['test_accuracy']:.4f}")
        print(f"Test Precision: {results['test_precision']:.4f}")
        print(f"Test Recall:    {results['test_recall']:.4f}")
        print(f"Test F1-Score:  {results['test_f1_score']:.4f}")
        
        print(f"\nClassification Report:")
        print("-" * 40)
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
    
    def plot_confusion_matrix(self, save_path: str = None) -> None:
        """
        Plot confusion matrix for the tuned model.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.tuning_results is None:
            raise ValueError("Tuned model not evaluated. Run evaluate_tuned_model first.")
        
        conf_matrix = self.tuning_results['confusion_matrix']
        
        plt.figure(figsize=FIGURE_SIZE)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=CONFUSION_MATRIX_CMAP,
                   xticklabels=['Reliable', 'Unreliable'],
                   yticklabels=['Reliable', 'Unreliable'])
        plt.title('Confusion Matrix: Tuned Random Forest')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def get_best_model(self) -> Any:
        """
        Get the best tuned model.
        
        Returns:
            Best Random Forest model
        """
        if self.best_model is None:
            raise ValueError("Best model not available. Run run_grid_search first.")
        
        return self.best_model
    
    def get_tuning_results(self) -> Dict[str, Any]:
        """
        Get complete tuning and evaluation results.
        
        Returns:
            Dict: Complete results
        """
        return self.tuning_results
