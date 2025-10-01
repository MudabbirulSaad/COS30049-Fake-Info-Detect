#!/usr/bin/env python3
"""
Hyperparameter Tuning Pipeline for Random Forest Optimization

This module implements a comprehensive hyperparameter tuning pipeline for
optimizing the Random Forest classifier performance on misinformation detection.

Author: Research Team
Date: 2025-01-23
"""

import logging
import warnings
import joblib
import json
from pathlib import Path
from .data_handler import DataHandler
from .hyperparameter_tuner import HyperparameterTuner
from .config import MODEL_OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TuningPipeline:
    """
    Implements comprehensive hyperparameter tuning pipeline for Random Forest.
    
    This class orchestrates the complete tuning process including data loading,
    hyperparameter optimization, evaluation, and comparison with baseline model.
    """
    
    def __init__(self):
        """Initialize the tuning pipeline."""
        self.data_handler = DataHandler()
        self.tuner = HyperparameterTuner()
        
        self.baseline_results = None
        self.tuned_results = None
        self.comparison_results = None
        
        logger.info("Hyperparameter tuning pipeline initialized")
    
    def load_baseline_results(self) -> dict:
        """Load baseline Random Forest results from previous evaluation."""
        results_path = MODEL_OUTPUT_DIR / "model_evaluation_results.json"
        
        if not results_path.exists():
            logger.warning("Baseline results not found. Using default values.")
            return {
                'accuracy': 0.9058,
                'precision': 0.9062,
                'recall': 0.9058,
                'f1_score': 0.9056
            }
        
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        
        if 'random_forest' in all_results:
            self.baseline_results = all_results['random_forest']
            logger.info("Baseline Random Forest results loaded")
        else:
            logger.warning("Random Forest results not found in baseline file")
            self.baseline_results = {
                'accuracy': 0.9058,
                'precision': 0.9062,
                'recall': 0.9058,
                'f1_score': 0.9056
            }
        
        return self.baseline_results
    
    def load_and_prepare_data(self):
        """Load dataset and prepare features for model training."""
        logger.info("Loading and preparing data for hyperparameter tuning")
        
        # Load dataset
        self.data_handler.load_dataset()
        
        # Split data
        self.data_handler.split_data()
        
        # Create TF-IDF features
        self.data_handler.create_tfidf_features()
        
        logger.info("Data preparation completed")
    
    def run_hyperparameter_tuning(self):
        """Execute hyperparameter tuning for Random Forest."""
        logger.info("Starting hyperparameter tuning phase")
        
        # Get training data
        X_train_tfidf, _, y_train, _, _ = self.data_handler.get_processed_data()
        
        # Define parameter grid
        self.tuner.define_parameter_grid()
        
        # Run grid search
        self.tuner.run_grid_search(X_train_tfidf, y_train)
        
        # Print tuning results
        self.tuner.print_tuning_results()
        
        logger.info("Hyperparameter tuning completed")
    
    def evaluate_tuned_model(self):
        """Evaluate the tuned Random Forest model."""
        logger.info("Evaluating tuned Random Forest model")
        
        # Get test data
        _, X_test_tfidf, _, y_test, _ = self.data_handler.get_processed_data()
        
        # Evaluate tuned model
        self.tuned_results = self.tuner.evaluate_tuned_model(X_test_tfidf, y_test)
        
        # Print evaluation results
        self.tuner.print_evaluation_results()
        
        # Plot confusion matrix
        try:
            self.tuner.plot_confusion_matrix()
        except Exception as e:
            logger.warning(f"Could not plot confusion matrix: {str(e)}")
        
        logger.info("Tuned model evaluation completed")
    
    def compare_models(self):
        """Compare baseline and tuned model performance."""
        logger.info("Comparing baseline and tuned model performance")
        
        if self.baseline_results is None:
            self.load_baseline_results()
        
        if self.tuned_results is None:
            raise ValueError("Tuned model results not available")
        
        # Create comparison
        self.comparison_results = {
            'baseline': {
                'model_name': 'Random Forest (Default)',
                'accuracy': self.baseline_results['accuracy'],
                'precision': self.baseline_results['precision'],
                'recall': self.baseline_results['recall'],
                'f1_score': self.baseline_results['f1_score']
            },
            'tuned': {
                'model_name': 'Random Forest (Tuned)',
                'accuracy': self.tuned_results['test_accuracy'],
                'precision': self.tuned_results['test_precision'],
                'recall': self.tuned_results['test_recall'],
                'f1_score': self.tuned_results['test_f1_score']
            }
        }
        
        # Calculate improvements
        improvements = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            baseline_val = self.comparison_results['baseline'][metric]
            tuned_val = self.comparison_results['tuned'][metric]
            improvement = tuned_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100
            improvements[metric] = {
                'absolute': improvement,
                'percentage': improvement_pct
            }
        
        self.comparison_results['improvements'] = improvements
        
        # Print comparison
        self.print_model_comparison()
        
        logger.info("Model comparison completed")
    
    def print_model_comparison(self):
        """Print detailed comparison between baseline and tuned models."""
        if self.comparison_results is None:
            raise ValueError("Model comparison not completed")
        
        print(f"\n{'='*80}")
        print("MODEL PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        # Create comparison table
        baseline = self.comparison_results['baseline']
        tuned = self.comparison_results['tuned']
        improvements = self.comparison_results['improvements']
        
        print(f"{'Metric':<15} {'Baseline':<12} {'Tuned':<12} {'Improvement':<15}")
        print("-" * 60)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            baseline_val = baseline[metric]
            tuned_val = tuned[metric]
            improvement = improvements[metric]['absolute']
            improvement_pct = improvements[metric]['percentage']
            
            print(f"{metric.replace('_', ' ').title():<15} "
                  f"{baseline_val:<12.4f} "
                  f"{tuned_val:<12.4f} "
                  f"{improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        print(f"\n{'='*80}")
        
        # Print best parameters
        if self.tuned_results and 'best_params' in self.tuned_results:
            print("Optimal Hyperparameters:")
            for param, value in self.tuned_results['best_params'].items():
                print(f"  {param}: {value}")
        
        print(f"{'='*80}")
    
    def save_tuned_model(self):
        """Save the tuned model and results."""
        logger.info("Saving tuned model and results")
        
        # Create output directory
        MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Save tuned model
        tuned_model = self.tuner.get_best_model()
        tuned_model_path = MODEL_OUTPUT_DIR / "tuned_random_forest.joblib"
        joblib.dump(tuned_model, tuned_model_path)
        
        # Save vectorizer (same as before)
        _, _, _, _, vectorizer = self.data_handler.get_processed_data()
        vectorizer_path = MODEL_OUTPUT_DIR / "tuned_tfidf_vectorizer.joblib"
        joblib.dump(vectorizer, vectorizer_path)
        
        # Save tuning results
        tuning_results_path = MODEL_OUTPUT_DIR / "hyperparameter_tuning_results.json"
        serializable_results = {
            'tuned_model': {
                'best_params': self.tuned_results['best_params'],
                'best_cv_score': float(self.tuned_results['best_cv_score']),
                'test_accuracy': float(self.tuned_results['test_accuracy']),
                'test_precision': float(self.tuned_results['test_precision']),
                'test_recall': float(self.tuned_results['test_recall']),
                'test_f1_score': float(self.tuned_results['test_f1_score'])
            },
            'comparison': self.comparison_results
        }
        
        with open(tuning_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nTuned model artifacts saved:")
        print(f"  Model: {tuned_model_path}")
        print(f"  Vectorizer: {vectorizer_path}")
        print(f"  Results: {tuning_results_path}")
        
        logger.info("Tuned model and results saved successfully")
    
    def run_complete_pipeline(self):
        """Execute the complete hyperparameter tuning pipeline."""
        logger.info("Starting complete hyperparameter tuning pipeline")
        
        try:
            # Step 1: Load baseline results
            self.load_baseline_results()
            
            # Step 2: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 3: Run hyperparameter tuning
            self.run_hyperparameter_tuning()
            
            # Step 4: Evaluate tuned model
            self.evaluate_tuned_model()
            
            # Step 5: Compare models
            self.compare_models()
            
            # Step 6: Save tuned model
            self.save_tuned_model()
            
            # Generate final summary
            self.print_pipeline_summary()
            
            logger.info("Pipeline execution completed successfully")
            return self.comparison_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def print_pipeline_summary(self):
        """Print final pipeline summary."""
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING PIPELINE SUMMARY")
        print(f"{'='*80}")
        
        if self.data_handler.data is not None:
            print(f"Dataset: {len(self.data_handler.data)} total records")
            print(f"Training set: {len(self.data_handler.y_train)} samples")
            print(f"Testing set: {len(self.data_handler.y_test)} samples")
        
        if self.comparison_results:
            baseline_f1 = self.comparison_results['baseline']['f1_score']
            tuned_f1 = self.comparison_results['tuned']['f1_score']
            improvement = self.comparison_results['improvements']['f1_score']['percentage']
            
            print(f"Baseline F1-Score: {baseline_f1:.4f}")
            print(f"Tuned F1-Score: {tuned_f1:.4f}")
            print(f"Performance Improvement: {improvement:+.2f}%")
        
        print(f"Status: Hyperparameter tuning completed")
        print(f"Optimized model ready for deployment")
        print(f"{'='*80}")


def main():
    """Execute the hyperparameter tuning pipeline."""
    print("Aura Misinformation Detection - Hyperparameter Tuning")
    print("="*60)
    
    try:
        # Initialize and run pipeline
        pipeline = TuningPipeline()
        comparison_results = pipeline.run_complete_pipeline()
        
        print(f"\nHyperparameter tuning completed successfully")
        tuned_f1 = comparison_results['tuned']['f1_score']
        improvement = comparison_results['improvements']['f1_score']['percentage']
        print(f"Tuned model F1-Score: {tuned_f1:.4f}")
        print(f"Performance improvement: {improvement:+.2f}%")
        
        return pipeline
        
    except Exception as e:
        print(f"Error during hyperparameter tuning: {str(e)}")
        logger.error(f"Hyperparameter tuning failed: {str(e)}")
        raise


if __name__ == "__main__":
    pipeline = main()
