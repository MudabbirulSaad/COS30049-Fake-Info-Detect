#!/usr/bin/env python3
"""
Machine Learning Model Evaluation Pipeline for Misinformation Detection Research

This module implements a comprehensive evaluation pipeline for multiple classification
algorithms applied to misinformation detection tasks.

Author: Research Team
Date: 2025-01-23
"""

import logging
import warnings
from pathlib import Path
from .data_handler import DataHandler
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .model_selector import ModelSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class EvaluationPipeline:
    """
    Implements comprehensive machine learning model evaluation pipeline.
    
    This class orchestrates the complete evaluation process including data loading,
    model training, evaluation, and selection procedures.
    """
    
    def __init__(self):
        """Initialize the evaluation pipeline."""
        self.data_handler = DataHandler()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.model_selector = ModelSelector()
        
        self.trained_models = None
        self.evaluation_results = None
        self.best_model_info = None
        
        logger.info("Evaluation pipeline initialized")
    
    def load_and_prepare_data(self):
        """Load dataset and prepare features for model training."""
        logger.info("Loading and preparing data")
        
        # Load dataset
        self.data_handler.load_dataset()
        
        # Split data
        self.data_handler.split_data()
        
        # Create TF-IDF features
        self.data_handler.create_tfidf_features()
        
        logger.info("Data preparation completed")
    
    def train_models(self):
        """Train all classification models."""
        logger.info("Starting model training phase")
        
        # Get processed data
        X_train_tfidf, _, y_train, _, _ = self.data_handler.get_processed_data()
        
        # Initialize and train models
        self.model_trainer.initialize_models()
        self.trained_models = self.model_trainer.train_all_models(X_train_tfidf, y_train)
        
        logger.info(f"Model training completed for {len(self.trained_models)} models")
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        logger.info("Starting model evaluation phase")
        
        # Get test data
        _, X_test_tfidf, _, y_test, _ = self.data_handler.get_processed_data()
        
        # Evaluate all models
        self.evaluation_results = self.model_evaluator.evaluate_all_models(
            self.trained_models, X_test_tfidf, y_test
        )
        
        logger.info("Model evaluation completed")
    
    def print_detailed_results(self):
        """Print detailed evaluation results for all models."""
        logger.info("Generating detailed evaluation reports")
        
        for model_name in self.evaluation_results.keys():
            self.model_evaluator.print_model_metrics(model_name)
            
            # Plot confusion matrix for each model
            try:
                self.model_evaluator.plot_confusion_matrix(model_name)
            except Exception as e:
                logger.warning(f"Could not plot confusion matrix for {model_name}: {str(e)}")
    
    def select_best_model(self):
        """Select and save the best performing model."""
        logger.info("Starting model selection phase")
        
        # Create comparison table
        self.model_selector.create_comparison_table(self.evaluation_results)
        self.model_selector.print_comparison_table()
        
        # Select best model
        best_model_name, best_model = self.model_selector.select_best_model(
            self.trained_models, self.evaluation_results
        )
        
        # Print recommendation
        self.model_selector.print_best_model_recommendation()
        
        # Save best model and vectorizer
        _, _, _, _, vectorizer = self.data_handler.get_processed_data()
        self.model_selector.save_best_model(vectorizer)
        
        # Save evaluation results
        self.model_selector.save_evaluation_results(self.evaluation_results)
        
        self.best_model_info = self.model_selector.get_best_model_info()
        
        logger.info("Model selection completed")
    
    def run_complete_pipeline(self):
        """Execute the complete evaluation pipeline."""
        logger.info("Starting complete model evaluation pipeline")
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Train models
            self.train_models()
            
            # Step 3: Evaluate models
            self.evaluate_models()
            
            # Step 4: Print detailed results
            self.print_detailed_results()
            
            # Step 5: Select best model
            self.select_best_model()
            
            # Generate final summary
            self.print_pipeline_summary()
            
            logger.info("Pipeline execution completed successfully")
            return self.best_model_info
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def print_pipeline_summary(self):
        """Print final pipeline summary."""
        print(f"\n{'='*80}")
        print("MODEL EVALUATION PIPELINE SUMMARY")
        print(f"{'='*80}")
        
        if self.data_handler.data is not None:
            print(f"Dataset: {len(self.data_handler.data)} total records")
            print(f"Training set: {len(self.data_handler.y_train)} samples")
            print(f"Testing set: {len(self.data_handler.y_test)} samples")
        
        if self.trained_models:
            print(f"Models trained: {len(self.trained_models)}")
            for model_name in self.trained_models.keys():
                print(f"  - {model_name.replace('_', ' ').title()}")
        
        if self.best_model_info:
            print(f"Best model: {self.best_model_info['model_name'].replace('_', ' ').title()}")
            print(f"Model artifacts saved to: models/")
        
        print(f"Status: Ready for deployment")
        print(f"{'='*80}")


def main():
    """Execute the model evaluation pipeline."""
    print("Aura Misinformation Detection - Model Evaluation Pipeline")
    print("="*65)
    
    try:
        # Initialize and run pipeline
        pipeline = EvaluationPipeline()
        best_model_info = pipeline.run_complete_pipeline()
        
        print(f"\nModel evaluation completed successfully")
        print(f"Best model: {best_model_info['model_name'].replace('_', ' ').title()}")
        print(f"Ready for deployment in web application")
        
        return pipeline
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        logger.error(f"Model evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    pipeline = main()
