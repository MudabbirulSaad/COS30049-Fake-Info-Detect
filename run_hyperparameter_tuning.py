#!/usr/bin/env python3
"""
Hyperparameter Tuning Runner for Misinformation Detection Research

This script executes the complete hyperparameter tuning pipeline for optimizing
the Random Forest classifier performance on the Aura Misinformation Detection System.

Usage:
    python run_hyperparameter_tuning.py

Author: Research Team
Date: 2025-01-23
"""

import sys
import os
from pathlib import Path

# Add the model_evaluation directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'model_evaluation'))

from model_evaluation import tuning_pipeline

def main():
    """Execute the hyperparameter tuning pipeline."""
    print("Aura Misinformation Detection - Hyperparameter Tuning")
    print("="*55)
    print("This process will optimize Random Forest hyperparameters")
    print("Expected duration: 10-30 minutes depending on system performance")
    print("="*55)
    
    try:
        # Initialize and run the tuning pipeline
        pipeline = tuning_pipeline.TuningPipeline()
        comparison_results = pipeline.run_complete_pipeline()
        
        print(f"\nHyperparameter tuning completed successfully")
        
        # Extract key results
        baseline_f1 = comparison_results['baseline']['f1_score']
        tuned_f1 = comparison_results['tuned']['f1_score']
        improvement = comparison_results['improvements']['f1_score']['percentage']
        
        print(f"Baseline Random Forest F1-Score: {baseline_f1:.4f}")
        print(f"Tuned Random Forest F1-Score: {tuned_f1:.4f}")
        print(f"Performance Improvement: {improvement:+.2f}%")
        print("Optimized model artifacts saved for deployment")
        
        return pipeline
        
    except Exception as e:
        print(f"Error during hyperparameter tuning: {str(e)}")
        return None

if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)
