#!/usr/bin/env python3
"""
Model Evaluation Runner for Misinformation Detection Research

This script executes the complete machine learning model evaluation pipeline
for the Aura Misinformation Detection System.

Usage:
    python run_model_evaluation.py

Author: Research Team
Date: 2025-01-23
"""

import sys
import os
from pathlib import Path

# Add the model_evaluation directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'model_evaluation'))

from model_evaluation import evaluation_pipeline

def main():
    """Execute the model evaluation pipeline."""
    print("Aura Misinformation Detection - Model Evaluation")
    print("="*50)
    
    try:
        # Initialize and run the evaluation pipeline
        pipeline = evaluation_pipeline.EvaluationPipeline()
        best_model_info = pipeline.run_complete_pipeline()
        
        print(f"\nModel evaluation completed successfully")
        print(f"Best model: {best_model_info['model_name'].replace('_', ' ').title()}")
        print("Model artifacts saved for deployment")
        
        return pipeline
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        return None

if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)
