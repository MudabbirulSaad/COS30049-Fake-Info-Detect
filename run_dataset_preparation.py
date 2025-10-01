#!/usr/bin/env python3
"""
Dataset Preparation Runner for Misinformation Detection Research

This script executes the complete dataset preparation pipeline for the 
Aura Misinformation Detection System.

Usage:
    python run_dataset_preparation.py

Author: Research Team
Date: 2025-01-23
"""

import sys
import os
from pathlib import Path

# Add the dataset_preparation directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'dataset_preparation'))

from dataset_preparation import DatasetPreparator

def main():
    """Execute the dataset preparation pipeline."""
    print("Aura Misinformation Detection - Dataset Preparation")
    print("="*55)
    
    try:
        # Initialize and run the preparation pipeline
        preparator = DatasetPreparator()
        output_path = preparator.run_complete_pipeline()
        
        print(f"\nDataset preparation completed successfully")
        print(f"Output file: {output_path}")
        print("Ready for machine learning model development")
        
        return preparator
        
    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")
        return None

if __name__ == "__main__":
    result = main()
    if result is None:
        sys.exit(1)
