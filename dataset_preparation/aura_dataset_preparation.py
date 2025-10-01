#!/usr/bin/env python3
"""
Dataset Preparation Pipeline for Misinformation Detection Research

This module implements a comprehensive dataset preparation pipeline for the
Aura Misinformation Detection System, incorporating standardized preprocessing
procedures and quality assessment protocols.

Author: Research Team
Date: 2025-01-23
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
import json
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Import pipeline components
from .data_loader import load_datasets
from .text_processor import process_texts
from .feature_engineer import create_features
from .quality_analyzer import analyze_quality
from .config import OUTPUT_DIR, PROCESSED_DATASET

class DatasetPreparator:
    """
    Implements comprehensive dataset preparation for misinformation detection research.

    This class provides a standardized pipeline for processing LIAR and ISOT datasets,
    including text preprocessing, feature engineering, and quality assessment.
    """

    def __init__(self):
        """Initialize the dataset preparation pipeline."""
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.quality_report = None

        logger.info("Dataset Preparator initialized")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        
    def load_data(self):
        """Load raw datasets from LIAR and ISOT sources."""
        logger.info("Loading datasets")
        self.raw_data = load_datasets()
        logger.info(f"Loaded {len(self.raw_data)} total records")
        return self.raw_data

    def process_text(self):
        """Apply text preprocessing procedures."""
        logger.info("Processing texts")
        if self.raw_data is None:
            self.load_data()

        self.processed_data = process_texts(self.raw_data)
        logger.info(f"Processed {len(self.processed_data)} records")
        return self.processed_data

    def create_features(self):
        """Generate feature representations for machine learning."""
        logger.info("Creating features")
        if self.processed_data is None:
            self.process_text()

        self.features = create_features(self.processed_data)
        logger.info("Feature engineering completed")
        return self.features

    def analyze_quality(self):
        """Perform comprehensive quality assessment."""
        logger.info("Analyzing dataset quality")
        if self.processed_data is None:
            self.process_text()

        self.quality_report = analyze_quality(self.processed_data, self.features)
        logger.info(f"Quality score: {self.quality_report['summary']['quality_score']}/100")
        return self.quality_report
    
    def save_dataset(self, filename: str = None):
        """Save the processed dataset to CSV format."""
        if filename is None:
            filename = PROCESSED_DATASET

        if self.processed_data is None:
            raise ValueError("No processed data available. Execute process_text() first.")

        # Extract essential columns for machine learning
        output_df = self.processed_data[['text', 'label']].copy()
        output_path = OUTPUT_DIR / filename
        output_df.to_csv(output_path, index=False)

        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Final dataset: {len(output_df)} records")
        return output_path

    def run_complete_pipeline(self):
        """Execute the complete dataset preparation pipeline."""
        logger.info("Starting dataset preparation pipeline")

        try:
            # Load raw data
            self.load_data()

            # Process text data
            self.process_text()

            # Generate features
            self.create_features()

            # Assess quality
            self.analyze_quality()

            # Save processed dataset
            output_path = self.save_dataset()

            # Generate summary report
            self.print_summary()

            logger.info("Pipeline execution completed successfully")
            return output_path

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def print_summary(self):
        """Generate summary report of the preparation process."""
        print("\n" + "="*60)
        print("DATASET PREPARATION SUMMARY")
        print("="*60)

        if self.raw_data is not None:
            print(f"Raw Data: {len(self.raw_data)} records")

        if self.processed_data is not None:
            print(f"Processed Data: {len(self.processed_data)} records")
            class_dist = self.processed_data['label'].value_counts()
            print(f"   - Reliable (0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(self.processed_data)*100:.1f}%)")
            print(f"   - Unreliable (1): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(self.processed_data)*100:.1f}%)")

        if self.features is not None:
            print(f"Features Generated:")
            print(f"   - TF-IDF Matrix: {self.features['tfidf_matrix'].shape}")
            print(f"   - BERT Embeddings: {self.features['bert_embeddings'].shape}")
            print(f"   - Linguistic Features: {self.features['linguistic_features'].shape}")

        if self.quality_report is not None:
            quality_score = self.quality_report['summary']['quality_score']
            print(f"Quality Score: {quality_score}/100")

            if quality_score >= 80:
                print("Dataset quality assessment: EXCELLENT")
            elif quality_score >= 60:
                print("Dataset quality assessment: GOOD")
            else:
                print("Dataset quality assessment: REQUIRES IMPROVEMENT")

            if self.quality_report['summary']['issues']:
                print(f"Issues Identified: {len(self.quality_report['summary']['issues'])}")
                for issue in self.quality_report['summary']['issues'][:3]:
                    print(f"   - {issue}")

        print(f"\nOutput Location: {OUTPUT_DIR / PROCESSED_DATASET}")
        print("Status: Ready for machine learning model training")
        print("="*60)

def main():
    """Execute the dataset preparation pipeline."""
    print("Aura Misinformation Detection - Dataset Preparation Pipeline")
    print("="*60)

    try:
        # Initialize preparator
        preparator = DatasetPreparator()

        # Execute complete pipeline
        output_path = preparator.run_complete_pipeline()

        print(f"\nDataset preparation completed successfully")
        print(f"Output file: {output_path}")
        print(f"Target accuracy threshold: 85% (ready for model training)")

        return preparator

    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    preparator = main()
