"""
Data handling module for model evaluation pipeline.
Implements data loading, preprocessing, and train-test splitting procedures.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import (
    DATA_DIR, DATASET_FILE, TEST_SIZE, RANDOM_STATE, 
    STRATIFY, TFIDF_PARAMS
)

logger = logging.getLogger(__name__)


class DataHandler:
    """Handles data loading and preprocessing for model evaluation."""
    
    def __init__(self):
        """Initialize the data handler."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the processed dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        dataset_path = DATA_DIR / DATASET_FILE
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"Loading dataset from {dataset_path}")
        self.data = pd.read_csv(dataset_path)
        
        # Handle missing values in text column
        if self.data['text'].isnull().any():
            missing_count = self.data['text'].isnull().sum()
            logger.warning(f"Found {missing_count} missing values in text column")
            self.data = self.data.dropna(subset=['text'])
            logger.info(f"Removed {missing_count} rows with missing text")
        
        logger.info(f"Dataset loaded successfully: {len(self.data)} records")
        logger.info(f"Class distribution: {self.data['label'].value_counts().to_dict()}")
        
        return self.data
    
    def split_data(self) -> Tuple[Any, Any, Any, Any]:
        """
        Split data into training and testing sets.
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        if self.data is None:
            self.load_dataset()
        
        X = self.data['text']
        y = self.data['label']
        
        stratify_param = y if STRATIFY else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify_param
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {len(self.X_train)} samples")
        logger.info(f"  Testing set: {len(self.X_test)} samples")
        logger.info(f"  Training class distribution: {self.y_train.value_counts().to_dict()}")
        logger.info(f"  Testing class distribution: {self.y_test.value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_tfidf_features(self) -> Tuple[Any, Any]:
        """
        Create TF-IDF feature vectors from text data.
        
        Returns:
            Tuple: X_train_tfidf, X_test_tfidf
        """
        if self.X_train is None:
            self.split_data()
        
        logger.info("Creating TF-IDF feature vectors")
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        
        # Fit on training data and transform both sets
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        logger.info(f"TF-IDF vectorization completed:")
        logger.info(f"  Feature dimensions: {self.X_train_tfidf.shape[1]}")
        logger.info(f"  Training matrix shape: {self.X_train_tfidf.shape}")
        logger.info(f"  Testing matrix shape: {self.X_test_tfidf.shape}")
        logger.info(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return self.X_train_tfidf, self.X_test_tfidf
    
    def get_processed_data(self) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Get all processed data components.
        
        Returns:
            Tuple: X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
        """
        if self.X_train_tfidf is None:
            self.create_tfidf_features()
        
        return (
            self.X_train_tfidf, 
            self.X_test_tfidf, 
            self.y_train, 
            self.y_test, 
            self.vectorizer
        )
