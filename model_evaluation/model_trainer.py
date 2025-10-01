"""
Model training module for misinformation detection research.
Implements training procedures for multiple classification algorithms.
"""

import logging
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from .config import MODEL_CONFIGS

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training of multiple classification models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.trained_models = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all classification models with configured parameters.
        
        Returns:
            Dict: Dictionary of initialized models
        """
        logger.info("Initializing classification models")
        
        self.models = {
            'logistic_regression': LogisticRegression(
                **MODEL_CONFIGS['logistic_regression']
            ),
            'naive_bayes': MultinomialNB(
                **MODEL_CONFIGS['naive_bayes']
            ),
            'linear_svm': LinearSVC(
                **MODEL_CONFIGS['linear_svm']
            ),
            'random_forest': RandomForestClassifier(
                **MODEL_CONFIGS['random_forest']
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models:")
        for model_name in self.models.keys():
            logger.info(f"  - {model_name}")
        
        return self.models
    
    def train_model(self, model_name: str, X_train: Any, y_train: Any) -> Any:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in initialized models")
        
        logger.info(f"Training {model_name}")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        self.trained_models[model_name] = model
        logger.info(f"Training completed for {model_name}")
        
        return model
    
    def train_all_models(self, X_train: Any, y_train: Any) -> Dict[str, Any]:
        """
        Train all initialized models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dict: Dictionary of trained models
        """
        if not self.models:
            self.initialize_models()
        
        logger.info("Starting training for all models")
        
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {str(e)}")
                continue
        
        logger.info(f"Training completed for {len(self.trained_models)} models")
        return self.trained_models
    
    def get_trained_models(self) -> Dict[str, Any]:
        """
        Get all trained models.
        
        Returns:
            Dict: Dictionary of trained models
        """
        return self.trained_models
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a specific trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model object
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained")
        
        return self.trained_models[model_name]
