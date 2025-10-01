"""
Model loading and prediction service for the Aura API.
"""

import joblib
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from .config import TUNED_MODEL_PATH, TUNED_VECTORIZER_PATH

logger = logging.getLogger(__name__)


class ModelService:
    """Service for loading and using the trained misinformation detection model."""
    
    def __init__(self):
        """Initialize the model service."""
        self.model = None
        self.vectorizer = None
        self._model_loaded = False
        self._vectorizer_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the trained Random Forest model and TF-IDF vectorizer.
        
        Returns:
            bool: True if both model and vectorizer loaded successfully
        """
        try:
            logger.info(f"Loading model from {TUNED_MODEL_PATH}")
            self.model = joblib.load(TUNED_MODEL_PATH)
            self._model_loaded = True
            logger.info("Model loaded successfully")
            
            logger.info(f"Loading vectorizer from {TUNED_VECTORIZER_PATH}")
            self.vectorizer = joblib.load(TUNED_VECTORIZER_PATH)
            self._vectorizer_loaded = True
            logger.info("Vectorizer loaded successfully")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model or vectorizer file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading model or vectorizer: {e}")
            return False
    
    def is_ready(self) -> bool:
        """
        Check if the model service is ready to make predictions.
        
        Returns:
            bool: True if both model and vectorizer are loaded
        """
        return self._model_loaded and self._vectorizer_loaded
    
    def predict_single(self, text: str, include_confidence: bool = True) -> Tuple[str, float, float]:
        """
        Make a prediction for a single text sample.
        
        Args:
            text: Text content to analyze
            include_confidence: Whether to calculate confidence score
            
        Returns:
            Tuple of (prediction, confidence, processing_time_ms)
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If text is invalid
        """
        if not self.is_ready():
            raise RuntimeError("Model service is not ready. Model or vectorizer not loaded.")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        start_time = time.time()
        
        try:
            # Transform text to TF-IDF features
            text_vectorized = self.vectorizer.transform([text])
            
            # Make prediction
            prediction_label = self.model.predict(text_vectorized)[0]
            
            # Get confidence score if requested
            if include_confidence:
                prediction_proba = self.model.predict_proba(text_vectorized)[0]
                confidence = float(prediction_proba[prediction_label])
            else:
                confidence = 1.0
            
            # Convert label to human-readable format
            prediction_text = "Reliable" if prediction_label == 0 else "Unreliable"
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return prediction_text, confidence, processing_time
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, texts: List[str], include_confidence: bool = True) -> List[Tuple[str, float, float]]:
        """
        Make predictions for multiple text samples.
        
        Args:
            texts: List of text contents to analyze
            include_confidence: Whether to calculate confidence scores
            
        Returns:
            List of tuples (prediction, confidence, processing_time_ms) for each text
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If texts list is invalid
        """
        if not self.is_ready():
            raise RuntimeError("Model service is not ready. Model or vectorizer not loaded.")
        
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        results = []
        
        for text in texts:
            try:
                result = self.predict_single(text, include_confidence)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text: {e}")
                # Return error result for this text
                results.append(("Error", 0.0, 0.0))
        
        return results
    
    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    @property
    def vectorizer_loaded(self) -> bool:
        """Check if vectorizer is loaded."""
        return self._vectorizer_loaded


# Global model service instance
model_service = ModelService()

