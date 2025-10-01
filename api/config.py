"""
Configuration settings for the Aura API.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "models"
TUNED_MODEL_PATH = MODEL_DIR / "tuned_random_forest.joblib"
TUNED_VECTORIZER_PATH = MODEL_DIR / "tuned_tfidf_vectorizer.joblib"
MODEL_RESULTS_PATH = MODEL_DIR / "hyperparameter_tuning_results.json"

# API settings
API_TITLE = "Aura Misinformation Detection API"
API_DESCRIPTION = """
Aura is a machine learning-powered API for detecting misinformation in textual content.

The system uses a tuned Random Forest classifier trained on over 50,000 articles
from the LIAR and ISOT datasets, achieving 90.70% accuracy.

## Features

* **Single Text Analysis**: Analyze individual text samples for reliability
* **Batch Processing**: Process multiple texts simultaneously
* **Confidence Scores**: Get probability scores for predictions
* **Fast Response**: Average prediction time under 20ms

## Use Cases

* Fact-checking assistance
* Content moderation
* News verification
* Social media monitoring
"""
API_VERSION = "1.0.0"

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://localhost:8080",
]

# Text processing limits
MAX_TEXT_LENGTH = 10000
MIN_TEXT_LENGTH = 10
MAX_BATCH_SIZE = 100

# Model metadata
MODEL_INFO = {
    "model_type": "Random Forest Classifier",
    "training_samples": 50648,
    "accuracy": 0.9070,
    "precision": 0.9074,
    "recall": 0.9070,
    "f1_score": 0.9068,
    "features": 10000,
    "hyperparameters": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
}

