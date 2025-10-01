"""
Configuration parameters for model evaluation pipeline.
"""

from pathlib import Path

# Data paths
DATA_DIR = Path("output")
DATASET_FILE = "aura_processed_dataset.csv"
MODEL_OUTPUT_DIR = Path("models")

# Data processing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY = True

# TF-IDF parameters
TFIDF_PARAMS = {
    'max_features': 10000,
    'ngram_range': (1, 2),
    'stop_words': 'english',
    'lowercase': True,
    'min_df': 2,
    'max_df': 0.95
}

# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'random_state': RANDOM_STATE,
        'max_iter': 1000,
        'C': 1.0
    },
    'naive_bayes': {
        'alpha': 1.0
    },
    'linear_svm': {
        'random_state': RANDOM_STATE,
        'max_iter': 1000,
        'C': 1.0
    },
    'random_forest': {
        'random_state': RANDOM_STATE,
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2
    }
}

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score'
]

# Output files
BEST_MODEL_FILE = "best_model.joblib"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
RESULTS_FILE = "model_evaluation_results.json"

# Visualization parameters
FIGURE_SIZE = (10, 8)
CONFUSION_MATRIX_CMAP = 'Blues'
