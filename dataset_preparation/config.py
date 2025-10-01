"""
Configuration parameters for dataset preparation pipeline.
Academic implementation for misinformation detection research.
"""

from pathlib import Path

# Dataset directory structure
DATA_DIR = Path("datasets")
LIAR_DIR = DATA_DIR / "LIAR"
ISOT_DIR = DATA_DIR / "ISOT"

# Output configuration
OUTPUT_DIR = Path("output")
PROCESSED_DATASET = "aura_processed_dataset.csv"
ANALYSIS_REPORT = "dataset_analysis_report.json"

# Text preprocessing parameters
TEXT_PROCESSING = {
    'min_text_length': 50,
    'max_text_length': 10000,
    'remove_duplicates': True,
    'normalize_whitespace': True,
    'remove_special_chars': True
}

# TF-IDF vectorization parameters
TFIDF_PARAMS = {
    'max_features': 15000,
    'ngram_range': (1, 3),
    'min_df': 3,
    'max_df': 0.9,
    'stop_words': 'english',
    'sublinear_tf': True,
    'use_idf': True
}

# BERT embedding parameters
BERT_PARAMS = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 512,
    'batch_size': 16,
    'sample_size': 100
}

# Quality assessment thresholds
QUALITY_THRESHOLDS = {
    'min_readability_score': 20,
    'max_readability_score': 80,
    'min_vocab_diversity': 0.005,
    'max_class_imbalance_ratio': 1.5,
    'max_duplicate_percentage': 0.02
}

# Model evaluation configuration
EVAL_PARAMS = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'stratify': True,
    'cv_folds': 5
}

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)
