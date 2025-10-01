"""
Feature engineering module for misinformation detection research.
Implements TF-IDF vectorization, BERT embeddings, and linguistic feature extraction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
import torch
from transformers import AutoTokenizer, AutoModel
from .config import TFIDF_PARAMS, BERT_PARAMS

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Implements feature engineering procedures for misinformation detection."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.feature_selector = None
        self.dimensionality_reducer = None
        
    def create_optimized_tfidf(self, texts: pd.Series) -> Tuple[np.ndarray, TfidfVectorizer]:
        """Create optimized TF-IDF features."""
        logger.info("Creating optimized TF-IDF features...")
        
        # Initialize with improved parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_PARAMS['max_features'],
            ngram_range=TFIDF_PARAMS['ngram_range'],
            min_df=TFIDF_PARAMS['min_df'],
            max_df=TFIDF_PARAMS['max_df'],
            stop_words=TFIDF_PARAMS['stop_words'],
            sublinear_tf=TFIDF_PARAMS['sublinear_tf'],
            use_idf=TFIDF_PARAMS['use_idf'],
            smooth_idf=True,
            norm='l2'
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return tfidf_matrix, self.tfidf_vectorizer
    
    def apply_feature_selection(self, X: np.ndarray, y: np.ndarray, k: int = 10000) -> Tuple[np.ndarray, Any]:
        """Apply feature selection to reduce dimensionality."""
        logger.info(f"Applying feature selection (k={k})...")
        
        self.feature_selector = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        logger.info(f"Features reduced from {X.shape[1]} to {X_selected.shape[1]}")
        
        return X_selected, self.feature_selector
    
    def apply_dimensionality_reduction(self, X: np.ndarray, n_components: int = 1000) -> Tuple[np.ndarray, Any]:
        """Apply SVD for dimensionality reduction."""
        logger.info(f"Applying SVD dimensionality reduction (n_components={n_components})...")
        
        self.dimensionality_reducer = TruncatedSVD(
            n_components=min(n_components, X.shape[1] - 1),
            random_state=42
        )
        X_reduced = self.dimensionality_reducer.fit_transform(X)
        
        explained_variance = self.dimensionality_reducer.explained_variance_ratio_.sum()
        logger.info(f"Explained variance ratio: {explained_variance:.4f}")
        
        return X_reduced, self.dimensionality_reducer
    
    def create_linguistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create linguistic and statistical features."""
        logger.info("Creating linguistic features...")
        
        features_df = df.copy()
        
        # Text length features
        features_df['text_length'] = features_df['text'].str.len()
        features_df['word_count'] = features_df['text'].str.split().str.len()
        features_df['avg_word_length'] = features_df['text_length'] / features_df['word_count']
        
        # Punctuation features
        features_df['exclamation_count'] = features_df['text'].str.count('!')
        features_df['question_count'] = features_df['text'].str.count('\?')
        features_df['quote_count'] = features_df['text'].str.count('"')
        features_df['caps_ratio'] = features_df['text'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # Readability approximation
        features_df['sentence_count'] = features_df['text'].str.count('[.!?]+')
        features_df['avg_sentence_length'] = features_df['word_count'] / (features_df['sentence_count'] + 1)
        
        # Source-based features
        features_df['is_liar_source'] = (features_df['source'] == 'LIAR').astype(int)
        features_df['is_isot_source'] = (features_df['source'] == 'ISOT').astype(int)
        
        logger.info(f"Created {len([col for col in features_df.columns if col not in df.columns])} linguistic features")
        
        return features_df
    
    def create_bert_embeddings_batch(self, texts: pd.Series, sample_size: int = None) -> np.ndarray:
        """Create BERT embeddings with batch processing."""
        logger.info("Creating BERT embeddings...")
        
        # Use sample for demonstration
        if sample_size is None:
            sample_size = BERT_PARAMS['sample_size']
        
        if len(texts) > sample_size:
            texts_sample = texts.sample(n=sample_size, random_state=42)
            logger.info(f"Using sample of {sample_size} texts for BERT embeddings")
        else:
            texts_sample = texts
        
        # Load model
        model_name = BERT_PARAMS['model_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        batch_size = BERT_PARAMS['batch_size']
        
        for i in range(0, len(texts_sample), batch_size):
            batch_texts = texts_sample.iloc[i:i+batch_size].tolist()
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=BERT_PARAMS['max_length']
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.extend(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(texts_sample)} texts")
        
        embeddings_matrix = np.array(embeddings)
        logger.info(f"BERT embeddings shape: {embeddings_matrix.shape}")
        
        return embeddings_matrix
    
    def create_all_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create all feature types."""
        logger.info("Creating comprehensive feature set...")
        
        # Create linguistic features
        features_df = self.create_linguistic_features(df)
        
        # Create TF-IDF features
        tfidf_matrix, tfidf_vectorizer = self.create_optimized_tfidf(features_df['text'])
        
        # Apply feature selection
        if tfidf_matrix.shape[1] > 10000:
            tfidf_selected, feature_selector = self.apply_feature_selection(
                tfidf_matrix, features_df['label'], k=10000
            )
        else:
            tfidf_selected = tfidf_matrix
            feature_selector = None
        
        # Create BERT embeddings (sample)
        bert_embeddings = self.create_bert_embeddings_batch(features_df['text'])
        
        # Extract linguistic feature matrix
        linguistic_features = features_df[[
            'text_length', 'word_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'quote_count', 'caps_ratio',
            'sentence_count', 'avg_sentence_length',
            'is_liar_source', 'is_isot_source'
        ]].fillna(0).values
        
        feature_dict = {
            'tfidf_matrix': tfidf_selected,
            'tfidf_vectorizer': tfidf_vectorizer,
            'feature_selector': feature_selector,
            'bert_embeddings': bert_embeddings,
            'linguistic_features': linguistic_features,
            'feature_names': {
                'linguistic': ['text_length', 'word_count', 'avg_word_length',
                             'exclamation_count', 'question_count', 'quote_count', 'caps_ratio',
                             'sentence_count', 'avg_sentence_length',
                             'is_liar_source', 'is_isot_source']
            },
            'processed_df': features_df
        }
        
        logger.info("Feature engineering completed successfully")
        
        return feature_dict

def create_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function for feature engineering."""
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)
