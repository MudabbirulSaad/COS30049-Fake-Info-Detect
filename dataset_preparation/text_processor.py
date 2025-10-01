"""
Text processing module for misinformation detection research.
Implements standardized preprocessing procedures for news text data.
"""

import pandas as pd
import numpy as np
import re
import string
import logging
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from .config import TEXT_PROCESSING, QUALITY_THRESHOLDS

logger = logging.getLogger(__name__)

class TextProcessor:
    """Implements standardized text preprocessing for news content analysis."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Domain-specific stopwords for news text
        news_stopwords = {
            'said', 'says', 'according', 'reported', 'report', 'reports',
            'news', 'article', 'story', 'breaking', 'update', 'latest'
        }
        self.stop_words.update(news_stopwords)

        # Standard contraction mappings
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
            "'m": " am", "it's": "it is", "that's": "that is"
        }
        
    def expand_contractions(self, text: str) -> str:
        """Expand contractions to their full forms."""
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def clean_text_advanced(self, text: str) -> str:
        """
        Apply comprehensive text cleaning procedures.

        Args:
            text: Raw text input

        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Expand contractions
        text = self.expand_contractions(text)

        # Remove URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, ' ', text)
        text = re.sub(r'www\.[^\s]+', ' ', text)

        # Remove HTML tags and entities
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)

        # Normalize quotes and punctuation
        text = re.sub(r'[""''`]', '"', text)
        text = re.sub(r'[–—]', '-', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Normalize excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)

        # Remove special characters while preserving basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text with quality filtering."""
        if not text:
            return []
        
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        processed_tokens = []
        for token in tokens:
            # Skip if stopword
            if token in self.stop_words:
                continue
                
            # Skip very short tokens
            if len(token) < 2:
                continue
                
            # Skip if all punctuation
            if all(c in string.punctuation for c in token):
                continue
                
            # Skip if all digits
            if token.isdigit():
                continue
                
            # Lemmatize
            try:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
            except:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def process_text(self, text: str) -> str:
        """Complete text processing pipeline."""
        # Clean text
        cleaned = self.clean_text_advanced(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned)
        
        # Rejoin tokens
        processed = ' '.join(tokens)
        
        return processed
    
    def filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter texts by length criteria."""
        logger.info("Filtering texts by length...")
        
        initial_size = len(df)
        min_len = TEXT_PROCESSING['min_text_length']
        max_len = TEXT_PROCESSING['max_text_length']
        
        # Calculate text lengths
        text_lengths = df['text'].str.len()
        
        # Apply filters
        length_mask = (text_lengths >= min_len) & (text_lengths <= max_len)
        filtered_df = df[length_mask].copy()
        
        removed = initial_size - len(filtered_df)
        logger.info(f"Removed {removed} texts outside length range [{min_len}, {max_len}]")
        
        return filtered_df
    
    def remove_near_duplicates(self, df: pd.DataFrame, similarity_threshold: float = 0.8) -> pd.DataFrame:
        """Remove near-duplicate texts using simple similarity."""
        logger.info("Removing near-duplicate texts...")
        
        initial_size = len(df)
        
        # Use first 200 characters for similarity check
        text_prefixes = df['text'].str[:200].str.lower()
        
        # Simple approach: remove texts with identical prefixes
        df_deduplicated = df.drop_duplicates(subset=['text'], keep='first')
        
        # Additional check for very similar prefixes
        seen_prefixes = set()
        indices_to_keep = []
        
        for idx, prefix in enumerate(text_prefixes):
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                indices_to_keep.append(idx)
        
        df_final = df.iloc[indices_to_keep].copy()
        
        removed = initial_size - len(df_final)
        logger.info(f"Removed {removed} near-duplicate texts")
        
        return df_final
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply complete text processing pipeline to dataset."""
        logger.info("Starting text processing pipeline")

        # Apply length filtering
        df = self.filter_by_length(df)

        # Remove near duplicates if configured
        if TEXT_PROCESSING['remove_duplicates']:
            df = self.remove_near_duplicates(df)

        # Apply text cleaning and preprocessing
        logger.info("Applying text cleaning procedures")
        df['processed_text'] = df['text'].apply(self.process_text)

        # Remove texts that became empty after processing
        initial_size = len(df)
        df = df[df['processed_text'].str.len() > 0]
        empty_removed = initial_size - len(df)

        if empty_removed > 0:
            logger.info(f"Removed {empty_removed} texts that became empty after processing")

        # Update text column with processed content
        df['original_text'] = df['text']
        df['text'] = df['processed_text']
        df = df.drop('processed_text', axis=1)

        logger.info(f"Text processing completed. Final dataset size: {len(df)}")

        return df

def process_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Process texts using the TextProcessor class."""
    processor = TextProcessor()
    return processor.process_dataset(df)
