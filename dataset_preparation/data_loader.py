"""
Data loading module for LIAR and ISOT datasets.
Implements standardized loading procedures for misinformation detection research.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
from .config import LIAR_DIR, ISOT_DIR

logger = logging.getLogger(__name__)

class DataLoader:
    """Implements standardized data loading procedures for research datasets."""

    def __init__(self):
        self.liar_files = ['train.tsv', 'test.tsv', 'valid.tsv']
        self.isot_files = ['True.csv', 'Fake.csv']

    def load_liar_dataset(self) -> pd.DataFrame:
        """
        Load and standardize the LIAR dataset for binary classification.

        Returns:
            pd.DataFrame: Processed LIAR dataset with binary labels
        """
        logger.info("Loading LIAR dataset")

        dfs = []
        for file in self.liar_files:
            file_path = LIAR_DIR / file
            if not file_path.exists():
                logger.warning(f"LIAR file not found: {file}")
                continue

            df = pd.read_csv(file_path, sep='\t', header=None,
                           names=['id', 'label', 'statement', 'subject', 'speaker',
                                 'job_title', 'state', 'party', 'barely_true_count',
                                 'false_count', 'half_true_count', 'mostly_true_count',
                                 'pants_fire_count', 'context'])
            dfs.append(df)
            logger.info(f"Loaded {len(df)} records from {file}")

        if not dfs:
            raise FileNotFoundError("No LIAR dataset files found")

        combined = pd.concat(dfs, ignore_index=True)

        # Binary label mapping for misinformation detection
        label_mapping = {
            'pants-fire': 1,    # Unreliable
            'false': 1,         # Unreliable
            'barely-true': 1,   # Unreliable
            'half-true': 0,     # Reliable
            'mostly-true': 0,   # Reliable
            'true': 0          # Reliable
        }

        # Create standardized dataframe
        result = pd.DataFrame({
            'text': combined['statement'],
            'label': combined['label'].map(label_mapping),
            'source': 'LIAR',
            'subject': combined['subject'],
            'speaker': combined['speaker']
        })

        # Remove unmapped labels
        result = result.dropna(subset=['label'])
        result['label'] = result['label'].astype(int)

        logger.info(f"LIAR dataset processed: {len(result)} records")
        return result
    
    def load_isot_dataset(self) -> pd.DataFrame:
        """
        Load and standardize the ISOT dataset for binary classification.

        Returns:
            pd.DataFrame: Processed ISOT dataset with binary labels
        """
        logger.info("Loading ISOT dataset")

        dfs = []
        labels = [0, 1]  # True=0, Fake=1

        for file, label in zip(self.isot_files, labels):
            file_path = ISOT_DIR / file
            if not file_path.exists():
                logger.warning(f"ISOT file not found: {file}")
                continue

            df = pd.read_csv(file_path)
            df['label'] = label
            df['source'] = 'ISOT'
            dfs.append(df)
            logger.info(f"Loaded {len(df)} records from {file}")

        if not dfs:
            raise FileNotFoundError("No ISOT dataset files found")

        combined = pd.concat(dfs, ignore_index=True)

        # Create standardized dataframe with combined title and text
        result = pd.DataFrame({
            'text': combined['title'].fillna('') + ' ' + combined['text'].fillna(''),
            'label': combined['label'],
            'source': 'ISOT',
            'subject': combined.get('subject', 'general'),
            'date': combined.get('date', '')
        })

        logger.info(f"ISOT dataset processed: {len(result)} records")
        return result
    
    def combine_datasets(self, liar_df: pd.DataFrame, isot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine LIAR and ISOT datasets with standardization.

        Args:
            liar_df: Processed LIAR dataset
            isot_df: Processed ISOT dataset

        Returns:
            pd.DataFrame: Combined and standardized dataset
        """
        logger.info("Combining datasets")

        # Ensure consistent columns
        common_cols = ['text', 'label', 'source']
        liar_subset = liar_df[common_cols + ['subject']].copy()
        isot_subset = isot_df[common_cols + ['subject']].copy()

        # Add dataset source information
        liar_subset['dataset_source'] = 'LIAR'
        isot_subset['dataset_source'] = 'ISOT'

        combined = pd.concat([liar_subset, isot_subset], ignore_index=True)

        # Remove empty text entries
        initial_size = len(combined)
        combined = combined[combined['text'].str.strip() != '']
        empty_removed = initial_size - len(combined)

        if empty_removed > 0:
            logger.info(f"Removed {empty_removed} empty text records")

        # Randomize dataset order
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Combined dataset: {len(combined)} records")
        logger.info(f"Class distribution: {combined['label'].value_counts().to_dict()}")

        return combined

    def load_all_datasets(self) -> pd.DataFrame:
        """
        Load and combine all available datasets.

        Returns:
            pd.DataFrame: Complete combined dataset
        """
        try:
            liar_df = self.load_liar_dataset()
            isot_df = self.load_isot_dataset()
            combined_df = self.combine_datasets(liar_df, isot_df)

            logger.info("All datasets loaded successfully")
            return combined_df

        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise

def load_datasets() -> pd.DataFrame:
    """Load all datasets using the DataLoader class."""
    loader = DataLoader()
    return loader.load_all_datasets()
