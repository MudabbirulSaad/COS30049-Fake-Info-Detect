"""
Dataset Preparation Package for Misinformation Detection Research

This package provides standardized tools for preprocessing LIAR and ISOT datasets
for misinformation detection research.
"""

from .aura_dataset_preparation import DatasetPreparator
from .data_loader import DataLoader, load_datasets
from .text_processor import TextProcessor, process_texts
from .feature_engineer import FeatureEngineer, create_features
from .quality_analyzer import QualityAnalyzer, analyze_quality

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    'DatasetPreparator',
    'DataLoader',
    'TextProcessor', 
    'FeatureEngineer',
    'QualityAnalyzer',
    'load_datasets',
    'process_texts',
    'create_features',
    'analyze_quality'
]
