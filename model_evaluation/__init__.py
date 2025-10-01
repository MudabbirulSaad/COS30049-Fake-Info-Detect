"""
Model Evaluation Package for Misinformation Detection Research

This package provides standardized tools for training and evaluating machine learning
models for misinformation detection tasks.
"""

from .data_handler import DataHandler
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .model_selector import ModelSelector
from .hyperparameter_tuner import HyperparameterTuner

__version__ = "1.1.0"
__author__ = "Research Team"

__all__ = [
    'DataHandler',
    'ModelTrainer',
    'ModelEvaluator',
    'ModelSelector',
    'HyperparameterTuner'
]
