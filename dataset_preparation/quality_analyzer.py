"""
Quality analysis module for dataset validation.
Implements comprehensive quality assessment procedures for research datasets.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import json
from .config import QUALITY_THRESHOLDS, OUTPUT_DIR

logger = logging.getLogger(__name__)

class QualityAnalyzer:
    """Implements comprehensive quality assessment for research datasets."""
    
    def __init__(self):
        self.quality_report = {}
        self.issues = []
        self.recommendations = []
        
    def analyze_class_balance(self, df: pd.DataFrame) -> Dict:
        """Analyze class distribution and balance."""
        logger.info("Analyzing class balance...")
        
        class_counts = df['label'].value_counts().sort_index()
        total = len(df)
        
        balance_info = {
            'class_0_count': int(class_counts.get(0, 0)),
            'class_1_count': int(class_counts.get(1, 0)),
            'class_0_ratio': class_counts.get(0, 0) / total,
            'class_1_ratio': class_counts.get(1, 0) / total,
            'imbalance_ratio': class_counts.get(0, 0) / class_counts.get(1, 1)
        }
        
        # Check for severe imbalance
        if balance_info['imbalance_ratio'] > QUALITY_THRESHOLDS['max_class_imbalance_ratio']:
            self.issues.append(f"Class imbalance detected: {balance_info['imbalance_ratio']:.2f}")
            self.recommendations.append("Consider class balancing techniques")
        
        return balance_info
    
    def analyze_text_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze text quality metrics."""
        logger.info("Analyzing text quality...")
        
        text_lengths = df['text'].str.len()
        word_counts = df['text'].str.split().str.len()
        
        quality_info = {
            'avg_text_length': float(text_lengths.mean()),
            'median_text_length': float(text_lengths.median()),
            'std_text_length': float(text_lengths.std()),
            'min_text_length': int(text_lengths.min()),
            'max_text_length': int(text_lengths.max()),
            'avg_word_count': float(word_counts.mean()),
            'median_word_count': float(word_counts.median())
        }
        
        # Check for quality issues
        if quality_info['min_text_length'] < 20:
            self.issues.append("Very short texts detected")
            self.recommendations.append("Filter out texts shorter than 20 characters")
        
        if quality_info['max_text_length'] > 20000:
            self.issues.append("Very long texts detected")
            self.recommendations.append("Consider truncating very long texts")
        
        return quality_info
    
    def analyze_vocabulary_diversity(self, df: pd.DataFrame) -> Dict:
        """Analyze vocabulary diversity and richness."""
        logger.info("Analyzing vocabulary diversity...")
        
        # Sample for efficiency
        sample_size = min(10000, len(df))
        sample_texts = df['text'].sample(n=sample_size, random_state=42)
        
        all_words = []
        for text in sample_texts:
            words = str(text).lower().split()
            all_words.extend(words)
        
        unique_words = set(all_words)
        total_words = len(all_words)
        
        vocab_info = {
            'total_words_sampled': total_words,
            'unique_words': len(unique_words),
            'vocabulary_diversity': len(unique_words) / total_words if total_words > 0 else 0,
            'sample_size': sample_size
        }
        
        # Check vocabulary diversity
        if vocab_info['vocabulary_diversity'] < QUALITY_THRESHOLDS['min_vocab_diversity']:
            self.issues.append("Low vocabulary diversity detected")
            self.recommendations.append("Check for repetitive or template content")
        
        return vocab_info
    
    def analyze_source_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze distribution across data sources."""
        logger.info("Analyzing source distribution...")
        
        source_counts = df['source'].value_counts()
        source_label_dist = df.groupby(['source', 'label']).size().unstack(fill_value=0)
        
        source_info = {
            'source_counts': source_counts.to_dict(),
            'source_label_distribution': source_label_dist.to_dict()
        }
        
        # Check for source bias
        for source in source_counts.index:
            source_data = df[df['source'] == source]
            source_balance = source_data['label'].value_counts()
            if len(source_balance) > 1:
                ratio = source_balance.iloc[0] / source_balance.iloc[1]
                if ratio > 2 or ratio < 0.5:
                    self.issues.append(f"Source bias detected in {source}")
                    self.recommendations.append(f"Balance labels within {source} source")
        
        return source_info
    
    def analyze_duplicate_content(self, df: pd.DataFrame) -> Dict:
        """Analyze duplicate and near-duplicate content."""
        logger.info("Analyzing duplicate content...")
        
        # Exact duplicates
        exact_duplicates = df.duplicated(subset=['text']).sum()
        
        # Near duplicates (first 100 characters)
        text_prefixes = df['text'].str[:100]
        near_duplicates = text_prefixes.duplicated().sum()
        
        duplicate_info = {
            'exact_duplicates': int(exact_duplicates),
            'near_duplicates': int(near_duplicates),
            'duplicate_percentage': exact_duplicates / len(df) * 100,
            'near_duplicate_percentage': near_duplicates / len(df) * 100
        }
        
        # Check for excessive duplicates
        if duplicate_info['duplicate_percentage'] > QUALITY_THRESHOLDS['max_duplicate_percentage'] * 100:
            self.issues.append(f"High duplicate content: {duplicate_info['duplicate_percentage']:.2f}%")
            self.recommendations.append("Implement more aggressive deduplication")
        
        return duplicate_info
    
    def evaluate_model_performance(self, X, y) -> Dict:
        """Quick model evaluation to assess data quality."""
        logger.info("Evaluating baseline model performance...")
        
        # Quick logistic regression test
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        performance_info = {
            'cv_mean_accuracy': float(cv_scores.mean()),
            'cv_std_accuracy': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist()
        }
        
        # Check if performance meets expectations
        if performance_info['cv_mean_accuracy'] < 0.75:
            self.issues.append("Low baseline model performance")
            self.recommendations.append("Improve feature engineering or data quality")
        
        return performance_info
    
    def analyze_clustering_quality(self, X, y) -> Dict:
        """Analyze clustering quality to understand data separability."""
        logger.info("Analyzing clustering quality...")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate metrics
        ari_score = adjusted_rand_score(y, cluster_labels)
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        clustering_info = {
            'adjusted_rand_index': float(ari_score),
            'silhouette_score': float(silhouette_avg),
            'cluster_centers_distance': float(np.linalg.norm(
                kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]
            ))
        }
        
        return clustering_info
    
    def generate_comprehensive_report(self, df: pd.DataFrame, features: Dict = None) -> Dict:
        """Generate comprehensive quality report."""
        logger.info("Generating comprehensive quality report...")
        
        # Basic dataset info
        self.quality_report['dataset_info'] = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Analyze different aspects
        self.quality_report['class_balance'] = self.analyze_class_balance(df)
        self.quality_report['text_quality'] = self.analyze_text_quality(df)
        self.quality_report['vocabulary'] = self.analyze_vocabulary_diversity(df)
        self.quality_report['source_distribution'] = self.analyze_source_distribution(df)
        self.quality_report['duplicates'] = self.analyze_duplicate_content(df)
        
        # Model performance analysis (if features provided)
        if features and 'tfidf_matrix' in features:
            X = features['tfidf_matrix']
            y = df['label'].values
            
            # Sample for efficiency
            if X.shape[0] > 5000:
                indices = np.random.choice(X.shape[0], 5000, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            self.quality_report['model_performance'] = self.evaluate_model_performance(X_sample, y_sample)
            self.quality_report['clustering'] = self.analyze_clustering_quality(X_sample, y_sample)
        
        # Calculate overall quality score
        quality_score = max(0, 100 - len(self.issues) * 5)
        
        self.quality_report['summary'] = {
            'quality_score': quality_score,
            'issues_count': len(self.issues),
            'issues': self.issues,
            'recommendations_count': len(self.recommendations),
            'recommendations': self.recommendations
        }
        
        # Save report
        report_path = OUTPUT_DIR / "quality_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.quality_report, f, indent=2)
        
        logger.info(f"Quality report saved to {report_path}")
        logger.info(f"Overall quality score: {quality_score}/100")
        
        return self.quality_report

def analyze_quality(df: pd.DataFrame, features: Dict = None) -> Dict:
    """Convenience function for quality analysis."""
    analyzer = QualityAnalyzer()
    return analyzer.generate_comprehensive_report(df, features)
