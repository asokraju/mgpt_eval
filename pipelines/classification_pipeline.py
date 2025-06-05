"""
Pure Config-Driven Binary Classification Pipeline for training classifiers on embeddings.

Simplified implementation for binary classification with config-driven approach.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple, Any, List, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from models.config_models import PipelineConfig
from utils.logging_utils import get_logger
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.base import clone


class ClassificationPipeline:
    """Pure config-driven binary classification pipeline for embeddings."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        
        # Setup directories first
        self.config.setup_directories()
        
        # Resolve log file path
        log_config = config.logging
        log_config.file = config.resolve_template_string(log_config.file)
        
        self.logger = get_logger("classification_pipeline", log_config)
        
        # Set random seed
        np.random.seed(self.config.job.random_seed)
        
        # Initialize classifiers
        self.classifiers = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.config.job.random_seed
            ),
            'svm': SVC(
                probability=True, 
                random_state=self.config.job.random_seed
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.config.job.random_seed
            )
        }
    
    def run(self, train_embeddings_path: Optional[str] = None, 
            test_embeddings_path: Optional[str] = None,
            classifier_type: Optional[str] = None) -> Dict[str, Any]:
        """Run the classification pipeline using config paths."""
        
        # Use config paths if not provided
        if train_embeddings_path is None:
            train_embeddings_path = self.config.resolve_template_string(
                "${output.embeddings_dir}/train_embeddings.csv"
            )
        if test_embeddings_path is None:
            test_embeddings_path = self.config.resolve_template_string(
                "${output.embeddings_dir}/test_embeddings.csv"
            )
        if classifier_type is None:
            classifier_type = self.config.classification.models[0]  # Use first model from config
            
        # Validate inputs
        train_path = Path(train_embeddings_path)
        test_path = Path(test_embeddings_path)
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_embeddings_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_embeddings_path}")
        if classifier_type not in self.classifiers:
            raise ValueError(f"Invalid classifier: {classifier_type}. Must be one of {list(self.classifiers.keys())}")
            
        # Create output directory
        models_dir = self.config.resolve_template_string(
            "${job.output_dir}/models/${job.name}_{classifier_type}"
        ).replace("{classifier_type}", classifier_type)
        output_path = Path(models_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting binary classification with {classifier_type}")
        
        # Load data
        X_train, y_train = self._load_embeddings(train_embeddings_path)
        X_test, y_test = self._load_embeddings(test_embeddings_path)
        
        # Validate binary classification
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)
        
        if not np.array_equal(sorted(train_classes), [0, 1]):
            raise ValueError(f"Training labels must be binary [0,1], found: {train_classes}")
        if len(test_classes) > 2 or not all(c in [0, 1] for c in test_classes):
            raise ValueError(f"Test labels must be binary [0,1], found: {test_classes}")
        
        # Check dimensions
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Feature dimension mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}")
            
        self.logger.info(f"Loaded {len(X_train)} training and {len(X_test)} test samples")
        self.logger.info(f"Feature dimension: {X_train.shape[1]}")
        self.logger.info(f"Training distribution: class_0={np.sum(y_train==0)}, class_1={np.sum(y_train==1)}")
        self.logger.info(f"Test distribution: class_0={np.sum(y_test==0)}, class_1={np.sum(y_test==1)}")
        
        # Check class imbalance
        train_counts = np.bincount(y_train)
        if len(train_counts) < 2:
            raise ValueError(f"Training data has only one class: {np.unique(y_train)}")
        imbalance_ratio = max(train_counts) / min(train_counts)
        use_balanced = imbalance_ratio > 3
        if use_balanced:
            self.logger.info(f"Class imbalance detected (ratio {imbalance_ratio:.2f}), using balanced weights")
        
        # Check minimum samples for CV
        min_class_count = min(train_counts)
        if min_class_count < self.config.classification.cross_validation.n_folds:
            raise ValueError(f"Not enough samples for {self.config.classification.cross_validation.n_folds}-fold CV. "
                           f"Minority class has only {min_class_count} samples")
        
        # Scale features
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Validate scaling didn't produce invalid values
            if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                raise ValueError("Scaling produced NaN or infinite values in training data")
            if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any():
                raise ValueError("Scaling produced NaN or infinite values in test data")
                
        except Exception as e:
            raise RuntimeError(f"Feature scaling failed: {e}")
        
        # Train classifier
        best_model, best_params, best_score = self._train_classifier(
            X_train_scaled, y_train, classifier_type, use_balanced
        )
        
        # Evaluate
        test_metrics = self._evaluate_model(best_model, X_test_scaled, y_test)
        
        # Save model
        model_path = self._save_model(
            best_model, scaler, classifier_type, best_params, 
            test_metrics, models_dir
        )
        
        return {
            'model_path': model_path,
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_metrics': test_metrics,
            'classifier_type': classifier_type,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_embeddings(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings from CSV."""
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file {filepath}: {e}")
        
        # Check for empty file
        if len(df) == 0:
            raise ValueError(f"CSV file is empty: {filepath}")
        
        # Validate columns
        required_cols = ['embedding', 'label']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {filepath}: {missing}")
        
        # Validate no missing data
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Found null values in {filepath}: {null_counts[null_counts > 0].to_dict()}")
        
        # Parse embeddings
        embeddings = []
        embedding_dim = None
        
        for idx, emb_str in enumerate(df['embedding']):
            try:
                emb = json.loads(emb_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Row {idx} in {filepath}: invalid JSON - {e}")
                
            if not isinstance(emb, list):
                raise ValueError(f"Row {idx} in {filepath}: embedding must be a list, got {type(emb)}")
            if len(emb) == 0:
                raise ValueError(f"Row {idx} in {filepath}: empty embedding")
            
            # Validate numeric values and consistent dimensions
            try:
                emb_array = np.array(emb, dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Row {idx} in {filepath}: embedding contains non-numeric values - {e}")
            
            if np.isnan(emb_array).any():
                raise ValueError(f"Row {idx} in {filepath}: embedding contains NaN values")
            if np.isinf(emb_array).any():
                raise ValueError(f"Row {idx} in {filepath}: embedding contains infinite values")
            
            # Check dimension consistency
            if embedding_dim is None:
                embedding_dim = len(emb_array)
            elif len(emb_array) != embedding_dim:
                raise ValueError(f"Row {idx} in {filepath}: inconsistent embedding dimension "
                               f"(expected {embedding_dim}, got {len(emb_array)})")
            
            embeddings.append(emb_array)
        
        # Convert to numpy array
        try:
            embeddings = np.stack(embeddings, axis=0)
        except Exception as e:
            raise ValueError(f"Failed to convert embeddings to numpy array: {e}")
        
        # Parse labels
        try:
            labels = df['label'].astype(np.int32).values
        except Exception as e:
            raise ValueError(f"Failed to convert labels to integers in {filepath}: {e}")
        
        # Validate binary labels
        unique_labels = np.unique(labels)
        if not all(l in [0, 1] for l in unique_labels):
            raise ValueError(f"Labels must be binary [0,1] in {filepath}, found: {unique_labels}")
        
        return embeddings, labels
    
    def _train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                         classifier_type: str, use_balanced: bool) -> Tuple[Any, Dict, float]:
        """Train classifier with grid search using config parameters."""
        
        # Clone classifier to avoid modifying the original
        classifier = clone(self.classifiers[classifier_type])
        
        # Apply balanced weights if needed
        if use_balanced and hasattr(classifier, 'class_weight'):
            classifier.set_params(class_weight='balanced')
        
        # Get hyperparameters from config
        param_grid = getattr(self.config.classification.hyperparameter_search, classifier_type)
        
        # Check for empty parameter grid
        if not param_grid:
            raise ValueError(f"Empty parameter grid for {classifier_type}")
        
        # Setup cross-validation using config
        cv = StratifiedKFold(
            n_splits=self.config.classification.cross_validation.n_folds,
            shuffle=True,
            random_state=self.config.job.random_seed
        )
        
        # Grid search using config
        grid_search = GridSearchCV(
            classifier,
            param_grid,
            cv=cv,
            scoring=self.config.classification.cross_validation.scoring,
            n_jobs=self.config.classification.cross_validation.n_jobs,
            verbose=1,
            error_score='raise',
            refit=True
        )
        
        self.logger.info("Starting grid search...")
        try:
            grid_search.fit(X_train, y_train)
        except Exception as e:
            raise RuntimeError(f"Grid search failed: {e}")
        
        # Validate results
        if not hasattr(grid_search, 'best_estimator_') or grid_search.best_estimator_ is None:
            raise RuntimeError("Grid search failed to find a valid model")
        
        if grid_search.best_score_ <= 0:
            raise RuntimeError(f"Grid search produced invalid best score: {grid_search.best_score_}")
        
        self.logger.info(f"Best params: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate binary classifier."""
        # Predictions
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")
        
        # Basic metrics
        try:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0)
            }
        except Exception as e:
            raise RuntimeError(f"Failed to compute basic metrics: {e}")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to compute confusion matrix: {e}")
        
        # ROC-AUC (only if test set has both classes)
        if len(np.unique(y_test)) == 2:
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                    metrics['roc_auc'] = roc_auc_score(y_test, y_scores)
                else:
                    self.logger.warning("Model does not support probability predictions, skipping ROC-AUC")
                    metrics['roc_auc'] = None
            except Exception as e:
                raise RuntimeError(f"Failed to compute ROC-AUC: {e}")
        else:
            self.logger.warning("Test set has only one class, skipping ROC-AUC")
            metrics['roc_auc'] = None
        
        self.logger.info(f"Test metrics: Acc={metrics['accuracy']:.4f}, "
                       f"F1={metrics['f1_score']:.4f}, "
                       f"AUC={metrics.get('roc_auc', 'N/A')}")
        
        return metrics
    
    def _save_model(self, model: Any, scaler: StandardScaler, classifier_type: str,
                   best_params: Dict, test_metrics: Dict, output_dir: str) -> str:
        """Save model and metadata using config-derived paths."""
        output_path = Path(output_dir)
        
        # Use config-derived timestamp format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model package
        model_data = {
            'model': model,
            'scaler': scaler,
            'classifier_type': classifier_type,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0.0',
            'job_name': self.config.job.name
        }
        
        # Save model using config naming
        model_file = output_path / f"{classifier_type}_model_{timestamp}.pkl"
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {model_file}: {e}")
        
        # Save metrics
        metrics_file = output_path / f"{classifier_type}_metrics_{timestamp}.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump({
                    'job_name': self.config.job.name,
                    'classifier_type': classifier_type,
                    'best_params': best_params,
                    'test_metrics': test_metrics,
                    'timestamp': model_data['timestamp'],
                    'pipeline_version': model_data['pipeline_version']
                }, f, indent=2)
        except Exception as e:
            # Clean up model file if metrics save fails
            try:
                model_file.unlink()
            except:
                pass
            raise RuntimeError(f"Failed to save metrics to {metrics_file}: {e}")
        
        self.logger.info(f"Model saved: {model_file}")
        self.logger.info(f"Metrics saved: {metrics_file}")
        
        return str(model_file)