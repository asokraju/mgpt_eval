"""
Binary Classification Pipeline for training classifiers on embeddings.

Focused implementation for binary classification only with hard failure on errors.
"""

import json
import logging
import yaml
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple, Any, List, Union
import pandas as pd
import numpy as np

from models.config_models import PipelineConfig
from datetime import datetime
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.base import clone

logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """Binary classification pipeline for embeddings."""
    
    def __init__(self, config):
        """Initialize pipeline with configuration."""
        # Load config
        if isinstance(config, str):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config}")
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, PipelineConfig):
            # Convert PipelineConfig to dict for backward compatibility
            self.config = config.model_dump()
        else:
            self.config = config
            
        if not isinstance(self.config, dict):
            raise TypeError("Configuration must be a dictionary or PipelineConfig object")
            
        if 'classification' not in self.config:
            raise ValueError("Configuration missing 'classification' section")
            
        self.class_config = self.config['classification']
        self.output_config = self.config.get('output', {})
        
        # Set random seed
        self.random_seed = self.config.get('job', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)
        
        # Initialize classifiers
        self.classifiers = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.random_seed
            ),
            'svm': SVC(
                probability=True, 
                random_state=self.random_seed
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_seed
            )
        }
    
    def run(self, train_embeddings: str, test_embeddings: str,
            classifier_type: str, output_dir: str) -> Dict[str, Any]:
        """Run the classification pipeline."""
        # Validate inputs
        train_path = Path(train_embeddings)
        test_path = Path(test_embeddings)
        output_path = Path(output_dir)
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_embeddings}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_embeddings}")
        if classifier_type not in self.classifiers:
            raise ValueError(f"Invalid classifier: {classifier_type}. Must be one of {list(self.classifiers.keys())}")
            
        # Create output directory and test writability early
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = output_path / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise RuntimeError(f"Cannot write to output directory {output_dir}: {e}")
        
        logger.info(f"Starting binary classification with {classifier_type}")
        
        # Load data
        X_train, y_train, train_metadata = self._load_embeddings(train_embeddings)
        X_test, y_test, test_metadata = self._load_embeddings(test_embeddings)
        
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
            
        logger.info(f"Loaded {len(X_train)} training and {len(X_test)} test samples")
        logger.info(f"Feature dimension: {X_train.shape[1]}")
        logger.info(f"Training distribution: class_0={np.sum(y_train==0)}, class_1={np.sum(y_train==1)}")
        logger.info(f"Test distribution: class_0={np.sum(y_test==0)}, class_1={np.sum(y_test==1)}")
        
        # Check class imbalance
        train_counts = np.bincount(y_train)
        if len(train_counts) < 2:
            raise ValueError(f"Training data has only one class: {np.unique(y_train)}")
        imbalance_ratio = max(train_counts) / min(train_counts)
        use_balanced = imbalance_ratio > 3
        if use_balanced:
            logger.info(f"Class imbalance detected (ratio {imbalance_ratio:.2f}), using balanced weights")
        
        # Check minimum samples for CV
        min_class_count = min(train_counts)
        if min_class_count < 5:
            raise ValueError(f"Not enough samples for 5-fold CV. Minority class has only {min_class_count} samples")
        
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
            test_metrics, output_dir
        )
        
        return {
            'model_path': model_path,
            'best_params': best_params,
            'best_cv_score': best_score,
            'test_metrics': test_metrics
        }
    
    def _load_embeddings(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load embeddings from CSV. Fails on any invalid data."""
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file {filepath}: {e}")
        
        # Check for empty file
        if len(df) == 0:
            raise ValueError(f"CSV file is empty: {filepath}")
        
        # Validate columns
        required_cols = ['embedding', 'label', 'mcid']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {filepath}: {missing}")
        
        # Validate no missing data
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Found null values in {filepath}: {null_counts[null_counts > 0].to_dict()}")
        
        # Parse embeddings more efficiently
        embeddings = []
        embedding_dim = None
        
        for idx, emb_str in enumerate(df['embedding']):
            try:
                emb = json.loads(emb_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Row {idx} in {filepath}: invalid JSON - {e}")
            except Exception as e:
                raise ValueError(f"Row {idx} in {filepath}: unexpected error parsing embedding - {e}")
                
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
        
        # Final validation
        if embeddings.shape[0] != len(df):
            raise ValueError(f"Embedding array size mismatch: {embeddings.shape[0]} vs {len(df)}")
        
        # Parse labels
        try:
            labels = df['label'].astype(np.int32).values
        except Exception as e:
            raise ValueError(f"Failed to convert labels to integers in {filepath}: {e}")
        
        # Validate binary labels
        unique_labels = np.unique(labels)
        if not all(l in [0, 1] for l in unique_labels):
            raise ValueError(f"Labels must be binary [0,1] in {filepath}, found: {unique_labels}")
        
        metadata = {'mcids': df['mcid'].tolist()}
        
        return embeddings, labels, metadata
    
    def _train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                         classifier_type: str, use_balanced: bool) -> Tuple[Any, Dict, float]:
        """Train classifier with grid search."""
        # Clone classifier to avoid modifying the original
        try:
            classifier = clone(self.classifiers[classifier_type])
        except Exception as e:
            raise RuntimeError(f"Failed to clone classifier: {e}")
        
        # Apply balanced weights if needed
        if use_balanced and hasattr(classifier, 'class_weight'):
            classifier.set_params(class_weight='balanced')
        
        # Get hyperparameters
        param_grid = self._get_hyperparameters(classifier_type)
        
        # Check for empty parameter grid
        if not param_grid:
            raise ValueError(f"Empty parameter grid for {classifier_type}")
        
        # Calculate number of parameter combinations correctly
        if isinstance(param_grid, dict):
            n_params = np.prod([len(v) for v in param_grid.values()])
        elif isinstance(param_grid, list):
            # Handle list of dicts (from _get_hyperparameters or config)
            n_params = len(param_grid)
        else:
            raise ValueError(f"Invalid parameter grid type: {type(param_grid)}")
            
        min_samples_per_param = 10  # Heuristic: at least 10 samples per parameter combination
        if len(y_train) < n_params * min_samples_per_param:
            logger.warning(f"Limited samples ({len(y_train)}) for {n_params} parameter combinations. "
                         f"Consider reducing parameter grid complexity.")
        
        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_seed
        )
        
        # Grid search
        grid_search = GridSearchCV(
            classifier,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            error_score='raise',  # Fail immediately on any error
            refit=True
        )
        
        logger.info("Starting grid search...")
        try:
            grid_search.fit(X_train, y_train)
        except ValueError as e:
            if "Solver" in str(e) and "supports only" in str(e):
                raise ValueError(f"Invalid solver/penalty combination for logistic regression: {e}")
            raise RuntimeError(f"Grid search failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Grid search failed: {e}")
        
        # Validate grid search results
        if not hasattr(grid_search, 'cv_results_'):
            raise RuntimeError("Grid search did not complete - no cv_results_ found")
        
        if not hasattr(grid_search, 'best_estimator_') or grid_search.best_estimator_ is None:
            raise RuntimeError("Grid search failed to find a valid model")
        
        if not hasattr(grid_search, 'best_params_') or grid_search.best_params_ is None:
            raise RuntimeError("Grid search failed to find best parameters")
        
        if grid_search.best_score_ <= 0:
            raise RuntimeError(f"Grid search produced invalid best score: {grid_search.best_score_}")
        
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate binary classifier."""
        # Predictions
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")
        
        # Validate predictions
        if len(y_pred) != len(y_test):
            raise RuntimeError(f"Prediction length mismatch: {len(y_pred)} vs {len(y_test)}")
        
        unique_preds = np.unique(y_pred)
        if not all(p in [0, 1] for p in unique_preds):
            raise RuntimeError(f"Model produced non-binary predictions: {unique_preds}")
        
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
                    logger.warning("Model does not support probability predictions, skipping ROC-AUC")
                    metrics['roc_auc'] = None
            except Exception as e:
                raise RuntimeError(f"Failed to compute ROC-AUC: {e}")
        else:
            logger.warning("Test set has only one class, skipping ROC-AUC")
            metrics['roc_auc'] = None
        
        logger.info(f"Test metrics: Acc={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}, "
                   f"AUC={metrics.get('roc_auc', 'N/A')}")
        
        return metrics
    
    def _save_model(self, model: Any, scaler: StandardScaler, classifier_type: str,
                   best_params: Dict, test_metrics: Dict, output_dir: str) -> str:
        """Save model and metadata."""
        output_path = Path(output_dir)
        # Use filesystem-safe timestamp (no colons)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model package
        model_data = {
            'model': model,
            'scaler': scaler,
            'classifier_type': classifier_type,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0.0'
        }
        
        # Save model
        model_file = output_path / f"{classifier_type}_model_{timestamp}.pkl"
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {model_file}: {e}")
        
        # Verify the file was saved
        if not model_file.exists():
            raise RuntimeError(f"Model file was not created: {model_file}")
        
        # Save metrics (for easy inspection without loading pickle)
        metrics_file = output_path / f"{classifier_type}_metrics_{timestamp}.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump({
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
        
        logger.info(f"Model saved: {model_file}")
        logger.info(f"Metrics saved: {metrics_file}")
        
        return str(model_file)
    
    def _get_hyperparameters(self, classifier_type: str) -> Union[Dict[str, list], List[Dict[str, Any]]]:
        """Get hyperparameter grid from config or defaults."""
        # Try config first
        config_params = self.class_config.get('hyperparameter_search', {})
        if classifier_type in config_params:
            params = config_params[classifier_type]
            
            # Validate parameter structure
            if isinstance(params, dict):
                for key, values in params.items():
                    if not isinstance(values, list):
                        raise ValueError(f"Hyperparameter '{key}' must be a list of values")
                    if len(values) == 0:
                        raise ValueError(f"Hyperparameter '{key}' has empty list of values")
            elif isinstance(params, list):
                # Support list of parameter combinations from config
                if len(params) == 0:
                    raise ValueError(f"Empty parameter list for {classifier_type}")
                for p in params:
                    if not isinstance(p, dict):
                        raise ValueError(f"Each parameter combination must be a dict, got {type(p)}")
            else:
                raise ValueError(f"Hyperparameters for {classifier_type} must be a dict or list of dicts")
                
            return params
        
        # Simplified defaults for binary classification
        # Using only compatible penalty/solver combinations
        defaults = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],  # Only l2 penalty for simplicity
                'solver': ['lbfgs']  # lbfgs works well with l2
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        }
        
        return defaults.get(classifier_type, {})