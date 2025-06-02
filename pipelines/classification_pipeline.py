"""
Classification Pipeline for training binary classifiers on embeddings.

This module implements a comprehensive pipeline for training various binary
classifiers on embeddings generated from the MediClaimGPT model. It includes
hyperparameter tuning, cross-validation, and model persistence.

Supported Classifiers:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Random Forest

Key Features:
    - Automatic hyperparameter tuning with GridSearchCV
    - Stratified k-fold cross-validation
    - Feature scaling with StandardScaler
    - Model serialization with pickle/joblib
    - Comprehensive metrics calculation

Example:
    >>> pipeline = ClassificationPipeline('config.yaml')
    >>> results = pipeline.run(
    ...     train_embeddings='train_embeddings.json',
    ...     test_embeddings='test_embeddings.json',
    ...     classifier_type='logistic_regression',
    ...     output_dir='outputs/models'
    ... )
"""

import json
import logging
import yaml
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
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

logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """
    Pipeline for training binary classifiers on embeddings.
    
    This class manages the entire classification workflow, from loading
    embeddings to saving trained models. It handles data preprocessing,
    model selection, hyperparameter tuning, and evaluation.
    
    Attributes:
        config (dict): Full configuration from YAML file
        class_config (dict): Classification-specific configuration
        output_config (dict): Output format configuration
        classifiers (dict): Dictionary of available classifier instances
    """
    
    def __init__(self, config):
        """Initialize the classification pipeline with configuration."""
        try:
            # Handle both config objects and config paths
            if isinstance(config, str):
                if not Path(config).exists():
                    raise FileNotFoundError(f"Configuration file not found: {config}")
                with open(config, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif hasattr(config, 'dict'):
                # Assume it's a PipelineConfig object
                self.config = config.dict()
            else:
                # Already a dictionary
                self.config = config
            
            if not self.config:
                raise ValueError("Configuration is empty or invalid")
            
            # Validate required configuration sections
            if 'classification' not in self.config:
                raise ValueError("Configuration missing required 'classification' section")
            
            self.class_config = self.config['classification']
            self.output_config = self.config.get('output', {})
            
        except Exception as e:
            logger.error(f"Failed to initialize ClassificationPipeline: {e}")
            raise
        
        # Initialize classifier instances with base parameters
        # These will be overridden during hyperparameter tuning
        # Get random seed from config for reproducible results
        random_seed = self.config.get('job', {}).get('random_seed', 42)
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        try:
            import random
            random.seed(random_seed)
        except ImportError:
            pass
        
        logger.info(f"Random seed set to {random_seed} for reproducible results")
        
        self.classifiers = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=random_seed
            ),
            'svm': SVC(
                probability=True, 
                random_state=random_seed
            ),
            'random_forest': RandomForestClassifier(
                random_state=random_seed
            )
        }
    
    def run(self, train_embeddings: str, test_embeddings: str,
            classifier_type: str, output_dir: str) -> Dict[str, Any]:
        """
        Run the classification pipeline.
        
        Args:
            train_embeddings: Path to training embeddings
            test_embeddings: Path to test embeddings
            classifier_type: Type of classifier to use
            output_dir: Directory to save models
            
        Returns:
            Dictionary with results including model path and performance metrics
        """
        # Comprehensive input validation
        if not train_embeddings or not isinstance(train_embeddings, str):
            raise ValueError("train_embeddings must be a non-empty string path")
        
        if not test_embeddings or not isinstance(test_embeddings, str):
            raise ValueError("test_embeddings must be a non-empty string path")
        
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("output_dir must be a non-empty string path")
        
        # Validate file paths exist
        train_path = Path(train_embeddings)
        test_path = Path(test_embeddings)
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training embeddings file not found: {train_embeddings}")
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test embeddings file not found: {test_embeddings}")
        
        # Validate file formats (note: parquet support is limited to basic embedding format)
        valid_extensions = {'.json', '.csv', '.parquet'}
        if train_path.suffix not in valid_extensions:
            raise ValueError(f"Unsupported train file format: {train_path.suffix}. Supported: {valid_extensions}")
        
        if test_path.suffix not in valid_extensions:
            raise ValueError(f"Unsupported test file format: {test_path.suffix}. Supported: {valid_extensions}")
        
        # Validate classifier type
        if classifier_type not in self.classifiers:
            valid_types = list(self.classifiers.keys())
            raise ValueError(f"Invalid classifier_type '{classifier_type}'. Valid options: {valid_types}")
        
        # Validate output directory is writable
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            # Test write permission by creating a temporary file
            test_file = output_path / '.write_test'
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Cannot write to output directory {output_dir}: {e}")
        
        logger.info(f"Starting classification pipeline with {classifier_type}")
        
        # Load embeddings
        X_train, y_train, train_metadata = self._load_embeddings(train_embeddings)
        X_test, y_test, test_metadata = self._load_embeddings(test_embeddings)
        
        logger.info(f"Loaded {len(X_train)} training and {len(X_test)} test samples")
        
        # Validate data consistency
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Feature dimension mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}")
        
        if len(X_train) < 2:
            raise ValueError("Training set too small (need at least 2 samples)")
        
        if len(X_test) < 1:
            raise ValueError("Test set too small (need at least 1 sample)")
        
        # Check class balance and validate data quality
        unique_train_labels = np.unique(y_train)
        unique_test_labels = np.unique(y_test)
        
        if len(unique_train_labels) < 2:
            raise ValueError(f"Training set needs both classes, found only: {unique_train_labels}")
        
        # Validate embeddings for NaN/infinity
        if not np.isfinite(X_train).all():
            raise ValueError("Training embeddings contain NaN or infinite values")
        
        if not np.isfinite(X_test).all():
            raise ValueError("Test embeddings contain NaN or infinite values")
        
        # Log class distributions
        train_counts = np.bincount(y_train)
        test_counts = np.bincount(y_test)
        
        logger.info(f"Training class distribution: {train_counts}")
        logger.info(f"Test class distribution: {test_counts}")
        
        # EARLY WARNING: Check test set class distribution before training
        if len(unique_test_labels) < 2:
            logger.warning(f"⚠️  EARLY WARNING: Test set contains only one class: {unique_test_labels}. "
                          "ROC-AUC and some metrics will not be meaningful. Consider using a different test set.")
        
        # Check for class imbalance (store in local variable, not instance state)
        class_imbalance_detected = False
        if len(train_counts) >= 2:
            imbalance_ratio = max(train_counts) / min(train_counts)
            if imbalance_ratio > 10:
                logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                             "Enabling balanced class weights automatically.")
                class_imbalance_detected = True
            elif imbalance_ratio > 3:
                logger.info(f"Moderate class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                          "Consider using balanced class weights if performance is poor.")
        
        # Preprocess data
        X_train_scaled, X_test_scaled, scaler = self._preprocess_data(X_train, X_test)
        
        # Train classifier with hyperparameter tuning
        best_model, best_params, best_score = self._train_classifier(
            X_train_scaled, y_train, classifier_type, class_imbalance_detected
        )
        
        # Evaluate on test set
        test_metrics = self._evaluate_model(best_model, X_test_scaled, y_test)
        
        # Save model and results
        model_path = self._save_model(
            best_model, scaler, classifier_type, best_params, test_metrics, 
            train_metadata, test_metadata, output_dir
        )
        
        return {
            'model_path': model_path,
            'best_params': best_params,
            'best_score': best_score,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_embeddings(self, embeddings_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load embeddings from file with metadata.
        
        Supports both JSON and CSV formats. For CSV format, attempts to load
        accompanying metadata file if it exists.
        
        Args:
            embeddings_path (str): Path to embeddings file
            
        Returns:
            Tuple containing:
                - embeddings (np.ndarray): Array of embedding vectors
                - labels (np.ndarray): Array of binary labels
                - metadata (dict): Dictionary with mcids and other metadata
                
        Raises:
            FileNotFoundError: If embeddings file doesn't exist
            json.JSONDecodeError: If file format is invalid
        """
        path = Path(embeddings_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        try:
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Validate required fields
                if 'embeddings' not in data:
                    raise ValueError("JSON file missing 'embeddings' field")
                if 'labels' not in data:
                    raise ValueError("JSON file missing 'labels' field")
                
                # Validate embeddings data
                raw_embeddings = data['embeddings']
                if not isinstance(raw_embeddings, list):
                    raise ValueError(f"Embeddings must be a list, got {type(raw_embeddings)}")
                
                if len(raw_embeddings) == 0:
                    raise ValueError("Embeddings list is empty")
                
                # Convert to numpy with validation
                try:
                    embeddings = np.array(raw_embeddings, dtype=np.float64)
                    if embeddings.ndim != 2:
                        raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid embeddings data format: {e}")
                
                # Validate labels data
                raw_labels = data['labels']
                if not isinstance(raw_labels, list):
                    raise ValueError(f"Labels must be a list, got {type(raw_labels)}")
                
                try:
                    labels = np.array(raw_labels, dtype=np.int32)
                    if labels.ndim != 1:
                        raise ValueError(f"Labels must be 1D array, got shape {labels.shape}")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid labels data format: {e}")
                
                # Handle metadata safely
                metadata = data.get('metadata', {})
                if not isinstance(metadata, dict):
                    logger.warning("Metadata is not a dictionary, using empty dict")
                    metadata = {}
                
                # Add mcids to metadata without overwriting existing data
                if 'mcids' not in metadata:
                    metadata['mcids'] = data.get('mcids', [])
                
            elif path.suffix == '.parquet':
                # Load parquet file (expects columns: mcid, label, embedding)
                df = pd.read_parquet(path)
                
                # Validate required columns
                required_cols = ['embedding', 'label', 'mcid']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Parquet file missing required columns: {missing_cols}")
                
                # Handle parquet embeddings - they should be stored as list columns
                if df['embedding'].dtype == 'object':
                    # Embeddings stored as list objects (most common case)
                    try:
                        embeddings = np.array(df['embedding'].tolist())
                        if embeddings.ndim != 2:
                            raise ValueError(f"Parquet embeddings must be 2D, got shape {embeddings.shape}")
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid parquet embedding format: {e}")
                else:
                    # Try to reshape if embeddings are flattened
                    try:
                        embedding_values = df['embedding'].values
                        # Attempt to infer dimensions
                        n_samples = len(df)
                        embedding_dim = len(embedding_values) // n_samples
                        embeddings = embedding_values.reshape(n_samples, embedding_dim)
                    except Exception as e:
                        raise ValueError(f"Cannot parse parquet embeddings: {e}")
                
                labels = df['label'].values
                metadata = {'mcids': df['mcid'].tolist()}
                
                # Load metadata if exists
                metadata_file = path.parent / f'{path.stem}_metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata.update(json.load(f))
                    except Exception as e:
                        logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
                
            else:  # CSV format
                df = pd.read_csv(path)
                
                # Validate required columns
                required_cols = ['embedding', 'label', 'mcid']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"CSV file missing required columns: {missing_cols}")
                
                # Parse embeddings with improved memory efficiency for large datasets
                total_rows = len(df)
                logger.info(f"Processing {total_rows} embeddings with memory-optimized approach")
                
                # First pass: determine embedding dimension and count valid embeddings
                sample_size = min(100, total_rows)  # Sample first 100 rows to determine dimension
                embedding_dim = None
                failed_indices = []
                
                for i in range(sample_size):
                    emb_str = df['embedding'].iloc[i]
                    if pd.isna(emb_str) or emb_str == '':
                        continue
                    
                    try:
                        emb = json.loads(emb_str)
                        if isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb):
                            embedding_dim = len(emb)
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
                
                if embedding_dim is None:
                    raise ValueError("Could not determine embedding dimension from first 100 samples")
                
                logger.info(f"Detected embedding dimension: {embedding_dim}")
                
                # Pre-allocate numpy array for better memory efficiency
                embeddings = np.zeros((total_rows, embedding_dim), dtype=np.float64)
                
                # Process embeddings in chunks directly into pre-allocated array
                chunk_size = 1000  # Process 1000 embeddings at a time
                
                for chunk_start in range(0, total_rows, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_rows)
                    
                    for i in range(chunk_start, chunk_end):
                        emb_str = df['embedding'].iloc[i]
                        
                        # Handle missing values
                        if pd.isna(emb_str) or emb_str == '':
                            logger.debug(f"Missing embedding at row {i}, using zero vector")
                            failed_indices.append(i)
                            # embeddings[i] is already zeros
                            continue
                        
                        try:
                            emb = json.loads(emb_str)
                            if not isinstance(emb, list):
                                raise ValueError(f"Embedding must be a list, got {type(emb)}")
                            if len(emb) != embedding_dim:
                                raise ValueError(f"Embedding dimension mismatch: expected {embedding_dim}, got {len(emb)}")
                            if not all(isinstance(x, (int, float)) for x in emb):
                                raise ValueError("Embedding must contain only numeric values")
                            
                            # Directly assign to pre-allocated array
                            embeddings[i] = emb
                            
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.debug(f"Invalid embedding format at row {i}: {e}, using zero vector")
                            failed_indices.append(i)
                            # embeddings[i] is already zeros
                
                # Log summary of failed embeddings
                if failed_indices:
                    logger.warning(f"Replaced {len(failed_indices)} invalid embeddings with zero vectors")
                
                # embeddings is already a properly shaped numpy array
                labels = df['label'].values
                metadata = {'mcids': df['mcid'].tolist()}
                
                # Load metadata if exists
                metadata_file = path.parent / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata.update(json.load(f))
            
            # Validate data consistency
            if len(embeddings) != len(labels):
                raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(labels)} labels")
            
            if len(embeddings) == 0:
                raise ValueError("No embeddings found in file")
            
            return embeddings, labels, metadata
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from {embeddings_path}: {e}")
            raise RuntimeError(f"Embedding loading failed for {embeddings_path}") from e
    
    def _preprocess_data(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Preprocess data by scaling features.
        
        Uses StandardScaler to normalize features to zero mean and unit variance.
        The scaler is fit only on training data to prevent data leakage.
        
        Args:
            X_train (np.ndarray): Training embeddings
            X_test (np.ndarray): Test embeddings
            
        Returns:
            Tuple containing:
                - X_train_scaled (np.ndarray): Scaled training data
                - X_test_scaled (np.ndarray): Scaled test data
                - scaler (StandardScaler): Fitted scaler for future use
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def _train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                         classifier_type: str, class_imbalance_detected: bool = False) -> Tuple[Any, Dict, float]:
        """
        Train classifier with hyperparameter tuning.
        
        Performs exhaustive grid search over specified parameter ranges using
        stratified k-fold cross-validation. Returns the best model based on
        the configured scoring metric (default: ROC-AUC).
        
        Args:
            X_train (np.ndarray): Scaled training features
            y_train (np.ndarray): Training labels
            classifier_type (str): Type of classifier to train
            
        Returns:
            Tuple containing:
                - best_model: Trained model with best parameters
                - best_params (dict): Best hyperparameters found
                - best_score (float): Best cross-validation score
                
        Note:
            Training progress is logged at verbose level 2
        """
        logger.info(f"Training {classifier_type} with hyperparameter tuning...")
        
        # Get classifier and parameters - CREATE FRESH INSTANCE to avoid state persistence
        if classifier_type not in self.classifiers:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Create a fresh classifier instance to avoid modifying the original
        # Use sklearn.base.clone() for proper deep copying
        classifier = clone(self.classifiers[classifier_type])
        
        # Apply class weights if severe imbalance detected
        if class_imbalance_detected:
            if classifier_type in ['logistic_regression', 'svm']:
                classifier.set_params(class_weight='balanced')
                logger.info(f"Applied balanced class weights to {classifier_type}")
            elif classifier_type == 'random_forest':
                classifier.set_params(class_weight='balanced')
                logger.info(f"Applied balanced class weights to {classifier_type}")
        
        # Set up cross-validation configuration first (needed for grid size calculation)
        # Handle both old and new config formats
        if 'cross_validation' in self.class_config:
            # New nested format
            cv_config = self.class_config['cross_validation']
            n_folds = cv_config.get('n_folds', 5)
            scoring = cv_config.get('scoring', 'roc_auc')
            n_jobs = cv_config.get('n_jobs', -1)
        else:
            # Old flat format (fallback)
            n_folds = self.class_config.get('cv_folds', 5)
            scoring = self.class_config.get('cv_scoring', 'roc_auc')
            n_jobs = self.class_config.get('cv_parallel_jobs', -1)
        
        # Get hyperparameters with fallback defaults
        hyperparams = self.class_config.get('hyperparameter_search', {})
        if classifier_type not in hyperparams:
            logger.warning(f"No hyperparameters found for {classifier_type}, using defaults")
            param_grid = self._get_default_hyperparameters(classifier_type)
        else:
            param_grid = hyperparams[classifier_type]
        
        # Validate hyperparameter combinations for early error detection
        try:
            self._validate_hyperparameter_grid(classifier_type, param_grid)
        except ValueError as e:
            logger.error(f"Invalid hyperparameter configuration: {e}")
            raise
        
        # Calculate and warn about grid search size
        grid_size = 1
        for param_values in param_grid.values():
            grid_size *= len(param_values)
        
        total_fits = grid_size * n_folds
        logger.info(f"Grid search will evaluate {grid_size} parameter combinations "
                   f"with {n_folds}-fold CV = {total_fits} total model fits")
        
        if total_fits > 1000:
            logger.warning(f"Large grid search ({total_fits} fits) may take significant time and memory")
        if total_fits > 5000:
            logger.error(f"Extremely large grid search ({total_fits} fits) - consider reducing parameter grid")
            raise ValueError(f"Grid search too large ({total_fits} fits). Please reduce hyperparameter grid size.")
        
        # Get random seed from config
        random_seed = self.config.get('job', {}).get('random_seed', 42)
        
        cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed
        )
        
        # Perform exhaustive grid search over parameter space
        grid_search = GridSearchCV(
            classifier,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=2
        )
        
        # Fit model with timing
        import time
        start_time = time.time()
        logger.info("Starting grid search training...")
        
        try:
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            logger.info(f"Grid search completed in {training_time:.2f} seconds")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Log additional diagnostics
            if hasattr(grid_search, 'cv_results_'):
                mean_fit_time = np.mean(grid_search.cv_results_['mean_fit_time'])
                logger.info(f"Average fit time per fold: {mean_fit_time:.3f} seconds")
            
            return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Grid search failed after {training_time:.2f} seconds: {e}")
            raise
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set with comprehensive error handling."""
        logger.info("Evaluating model on test set...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Determine if binary or multiclass classification
            n_classes = len(np.unique(np.concatenate([y_test, y_pred])))
            is_binary = n_classes == 2
            
            # Calculate basic metrics with appropriate averaging
            if is_binary:
                average_method = 'binary'
                logger.debug("Using binary classification metrics")
            else:
                average_method = 'macro'  # Use macro-averaging for multiclass
                logger.info(f"Detected {n_classes} classes, using macro-averaged metrics")
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average=average_method, zero_division=0),
                'recall': recall_score(y_test, y_pred, average=average_method, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average=average_method, zero_division=0),
                'n_classes': n_classes,
                'is_binary': is_binary
            }
            
            # Add confusion matrix with safe division for normalization
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Add normalized confusion matrix with division by zero protection
            cm_row_sums = cm.sum(axis=1)
            # Avoid division by zero by replacing zeros with ones (empty classes)
            cm_row_sums_safe = np.where(cm_row_sums == 0, 1, cm_row_sums)
            cm_normalized = cm / cm_row_sums_safe[:, np.newaxis]
            metrics['confusion_matrix_normalized'] = cm_normalized.tolist()
            
            # Try to get probability predictions for ROC-AUC
            try:
                # Check if test set has both classes (required for ROC-AUC)
                unique_test_labels = np.unique(y_test)
                if len(unique_test_labels) < 2:
                    logger.warning(f"Test set has only one class ({unique_test_labels}), skipping ROC-AUC")
                    metrics['roc_auc'] = None
                elif hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    # Handle both binary and multiclass cases
                    if y_pred_proba.shape[1] == 2:
                        # Binary classification - use positive class probability
                        y_pred_proba = y_pred_proba[:, 1]
                    elif y_pred_proba.shape[1] > 2:
                        # Multiclass - ROC AUC requires special handling
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                        logger.info("Using multiclass ROC-AUC (macro-averaged)")
                    else:
                        logger.warning("Unexpected predict_proba shape, skipping ROC-AUC")
                        metrics['roc_auc'] = None
                    
                    if y_pred_proba.ndim == 1:  # Binary case
                        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                elif hasattr(model, 'decision_function'):
                    # For SVM without probability=True
                    y_scores = model.decision_function(X_test)
                    metrics['roc_auc'] = roc_auc_score(y_test, y_scores)
                else:
                    logger.warning("Model doesn't support probability prediction - ROC-AUC not calculated")
                    metrics['roc_auc'] = None
            except Exception as e:
                logger.warning(f"Failed to calculate ROC-AUC: {e}")
                metrics['roc_auc'] = None
            
            logger.info(f"Test metrics: Accuracy={metrics['accuracy']:.4f}, "
                       f"F1={metrics['f1_score']:.4f}, ROC-AUC={metrics.get('roc_auc', 'N/A')}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _save_model(self, model: Any, scaler: StandardScaler, classifier_type: str,
                   best_params: Dict, test_metrics: Dict, train_metadata: Dict,
                   test_metadata: Dict, output_dir: str) -> str:
        """Save trained model and metadata with error handling."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare comprehensive model data package for serialization
            model_data = {
                'model': model,
                'scaler': scaler,
                'classifier_type': classifier_type,
                'best_params': best_params,
                'test_metrics': test_metrics,
                'train_metadata': train_metadata,
                'test_metadata': test_metadata,
                'timestamp': datetime.now().isoformat(),
                'sklearn_version': getattr(__import__('sklearn'), '__version__', 'unknown')
            }
            
            # Calculate model size BEFORE adding to dict to avoid circular reference
            model_size_mb = self._estimate_model_size(model_data)
            model_data['model_size_mb'] = model_size_mb
            
            # Generate unique filename to avoid collisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            counter = 0
            while True:
                if counter == 0:
                    model_filename = f"{classifier_type}_model_{timestamp}.pkl"
                else:
                    model_filename = f"{classifier_type}_model_{timestamp}_{counter}.pkl"
                
                model_path = output_path / model_filename
                if not model_path.exists():
                    break
                counter += 1
                
                if counter > 100:  # Prevent infinite loop
                    raise RuntimeError("Could not generate unique filename after 100 attempts")
            
            # Save model with format fallback and error handling
            model_format = self.output_config.get('model_format', 'pickle')
            
            if model_format == 'joblib':
                joblib.dump(model_data, model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
            
            # Verify file was saved successfully
            if not model_path.exists() or model_path.stat().st_size == 0:
                raise RuntimeError(f"Model file was not saved properly: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        
        # Optionally save a separate "best model" file for easy access
        if self.output_config.get('save_best_model_only', False):
            best_model_path = output_path / f"best_{classifier_type}_model.pkl"
            if model_format == 'joblib':
                joblib.dump(model_data, best_model_path)
            else:
                with open(best_model_path, 'wb') as f:
                    pickle.dump(model_data, f)
        
        # Save metrics summary
        metrics_file = output_path / f"{classifier_type}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'classifier_type': classifier_type,
                'best_params': best_params,
                'test_metrics': test_metrics,
                'timestamp': model_data['timestamp']
            }, f, indent=2)
        
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    def _get_default_hyperparameters(self, classifier_type: str) -> Dict[str, List]:
        """Get default hyperparameters for a classifier type."""
        defaults = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        }
        return defaults.get(classifier_type, {})
    
    def _validate_hyperparameter_grid(self, classifier_type: str, param_grid: Dict[str, List]) -> None:
        """Validate hyperparameter combinations to catch invalid configurations early."""
        if not param_grid:
            return
        
        # Validate logistic regression hyperparameters
        if classifier_type == 'logistic_regression':
            penalties = param_grid.get('penalty', ['l2'])
            solvers = param_grid.get('solver', ['liblinear'])
            
            # Check penalty-solver compatibility
            for penalty in penalties:
                for solver in solvers:
                    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                        raise ValueError(f"L1 penalty only supported by 'liblinear' and 'saga' solvers, got '{solver}'")
                    elif penalty == 'elasticnet' and solver != 'saga':
                        raise ValueError(f"Elasticnet penalty only supported by 'saga' solver, got '{solver}'")
                    elif penalty == 'none' and solver in ['liblinear']:
                        raise ValueError(f"No penalty not supported by 'liblinear' solver")
        
        # Validate SVM hyperparameters
        elif classifier_type == 'svm':
            kernels = param_grid.get('kernel', ['rbf'])
            gammas = param_grid.get('gamma', ['scale'])
            
            # Check gamma-kernel compatibility
            for kernel in kernels:
                if kernel == 'linear':
                    for gamma in gammas:
                        if gamma not in ['scale', 'auto'] and isinstance(gamma, (int, float)):
                            logger.warning(f"Gamma parameter is ignored for linear kernel, but {gamma} was specified")
        
        # Validate Random Forest hyperparameters
        elif classifier_type == 'random_forest':
            max_depths = param_grid.get('max_depth', [None])
            min_samples_splits = param_grid.get('min_samples_split', [2])
            
            # Check logical constraints
            for max_depth in max_depths:
                if max_depth is not None and max_depth <= 0:
                    raise ValueError(f"max_depth must be positive or None, got {max_depth}")
            
            for min_split in min_samples_splits:
                if isinstance(min_split, int) and min_split < 2:
                    raise ValueError(f"min_samples_split must be at least 2, got {min_split}")
                elif isinstance(min_split, float) and (min_split <= 0 or min_split > 1):
                    raise ValueError(f"min_samples_split as float must be in (0, 1], got {min_split}")
        
        # Validate common numeric parameters
        for param_name, param_values in param_grid.items():
            if param_name == 'C':  # Regularization parameter
                for c_val in param_values:
                    if c_val <= 0:
                        raise ValueError(f"C parameter must be positive, got {c_val}")
            elif param_name == 'n_estimators':  # Number of trees
                for n_est in param_values:
                    if n_est <= 0:
                        raise ValueError(f"n_estimators must be positive, got {n_est}")
            elif param_name == 'max_iter':  # Maximum iterations
                for max_iter in param_values:
                    if max_iter <= 0:
                        raise ValueError(f"max_iter must be positive, got {max_iter}")
        
        logger.debug(f"Hyperparameter grid validation passed for {classifier_type}")
    
    def _estimate_model_size(self, model_data: Dict) -> float:
        """Estimate model size in MB for logging purposes."""
        try:
            import sys
            
            def get_size(obj, seen=None):
                """Recursively calculate size of objects."""
                size = sys.getsizeof(obj)
                if seen is None:
                    seen = set()
                
                obj_id = id(obj)
                if obj_id in seen:
                    return 0
                
                # Important mark as seen *before* entering recursion
                seen.add(obj_id)
                
                if isinstance(obj, dict):
                    size += sum([get_size(v, seen) for v in obj.values()])
                    size += sum([get_size(k, seen) for k in obj.keys()])
                elif hasattr(obj, '__dict__'):
                    size += get_size(obj.__dict__, seen)
                elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                    size += sum([get_size(i, seen) for i in obj])
                
                return size
            
            total_bytes = get_size(model_data)
            return round(total_bytes / (1024 * 1024), 2)  # Convert to MB
            
        except Exception:
            return 0.0  # Fallback if size estimation fails