"""
Evaluation Pipeline for comprehensive model evaluation and visualization.
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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Pipeline for comprehensive model evaluation."""
    
    def __init__(self, config):
        """Initialize the evaluation pipeline with configuration."""
        # Handle both config objects and config paths with robust error handling
        try:
            if isinstance(config, str):
                # Config is a file path
                if not Path(config).exists():
                    raise FileNotFoundError(f"Configuration file not found: {config}")
                with open(config, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif hasattr(config, 'dict') and callable(getattr(config, 'dict')):
                # Pydantic model with dict() method
                self.config = config.dict()
            elif hasattr(config, '__dict__'):
                # Object with __dict__ attribute
                self.config = config.__dict__
            elif isinstance(config, dict):
                # Already a dictionary
                self.config = config
            else:
                # Try to convert to dict or fail gracefully
                try:
                    self.config = dict(config)
                except (TypeError, ValueError) as e:
                    raise TypeError(f"Config must be a dict, file path, or object with dict() method. Got {type(config)}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to initialize EvaluationPipeline configuration: {e}")
            raise
        
        # Validate and extract configuration with proper error handling
        try:
            self.eval_config = self.config['evaluation']
        except KeyError:
            raise ValueError("Configuration missing required 'evaluation' section")
        
        self.output_config = self.config.get('output', {})
        
        # Validate required evaluation config sections
        self._validate_config()
        
        # Set plotting style with fallback for different matplotlib/seaborn versions
        style_applied = False
        style_attempts = [
            'seaborn-v0_8-darkgrid',  # Latest seaborn style
            'seaborn-darkgrid',       # Older seaborn style  
            'seaborn',                # Basic seaborn
            'ggplot',                 # Matplotlib built-in
            'default'                 # Matplotlib default
        ]
        
        for style in style_attempts:
            try:
                plt.style.use(style)
                logger.debug(f"Successfully applied matplotlib style: {style}")
                style_applied = True
                break
            except (OSError, ValueError) as e:
                logger.debug(f"Failed to apply style '{style}': {e}")
                continue
        
        if not style_applied:
            logger.warning("Could not apply any matplotlib style, using default")
        
        # Set seaborn palette with error handling
        try:
            sns.set_palette("husl")
        except Exception as e:
            logger.warning(f"Could not set seaborn palette: {e}")
    
    def _validate_config(self):
        """Validate evaluation configuration to prevent runtime errors."""
        # Validate metrics section
        if 'metrics' not in self.eval_config:
            logger.warning("No 'metrics' section in config, using default metrics")
            self.eval_config['metrics'] = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Validate visualization section
        if 'visualization' not in self.eval_config:
            logger.warning("No 'visualization' section in config, using defaults")
            self.eval_config['visualization'] = {
                'generate_plots': True,
                'plot_formats': ['png'],
                'dpi': 300
            }
        else:
            viz_config = self.eval_config['visualization']
            
            # Validate plot_formats
            if 'plot_formats' not in viz_config:
                logger.warning("No 'plot_formats' in visualization config, using ['png']")
                viz_config['plot_formats'] = ['png']
            
            # Validate DPI
            if 'dpi' not in viz_config:
                logger.warning("No 'dpi' in visualization config, using 300")
                viz_config['dpi'] = 300
            
            # Validate generate_plots
            if 'generate_plots' not in viz_config:
                logger.warning("No 'generate_plots' in visualization config, using True")
                viz_config['generate_plots'] = True
        
        logger.debug("Configuration validation completed")
    
    def run(self, model_path: str, test_embeddings: str, 
            output_dir: str) -> Dict[str, Any]:
        """
        Run the evaluation pipeline.
        
        Args:
            model_path: Path to trained model
            test_embeddings: Path to test embeddings
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting evaluation pipeline...")
        
        # Load model and test data
        model_data = self._load_model(model_path)
        X_test, y_test, test_metadata = self._load_embeddings(test_embeddings)
        
        # Apply scaling with validation
        if 'scaler' not in model_data:
            raise ValueError("Model data missing required 'scaler' component")
        
        try:
            X_test_scaled = model_data['scaler'].transform(X_test)
        except Exception as e:
            raise ValueError(f"Failed to apply scaler to test data: {e}")
        
        # Get predictions with model validation
        if 'model' not in model_data:
            raise ValueError("Model data missing required 'model' component")
        
        y_pred, y_pred_proba = self._get_predictions(model_data['model'], X_test_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate visualizations if enabled
        if self.eval_config['visualization']['generate_plots']:
            viz_paths = self._generate_visualizations(
                y_test, y_pred, y_pred_proba, output_dir
            )
            metrics['visualizations'] = viz_paths
        
        # Generate detailed report
        report_path = self._generate_report(
            metrics, model_data, test_metadata, output_dir
        )
        
        # Save per-sample predictions
        predictions_path = self._save_predictions(
            test_metadata, y_test, y_pred, y_pred_proba, output_dir
        )
        
        return {
            'metrics': metrics,
            'report_path': report_path,
            'predictions_path': predictions_path,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_model(self, model_path: str) -> Dict[str, Any]:
        """Load trained model from file."""
        path = Path(model_path)
        
        if path.suffix == '.pkl':
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
            except:
                model_data = joblib.load(path)
        else:
            model_data = joblib.load(path)
        
        return model_data
    
    def _load_embeddings(self, embeddings_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load embeddings from file with comprehensive error handling."""
        path = Path(embeddings_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        try:
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                
                # Validate required fields
                if 'embeddings' not in data:
                    raise ValueError("JSON file missing required 'embeddings' field")
                if 'labels' not in data:
                    raise ValueError("JSON file missing required 'labels' field")
                
                try:
                    embeddings = np.array(data['embeddings'])
                    labels = np.array(data['labels'])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid data format in JSON file: {e}")
                
                metadata = data.get('metadata', {})
                if not isinstance(metadata, dict):
                    logger.warning("Metadata is not a dictionary, using empty dict")
                    metadata = {}
                
                # Safely add mcids
                if 'mcids' not in metadata:
                    metadata['mcids'] = data.get('mcids', [])
                
            elif path.suffix == '.csv':
                df = pd.read_csv(path)
                
                # Validate required columns
                required_cols = ['embedding', 'label', 'mcid']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"CSV file missing required columns: {missing_cols}")
                
                # Parse embeddings safely with error handling
                embeddings = []
                failed_indices = []
                
                for i, emb_str in enumerate(df['embedding']):
                    try:
                        if pd.isna(emb_str):
                            raise ValueError("Missing embedding")
                        emb = json.loads(emb_str)
                        if not isinstance(emb, list):
                            raise ValueError(f"Embedding must be a list, got {type(emb)}")
                        embeddings.append(emb)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Invalid embedding at row {i}: {e}")
                        failed_indices.append(i)
                        # Use dynamic embedding dimension based on first valid embedding found
                        default_dim = self._get_default_embedding_dimension(embeddings)
                        embeddings.append([0.0] * default_dim)
                
                if failed_indices:
                    logger.warning(f"Replaced {len(failed_indices)} invalid embeddings with zero vectors")
                
                embeddings = np.array(embeddings)
                labels = df['label'].values
                metadata = {'mcids': df['mcid'].tolist()}
                
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}. Supported: .json, .csv")
            
            # Validate data consistency
            if len(embeddings) != len(labels):
                raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(labels)} labels")
            
            if len(embeddings) == 0:
                raise ValueError("No embeddings found in file")
            
            return embeddings, labels, metadata
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from {embeddings_path}: {e}")
            raise RuntimeError(f"Embedding loading failed for {embeddings_path}") from e
    
    def _get_predictions(self, model: Any, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions with comprehensive validation."""
        # Validate model has required methods
        if not hasattr(model, 'predict'):
            raise ValueError("Model does not have a 'predict' method")
        
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")
        
        # Handle probability predictions with fallback
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba_full = model.predict_proba(X_test)
                
                # Handle different probability shapes
                if y_pred_proba_full.shape[1] == 2:
                    # Binary classification - use positive class probability
                    y_pred_proba = y_pred_proba_full[:, 1]
                elif y_pred_proba_full.shape[1] > 2:
                    # Multiclass - use max probability
                    y_pred_proba = np.max(y_pred_proba_full, axis=1)
                    logger.warning("Using max probability for multiclass problem")
                else:
                    raise ValueError(f"Unexpected predict_proba shape: {y_pred_proba_full.shape}")
                    
            except Exception as e:
                logger.warning(f"Failed to get probability predictions: {e}")
                
        elif hasattr(model, 'decision_function'):
            try:
                # Use decision function as probability substitute
                y_pred_proba = model.decision_function(X_test)
                # Normalize to [0, 1] range using sigmoid
                y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
                logger.info("Using decision function as probability substitute")
            except Exception as e:
                logger.warning(f"Failed to get decision function values: {e}")
        
        if y_pred_proba is None:
            # Create dummy probabilities for compatibility
            y_pred_proba = np.ones(len(y_pred)) * 0.5
            logger.warning("Model does not support probability prediction, using dummy values (0.5)")
        
        return y_pred, y_pred_proba
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        if 'accuracy' in self.eval_config['metrics']:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Determine if binary or multiclass for appropriate averaging
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique_labels) == 2
        average_method = 'binary' if is_binary else 'macro'
        
        if 'precision' in self.eval_config['metrics']:
            metrics['precision'] = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        
        if 'recall' in self.eval_config['metrics']:
            metrics['recall'] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        
        if 'f1_score' in self.eval_config['metrics']:
            metrics['f1_score'] = f1_score(y_true, y_pred, average=average_method, zero_division=0)
        
        # Add metadata about classification type
        metrics['classification_type'] = 'binary' if is_binary else 'multiclass'
        metrics['n_classes'] = len(unique_labels)
        metrics['class_labels'] = sorted(unique_labels.tolist())
        
        if 'roc_auc' in self.eval_config['metrics']:
            try:
                if is_binary:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    # Multiclass ROC-AUC requires different handling
                    if hasattr(y_pred_proba, 'shape') and len(y_pred_proba.shape) == 2:
                        # Multi-class probabilities available
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                    else:
                        # Only max probabilities available for multiclass
                        logger.warning("Cannot calculate ROC-AUC for multiclass with single probability values")
                        metrics['roc_auc'] = None
            except Exception as e:
                logger.warning(f"Failed to calculate ROC-AUC: {e}")
                metrics['roc_auc'] = None
        
        # Confusion matrix
        if 'confusion_matrix' in self.eval_config['metrics']:
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Safe normalization to avoid division by zero
            cm_row_sums = cm.sum(axis=1)
            # Replace zero row sums with 1 to avoid division by zero
            cm_row_sums_safe = np.where(cm_row_sums == 0, 1, cm_row_sums)
            cm_normalized = cm / cm_row_sums_safe[:, np.newaxis]
            metrics['confusion_matrix_normalized'] = cm_normalized.tolist()
        
        # Additional metrics
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )
        
        # Calculate confidence intervals if sample size allows
        if len(y_true) >= 100:
            metrics['confidence_intervals'] = self._calculate_confidence_intervals(
                y_true, y_pred, y_pred_proba
            )
        
        return metrics
    
    def _calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_pred_proba: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals using bootstrap."""
        n_samples = len(y_true)
        metrics_bootstrap = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': []
        }
        
        # Bootstrap sampling with edge case handling
        for bootstrap_iter in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            y_pred_proba_boot = y_pred_proba[indices]
            
            # Check if bootstrap sample has both classes
            unique_true = np.unique(y_true_boot)
            unique_pred = np.unique(y_pred_boot)
            
            try:
                metrics_bootstrap['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            except Exception as e:
                logger.debug(f"Bootstrap {bootstrap_iter}: accuracy calculation failed: {e}")
                continue
            
            # Precision/Recall/F1 with zero_division handling
            try:
                metrics_bootstrap['precision'].append(
                    precision_score(y_true_boot, y_pred_boot, average='binary', zero_division=0)
                )
            except Exception as e:
                logger.debug(f"Bootstrap {bootstrap_iter}: precision calculation failed: {e}")
                metrics_bootstrap['precision'].append(0.0)
            
            try:
                metrics_bootstrap['recall'].append(
                    recall_score(y_true_boot, y_pred_boot, average='binary', zero_division=0)
                )
            except Exception as e:
                logger.debug(f"Bootstrap {bootstrap_iter}: recall calculation failed: {e}")
                metrics_bootstrap['recall'].append(0.0)
            
            try:
                metrics_bootstrap['f1_score'].append(
                    f1_score(y_true_boot, y_pred_boot, average='binary', zero_division=0)
                )
            except Exception as e:
                logger.debug(f"Bootstrap {bootstrap_iter}: f1_score calculation failed: {e}")
                metrics_bootstrap['f1_score'].append(0.0)
            
            # ROC-AUC only if both classes present
            try:
                if len(unique_true) >= 2:
                    metrics_bootstrap['roc_auc'].append(roc_auc_score(y_true_boot, y_pred_proba_boot))
                else:
                    logger.debug(f"Bootstrap {bootstrap_iter}: only one class present, skipping ROC-AUC")
                    # Skip this bootstrap sample for ROC-AUC to maintain meaningful statistics
            except Exception as e:
                logger.debug(f"Bootstrap {bootstrap_iter}: roc_auc calculation failed: {e}")
                # Skip this bootstrap sample for ROC-AUC
        
        # Calculate confidence intervals with edge case handling
        confidence_intervals = {}
        for metric, values in metrics_bootstrap.items():
            if len(values) == 0:
                # No valid bootstrap samples for this metric
                confidence_intervals[metric] = {
                    'mean': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n_valid_samples': 0
                }
            elif len(values) < 10:
                # Too few samples for reliable confidence intervals
                confidence_intervals[metric] = {
                    'mean': np.mean(values),
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'n_valid_samples': len(values),
                    'warning': 'Too few bootstrap samples for reliable CI'
                }
            else:
                # Normal case with sufficient samples
                confidence_intervals[metric] = {
                    'mean': np.mean(values),
                    'ci_lower': np.percentile(values, 2.5),
                    'ci_upper': np.percentile(values, 97.5),
                    'n_valid_samples': len(values)
                }
        
        return confidence_intervals
    
    def _generate_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray, output_dir: str) -> Dict[str, str]:
        """Generate evaluation visualizations with comprehensive error handling."""
        viz_dir = Path(output_dir) / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        viz_paths = {}
        viz_config = self.eval_config['visualization']
        failed_plots = []
        
        # Plot generation functions and their names
        plot_functions = [
            ('roc_curve', self._plot_roc_curve, y_true, y_pred_proba, viz_dir),
            ('precision_recall_curve', self._plot_precision_recall_curve, y_true, y_pred_proba, viz_dir),
            ('confusion_matrix', self._plot_confusion_matrix, y_true, y_pred, viz_dir),
            ('probability_distribution', self._plot_probability_distribution, y_true, y_pred_proba, viz_dir)
        ]
        
        for plot_name, plot_func, *args in plot_functions:
            try:
                plot_path = plot_func(*args)
                viz_paths[plot_name] = str(plot_path)
                logger.debug(f"Successfully generated {plot_name} plot")
            except Exception as e:
                logger.error(f"Failed to generate {plot_name} plot: {e}")
                failed_plots.append(plot_name)
                # Clean up any partial files for this plot
                self._cleanup_failed_plot(viz_dir, plot_name, viz_config)
                # Continue with other plots
                continue
        
        if failed_plots:
            logger.warning(f"Some plots failed to generate: {failed_plots}")
            viz_paths['failed_plots'] = failed_plots
        
        return viz_paths
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       output_dir: Path) -> Path:
        """Plot ROC curve with comprehensive error handling."""
        try:
            # Validate inputs
            if len(np.unique(y_true)) < 2:
                raise ValueError("ROC curve requires at least 2 classes in y_true")
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            
            # Create figure with error handling
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            # Save in multiple formats with error handling
            saved_files = []
            viz_config = self.eval_config['visualization']
            
            for fmt in viz_config['plot_formats']:
                try:
                    path = output_dir / f'roc_curve.{fmt}'
                    plt.savefig(path, dpi=viz_config['dpi'], bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    saved_files.append(path)
                except Exception as save_error:
                    logger.warning(f"Failed to save ROC curve in {fmt} format: {save_error}")
                    continue
            
            plt.close(fig)
            
            if not saved_files:
                raise RuntimeError("Failed to save ROC curve in any format")
            
            return saved_files[0]  # Return first successfully saved file
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            logger.error(f"Failed to generate ROC curve: {e}")
            raise RuntimeError(f"ROC curve generation failed: {e}") from e
    
    def _plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   output_dir: Path) -> Path:
        """Plot precision-recall curve with comprehensive error handling."""
        try:
            # Validate inputs
            if len(np.unique(y_true)) < 2:
                raise ValueError("Precision-Recall curve requires at least 2 classes in y_true")
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            # Create figure with error handling
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(recall, precision, color='b', lw=2,
                   label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            
            # Save in multiple formats with error handling
            saved_files = []
            viz_config = self.eval_config['visualization']
            
            for fmt in viz_config['plot_formats']:
                try:
                    path = output_dir / f'precision_recall_curve.{fmt}'
                    plt.savefig(path, dpi=viz_config['dpi'], bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    saved_files.append(path)
                except Exception as save_error:
                    logger.warning(f"Failed to save PR curve in {fmt} format: {save_error}")
                    continue
            
            plt.close(fig)
            
            if not saved_files:
                raise RuntimeError("Failed to save Precision-Recall curve in any format")
            
            return saved_files[0]
            
        except Exception as e:
            plt.close('all')
            logger.error(f"Failed to generate Precision-Recall curve: {e}")
            raise RuntimeError(f"Precision-Recall curve generation failed: {e}") from e
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              output_dir: Path) -> Path:
        """Plot confusion matrix with dynamic class labels and error handling."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            # Get unique class labels dynamically
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            
            # Create appropriate class labels
            if len(unique_labels) == 2:
                # Binary classification - check if labels are 0/1 or custom
                if set(unique_labels) == {0, 1}:
                    class_labels = ['Negative', 'Positive']
                else:
                    class_labels = [f'Class {label}' for label in sorted(unique_labels)]
            else:
                # Multiclass classification
                class_labels = [f'Class {label}' for label in sorted(unique_labels)]
            
            # Create figure with error handling
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Use seaborn with fallback to matplotlib
            try:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_labels,
                           yticklabels=class_labels, ax=ax)
            except Exception as heatmap_error:
                logger.warning(f"Failed to create seaborn heatmap, using matplotlib: {heatmap_error}")
                # Fallback to matplotlib
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.set_xticks(range(len(class_labels)))
                ax.set_yticks(range(len(class_labels)))
                ax.set_xticklabels(class_labels)
                ax.set_yticklabels(class_labels)
                
                # Add text annotations
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title('Confusion Matrix')
            
            # Save in multiple formats with error handling
            saved_files = []
            viz_config = self.eval_config['visualization']
            
            for fmt in viz_config['plot_formats']:
                try:
                    path = output_dir / f'confusion_matrix.{fmt}'
                    plt.savefig(path, dpi=viz_config['dpi'], bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    saved_files.append(path)
                except Exception as save_error:
                    logger.warning(f"Failed to save confusion matrix in {fmt} format: {save_error}")
                    continue
            
            plt.close(fig)
            
            if not saved_files:
                raise RuntimeError("Failed to save confusion matrix in any format")
            
            return saved_files[0]
            
        except Exception as e:
            plt.close('all')
            logger.error(f"Failed to generate confusion matrix: {e}")
            raise RuntimeError(f"Confusion matrix generation failed: {e}") from e
    
    def _plot_probability_distribution(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                     output_dir: Path) -> Path:
        """Plot probability distribution for classes with dynamic labeling and error handling."""
        try:
            # Validate inputs
            if len(y_pred_proba) == 0:
                raise ValueError("Empty probability array provided")
            
            # Create figure with error handling
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get unique class labels
            unique_labels = np.unique(y_true)
            
            if len(unique_labels) == 2:
                # Binary classification
                label_0, label_1 = sorted(unique_labels)
                probs_0 = y_pred_proba[y_true == label_0]
                probs_1 = y_pred_proba[y_true == label_1]
                
                # Validate we have probabilities for both classes
                if len(probs_0) == 0 or len(probs_1) == 0:
                    raise ValueError("Missing probability data for one or both classes")
                
                # Create appropriate class names
                if set(unique_labels) == {0, 1}:
                    class_0_name, class_1_name = 'Negative class', 'Positive class'
                    xlabel = 'Predicted Probability of Positive Class'
                else:
                    class_0_name, class_1_name = f'Class {label_0}', f'Class {label_1}'
                    xlabel = f'Predicted Probability of Class {label_1}'
                
                # Plot histograms with error handling
                try:
                    ax.hist(probs_0, bins=min(50, len(probs_0)//2+1), alpha=0.7, 
                           label=class_0_name, color='blue', density=True)
                    ax.hist(probs_1, bins=min(50, len(probs_1)//2+1), alpha=0.7, 
                           label=class_1_name, color='red', density=True)
                except ValueError as hist_error:
                    logger.warning(f"Histogram creation failed, using simpler approach: {hist_error}")
                    # Fallback with fewer bins
                    ax.hist(probs_0, bins=10, alpha=0.7, label=class_0_name, color='blue', density=True)
                    ax.hist(probs_1, bins=10, alpha=0.7, label=class_1_name, color='red', density=True)
                
                ax.set_xlabel(xlabel)
            else:
                # Multiclass - plot max probability distribution
                for i, label in enumerate(sorted(unique_labels)):
                    class_probs = y_pred_proba[y_true == label]
                    if len(class_probs) == 0:
                        logger.warning(f"No probability data for class {label}")
                        continue
                    
                    color = plt.cm.tab10(i % 10)  # Use different colors
                    try:
                        ax.hist(class_probs, bins=min(30, len(class_probs)//2+1), alpha=0.6, 
                               label=f'Class {label}', color=color, density=True)
                    except ValueError as hist_error:
                        logger.warning(f"Histogram for class {label} failed: {hist_error}")
                        continue
                
                ax.set_xlabel('Predicted Probability (Max Class)')
            
            ax.set_ylabel('Density')
            ax.set_title('Probability Distribution by True Class')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save in multiple formats with error handling
            saved_files = []
            viz_config = self.eval_config['visualization']
            
            for fmt in viz_config['plot_formats']:
                try:
                    path = output_dir / f'probability_distribution.{fmt}'
                    plt.savefig(path, dpi=viz_config['dpi'], bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    saved_files.append(path)
                except Exception as save_error:
                    logger.warning(f"Failed to save probability distribution in {fmt} format: {save_error}")
                    continue
            
            plt.close(fig)
            
            if not saved_files:
                raise RuntimeError("Failed to save probability distribution in any format")
            
            return saved_files[0]
            
        except Exception as e:
            plt.close('all')
            logger.error(f"Failed to generate probability distribution: {e}")
            raise RuntimeError(f"Probability distribution generation failed: {e}") from e
    
    def _generate_report(self, metrics: Dict[str, Any], model_data: Dict[str, Any],
                        test_metadata: Dict, output_dir: str) -> str:
        """Generate comprehensive evaluation report with safe key access."""
        report_path = Path(output_dir) / 'evaluation_report.json'
        
        # Safely extract model information with defaults
        model_info = {
            'classifier_type': model_data.get('classifier_type', 'unknown'),
            'best_params': model_data.get('best_params', {}),
            'training_timestamp': model_data.get('timestamp', 'unknown'),
            'model_size_mb': model_data.get('model_size_mb', 'unknown'),
            'sklearn_version': model_data.get('sklearn_version', 'unknown')
        }
        
        # Safely extract test data information
        test_data_info = {
            'n_samples': len(test_metadata.get('mcids', [])),
            'embedding_dim': test_metadata.get('embedding_dim', 'unknown'),
            'data_source': test_metadata.get('data_source', 'unknown')
        }
        
        report = {
            'model_info': model_info,
            'test_data_info': test_data_info,
            'metrics': metrics,
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_config': {
                'metrics_calculated': list(metrics.keys()),
                'visualization_enabled': self.eval_config.get('visualization', {}).get('generate_plots', False)
            }
        }
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)  # Handle non-serializable objects
            
            # Verify the report was written successfully
            if not report_path.exists() or report_path.stat().st_size == 0:
                raise RuntimeError(f"Report file was not saved properly: {report_path}")
                
            logger.info(f"Evaluation report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
            # Create a minimal backup report
            backup_report = {
                'error': f"Failed to generate full report: {e}",
                'metrics_summary': {
                    'accuracy': metrics.get('accuracy', 'N/A'),
                    'timestamp': datetime.now().isoformat()
                }
            }
            backup_path = Path(output_dir) / 'evaluation_report_backup.json'
            try:
                with open(backup_path, 'w') as f:
                    json.dump(backup_report, f, indent=2)
                logger.warning(f"Saved backup report to: {backup_path}")
                return str(backup_path)
            except Exception as backup_error:
                logger.error(f"Failed to save backup report: {backup_error}")
                raise RuntimeError(f"Could not save evaluation report: {e}") from e
    
    def _save_predictions(self, test_metadata: Dict, y_true: np.ndarray,
                         y_pred: np.ndarray, y_pred_proba: np.ndarray,
                         output_dir: str) -> str:
        """Save per-sample predictions with comprehensive error handling."""
        try:
            predictions_path = Path(output_dir) / 'predictions.csv'
            
            # Safely get MCIDs with fallback
            mcids = test_metadata.get('mcids', [])
            if len(mcids) != len(y_true):
                logger.warning(f"MCID count mismatch: {len(mcids)} vs {len(y_true)}. Using indices.")
                mcids = list(range(len(y_true)))
            
            # Handle probability computation based on data type
            if hasattr(y_pred_proba, 'shape') and len(y_pred_proba.shape) == 2:
                # Multi-class probabilities - use max probability
                prob_positive = np.max(y_pred_proba, axis=1)
                prob_negative = 1 - prob_positive
            else:
                # Binary probabilities
                prob_positive = y_pred_proba
                prob_negative = 1 - y_pred_proba
            
            # Create predictions dataframe with safe data types
            predictions_df = pd.DataFrame({
                'mcid': mcids,
                'true_label': y_true.astype(int),
                'predicted_label': y_pred.astype(int),
                'probability_positive': prob_positive.astype(float),
                'probability_negative': prob_negative.astype(float),
                'correct': (y_true == y_pred).astype(bool)
            })
            
            # Save with error handling
            try:
                predictions_df.to_csv(predictions_path, index=False)
            except Exception as save_error:
                # Try alternative formats if CSV fails
                logger.warning(f"CSV save failed: {save_error}. Trying JSON format.")
                json_path = Path(output_dir) / 'predictions.json'
                predictions_df.to_json(json_path, orient='records', indent=2)
                predictions_path = json_path
            
            # Verify file was saved
            if not predictions_path.exists() or predictions_path.stat().st_size == 0:
                raise RuntimeError(f"Predictions file was not saved properly: {predictions_path}")
            
            logger.info(f"Predictions saved to: {predictions_path}")
            return str(predictions_path)
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            # Create minimal backup
            try:
                backup_path = Path(output_dir) / 'predictions_summary.txt'
                with open(backup_path, 'w') as f:
                    f.write(f"Prediction Summary\n")
                    f.write(f"Total samples: {len(y_true)}\n")
                    f.write(f"Accuracy: {np.mean(y_true == y_pred):.4f}\n")
                    f.write(f"Error: Failed to save detailed predictions: {e}\n")
                logger.warning(f"Saved prediction summary to: {backup_path}")
                return str(backup_path)
            except Exception as backup_error:
                logger.error(f"Failed to save prediction backup: {backup_error}")
                raise RuntimeError(f"Could not save predictions: {e}") from e
    
    def _cleanup_failed_plot(self, viz_dir: Path, plot_name: str, viz_config: Dict):
        """Clean up partial files from failed plot generation."""
        try:
            plot_formats = viz_config.get('plot_formats', ['png'])
            for fmt in plot_formats:
                plot_file = viz_dir / f'{plot_name}.{fmt}'
                if plot_file.exists():
                    plot_file.unlink()
                    logger.debug(f"Cleaned up partial plot file: {plot_file}")
        except Exception as e:
            logger.warning(f"Failed to clean up partial plot files for {plot_name}: {e}")
    
    def _get_default_embedding_dimension(self, embeddings: List[List[float]]) -> int:
        """Get default embedding dimension based on first valid embedding or common defaults."""
        # Try to find dimension from first valid embedding
        for emb in embeddings:
            if isinstance(emb, list) and len(emb) > 0:
                return len(emb)
        
        # Common embedding dimensions as fallback
        common_dims = [768, 512, 384, 256, 128]
        default_dim = 768  # Most common for transformer models
        
        logger.warning(f"Could not determine embedding dimension, using default: {default_dim}")
        return default_dim