"""
End-to-End Pipeline for complete binary classification workflow.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from models.config_models import PipelineConfig
from models.pipeline_models import PipelineJob, PipelineResult, EndToEndConfig
from models.data_models import Dataset
from utils.logging_utils import get_logger, LogContext
from pipelines.embedding_pipeline import EmbeddingPipeline
from pipelines.classification_pipeline import ClassificationPipeline
from pipelines.evaluation_pipeline import EvaluationPipeline
from evaluation.target_word_evaluator import TargetWordEvaluator
import gc  # Used in cleanup_stage_data but not imported
import pandas as pd  # Used in validate_configuration but not imported
import requests  # Used in validate_configuration but not imported

class EndToEndPipeline:
    """
    Complete end-to-end pipeline for binary classification.
    
    Orchestrates the entire workflow from data input to final reports,
    including embeddings generation, classifier training, evaluation,
    and target word analysis.
    """
    
    def __init__(self, config: PipelineConfig, job_config: PipelineJob):
        """
        Initialize the end-to-end pipeline.
        
        Args:
            config: Pipeline configuration
            job_config: Job-specific configuration
        """
        self.config = config
        self.job_config = job_config
        self.logger = get_logger("end_to_end_pipeline", config.logging)
        
        # Set up output directories
        self.output_structure = self._get_output_structure()
        self._create_output_directories()
        
        # Initialize all stage results to prevent AttributeError
        self.embedding_results = None
        self.classification_results = None
        self.evaluation_results = None
        self.target_word_results = None
        
        # Memory management option
        self.enable_memory_cleanup = getattr(config.job, 'enable_memory_cleanup', False)
        
        # Initialize results tracking
        self.results = PipelineResult(
            job_name=job_config.job_name,
            start_time=datetime.now().isoformat(),
            end_time="",
            total_duration=0.0,
            success=False,
            output_directory=str(Path(job_config.output_base_dir) / job_config.job_name)
        )
    
    def _get_output_structure(self) -> Dict[str, str]:
        """Get output directory structure."""
        base_dir = Path(self.job_config.output_base_dir) / self.job_config.job_name
        return {
            'embeddings': str(base_dir / 'embeddings'),
            'models': str(base_dir / 'models'),
            'metrics': str(base_dir / 'metrics'),
            'summary': str(base_dir / 'summary')
        }
    
    def _create_output_directories(self):
        """Create all required output directories."""
        for dir_path in self.output_structure.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _can_run_stage(self, stage: str) -> bool:
        """
        Check if a stage can run based on dependencies and previous stage results.
        
        Args:
            stage: Stage name to check ('classification', 'evaluation', 'target_word', 'comparison')
            
        Returns:
            True if stage can run, False otherwise
        """
        dependencies = {
            'classification': ['embedding_results'],
            'evaluation': ['classification_results'],
            'target_word': [],  # Independent stage
            'comparison': ['evaluation_results']  # target_word_results is optional
        }
        
        required_deps = dependencies.get(stage, [])
        for dep in required_deps:
            if not hasattr(self, dep) or getattr(self, dep) is None:
                self.logger.warning(f"Cannot run {stage}: missing dependency {dep}")
                return False
            
            # Check if the dependency has an error state
            dep_result = getattr(self, dep)
            if isinstance(dep_result, dict) and dep_result.get('error'):
                self.logger.warning(f"Cannot run {stage}: dependency {dep} failed with error: {dep_result['error']}")
                return False
        
        return True
    
    def _run_stage_safely(self, stage_name: str, stage_func, *args, **kwargs):
        """
        Run a pipeline stage with proper error handling and dependency checking.
        
        Args:
            stage_name: Name of the stage for logging
            stage_func: Function to execute
            *args, **kwargs: Arguments to pass to stage function
            
        Returns:
            Stage result or None if failed/skipped
        """
        # Map stage names to correct result attribute names
        stage_to_attr = {
            'embedding': 'embedding_results',
            'classification': 'classification_results',
            'evaluation': 'evaluation_results',
            'target_word': 'target_word_results'
        }
        attr_name = stage_to_attr.get(stage_name, f"{stage_name}_results")
        
        try:
            # Check dependencies first
            if not self._can_run_stage(stage_name):
                error_result = {
                    'error': f'Stage dependencies not met for {stage_name}', 
                    'success': False,
                    'stage': stage_name,
                    'timestamp': datetime.now().isoformat()
                }
                setattr(self, attr_name, error_result)
                self.logger.warning(f"{stage_name} stage skipped due to unmet dependencies")
                return error_result
            
            self.logger.info(f"Starting {stage_name} stage...")
            result = stage_func(*args, **kwargs)
            
            # Ensure successful results have consistent structure
            if isinstance(result, dict) and 'success' not in result:
                result['success'] = True
                result['stage'] = stage_name
                result['timestamp'] = datetime.now().isoformat()
            
            setattr(self, attr_name, result)
            self.logger.info(f"{stage_name} stage completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"{stage_name} stage failed: {e}", exc_info=True)
            error_result = {
                'error': str(e), 
                'success': False,
                'stage': stage_name,
                'timestamp': datetime.now().isoformat(),
                'error_type': type(e).__name__
            }
            setattr(self, attr_name, error_result)
            
            # Check if we should stop on error (with proper config access)
            # Note: stop_on_error configuration controls whether pipeline continues after stage failure
            # Default: True (stop pipeline), set to False to continue with remaining stages
            stop_on_error = getattr(self.job_config.end_to_end, 'stop_on_error', True)
            if stop_on_error:
                self.logger.critical(f"Pipeline stopped due to {stage_name} failure (stop_on_error=True)")
                raise
            else:
                self.logger.warning(f"Continuing pipeline despite {stage_name} failure (stop_on_error=False)")
            return error_result
    
    def cleanup_stage_data(self, stage: str, keep_summary: bool = True):
        """
        Free memory from completed stage while keeping essential info.
        
        Args:
            stage: Stage name ('embedding', 'classification', 'evaluation', 'target_word')
            keep_summary: Whether to keep a summary of key metrics
        """
        if not self.enable_memory_cleanup:
            return
            
        stage_to_attr = {
            'embedding': 'embedding_results',
            'classification': 'classification_results',
            'evaluation': 'evaluation_results',
            'target_word': 'target_word_results'
        }
        
        attr_name = stage_to_attr.get(stage)
        if not attr_name or not hasattr(self, attr_name):
            return
            
        results = getattr(self, attr_name)
        if keep_summary and isinstance(results, dict):
            # Keep only essential information
            summary = {
                'success': results.get('success', True),
                'error': results.get('error'),
                'stage': results.get('stage'),
                'timestamp': results.get('timestamp'),
                'key_metrics': self._extract_key_metrics(stage, results)
            }
            setattr(self, f"{attr_name}_summary", summary)
            self.logger.info(f"Created summary for {stage} stage, clearing detailed data")
        
        # Clear the large data
        setattr(self, attr_name, None)
        
        # Force garbage collection
        import gc
        gc.collect()
        self.logger.debug(f"Cleaned up memory for {stage} stage")
    
    def _extract_key_metrics(self, stage: str, results: dict) -> dict:
        """Extract essential metrics to keep in memory."""
        if not isinstance(results, dict):
            return {}
            
        key_metrics = {}
        
        if stage == 'embedding':
            # Keep embedding dimension and sample counts
            if 'train' in results:
                key_metrics['n_train'] = results.get('n_train', 0)
                key_metrics['n_test'] = results.get('n_test', 0)
                key_metrics['embedding_dim'] = results.get('train', {}).get('embedding_dim', 0)
            else:
                key_metrics.update({
                    'n_samples': results.get('n_samples', 0),
                    'embedding_dim': results.get('embedding_dim', 0)
                })
                
        elif stage == 'classification':
            # Keep model paths and best scores
            for classifier_type, result in results.items():
                if isinstance(result, dict) and 'best_score' in result:
                    key_metrics[classifier_type] = {
                        'model_path': result.get('model_path'),
                        'best_score': result.get('best_score'),
                        'best_params': result.get('best_params', {})
                    }
                    
        elif stage == 'evaluation':
            # Keep test metrics only
            for classifier_type, result in results.items():
                if isinstance(result, dict) and 'metrics' in result:
                    metrics = result['metrics']
                    key_metrics[classifier_type] = {
                        'accuracy': metrics.get('accuracy'),
                        'roc_auc': metrics.get('roc_auc'),
                        'f1_score': metrics.get('f1_score'),
                        'precision': metrics.get('precision'),
                        'recall': metrics.get('recall')
                    }
                    
        elif stage == 'target_word':
            # Keep main accuracy score
            key_metrics = {
                'accuracy': results.get('accuracy'),
                'n_samples': results.get('n_samples'),
                'target_words_count': len(results.get('target_words', []))
            }
            
        return key_metrics
    
    def validate_configuration(self) -> List[str]:
        """
        Validate pipeline configuration without execution.
        
        Returns:
            List of validation issues found (empty if all valid)
        """
        issues = []
        
        try:
            # Check dataset accessibility
            if self.job_config.dataset_path:
                if not Path(self.job_config.dataset_path).exists():
                    issues.append(f"Dataset not found: {self.job_config.dataset_path}")
                else:
                    # Try to read a few lines to validate format
                    try:
                        df = pd.read_csv(self.job_config.dataset_path, nrows=1)
                        required_cols = ['claims', 'mcid', 'label']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            issues.append(f"Dataset missing required columns: {missing_cols}")
                    except Exception as e:
                        issues.append(f"Cannot read dataset: {e}")
            
            # Check separate train/test datasets if specified
            if self.job_config.train_dataset_path:
                if not Path(self.job_config.train_dataset_path).exists():
                    issues.append(f"Train dataset not found: {self.job_config.train_dataset_path}")
                    
            if self.job_config.test_dataset_path:
                if not Path(self.job_config.test_dataset_path).exists():
                    issues.append(f"Test dataset not found: {self.job_config.test_dataset_path}")
            
            # Check output directory permissions
            try:
                output_dir = Path(self.job_config.output_base_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                test_file = output_dir / '.write_test'
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                issues.append(f"Cannot write to output directory {self.job_config.output_base_dir}: {e}")
            
            # Check model endpoint reachability (basic check)
            if self.job_config.model_endpoint:
                try:
                    import requests
                    # Just check if the base URL is reachable
                    response = requests.get(self.job_config.model_endpoint, timeout=5)
                    if response.status_code >= 500:
                        issues.append(f"Model endpoint returned server error: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    issues.append(f"Cannot connect to model endpoint: {self.job_config.model_endpoint}")
                except requests.exceptions.Timeout:
                    issues.append(f"Timeout connecting to model endpoint: {self.job_config.model_endpoint}")
                except Exception as e:
                    issues.append(f"Error checking model endpoint: {e}")
            
            # Check target words if target word evaluation is enabled
            if self.job_config.end_to_end.target_word_evaluation:
                target_words = getattr(self.job_config.end_to_end, 'target_words', [])
                if not target_words:
                    issues.append("Target word evaluation enabled but no target words specified")
                else:
                    valid_words = [w for w in target_words if w and isinstance(w, str) and w.strip()]
                    if not valid_words:
                        issues.append("No valid target words found (all empty or invalid)")
                    elif len(valid_words) < len(target_words):
                        issues.append(f"Some target words are invalid ({len(target_words) - len(valid_words)} out of {len(target_words)})")
            
            # Check classifier types
            classifier_types = getattr(self.job_config.end_to_end, 'classifier_types', [])
            if self.job_config.end_to_end.train_classifiers and not classifier_types:
                issues.append("Classifier training enabled but no classifier types specified")
            
            # Check split ratio
            if self.job_config.dataset_path and hasattr(self.job_config, 'split_ratio'):
                split_ratio = self.job_config.split_ratio
                if not 0.01 <= split_ratio <= 0.99:
                    issues.append(f"Split ratio must be between 0.01 and 0.99, got {split_ratio}")
                    
        except Exception as e:
            issues.append(f"Validation failed with error: {e}")
            
        return issues
    
    def run(self, progress_callback=None) -> PipelineResult:
        """
        Execute the complete end-to-end pipeline.
        
        Args:
            progress_callback: Optional callable(stage_name, percent_complete, message)
            
        Returns:
            Complete pipeline results with all outputs and metrics
        """
        start_time = time.time()
        
        # Validate configuration first if requested
        if getattr(self.config.job, 'validate_before_run', True):
            validation_issues = self.validate_configuration()
            if validation_issues:
                error_msg = f"Configuration validation failed: {'; '.join(validation_issues)}"
                self.logger.error(error_msg)
                self.results.success = False
                self.results.end_time = datetime.now().isoformat()
                self.results.total_duration = time.time() - start_time
                raise ValueError(error_msg)
        
        try:
            with LogContext(self.logger, "end_to_end_pipeline", 
                           job_name=self.job_config.job_name) as ctx:
                
                # Initialize result attributes
                self.results.best_classifier = None
                self.results.best_embedding_score = None
                self.results.target_word_score = None
                
                # Stage 1: Generate Embeddings
                if self.job_config.end_to_end.generate_embeddings:
                    if progress_callback:
                        progress_callback("embedding", 0, "Starting embedding generation...")
                    
                    stage_start = time.time()
                    embedding_results = self._run_stage_safely("embedding", self._run_embedding_generation)
                    stage_duration = time.time() - stage_start
                    
                    if embedding_results and not embedding_results.get('error'):
                        ctx.log_metrics({
                            "embeddings_generated": True, 
                            "embedding_stage_success": True,
                            "embedding_duration": stage_duration
                        })
                        if progress_callback:
                            progress_callback("embedding", 100, "Embedding generation completed")
                    else:
                        ctx.log_metrics({
                            "embeddings_generated": False, 
                            "embedding_stage_success": False,
                            "embedding_duration": stage_duration,
                            "embedding_error": embedding_results.get('error', 'Unknown error') if embedding_results else 'No result'
                        })
                        if progress_callback:
                            progress_callback("embedding", 100, f"Embedding generation failed: {embedding_results.get('error', 'Unknown error') if embedding_results else 'No result'}")
                    
                    # Clean up memory if enabled
                    if self.enable_memory_cleanup and embedding_results and not embedding_results.get('error'):
                        self.cleanup_stage_data("embedding")
                
                # Stage 2: Train Classifiers  
                if self.job_config.end_to_end.train_classifiers:
                    if progress_callback:
                        progress_callback("classification", 0, "Starting classifier training...")
                    
                    stage_start = time.time()
                    classification_results = self._run_stage_safely("classification", self._run_classification_training)
                    stage_duration = time.time() - stage_start
                    
                    if classification_results and not classification_results.get('error'):
                        ctx.log_metrics({
                            "classifiers_trained": len(classification_results), 
                            "classification_stage_success": True,
                            "classification_duration": stage_duration
                        })
                        if progress_callback:
                            progress_callback("classification", 100, f"Trained {len(classification_results)} classifiers")
                        # Clean up memory if enabled
                        if self.enable_memory_cleanup:
                            self.cleanup_stage_data("classification")
                    else:
                        ctx.log_metrics({
                            "classifiers_trained": 0, 
                            "classification_stage_success": False,
                            "classification_duration": stage_duration,
                            "classification_error": classification_results.get('error', 'Unknown error') if classification_results else 'No result'
                        })
                        if progress_callback:
                            progress_callback("classification", 100, f"Classification training failed")
                
                # Stage 3: Evaluate Models
                if self.job_config.end_to_end.evaluate_models:
                    if progress_callback:
                        progress_callback("evaluation", 0, "Starting model evaluation...")
                    
                    stage_start = time.time()
                    evaluation_results = self._run_stage_safely("evaluation", self._run_model_evaluation)
                    stage_duration = time.time() - stage_start
                    
                    if evaluation_results and not evaluation_results.get('error'):
                        ctx.log_metrics({
                            "models_evaluated": len(evaluation_results), 
                            "evaluation_stage_success": True,
                            "evaluation_duration": stage_duration
                        })
                        if progress_callback:
                            progress_callback("evaluation", 100, f"Evaluated {len(evaluation_results)} models")
                        # Clean up memory if enabled (keep this stage for comparison)
                        if self.enable_memory_cleanup:
                            self.cleanup_stage_data("evaluation", keep_summary=True)
                    else:
                        ctx.log_metrics({
                            "models_evaluated": 0, 
                            "evaluation_stage_success": False,
                            "evaluation_duration": stage_duration,
                            "evaluation_error": evaluation_results.get('error', 'Unknown error') if evaluation_results else 'No result'
                        })
                        if progress_callback:
                            progress_callback("evaluation", 100, "Model evaluation failed")
                
                # Stage 4: Target Word Evaluation
                if self.job_config.end_to_end.target_word_evaluation:
                    if progress_callback:
                        progress_callback("target_word", 0, "Starting target word evaluation...")
                    
                    stage_start = time.time()
                    target_word_results = self._run_stage_safely("target_word", self._run_target_word_evaluation)
                    stage_duration = time.time() - stage_start
                    
                    if target_word_results and not target_word_results.get('error'):
                        target_words = getattr(self.job_config.end_to_end, 'target_words', [])
                        ctx.log_metrics({
                            "target_words_evaluated": len(target_words), 
                            "target_word_stage_success": True,
                            "target_word_duration": stage_duration
                        })
                        if progress_callback:
                            progress_callback("target_word", 100, f"Evaluated {len(target_words)} target words")
                    else:
                        ctx.log_metrics({
                            "target_words_evaluated": 0, 
                            "target_word_stage_success": False,
                            "target_word_duration": stage_duration,
                            "target_word_error": target_word_results.get('error', 'Unknown error') if target_word_results else 'No result'
                        })
                        if progress_callback:
                            progress_callback("target_word", 100, "Target word evaluation failed")
                
                # Stage 5: Generate Summary Report
                if self.job_config.end_to_end.create_summary_report:
                    self.logger.info("Stage 5: Creating summary report...")
                    self._generate_summary_report()
                
                # Stage 6: Compare Methods
                if self.job_config.end_to_end.compare_methods:
                    self.logger.info("Stage 6: Comparing methods...")
                    comparison = self._compare_methods()
                    self._save_comparison_report(comparison)
                
                # Finalize results
                end_time = time.time()
                self.results.end_time = datetime.now().isoformat()
                self.results.total_duration = end_time - start_time
                self.results.success = True
                
                self.logger.info(f"Pipeline completed successfully in {self.results.total_duration:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.results.success = False
            self.results.error_message = str(e)
            self.results.end_time = datetime.now().isoformat()
            self.results.total_duration = time.time() - start_time
            raise
        
        return self.results
    
    def _run_embedding_generation(self) -> Dict[str, Any]:
        """Run embedding generation stage."""
        pipeline = EmbeddingPipeline(self.config)
        
        # Determine input strategy
        if self.job_config.dataset_path:
            # Single dataset with split (split_ratio triggers auto-splitting)
            results = pipeline.run(
                dataset_path=self.job_config.dataset_path,
                output_dir=self.output_structure['embeddings'],
                model_endpoint=self.job_config.model_endpoint,
                split_ratio=self.job_config.split_ratio
            )
        else:
            # Separate train/test datasets
            train_results = pipeline.run(
                dataset_path=self.job_config.train_dataset_path,
                output_dir=self.output_structure['embeddings'],
                output_prefix="train",
                model_endpoint=self.job_config.model_endpoint
            )
            
            test_results = pipeline.run(
                dataset_path=self.job_config.test_dataset_path,
                output_dir=self.output_structure['embeddings'],
                output_prefix="test",
                model_endpoint=self.job_config.model_endpoint
            )
            
            results = {
                'train': train_results,
                'test': test_results
            }
        
        return results
    
    def _run_classification_training(self) -> Dict[str, Any]:
        """Run classification training for all specified classifiers."""
        pipeline = ClassificationPipeline(self.config)
        results = {}
        
        # Get embedding paths - check config first, then look for files
        train_embeddings = None
        test_embeddings = None
        
        # Option 1: Use paths from config if provided
        if (hasattr(self.job_config, 'train_embeddings_path') and 
            hasattr(self.job_config, 'test_embeddings_path') and
            self.job_config.train_embeddings_path and 
            self.job_config.test_embeddings_path):
            
            train_embeddings = self.job_config.train_embeddings_path
            test_embeddings = self.job_config.test_embeddings_path
            self.logger.info(f"Using embedding paths from config: {train_embeddings}, {test_embeddings}")
            
            # Validate files exist
            if not Path(train_embeddings).exists():
                raise FileNotFoundError(f"Train embeddings not found: {train_embeddings}")
            if not Path(test_embeddings).exists():
                raise FileNotFoundError(f"Test embeddings not found: {test_embeddings}")
        
        # Option 2: Look for generated files in output directory
        else:
            embeddings_dir = Path(self.output_structure['embeddings'])
            
            # Check for specific formats based on config
            output_format = self.config.data_processing.output_format.lower()
            format_extensions = {
                'json': '.json',
                'csv': '.csv', 
                'parquet': '.parquet',
                'numpy': '.npy'
            }
            
            expected_ext = format_extensions.get(output_format, '.json')
            train_embeddings_file = embeddings_dir / f'train_embeddings{expected_ext}'
            test_embeddings_file = embeddings_dir / f'test_embeddings{expected_ext}'
            
            # First try specific format, then fall back to glob pattern
            if train_embeddings_file.exists() and test_embeddings_file.exists():
                train_embeddings = str(train_embeddings_file)
                test_embeddings = str(test_embeddings_file)
                self.logger.info(f"Using embedding files with expected format {output_format}: {train_embeddings}, {test_embeddings}")
            else:
                # Fallback to glob pattern
                train_files = list(embeddings_dir.glob('train_embeddings.*'))
                test_files = list(embeddings_dir.glob('test_embeddings.*'))
                
                if not train_files or not test_files:
                    train_file_names = ', '.join([f.name for f in train_files]) if train_files else 'none'
                    test_file_names = ', '.join([f.name for f in test_files]) if test_files else 'none'
                    
                    raise FileNotFoundError(
                        f"Could not find embedding files in {embeddings_dir}. "
                        f"Expected train_embeddings{expected_ext} and test_embeddings{expected_ext} files "
                        f"or provide train_embeddings_path/test_embeddings_path in config. "
                        f"Found train files: {train_file_names}, "
                        f"test files: {test_file_names}"
                    )
                
                train_embeddings = str(train_files[0])
                test_embeddings = str(test_files[0])
                self.logger.info(f"Using found embedding files: {train_embeddings}, {test_embeddings}")
        
        # Train each classifier type
        for classifier_type in self.job_config.end_to_end.classifier_types:
            self.logger.info(f"Training {classifier_type}...")
            
            try:
                result = pipeline.run(
                    train_embeddings=train_embeddings,
                    test_embeddings=test_embeddings,
                    classifier_type=classifier_type,
                    output_dir=self.output_structure['models']
                )
                results[classifier_type] = result
                
            except Exception as e:
                self.logger.error(f"Failed to train {classifier_type}: {e}")
                results[classifier_type] = {'error': str(e)}
        
        return results
    
    def _run_model_evaluation(self) -> Dict[str, Any]:
        """Run detailed evaluation for all trained models."""
        pipeline = EvaluationPipeline(self.config)
        results = {}
        
        # Get test embeddings path - check config first, then look for files
        test_embeddings = None
        
        # Option 1: Use path from config if provided
        if (hasattr(self.job_config, 'test_embeddings_path') and 
            self.job_config.test_embeddings_path):
            test_embeddings = self.job_config.test_embeddings_path
            self.logger.info(f"Using test embeddings from config: {test_embeddings}")
            
            # Validate file exists
            if not Path(test_embeddings).exists():
                raise FileNotFoundError(f"Test embeddings not found: {test_embeddings}")
        
        # Option 2: Look for generated file in output directory
        else:
            embeddings_dir = Path(self.output_structure['embeddings'])
            test_files = list(embeddings_dir.glob('test_embeddings.*'))
            
            if not test_files:
                raise FileNotFoundError(
                    f"Could not find test embeddings in {embeddings_dir}. "
                    f"Expected test_embeddings.* file or provide test_embeddings_path in config."
                )
            
            test_embeddings = str(test_files[0])
            self.logger.info(f"Using generated test embeddings: {test_embeddings}")
        
        # Evaluate each model (trained in current run or pre-trained from config)
        for classifier_type in self.job_config.end_to_end.classifier_types:
            model_path = None
            
            # Option 1: Use model trained in current pipeline run
            if hasattr(self, 'classification_results') and classifier_type in self.classification_results:
                classification_result = self.classification_results[classifier_type]
                if 'model_path' in classification_result:
                    model_path = classification_result['model_path']
                    self.logger.info(f"Evaluating {classifier_type} model trained in current run...")
            
            # Option 2: Use pre-trained model from config
            elif self.job_config.model_paths and classifier_type in self.job_config.model_paths:
                model_path = self.job_config.model_paths[classifier_type]
                self.logger.info(f"Evaluating pre-trained {classifier_type} model from config: {model_path}")
                
                # Validate model file exists
                if not Path(model_path).exists():
                    self.logger.error(f"Pre-trained model not found: {model_path}")
                    continue
            
            # Option 3: Look for model in models directory from config
            elif self.job_config.models_directory:
                models_dir = Path(self.job_config.models_directory)
                potential_files = list(models_dir.glob(f"{classifier_type}_model.*"))
                
                if potential_files:
                    model_path = str(potential_files[0])
                    self.logger.info(f"Found {classifier_type} model in directory: {model_path}")
                else:
                    self.logger.warning(f"No {classifier_type} model found in {models_dir}")
                    continue
            
            # Evaluate the model if we found one
            if model_path:
                try:
                    # Create classifier-specific output directory
                    eval_output_dir = Path(self.output_structure['metrics']) / classifier_type
                    eval_output_dir.mkdir(exist_ok=True)
                    
                    result = pipeline.run(
                        model_path=model_path,
                        test_embeddings=test_embeddings,
                        output_dir=str(eval_output_dir)
                    )
                    results[classifier_type] = result
                        
                except Exception as e:
                    self.logger.error(f"Failed to evaluate {classifier_type}: {e}")
                    results[classifier_type] = {'error': str(e)}
        
        return results
    
    def _run_target_word_evaluation(self) -> Dict[str, Any]:
        """Run target word evaluation with comprehensive validation."""
        evaluator = TargetWordEvaluator(self.config)
        
        # Get dataset path with validation
        dataset_path = (self.job_config.dataset_path or 
                       self.job_config.test_dataset_path)
        
        # Validation
        if not dataset_path:
            raise ValueError("No dataset path available for target word evaluation")
        
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        # Validate target words are provided and valid
        target_words = getattr(self.job_config.end_to_end, 'target_words', [])
        if not target_words:
            raise ValueError("No target words specified for target word evaluation")
        
        # Validate target words are non-empty strings
        valid_target_words = []
        for i, word in enumerate(target_words):
            if not word or not isinstance(word, str) or not word.strip():
                self.logger.warning(f"Skipping invalid target word at index {i}: {repr(word)}")
                continue
            valid_target_words.append(word.strip())
        
        if not valid_target_words:
            raise ValueError("No valid target words found (all were empty or invalid)")
        
        try:
            results = evaluator.evaluate(
                dataset_path=dataset_path,
                target_words=valid_target_words,  # Use the validated list
                n_samples=self.job_config.end_to_end.target_word_samples,
                max_tokens=self.job_config.end_to_end.target_word_max_tokens,
                model_endpoint=self.job_config.model_endpoint
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Target word evaluation failed: {e}")
            return {'error': str(e)}
    
    def _find_best_classifier(self) -> Tuple[str, float]:
        """Find the best performing classifier based on test metrics."""
        best_classifier = None
        best_score = 0.0
        
        if hasattr(self, 'evaluation_results'):
            for classifier_type, eval_result in self.evaluation_results.items():
                if 'metrics' in eval_result and 'roc_auc' in eval_result['metrics']:
                    score = eval_result['metrics']['roc_auc']
                    if score > best_score:
                        best_score = score
                        best_classifier = classifier_type
        
        return best_classifier, best_score
    
    def _compare_methods(self) -> Dict[str, Any]:
        """
        Compare embedding-based vs target word methods using consistent metrics.
        
        Note: Uses accuracy for both methods since target word evaluation only provides accuracy.
        ROC-AUC comparison would be more appropriate but requires target word method to provide
        probability scores rather than binary predictions.
        """
        # Find best embedding method using accuracy for fair comparison
        best_classifier, best_score_roc = self._find_best_classifier()
        
        # Get embedding method accuracy score for fair comparison
        best_score_accuracy = 0.0
        if (best_classifier and 
            hasattr(self, 'evaluation_results') and 
            self.evaluation_results and 
            isinstance(self.evaluation_results, dict) and
            best_classifier in self.evaluation_results and
            self.evaluation_results[best_classifier] and
            'metrics' in self.evaluation_results[best_classifier]):
            metrics = self.evaluation_results[best_classifier]['metrics']
            best_score_accuracy = metrics.get('accuracy', 0.0)
        
        # Get target word method score
        target_word_score = 0.0
        if (hasattr(self, 'target_word_results') and self.target_word_results and 
            'accuracy' in self.target_word_results):
            target_word_score = self.target_word_results['accuracy']
        
        # Use accuracy for comparison (both methods provide this metric)
        best_score = best_score_accuracy
        
        # Create comparison with safe access
        embedding_metrics = {}
        if (best_classifier and 
            hasattr(self, 'evaluation_results') and 
            self.evaluation_results and 
            isinstance(self.evaluation_results, dict) and
            best_classifier in self.evaluation_results and
            self.evaluation_results[best_classifier] and
            'metrics' in self.evaluation_results[best_classifier]):
            embedding_metrics = self.evaluation_results[best_classifier]['metrics']
        
        target_word_metrics = getattr(self, 'target_word_results', {}) or {}
        
        # Determine better method and calculate improvement safely
        if best_score == 0 and target_word_score == 0:
            better_method = "tie"
            improvement = 0.0
        elif best_score == target_word_score:
            better_method = "tie"
            improvement = 0.0
        else:
            better_method = "embedding" if best_score > target_word_score else "target_word"
            denominator = max(best_score, target_word_score)
            if denominator == 0:
                improvement = 100.0  # One method has 100% improvement over 0
            else:
                improvement = abs(best_score - target_word_score) / denominator * 100
        
        # Generate recommendations
        recommendations = []
        if best_score > target_word_score + 0.05:
            recommendations.append("Embedding-based method shows significantly better performance")
            recommendations.append("Consider using the trained classifier for production")
        elif target_word_score > best_score + 0.05:
            recommendations.append("Target word method performs better")
            recommendations.append("Consider expanding the target word list")
        else:
            recommendations.append("Both methods show similar performance")
            recommendations.append("Consider ensemble approach or domain-specific evaluation")
        
        # Per-classifier comparison with safe access
        per_classifier = {}
        if (hasattr(self, 'evaluation_results') and 
            self.evaluation_results and 
            isinstance(self.evaluation_results, dict)):
            for classifier_type, eval_result in self.evaluation_results.items():
                if eval_result and isinstance(eval_result, dict) and 'metrics' in eval_result:
                    per_classifier[classifier_type] = {
                        'roc_auc': eval_result['metrics'].get('roc_auc', 0.0),
                        'accuracy': eval_result['metrics'].get('accuracy', 0.0),
                        'f1_score': eval_result['metrics'].get('f1_score', 0.0)
                    }
        
        return {
            'embedding_method': {
                'best_classifier': best_classifier, 
                'accuracy': best_score,  # Used for comparison (consistent with target word method)
                'roc_auc': best_score_roc,  # ROC-AUC score (different metric)
                'metrics': embedding_metrics  # Full metrics for reference
            },
            'target_word_method': {
                'accuracy': target_word_score,
                'metrics': target_word_metrics  # Full metrics for reference
            },
            'comparison_metric': 'accuracy',  # Document which metric was used for comparison
            'better_method': better_method,
            'improvement': improvement,
            'recommendations': recommendations,
            'per_classifier_comparison': per_classifier
        }
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        summary = {
            'job_config': self.job_config.dict(),
            'pipeline_results': self.results.dict(),
            'best_classifier': None,
            'best_score': 0.0,
            'target_word_score': 0.0,
            'summary_metrics': {},
            'recommendations': []
        }
        
        # Add best classifier info
        if hasattr(self, 'evaluation_results'):
            best_classifier, best_score = self._find_best_classifier()
            summary['best_classifier'] = best_classifier
            summary['best_score'] = best_score
            self.results.best_classifier = best_classifier
            self.results.best_embedding_score = best_score
        
        # Add target word score
        if hasattr(self, 'target_word_results') and self.target_word_results:
            target_score = self.target_word_results.get('accuracy', 0.0)
            summary['target_word_score'] = target_score
            self.results.target_word_score = target_score
        
        # Save summary
        summary_path = Path(self.output_structure['summary']) / 'pipeline_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary report saved to: {summary_path}")
    
    def _save_comparison_report(self, comparison: Dict[str, Any]):
        """Save method comparison report."""
        comparison_path = Path(self.output_structure['summary']) / 'method_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        self.logger.info(f"Comparison report saved to: {comparison_path}")
        
        # Log key findings
        self.logger.info(f"Best method: {comparison['better_method']}")
        self.logger.info(f"Performance improvement: {comparison['improvement']:.2f}%")
        for rec in comparison['recommendations']:
            self.logger.info(f"Recommendation: {rec}")