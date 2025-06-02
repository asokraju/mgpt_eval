#!/usr/bin/env python3
"""
Main entry point for the Binary Classifier Pipeline using MediClaimGPT embeddings.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import traceback
import pandas as pd

# Add the parent directory to the path to import from pipelines
sys.path.append(str(Path(__file__).parent))

from models.config_models import PipelineConfig
from models.data_models import Dataset
from models.pipeline_models import PipelineJob, EndToEndConfig
from utils.logging_utils import get_logger, LogContext


def setup_pipeline(config_path: str) -> tuple[PipelineConfig, object]:
    """Setup pipeline configuration and logging."""
    # Load configuration
    try:
        config = PipelineConfig.from_yaml(config_path)
        config.setup_directories()
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)
    
    # Setup logger
    logger = get_logger("mgpt_eval", config.logging)
    logger.log_config(config.model_dump())
    
    return config, logger


def get_config_value(config: PipelineConfig, config_path: str, fallback=None):
    """Get a value from nested config structure using dot notation."""
    keys = config_path.split('.')
    value = config
    try:
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            else:
                return fallback
        return value
    except (AttributeError, KeyError):
        return fallback


def load_target_words_from_file(filepath: str) -> List[str]:
    """Load target words from file, one per line, ignoring comments."""
    target_words = []
    codes_file = Path(filepath)
    
    if not codes_file.exists():
        raise FileNotFoundError(f"Target codes file not found: {filepath}")
    
    with open(codes_file, 'r') as f:
        for line in f:
            code = line.strip()
            if code and not code.startswith('#'):
                target_words.append(code)
    
    if not target_words:
        raise ValueError(f"No valid target words found in {filepath}")
    
    return target_words


def safe_get_config_value(config: PipelineConfig, path: str, default=None):
    """Safely get nested config value using dot notation, supporting list indexing."""
    try:
        value = config
        for key in path.split('.'):
            # Handle list indexing (e.g., "0", "1", etc.)
            if key.isdigit():
                index = int(key)
                if isinstance(value, (list, tuple)) and 0 <= index < len(value):
                    value = value[index]
                else:
                    return default
            # Handle attribute access
            elif hasattr(value, key):
                value = getattr(value, key)
            else:
                return default
        return value if value is not None else default
    except (AttributeError, KeyError, IndexError, ValueError):
        return default


def setup_argument_parser_with_config(config: PipelineConfig) -> argparse.ArgumentParser:
    """Setup argument parser with config file values as defaults."""
    # Extract config values for easier access
    input_config = getattr(config, 'input', {})
    job_config = getattr(config, 'job', {})
    pipeline_config = getattr(config, 'pipeline_stages', {})
    target_eval_config = getattr(config, 'target_word_evaluation', {})
    
    parser = argparse.ArgumentParser(
        description="Binary Classifier Pipeline using MediClaimGPT Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings from single dataset
  python main.py generate-embeddings --dataset-path data.csv --output-dir outputs/

  # Generate embeddings from separate train/test files
  python main.py generate-embeddings --train-dataset train.csv --test-dataset test.csv --output-dir outputs/

  # Train classifier
  python main.py train-classifier --train-embeddings train.json --test-embeddings test.json --output-dir models/

  # Evaluate model
  python main.py evaluate --model-path model.pkl --test-embeddings test.json --output-dir metrics/
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default_config.yaml',
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline commands')
    
    # Generate embeddings command
    embed_parser = subparsers.add_parser('generate-embeddings', 
                                        help='Generate embeddings from dataset')
    # Option 1: Single dataset (will be split based on config)
    embed_parser.add_argument('--dataset-path', 
                             default=getattr(input_config, 'dataset_path', None),
                             help='Path to single dataset file (will be split based on config)')
    # Option 2: Separate train/test datasets
    embed_parser.add_argument('--train-dataset', 
                             default=getattr(input_config, 'train_dataset_path', None),
                             help='Path to training dataset file')
    embed_parser.add_argument('--test-dataset', 
                             default=getattr(input_config, 'test_dataset_path', None),
                             help='Path to test dataset file')
    # Common arguments
    embed_parser.add_argument('--output-dir', 
                             default=getattr(job_config, 'output_base_dir', 'outputs'),
                             help='Output directory for embeddings')
    embed_parser.add_argument('--model-endpoint', 
                             default=config.model_api.base_url,
                             help='Model API endpoint (overrides config)')
    embed_parser.add_argument('--resume', action='store_true', 
                             help='Resume from checkpoint')
    embed_parser.add_argument('--split-ratio', type=float, 
                             default=getattr(input_config, 'split_ratio', 0.8),
                             help='Override train/test split ratio from config')
    
    # Train classifier command
    train_parser = subparsers.add_parser('train-classifier', 
                                        help='Train binary classifier')
    train_parser.add_argument('--train-embeddings', required=True, 
                             help='Path to training embeddings')
    train_parser.add_argument('--test-embeddings', required=True, 
                             help='Path to test embeddings')
    train_parser.add_argument('--classifier-type', 
                             default=safe_get_config_value(config, 'classification.classifier_types.0', 'logistic_regression'),
                             choices=['logistic_regression', 'svm', 'random_forest'],
                             help='Type of classifier to train')
    train_parser.add_argument('--output-dir', 
                             default=getattr(job_config, 'output_base_dir', 'outputs'),
                             help='Output directory for model')
    
    # Evaluate model command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', required=True, 
                            help='Path to trained model')
    eval_parser.add_argument('--test-embeddings', required=True, 
                            help='Path to test embeddings')
    eval_parser.add_argument('--output-dir', 
                            default=getattr(job_config, 'output_base_dir', 'outputs'),
                            help='Output directory for metrics')
    
    # Target word evaluation command
    target_parser = subparsers.add_parser('evaluate-target-word', 
                                         help='Evaluate using target word presence')
    dataset_path_default = getattr(input_config, 'dataset_path', None)
    target_parser.add_argument('--dataset-path', 
                              default=dataset_path_default,
                              required=(dataset_path_default is None),
                              help='Path to input dataset')
    target_parser.add_argument('--target-words', nargs='+',
                              default=getattr(target_eval_config, 'target_codes', None),
                              help='Target medical codes to search for (space-separated)')
    target_parser.add_argument('--target-codes-file', type=str,
                              default=getattr(target_eval_config, 'target_codes_file', None),
                              help='Path to file containing target codes (one per line)')
    target_parser.add_argument('--n-samples', type=int, 
                              default=getattr(target_eval_config, 'n_generations', 10),
                              help='Number of generations per prompt')
    target_parser.add_argument('--max-tokens', type=int, 
                              default=getattr(target_eval_config, 'max_new_tokens', 200),
                              help='Maximum tokens to generate')
    target_parser.add_argument('--model-endpoint', 
                              default=config.model_api.base_url,
                              help='Model API endpoint (overrides config)')
    
    # End-to-end pipeline command
    e2e_parser = subparsers.add_parser('run-all', 
                                      help='Run complete end-to-end pipeline')
    
    # Data inputs (same validation as generate-embeddings)
    e2e_parser.add_argument('--dataset-path', 
                           default=getattr(input_config, 'dataset_path', None),
                           help='Path to single dataset file (will be split)')
    e2e_parser.add_argument('--train-dataset', 
                           default=getattr(input_config, 'train_dataset_path', None),
                           help='Path to training dataset file')
    e2e_parser.add_argument('--test-dataset', 
                           default=getattr(input_config, 'test_dataset_path', None),
                           help='Path to test dataset file')
    
    # Job configuration
    output_dir_default = getattr(job_config, 'output_base_dir', 'outputs')
    e2e_parser.add_argument('--output-dir', 
                           default=output_dir_default,
                           required=(output_dir_default is None),
                           help='Base output directory')
    e2e_parser.add_argument('--job-name', 
                           default=getattr(job_config, 'job_name', 'pipeline_job'),
                           help='Job name for organizing outputs')
    e2e_parser.add_argument('--model-endpoint', 
                           default=config.model_api.base_url,
                           help='Model API endpoint (overrides config)')
    e2e_parser.add_argument('--split-ratio', type=float,
                           default=getattr(input_config, 'split_ratio', 0.8),
                           help='Train/test split ratio (overrides config)')
    
    # Pipeline stages (defaults from config)
    pipeline_default_generate = getattr(pipeline_config, 'generate_embeddings', True)
    pipeline_default_train = getattr(pipeline_config, 'train_classifiers', True)
    pipeline_default_evaluate = getattr(pipeline_config, 'evaluate_models', True)
    pipeline_default_target = getattr(pipeline_config, 'target_word_evaluation', True)
    pipeline_default_summary = getattr(pipeline_config, 'create_summary_report', True)
    pipeline_default_compare = getattr(pipeline_config, 'compare_methods', True)
    
    e2e_parser.add_argument('--no-embeddings', action='store_false', dest='generate_embeddings',
                           help='Skip embedding generation')
    e2e_parser.add_argument('--no-training', action='store_false', dest='train_classifiers',
                           help='Skip classifier training')
    e2e_parser.add_argument('--no-evaluation', action='store_false', dest='evaluate_models',
                           help='Skip model evaluation')
    e2e_parser.add_argument('--no-target-words', action='store_false', dest='target_word_evaluation',
                           help='Skip target word evaluation')
    e2e_parser.add_argument('--no-summary', action='store_false', dest='create_summary_report',
                           help='Skip summary report generation')
    e2e_parser.add_argument('--no-comparison', action='store_false', dest='compare_methods',
                           help='Skip method comparison')
    
    # Classifier selection
    e2e_parser.add_argument('--classifier-types', nargs='+',
                           default=safe_get_config_value(config, 'classification.classifier_types', ['logistic_regression']),
                           choices=['logistic_regression', 'svm', 'random_forest'],
                           help='Classifier types to train')
    
    # Target word configuration
    e2e_parser.add_argument('--target-words', nargs='+',
                           default=getattr(target_eval_config, 'target_codes', None),
                           help='Target words for alternative evaluation')
    e2e_parser.add_argument('--target-codes-file', type=str,
                           default=getattr(target_eval_config, 'target_codes_file', None),
                           help='Path to file containing target codes (one per line)')
    e2e_parser.add_argument('--target-word-samples', type=int, 
                           default=getattr(target_eval_config, 'n_generations', 10),
                           help='Number of generations per prompt for target word eval')
    e2e_parser.add_argument('--target-word-max-tokens', type=int, 
                           default=getattr(target_eval_config, 'max_new_tokens', 200),
                           help='Max tokens for target word evaluation')
    
    # Set defaults for boolean flags from config
    e2e_parser.set_defaults(
        generate_embeddings=pipeline_default_generate,
        train_classifiers=pipeline_default_train,
        evaluate_models=pipeline_default_evaluate,
        target_word_evaluation=pipeline_default_target,
        create_summary_report=pipeline_default_summary,
        compare_methods=pipeline_default_compare
    )
    
    return parser


def validate_embedding_args(args) -> Optional[str]:
    """Validate arguments for generate-embeddings command."""
    # Check if we have any input at all
    has_single_dataset = args.dataset_path is not None
    has_train_dataset = args.train_dataset is not None
    has_test_dataset = args.test_dataset is not None
    
    if has_single_dataset and (has_train_dataset or has_test_dataset):
        return "Cannot specify both --dataset-path and --train-dataset/--test-dataset"
    elif not has_single_dataset and not (has_train_dataset and has_test_dataset):
        return "Must specify either --dataset-path OR both --train-dataset and --test-dataset"
    elif (has_train_dataset and not has_test_dataset) or (has_test_dataset and not has_train_dataset):
        return "When using separate datasets, both --train-dataset and --test-dataset must be provided"
    return None


def generate_embeddings(args, config: PipelineConfig, logger) -> dict:
    """Generate embeddings for the dataset."""
    from pipelines.embedding_pipeline import EmbeddingPipeline
    
    with LogContext(logger, "embedding_generation", 
                   config_path=args.config) as ctx:
        
        pipeline = EmbeddingPipeline(config)
        
        # Check if separate train/test files are provided
        if args.train_dataset and args.test_dataset:
            logger.info("Using separate train and test datasets")
            
            # Validate datasets
            try:
                Dataset.from_file(args.train_dataset)
                Dataset.from_file(args.test_dataset)
            except Exception as e:
                logger.error(f"Dataset validation failed: {e}")
                raise
            
            # Generate embeddings for train dataset
            train_results = pipeline.run(
                dataset_path=args.train_dataset,
                output_dir=args.output_dir,
                output_prefix="train",
                model_endpoint=args.model_endpoint,
                resume=args.resume
            )
            
            # Generate embeddings for test dataset
            test_results = pipeline.run(
                dataset_path=args.test_dataset,
                output_dir=args.output_dir,
                output_prefix="test",
                model_endpoint=args.model_endpoint,
                resume=args.resume
            )
            
            results = {
                'train': train_results,
                'test': test_results
            }
            logger.info(f"Train embeddings saved to: {train_results['output_path']}")
            logger.info(f"Test embeddings saved to: {test_results['output_path']}")
        else:
            # Single dataset - will be split based on config
            logger.info("Using single dataset with automatic train/test split")
            
            # Validate dataset
            try:
                dataset = Dataset.from_file(args.dataset_path)
                logger.info(f"Validated dataset with {len(dataset.records)} records")
            except Exception as e:
                logger.error(f"Dataset validation failed: {e}")
                raise
            
            results = pipeline.run(
                dataset_path=args.dataset_path,
                output_dir=args.output_dir,
                model_endpoint=args.model_endpoint,
                resume=args.resume,
                split_ratio=args.split_ratio
            )
            
            if 'train' in results and 'test' in results:
                logger.info(f"Train embeddings saved to: {results['train']['output_path']}")
                logger.info(f"Test embeddings saved to: {results['test']['output_path']}")
            else:
                logger.info(f"Embeddings saved to: {results['output_path']}")
        
        ctx.log_metrics({'results': results})
        return results


def train_classifier(args, config: PipelineConfig, logger) -> dict:
    """Train binary classifier on embeddings."""
    from pipelines.classification_pipeline import ClassificationPipeline
    
    with LogContext(logger, "classification_training") as ctx:
        
        # Validate required arguments
        if not args.train_embeddings or not Path(args.train_embeddings).exists():
            logger.error(f"Train embeddings file not found: {args.train_embeddings}")
            raise FileNotFoundError(f"Train embeddings file not found: {args.train_embeddings}")
            
        if not args.test_embeddings or not Path(args.test_embeddings).exists():
            logger.error(f"Test embeddings file not found: {args.test_embeddings}")
            raise FileNotFoundError(f"Test embeddings file not found: {args.test_embeddings}")
        
        logger.info(f"Training {args.classifier_type} classifier")
        
        pipeline = ClassificationPipeline(config)
        
        results = pipeline.run(
            train_embeddings=args.train_embeddings,
            test_embeddings=args.test_embeddings,
            classifier_type=args.classifier_type,
            output_dir=args.output_dir
        )
        
        logger.info(f"Model saved to: {results['model_path']}")
        ctx.log_metrics({
            'best_score': results['best_score'],
            'test_metrics': results['test_metrics']
        })
        
        return results


def evaluate_model(args, config: PipelineConfig, logger) -> dict:
    """Evaluate trained model."""
    from pipelines.evaluation_pipeline import EvaluationPipeline
    
    with LogContext(logger, "model_evaluation") as ctx:
        
        # Validate required arguments
        if not args.model_path or not Path(args.model_path).exists():
            logger.error(f"Model file not found: {args.model_path}")
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
            
        if not args.test_embeddings or not Path(args.test_embeddings).exists():
            logger.error(f"Test embeddings file not found: {args.test_embeddings}")
            raise FileNotFoundError(f"Test embeddings file not found: {args.test_embeddings}")
        
        pipeline = EvaluationPipeline(config)
        
        results = pipeline.run(
            model_path=args.model_path,
            test_embeddings=args.test_embeddings,
            output_dir=args.output_dir
        )
        
        logger.info("Evaluation Results:")
        for metric, value in results['metrics'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        ctx.log_metrics(results['metrics'])
        return results


def evaluate_target_word(args, config: PipelineConfig, logger) -> dict:
    """Run alternative evaluation based on target word presence."""
    from evaluation.target_word_evaluator import TargetWordEvaluator
    
    with LogContext(logger, "target_word_evaluation") as ctx:
        
        # Validate request
        try:
            # Load target codes from command line args or config
            target_words = []
            
            # Priority: command line args > config file
            if args.target_words:
                target_words = args.target_words if isinstance(args.target_words, list) else [args.target_words]
            elif hasattr(args, 'target_codes_file') and args.target_codes_file:
                # Load from file using helper function
                target_words = load_target_words_from_file(args.target_codes_file)
            else:
                # Try to get from config using safe access
                try:
                    if hasattr(config, 'target_word_evaluation'):
                        if hasattr(config.target_word_evaluation, 'target_codes'):
                            target_words = config.target_word_evaluation.target_codes or []
                        elif hasattr(config.target_word_evaluation, 'target_codes_file'):
                            target_codes_file = config.target_word_evaluation.target_codes_file
                            if target_codes_file:
                                target_words = load_target_words_from_file(target_codes_file)
                except (AttributeError, FileNotFoundError) as e:
                    logger.debug(f"Could not load target codes from config: {e}")
                    target_words = []
            
            # Validate target words are provided
            if not target_words or len(target_words) == 0:
                raise ValueError(
                    "Target codes must be specified! Use one of these methods:\n"
                    "1. Command line codes: --target-words E119 76642 N6320\n"
                    "2. Command line file: --target-codes-file path/to/codes.txt\n"
                    "3. Config file: Set target_codes or target_codes_file in pipeline_config.yaml"
                )
            
            logger.info(f"Evaluating {len(target_words)} target codes: {target_words[:5]}{'...' if len(target_words) > 5 else ''}")
        except Exception as e:
            logger.error(f"Invalid target word evaluation request: {e}")
            raise
        
        evaluator = TargetWordEvaluator(config)
        
        results = evaluator.evaluate(
            dataset_path=args.dataset_path,
            target_words=target_words,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            model_endpoint=args.model_endpoint
        )
        
        logger.info(f"Target words {target_words} evaluation:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall: {results['recall']:.4f}")
        logger.info(f"  F1 Score: {results['f1_score']:.4f}")
        
        ctx.log_metrics({
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score']
        })
        
        return results


def run_end_to_end_pipeline(args, config: PipelineConfig, logger) -> dict:
    """Run complete end-to-end pipeline."""
    from pipelines.end_to_end_pipeline import EndToEndPipeline
    
    with LogContext(logger, "end_to_end_execution") as ctx:
        
        # Create job configuration
        try:
            # Load and validate target words if target word evaluation is enabled
            target_words = []
            if args.target_word_evaluation:
                if args.target_words:
                    target_words = args.target_words
                elif hasattr(args, 'target_codes_file') and args.target_codes_file:
                    # Load from file using helper function
                    target_words = load_target_words_from_file(args.target_codes_file)
                else:
                    # Try to get from config using safe access
                    try:
                        if hasattr(config, 'target_word_evaluation'):
                            if hasattr(config.target_word_evaluation, 'target_codes'):
                                target_words = config.target_word_evaluation.target_codes or []
                            elif hasattr(config.target_word_evaluation, 'target_codes_file'):
                                target_codes_file = config.target_word_evaluation.target_codes_file
                                if target_codes_file:
                                    target_words = load_target_words_from_file(target_codes_file)
                    except (AttributeError, FileNotFoundError) as e:
                        logger.debug(f"Could not load target codes from config: {e}")
                        target_words = []
                
                if not target_words or len(target_words) == 0:
                    raise ValueError(
                        "Target codes must be specified when target word evaluation is enabled! Use one of these methods:\n"
                        "1. Command line codes: --target-words E119 76642 N6320\n"
                        "2. Command line file: --target-codes-file path/to/codes.txt\n"
                        "3. Config file: Set target_codes or target_codes_file in pipeline_config.yaml"
                    )
            
            job_config = PipelineJob(
                dataset_path=getattr(args, 'dataset_path', None) or safe_get_config_value(config, 'input.dataset_path'),
                train_dataset_path=getattr(args, 'train_dataset', None) or safe_get_config_value(config, 'input.train_dataset_path'),
                test_dataset_path=getattr(args, 'test_dataset', None) or safe_get_config_value(config, 'input.test_dataset_path'),
                train_embeddings_path=safe_get_config_value(config, 'input.train_embeddings_path'),
                test_embeddings_path=safe_get_config_value(config, 'input.test_embeddings_path'),
                model_paths=safe_get_config_value(config, 'input.model_paths'),
                models_directory=safe_get_config_value(config, 'input.models_directory'),
                model_endpoint=args.model_endpoint or safe_get_config_value(config, 'model_api.base_url'),
                output_base_dir=args.output_dir,
                job_name=args.job_name,
                split_ratio=getattr(args, 'split_ratio', None) or safe_get_config_value(config, 'input.split_ratio'),
                end_to_end=EndToEndConfig(
                    generate_embeddings=args.generate_embeddings,
                    train_classifiers=args.train_classifiers,
                    evaluate_models=args.evaluate_models,
                    target_word_evaluation=args.target_word_evaluation,
                    classifier_types=args.classifier_types,
                    target_words=target_words,
                    target_word_samples=args.target_word_samples,
                    target_word_max_tokens=args.target_word_max_tokens,
                    create_summary_report=args.create_summary_report,
                    compare_methods=args.compare_methods
                )
            )
            logger.info(f"Created job configuration for: {job_config.job_name}")
        except Exception as e:
            logger.error(f"Invalid end-to-end job configuration: {e}")
            raise
        
        # Run pipeline
        pipeline = EndToEndPipeline(config, job_config)
        results = pipeline.run()
        
        # Log summary
        logger.info("=== PIPELINE COMPLETE ===")
        logger.info(f"Job: {results.job_name}")
        logger.info(f"Duration: {results.total_duration:.2f}s")
        logger.info(f"Success: {results.success}")
        
        if results.best_classifier:
            logger.info(f"Best Classifier: {results.best_classifier} (AUC: {results.best_embedding_score:.4f})")
        
        if results.target_word_score:
            logger.info(f"Target Word Method: {results.target_word_score:.4f}")
        
        ctx.log_metrics({
            'total_duration': results.total_duration,
            'success': results.success,
            'best_score': results.best_embedding_score,
            'target_word_score': results.target_word_score
        })
        
        return results.dict()


def main():
    """Main entry point."""
    # First, we need to load the config to setup argument parser with proper defaults
    # Parse config argument first
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    temp_args, _ = temp_parser.parse_known_args()
    
    # Load configuration
    try:
        config, logger = setup_pipeline(temp_args.config)
    except Exception as e:
        print(f"Setup failed: {e}")
        sys.exit(1)
    
    # Now setup the full parser with config-based defaults
    parser = setup_argument_parser_with_config(config)
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Validate arguments for generate-embeddings and run-all commands
    if args.command in ['generate-embeddings', 'run-all']:
        error_msg = validate_embedding_args(args)
        if error_msg:
            parser.error(error_msg)
    
    # Override log level if verbose
    if args.verbose:
        config.logging.level = "DEBUG"
        config.logging.console_level = "DEBUG"
        # Recreate logger with new config
        logger = get_logger("mgpt_eval", config.logging)
    
    logger.info(f"Starting {args.command} pipeline")
    logger.debug(f"Arguments: {vars(args)}")
    logger.info(f"Using config file: {args.config}")
    
    # Execute command
    try:
        if args.command == 'generate-embeddings':
            results = generate_embeddings(args, config, logger)
        elif args.command == 'train-classifier':
            results = train_classifier(args, config, logger)
        elif args.command == 'evaluate':
            results = evaluate_model(args, config, logger)
        elif args.command == 'evaluate-target-word':
            results = evaluate_target_word(args, config, logger)
        elif args.command == 'run-all':
            results = run_end_to_end_pipeline(args, config, logger)
        
        logger.info(f"Pipeline {args.command} completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Pipeline failed with error: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()