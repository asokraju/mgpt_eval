"""
Embedding Pipeline for generating embeddings from MediClaimGPT model.

This module provides a comprehensive pipeline for generating embeddings from text data
using the MediClaimGPT API. It includes features for batch processing, checkpoint
resumption, error handling, and progress tracking.

Key Features:
    - Batch processing with configurable batch sizes
    - Checkpoint/resume functionality for long-running jobs
    - Support for multiple data formats (CSV, JSON, Parquet)
    - Automatic retry logic for API failures
    - Progress tracking with structured logging
    - Flexible output formats
    - Pydantic data validation

Example:
    >>> config = PipelineConfig.from_yaml('config.yaml')
    >>> pipeline = EmbeddingPipeline(config)
    >>> results = pipeline.run(
    ...     dataset_path='data.csv',
    ...     output_dir='outputs/embeddings',
    ...     resume=True
    ... )
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.config_models import PipelineConfig
from utils.logging_utils import get_logger

# Check if psutil is available for memory monitoring
try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class EmbeddingPipeline:
    """
    Pipeline for generating embeddings from text data using MediClaimGPT API.
    
    This class orchestrates the entire embedding generation process, from loading
    data to saving results. It handles API communication, error recovery, and
    progress tracking with comprehensive logging.
    
    Attributes:
        config (PipelineConfig): Pydantic configuration object
        logger: Structured logger instance
        checkpoint_dir (Path): Directory for saving checkpoints
        tokenizer: AutoTokenizer instance for accurate text processing
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the embedding pipeline with configuration.
        
        Args:
            config (PipelineConfig): Validated Pydantic configuration object
        """
        self.config = config
        self.logger = get_logger("embedding_pipeline", config.logging)
        
        # Set up checkpoint directory
        self.checkpoint_dir = Path(config.embedding_generation.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer with proper error handling."""
        tokenizer_path = self.config.embedding_generation.tokenizer_path
        
        if not tokenizer_path:
            self.logger.warning("No tokenizer path specified, using character-based truncation")
            self.tokenizer = None
            return
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer from {tokenizer_path}: {e}")
            self.logger.warning("Will use character-based truncation as fallback")
            self.tokenizer = None
        
    def run(self, dataset_path: str, output_dir: str, 
            model_endpoint: Optional[str] = None, resume: bool = False,
            output_prefix: str = "",
            split_ratio: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the embedding generation pipeline.
        
        Args:
            dataset_path: Path to the input dataset
            output_dir: Directory to save embeddings
            model_endpoint: Override model endpoint from config
            resume: Resume from checkpoint if available
            output_prefix: Prefix for output files (e.g., 'train', 'test')
            split_ratio: Override split ratio from config
            
        Returns:
            Dictionary with results including output paths and statistics
        """
        self.logger.info(f"Starting embedding generation for dataset: {dataset_path}")
        
        # Validate inputs
        self._validate_inputs(dataset_path, split_ratio)
        
        # Load dataset
        data = self._load_dataset(dataset_path)
        self.logger.info(f"Loaded {len(data)} samples from dataset")
        
        # Handle dataset splitting if split_ratio is provided
        if split_ratio is not None or self.config.input.split_ratio is not None:
            # Use provided split_ratio or fall back to config
            ratio = split_ratio if split_ratio is not None else self.config.input.split_ratio
            if ratio is None:
                raise ValueError("split_ratio must be provided either as parameter or in config.input.split_ratio")
            
            # Stratified split based on labels
            train_data, test_data = train_test_split(
                data,
                train_size=ratio,
                random_state=self.config.job.random_seed,
                stratify=data['label'] if 'label' in data.columns else None
            )
            
            self.logger.info(f"Split dataset: {len(train_data)} train, {len(test_data)} test samples")
            
            # Generate embeddings for both splits
            train_results = self._process_data_split(
                train_data, output_dir, "train", model_endpoint, resume
            )
            test_results = self._process_data_split(
                test_data, output_dir, "test", model_endpoint, resume
            )
            
            return {
                'train': train_results,
                'test': test_results,
                'split_ratio': ratio,
                'n_train': len(train_data),
                'n_test': len(test_data)
            }
        else:
            # Process single dataset without splitting
            return self._process_data_split(
                data, output_dir, output_prefix, model_endpoint, resume
            )
    
    def _process_data_split(self, data: pd.DataFrame, output_dir: str, 
                           prefix: str, model_endpoint: Optional[str], 
                           resume: bool) -> Dict[str, Any]:
        """
        Process a single data split (train or test).
        
        Args:
            data: DataFrame with the data to process
            output_dir: Base output directory
            prefix: Prefix for output files (e.g., 'train', 'test')
            model_endpoint: Model API endpoint
            resume: Whether to resume from checkpoint
            
        Returns:
            Dictionary with results
        """
        # Check for checkpoint if resuming
        start_idx = 0
        if resume:
            checkpoint_name = f'embedding_checkpoint_{prefix}.json' if prefix else 'embedding_checkpoint.json'
            start_idx, _ = self._load_checkpoint(checkpoint_name)
            
            if start_idx > 0:
                self.logger.info(f"Resuming {prefix} from index {start_idx}")
                # Check if we already have a partial output file to append to
                partial_output_exists = self._check_partial_output_exists(output_dir, prefix, start_idx)
                if not partial_output_exists:
                    self.logger.warning(f"No partial output found for resume at index {start_idx}, restarting from beginning")
                    start_idx = 0
            else:
                self.logger.info(f"No valid checkpoint found for {prefix}, starting from beginning")
        
        # Generate embeddings (only new ones if resuming)
        new_embeddings = self._generate_embeddings(
            data, 
            model_endpoint or self.config.model_api.base_url,
            start_idx,
            prefix
        )
        
        # Handle output based on whether we're resuming or starting fresh
        if start_idx == 0:
            # Fresh run: save all embeddings
            all_embeddings = new_embeddings
            embedding_dim = len(all_embeddings[0]) if all_embeddings else 0
            output_path = self._save_embeddings(all_embeddings, data, output_dir, prefix)
        else:
            # Resume scenario: append new embeddings to existing partial output
            if new_embeddings:
                embedding_dim = len(new_embeddings[0])
                output_path = self._append_embeddings_to_existing(new_embeddings, data, output_dir, prefix, start_idx)
                all_embeddings = new_embeddings  # For stats calculation
            else:
                # No new embeddings needed (job was already complete)
                embedding_dim = 0
                output_path = self._get_existing_output_path(output_dir, prefix)
                all_embeddings = []
        
        # Validate total embeddings count based on the processing scenario
        expected_total = len(data)
        if start_idx == 0:
            # Fresh run: new_embeddings should equal total data length
            actual_total = len(new_embeddings)
        else:
            # Resume scenario: start_idx + new_embeddings should equal total data length
            actual_total = start_idx + len(new_embeddings)
        
        if actual_total != expected_total:
            self.logger.warning(f"Total embeddings ({actual_total}) doesn't match expected ({expected_total}). "
                              f"Resume from index: {start_idx}, new embeddings: {len(new_embeddings)}")
        
        # Calculate embedding statistics
        embedding_stats = self._calculate_embedding_stats(all_embeddings) if all_embeddings else {}
        
        # Save counts before memory cleanup
        n_new_samples = len(new_embeddings)
        
        # Clean up checkpoint
        checkpoint_name = f'embedding_checkpoint_{prefix}.json' if prefix else 'embedding_checkpoint.json'
        self._cleanup_checkpoint(checkpoint_name)
        
        # Memory optimization
        del new_embeddings
        if 'all_embeddings' in locals():
            del all_embeddings
        
        # Log memory usage
        self._log_memory_usage(f"Completed processing {prefix} split")
        
        return {
            'output_path': output_path,
            'n_samples': len(data),
            'n_new_samples': n_new_samples,
            'resumed_from_index': start_idx,
            'embedding_dim': embedding_dim,
            'embedding_stats': embedding_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset from file with validation.
        
        Supports CSV, JSON, and Parquet formats. Validates that required
        columns (claims, mcid, label) are present.
        
        Args:
            dataset_path (str): Path to the dataset file
            
        Returns:
            pd.DataFrame: Loaded dataset with required columns
            
        Raises:
            ValueError: If file format is unsupported or required columns missing
            FileNotFoundError: If dataset file doesn't exist
        """
        path = Path(dataset_path)
        
        if path.suffix == '.csv':
            data = pd.read_csv(path)
        elif path.suffix == '.json':
            data = pd.read_json(path)
        elif path.suffix == '.parquet':
            data = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Validate required columns for medical claims data
        required_cols = ['claims', 'mcid', 'label']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return data
    
    def _generate_embeddings(self, data: pd.DataFrame, model_endpoint: str,
                           start_idx: int = 0, prefix: str = "") -> List[List[float]]:
        """
        Generate embeddings for texts starting from start_idx.
        
        Args:
            data (pd.DataFrame): Dataset containing texts to embed
            model_endpoint (str): Base URL for the model API
            start_idx (int): Index to start processing from (for resumption)
            prefix (str): Prefix for checkpoint naming
            
        Returns:
            List[List[float]]: List of embedding vectors for texts from start_idx onwards
        """
        # Get configuration
        batch_size = self.config.embedding_generation.batch_size
        save_interval = self.config.embedding_generation.save_interval
        
        # Initialize for new embeddings only (don't load previous ones in memory)
        new_embeddings = []
        cache = {}
        
        # Build endpoint
        endpoint = f"{model_endpoint}{self.config.model_api.endpoints['embeddings_batch']}"
        
        # Calculate total batches for progress tracking
        remaining_items = len(data) - start_idx
        total_batches = (remaining_items + batch_size - 1) // batch_size
        
        if remaining_items <= 0:
            self.logger.info("No new embeddings to generate - job already complete")
            return []
        
        self.logger.info(f"Generating {remaining_items} embeddings starting from index {start_idx}")
        
        # Process from start_idx to end
        with tqdm(total=total_batches, desc=f"Generating embeddings{f' ({prefix})' if prefix else ''}") as pbar:
            for i in range(start_idx, len(data), batch_size):
                batch_end = min(i + batch_size, len(data))
                batch_claims = data['claims'].iloc[i:batch_end].tolist()
                
                try:
                    # Truncate texts if needed
                    if self.config.data_processing.max_sequence_length:
                        batch_claims = self._truncate_claims(batch_claims)
                    
                    # Call embedding API
                    try:
                        response = self._call_embedding_api(endpoint, batch_claims)
                        batch_embeddings = response['embeddings']
                        
                        # Add to results
                        new_embeddings.extend(batch_embeddings)
                        
                        # Update cache for checkpoint (use relative indexing)
                        for j, emb in enumerate(batch_embeddings):
                            cache_key = str(i + j - start_idx)  # Relative to start_idx
                            cache[cache_key] = emb
                    
                    except Exception as batch_error:
                        # Batch failed - try processing items one by one
                        self.logger.warning(f"Batch processing failed at index {i}: {batch_error}")
                        self.logger.info(f"Attempting single-item fallback for {len(batch_claims)} items...")
                        
                        batch_embeddings = self._process_batch_individually(endpoint, batch_claims, i)
                        
                        # Add successful embeddings to results
                        valid_embeddings = [emb for emb in batch_embeddings if emb is not None]
                        if len(valid_embeddings) != len(batch_embeddings):
                            raise RuntimeError(f"Failed to generate embeddings for some items in batch at index {i}")
                        
                        new_embeddings.extend(valid_embeddings)
                        
                        # Update cache
                        for j, emb in enumerate(valid_embeddings):
                            cache_key = str(i + j - start_idx)
                            cache[cache_key] = emb
                    
                    # Save checkpoint periodically
                    processed_count = len(new_embeddings)
                    if processed_count > 0 and processed_count % save_interval == 0:
                        checkpoint_name = f'embedding_checkpoint_{prefix}.json' if prefix else 'embedding_checkpoint.json'
                        self._save_checkpoint(start_idx + processed_count, cache, checkpoint_name)
                        self.logger.info(f"Checkpoint saved at index {start_idx + processed_count}")
                    
                    # Update progress bar
                    pbar.update(1)
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch at index {i}: {e}")
                    # Save checkpoint before raising
                    checkpoint_name = f'embedding_checkpoint_{prefix}.json' if prefix else 'embedding_checkpoint.json'
                    self._save_checkpoint(start_idx + len(new_embeddings), cache, checkpoint_name)
                    raise
        
        self.logger.info(f"Generated {len(new_embeddings)} new embeddings")
        return new_embeddings
    
    def _truncate_claims(self, claims: List[str]) -> List[str]:
        """Truncate claims based on configuration."""
        max_length = self.config.data_processing.max_sequence_length
        truncated_claims = []
        
        for claim in claims:
            if self.tokenizer:
                # Use tokenizer for accurate truncation
                tokens = self.tokenizer.encode(claim, add_special_tokens=True)
                if len(tokens) > max_length:
                    # Choose truncation strategy
                    truncation_strategy = getattr(self.config.data_processing, 'truncation_strategy', 'keep_last')
                    
                    if truncation_strategy == 'keep_first':
                        tokens = tokens[:max_length]
                    elif truncation_strategy == 'keep_last':
                        tokens = tokens[-max_length:]
                    elif truncation_strategy == 'keep_middle':
                        mid_start = (len(tokens) - max_length) // 2
                        tokens = tokens[mid_start:mid_start + max_length]
                    else:
                        tokens = tokens[-max_length:]
                    
                    truncated_claim = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    truncated_claims.append(truncated_claim)
                    self.logger.debug(f"Truncated claim using '{truncation_strategy}' from {len(self.tokenizer.encode(claim))} to {len(tokens)} tokens")
                else:
                    truncated_claims.append(claim)
            else:
                # Character-based truncation fallback
                max_chars = max_length * 4  # Rough estimate
                if len(claim) > max_chars:
                    truncation_strategy = getattr(self.config.data_processing, 'truncation_strategy', 'keep_last')
                    
                    if truncation_strategy == 'keep_first':
                        truncated_claim = claim[:max_chars]
                    elif truncation_strategy == 'keep_last':
                        truncated_claim = claim[-max_chars:]
                    elif truncation_strategy == 'keep_middle':
                        mid_start = (len(claim) - max_chars) // 2
                        truncated_claim = claim[mid_start:mid_start + max_chars]
                    else:
                        truncated_claim = claim[-max_chars:]
                    
                    truncated_claims.append(truncated_claim)
                    self.logger.debug(f"Truncated claim using '{truncation_strategy}' from {len(claim)} to {max_chars} chars")
                else:
                    truncated_claims.append(claim)
        
        return truncated_claims
    
    def _call_embedding_api(self, endpoint: str, claims: List[str]) -> Dict[str, Any]:
        """
        Call the embedding API endpoint with retry logic.
        """
        payload = {
            'claims': claims,
            'batch_size': len(claims)
        }
        
        base_delay = 1.0
        timeout = self.config.model_api.timeout
        
        for attempt in range(self.config.model_api.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Validate API response structure
                self._validate_api_response(result, len(claims))
                return result
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API call failed (attempt {attempt + 1}/{self.config.model_api.max_retries}): {e}")
                if attempt == self.config.model_api.max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                
                # Increase timeout but cap it
                max_timeout = self.config.model_api.timeout * 5
                timeout = min(int(timeout * 1.5), max_timeout)
    
    def _save_embeddings(self, embeddings: List[List[float]], 
                        data: pd.DataFrame, output_dir: str, 
                        prefix: str = "") -> str:
        """Save embeddings with metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics before saving
        embedding_stats = self._calculate_embedding_stats(embeddings)
        
        # Prepare output data
        output_data = {
            'mcids': data['mcid'].tolist(),
            'labels': data['label'].tolist(),
            'embeddings': embeddings,
            'metadata': {
                'n_samples': len(data),
                'embedding_dim': len(embeddings[0]) if embeddings else 0,
                'embedding_stats': embedding_stats,
                'timestamp': datetime.now().isoformat(),
                'truncation_strategy': getattr(self.config.data_processing, 'truncation_strategy', 'keep_last'),
                'model_endpoint': self.config.model_api.base_url,
                'config': self.config.embedding_generation.dict()
            }
        }
        
        # Save embeddings in the configured format
        output_format = self.config.data_processing.output_format.lower()
        
        if output_format == 'json':
            filename = f'{prefix}_embeddings.json' if prefix else 'embeddings.json'
            output_file = output_path / filename
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        
        elif output_format == 'parquet':
            filename = f'{prefix}_embeddings.parquet' if prefix else 'embeddings.parquet'
            output_file = output_path / filename
            
            df_output = pd.DataFrame({
                'mcid': data['mcid'],
                'label': data['label'],
                'embedding': embeddings
            })
            df_output.to_parquet(output_file, index=False)
            
            # Save metadata separately
            metadata_file = output_path / f'{prefix}_metadata.json' if prefix else 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(output_data['metadata'], f, indent=2)
        
        elif output_format == 'numpy':
            filename_base = f'{prefix}_embeddings' if prefix else 'embeddings'
            
            embeddings_file = output_path / f'{filename_base}.npy'
            np.save(embeddings_file, np.array(embeddings))
            
            metadata_file = output_path / f'{filename_base}_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(output_data['metadata'], f, indent=2)
            
            info_file = output_path / f'{filename_base}_info.csv'
            pd.DataFrame({
                'mcid': data['mcid'],
                'label': data['label']
            }).to_csv(info_file, index=False)
            
            output_file = embeddings_file
        
        else:
            # Legacy CSV format
            self.logger.warning("CSV format for embeddings is inefficient. Consider using 'parquet' or 'numpy'.")
            filename = f'{prefix}_embeddings.csv' if prefix else 'embeddings.csv'
            output_file = output_path / filename
            df_output = pd.DataFrame({
                'mcid': data['mcid'],
                'label': data['label'],
                'embedding': [json.dumps(emb) for emb in embeddings]
            })
            df_output.to_csv(output_file, index=False)
            
            metadata_file = output_path / f'{prefix}_metadata.json' if prefix else 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(output_data['metadata'], f, indent=2)
        
        self.logger.info(f"Embeddings saved to: {output_file}")
        return str(output_file)
    
    def _append_embeddings_to_existing(self, new_embeddings: List[List[float]], 
                                     data: pd.DataFrame, output_dir: str, 
                                     prefix: str, start_idx: int) -> str:
        """
        Append new embeddings to existing partial output by loading and merging.
        
        This properly loads existing embeddings and appends new ones for production use.
        """
        self.logger.info(f"Appending {len(new_embeddings)} embeddings to existing output")
        
        output_path = Path(output_dir)
        output_format = self.config.data_processing.output_format.lower()
        
        try:
            if output_format == 'numpy':
                # Load existing embeddings from numpy file
                filename_base = f'{prefix}_embeddings' if prefix else 'embeddings'
                embeddings_file = output_path / f'{filename_base}.npy'
                
                if embeddings_file.exists():
                    existing_embeddings = np.load(embeddings_file).tolist()
                    # Validate we have the expected number of existing embeddings
                    if len(existing_embeddings) < start_idx:
                        raise ValueError(f"Existing file has {len(existing_embeddings)} embeddings but expected at least {start_idx}")
                    
                    # Take only the first start_idx embeddings and append new ones
                    all_embeddings = existing_embeddings[:start_idx] + new_embeddings
                    
                    # Save combined embeddings
                    np.save(embeddings_file, np.array(all_embeddings))
                    
                    # Verify append success
                    self._verify_append_success(embeddings_file, len(all_embeddings), 'numpy')
                    
                    # Update metadata
                    self._update_metadata(output_path, prefix, len(all_embeddings), new_embeddings)
                    
                    self.logger.info(f"Successfully appended {len(new_embeddings)} embeddings to existing numpy file")
                    return str(embeddings_file)
                else:
                    raise FileNotFoundError(f"Expected partial output at {embeddings_file}")
            
            elif output_format == 'parquet':
                # Load existing embeddings from parquet file
                filename = f'{prefix}_embeddings.parquet' if prefix else 'embeddings.parquet'
                parquet_file = output_path / filename
                
                if parquet_file.exists():
                    existing_df = pd.read_parquet(parquet_file)
                    # Validate we have the expected number of existing embeddings
                    if len(existing_df) < start_idx:
                        raise ValueError(f"Existing file has {len(existing_df)} embeddings but expected at least {start_idx}")
                    
                    # Create dataframe for new embeddings
                    new_df = pd.DataFrame({
                        'mcid': data['mcid'].iloc[start_idx:].tolist(),
                        'label': data['label'].iloc[start_idx:].tolist(),
                        'embedding': new_embeddings
                    })
                    
                    # Combine existing (up to start_idx) with new embeddings
                    combined_df = pd.concat([
                        existing_df.iloc[:start_idx],
                        new_df
                    ], ignore_index=True)
                    
                    # Save combined dataframe
                    combined_df.to_parquet(parquet_file, index=False)
                    
                    # Verify append success
                    self._verify_append_success(parquet_file, len(combined_df), 'parquet')
                    
                    # Update metadata
                    self._update_metadata(output_path, prefix, len(combined_df), new_embeddings)
                    
                    self.logger.info(f"Successfully appended {len(new_embeddings)} embeddings to existing parquet file")
                    return str(parquet_file)
                else:
                    raise FileNotFoundError(f"Expected partial output at {parquet_file}")
            
            elif output_format == 'json':
                # Load existing embeddings from JSON file
                filename = f'{prefix}_embeddings.json' if prefix else 'embeddings.json'
                json_file = output_path / filename
                
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        existing_data = json.load(f)
                    
                    existing_embeddings = existing_data.get('embeddings', [])
                    # Validate we have the expected number of existing embeddings
                    if len(existing_embeddings) < start_idx:
                        raise ValueError(f"Existing file has {len(existing_embeddings)} embeddings but expected at least {start_idx}")
                    
                    # Combine existing (up to start_idx) with new embeddings
                    all_embeddings = existing_embeddings[:start_idx] + new_embeddings
                    
                    # Update the data structure
                    updated_data = {
                        'mcids': data['mcid'].tolist(),
                        'labels': data['label'].tolist(),
                        'embeddings': all_embeddings,
                        'metadata': {
                            'n_samples': len(data),
                            'embedding_dim': len(new_embeddings[0]) if new_embeddings else len(existing_embeddings[0]),
                            'embedding_stats': self._calculate_embedding_stats(all_embeddings),
                            'timestamp': datetime.now().isoformat(),
                            'resumed_from_index': start_idx,
                            'truncation_strategy': getattr(self.config.data_processing, 'truncation_strategy', 'keep_last'),
                            'model_endpoint': self.config.model_api.base_url,
                            'config': self.config.embedding_generation.dict()
                        }
                    }
                    
                    # Save combined data
                    with open(json_file, 'w') as f:
                        json.dump(updated_data, f, indent=2)
                    
                    # Verify append success
                    self._verify_append_success(json_file, len(all_embeddings), 'json')
                    
                    self.logger.info(f"Successfully appended {len(new_embeddings)} embeddings to existing JSON file")
                    return str(json_file)
                else:
                    raise FileNotFoundError(f"Expected partial output at {json_file}")
            
            else:  # CSV format
                # Load existing embeddings from CSV file
                filename = f'{prefix}_embeddings.csv' if prefix else 'embeddings.csv'
                csv_file = output_path / filename
                
                if csv_file.exists():
                    existing_df = pd.read_csv(csv_file)
                    # Parse existing embeddings (they're stored as JSON strings in CSV)
                    existing_embeddings = [json.loads(emb) for emb in existing_df['embedding'].iloc[:start_idx]]
                    
                    # Validate we have the expected number of existing embeddings
                    if len(existing_embeddings) < start_idx:
                        raise ValueError(f"Existing file has {len(existing_embeddings)} embeddings but expected at least {start_idx}")
                    
                    # Combine embeddings
                    all_embeddings = existing_embeddings + new_embeddings
                    
                    # Create updated dataframe
                    updated_df = pd.DataFrame({
                        'mcid': data['mcid'],
                        'label': data['label'],
                        'embedding': [json.dumps(emb) for emb in all_embeddings]
                    })
                    
                    # Save combined dataframe
                    updated_df.to_csv(csv_file, index=False)
                    
                    # Verify append success
                    self._verify_append_success(csv_file, len(all_embeddings), 'csv')
                    
                    # Update metadata
                    self._update_metadata(output_path, prefix, len(all_embeddings), new_embeddings)
                    
                    self.logger.info(f"Successfully appended {len(new_embeddings)} embeddings to existing CSV file")
                    return str(csv_file)
                else:
                    raise FileNotFoundError(f"Expected partial output at {csv_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to append embeddings to existing file: {e}")
            self.logger.warning("Falling back to creating new complete file with placeholder embeddings")
            
            # Fallback: create placeholder embeddings as before
            if new_embeddings:
                embedding_dim = len(new_embeddings[0])
            else:
                embedding_dim = 768  # Default dimension
            
            placeholder_embeddings = [[0.0] * embedding_dim for _ in range(start_idx)]
            all_embeddings = placeholder_embeddings + new_embeddings
            
            self.logger.warning(f"Using {start_idx} placeholder embeddings due to append failure")
            return self._save_embeddings(all_embeddings, data, output_dir, prefix)
    
    def _update_metadata(self, output_path: Path, prefix: str, total_count: int, new_embeddings: List[List[float]]):
        """Update metadata file for appended embeddings."""
        try:
            metadata_file = output_path / f'{prefix}_metadata.json' if prefix else 'metadata.json'
            
            # Load existing metadata if it exists
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Update metadata with current information
            metadata.update({
                'n_samples': total_count,
                'embedding_dim': len(new_embeddings[0]) if new_embeddings else metadata.get('embedding_dim', 0),
                'last_updated': datetime.now().isoformat(),
                'append_operation': True,
                'new_embeddings_added': len(new_embeddings),
                'total_embeddings': total_count
            })
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to update metadata: {e}")
    
    def _verify_append_success(self, output_file: Path, expected_count: int, format: str):
        """Verify the appended file has the expected number of embeddings."""
        try:
            if format == 'numpy':
                actual_count = len(np.load(output_file))
            elif format == 'parquet':
                actual_count = len(pd.read_parquet(output_file))
            elif format == 'json':
                with open(output_file, 'r') as f:
                    data = json.load(f)
                actual_count = len(data.get('embeddings', []))
            elif format == 'csv':
                actual_count = len(pd.read_csv(output_file))
            else:
                self.logger.warning(f"Unknown format for verification: {format}")
                return
            
            if actual_count != expected_count:
                raise ValueError(f"Append verification failed: expected {expected_count}, got {actual_count}")
            
            self.logger.debug(f"Append verification successful: {actual_count} embeddings in {format} file")
                
        except Exception as e:
            self.logger.error(f"Failed to verify append for {format} file: {e}")
            raise
    
    def _check_partial_output_exists(self, output_dir: str, prefix: str, expected_count: int) -> bool:
        """Check if partial output exists for resuming."""
        output_path = Path(output_dir)
        
        # Check for existing output files based on format
        output_format = self.config.data_processing.output_format.lower()
        
        if output_format == 'json':
            filename = f'{prefix}_embeddings.json' if prefix else 'embeddings.json'
            output_file = output_path / filename
        elif output_format == 'parquet':
            filename = f'{prefix}_embeddings.parquet' if prefix else 'embeddings.parquet'
            output_file = output_path / filename
        elif output_format == 'numpy':
            filename = f'{prefix}_embeddings.npy' if prefix else 'embeddings.npy'
            output_file = output_path / filename
        else:
            filename = f'{prefix}_embeddings.csv' if prefix else 'embeddings.csv'
            output_file = output_path / filename
        
        exists = output_file.exists()
        if exists:
            self.logger.info(f"Found existing partial output: {output_file}")
            # Add: Check how many embeddings are already saved
            try:
                existing_count = self._count_existing_embeddings(output_file, output_format)
                self.logger.info(f"Existing file contains {existing_count} embeddings")
                
                # Warn if existing count doesn't match expected checkpoint position
                if existing_count < expected_count:
                    self.logger.warning(f"Existing file has {existing_count} embeddings but checkpoint suggests {expected_count}")
                elif existing_count > expected_count:
                    self.logger.info(f"Existing file has {existing_count} embeddings, more than checkpoint position {expected_count}")
                
            except Exception as e:
                self.logger.warning(f"Could not count existing embeddings: {e}")
        
        return exists
    
    def _count_existing_embeddings(self, output_file: Path, output_format: str) -> int:
        """Count how many embeddings are in an existing output file."""
        if output_format == 'numpy':
            return len(np.load(output_file))
        elif output_format == 'parquet':
            return len(pd.read_parquet(output_file))
        elif output_format == 'json':
            with open(output_file, 'r') as f:
                data = json.load(f)
            return len(data.get('embeddings', []))
        elif output_format == 'csv':
            return len(pd.read_csv(output_file))
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _get_existing_output_path(self, output_dir: str, prefix: str) -> str:
        """Get path to existing output file."""
        output_path = Path(output_dir)
        output_format = self.config.data_processing.output_format.lower()
        
        if output_format == 'json':
            filename = f'{prefix}_embeddings.json' if prefix else 'embeddings.json'
        elif output_format == 'parquet':
            filename = f'{prefix}_embeddings.parquet' if prefix else 'embeddings.parquet'
        elif output_format == 'numpy':
            filename = f'{prefix}_embeddings.npy' if prefix else 'embeddings.npy'
        else:
            filename = f'{prefix}_embeddings.csv' if prefix else 'embeddings.csv'
        
        return str(output_path / filename)
    
    def _save_checkpoint(self, current_idx: int, embeddings: Dict[str, List[float]], 
                        checkpoint_name: str = 'embedding_checkpoint.json'):
        """Save lightweight checkpoint for resuming."""
        checkpoint_file = self.checkpoint_dir / checkpoint_name
        checkpoint_data = {
            'current_idx': current_idx,
            'timestamp': datetime.now().isoformat(),
            'note': 'Lightweight checkpoint - only stores progress index'
        }
        
        # Save a small sample for validation if available
        sample_embeddings = {}
        for i in range(min(5, len(embeddings))):
            if str(i) in embeddings:
                sample_embeddings[str(i)] = embeddings[str(i)]
        
        if sample_embeddings:
            checkpoint_data['sample_embeddings'] = sample_embeddings
            checkpoint_data['embedding_dim'] = len(next(iter(sample_embeddings.values())))
        
        # Atomic write using temporary file
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        temp_file.rename(checkpoint_file)
    
    def _load_checkpoint(self, checkpoint_name: str = 'embedding_checkpoint.json') -> tuple:
        """Load checkpoint if exists. Returns (current_idx, sample_cache)."""
        checkpoint_file = self.checkpoint_dir / checkpoint_name
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                current_idx = data.get('current_idx', 0)
                sample_embeddings = data.get('sample_embeddings', {})
                
                # Checkpoint integrity check - validate embedding dimensions if we have samples
                if sample_embeddings and current_idx > 0:
                    try:
                        sample_dim = len(next(iter(sample_embeddings.values())))
                        # Try to get dimension from existing output if possible
                        output_format = self.config.data_processing.output_format.lower()
                        partial_output_file = self._get_partial_output_file(checkpoint_name)
                        
                        if partial_output_file and partial_output_file.exists():
                            existing_count = self._count_existing_embeddings(partial_output_file, output_format)
                            if existing_count > 0:
                                # Get first embedding to check dimension
                                if output_format == 'numpy':
                                    first_embedding = np.load(partial_output_file)[0]
                                    existing_dim = len(first_embedding)
                                elif output_format == 'json':
                                    with open(partial_output_file, 'r') as f:
                                        file_data = json.load(f)
                                    if file_data.get('embeddings'):
                                        existing_dim = len(file_data['embeddings'][0])
                                    else:
                                        existing_dim = sample_dim  # No existing data to compare
                                else:
                                    existing_dim = sample_dim  # Skip check for complex formats
                                
                                if existing_dim != sample_dim:
                                    self.logger.warning(f"Dimension mismatch: checkpoint sample {sample_dim} vs existing {existing_dim}")
                                    self.logger.warning("Checkpoint may be incompatible - consider restarting")
                    except Exception as e:
                        self.logger.debug(f"Could not perform checkpoint integrity check: {e}")
                
                self.logger.info(f"Loaded checkpoint: resuming from index {current_idx}")
                return current_idx, sample_embeddings
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
                return 0, {}
        
        return 0, {}
    
    def _get_partial_output_file(self, checkpoint_name: str) -> Optional[Path]:
        """Get the path to the partial output file based on checkpoint name."""
        try:
            # Extract prefix from checkpoint name
            if 'train' in checkpoint_name:
                prefix = 'train'
            elif 'test' in checkpoint_name:
                prefix = 'test'
            else:
                prefix = ''
            
            # Get output directory (assuming it's relative to checkpoint dir)
            output_dir = self.checkpoint_dir.parent / 'embeddings'
            
            output_format = self.config.data_processing.output_format.lower()
            
            if output_format == 'numpy':
                filename = f'{prefix}_embeddings.npy' if prefix else 'embeddings.npy'
            elif output_format == 'parquet':
                filename = f'{prefix}_embeddings.parquet' if prefix else 'embeddings.parquet'
            elif output_format == 'json':
                filename = f'{prefix}_embeddings.json' if prefix else 'embeddings.json'
            else:
                filename = f'{prefix}_embeddings.csv' if prefix else 'embeddings.csv'
            
            return output_dir / filename
        except Exception:
            return None
    
    def _cleanup_checkpoint(self, checkpoint_name: str = 'embedding_checkpoint.json'):
        """Remove checkpoint file after successful completion."""
        checkpoint_file = self.checkpoint_dir / checkpoint_name
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            self.logger.info(f"Checkpoint {checkpoint_name} cleaned up")
    
    def _validate_inputs(self, dataset_path: str, split_ratio: Optional[float]):
        """Validate input parameters."""
        if not dataset_path or not isinstance(dataset_path, str):
            raise ValueError("dataset_path must be a non-empty string")
        
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        if split_ratio is not None:
            if not isinstance(split_ratio, (int, float)):
                raise TypeError("split_ratio must be a number")
            if not 0.01 <= split_ratio <= 0.99:
                raise ValueError("split_ratio must be between 0.01 and 0.99")
        
        # Validate required config sections
        required_sections = ['embedding_generation', 'model_api', 'data_processing', 'job']
        for section in required_sections:
            if not hasattr(self.config, section):
                raise ValueError(f"Config missing required '{section}' section")
        
        # Validate config values
        if self.config.embedding_generation.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        
        if self.config.embedding_generation.save_interval <= 0:
            raise ValueError("save_interval must be greater than 0")
        
        if self.config.model_api.max_retries <= 0:
            raise ValueError("max_retries must be greater than 0")
        
        if self.config.model_api.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        
        # Validate endpoints configuration
        if not hasattr(self.config.model_api, 'endpoints'):
            raise ValueError("model_api must have 'endpoints' configuration")
        
        if 'embeddings_batch' not in self.config.model_api.endpoints:
            raise ValueError("model_api.endpoints must contain 'embeddings_batch' endpoint")
        
        # Validate output format
        valid_formats = ['json', 'csv', 'parquet', 'numpy']
        if self.config.data_processing.output_format.lower() not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}")
    
    def _validate_api_response(self, result: dict, expected_count: int):
        """Validate API response structure and content."""
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict response from API, got {type(result)}")
        
        if 'embeddings' not in result:
            available_keys = list(result.keys())
            raise ValueError(f"API response missing 'embeddings' field. Available fields: {available_keys}")
        
        embeddings = result['embeddings']
        if not isinstance(embeddings, list):
            raise ValueError(f"Expected 'embeddings' to be a list, got {type(embeddings)}")
        
        if len(embeddings) != expected_count:
            raise ValueError(f"API returned {len(embeddings)} embeddings but expected {expected_count}")
        
        # Validate embedding structure
        if embeddings:
            first_embedding = embeddings[0]
            if not isinstance(first_embedding, list) or not all(isinstance(x, (int, float)) for x in first_embedding):
                raise ValueError("Invalid embedding format: expected list of numbers")
            
            # Check all embeddings have same dimension
            expected_dim = len(first_embedding)
            for i, emb in enumerate(embeddings):
                if len(emb) != expected_dim:
                    raise ValueError(f"Inconsistent embedding dimensions: embedding {i} has {len(emb)} dims, expected {expected_dim}")
        
        self.logger.debug(f"API response validated: {len(embeddings)} embeddings")
    
    def _process_batch_individually(self, endpoint: str, batch_claims: List[str], 
                                  batch_start_idx: int) -> List[List[float]]:
        """
        Process claims one by one when batch processing fails.
        
        Returns embeddings for all items, raising an exception if any fail.
        """
        batch_embeddings = []
        
        for idx, claim in enumerate(batch_claims):
            try:
                single_response = self._call_embedding_api(endpoint, [claim])
                embedding = single_response['embeddings'][0]
                batch_embeddings.append(embedding)
                self.logger.debug(f"Successfully processed individual item {batch_start_idx + idx}")
                
            except Exception as single_error:
                self.logger.error(f"Failed to process claim at index {batch_start_idx + idx}: {single_error}")
                # For embedding pipeline, we need all embeddings to succeed
                raise RuntimeError(f"Cannot process item at index {batch_start_idx + idx}: {single_error}")
        
        successful_count = len(batch_embeddings)
        self.logger.info(f"Single-item fallback completed: {successful_count}/{len(batch_claims)} items processed")
        
        return batch_embeddings
    
    def _calculate_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Calculate statistics about the embeddings."""
        if not embeddings:
            return {}
        
        # Convert to numpy for computation
        embeddings_array = np.array(embeddings)
        
        # Calculate norms
        norms = np.linalg.norm(embeddings_array, axis=1)
        
        # Count zero embeddings
        zero_count = np.sum(np.all(embeddings_array == 0, axis=1))
        
        return {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'zero_embeddings': int(zero_count),
            'zero_percentage': float(zero_count / len(embeddings) * 100)
        }
    
    def _log_memory_usage(self, context: str):
        """Log current memory usage for monitoring."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory usage at {context}: {memory_mb:.1f} MB")
        else:
            self.logger.debug("psutil not available for memory monitoring")