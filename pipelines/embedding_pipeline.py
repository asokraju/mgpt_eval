"""
Simplified Embedding Pipeline for generating embeddings from MediClaimGPT model.

This module provides a lean pipeline for generating embeddings from CSV data
using the MediClaimGPT API with strict fail-fast behavior.

Key Principles:
    - CSV input and output only
    - Fail fast on any error - no placeholders or zero embeddings
    - No checkpoint or resume functionality
    - Tokenizer is required - no character-based fallback
    - All embeddings must be successfully generated or the entire job fails
"""

import json
import random
import time
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from models.config_models import PipelineConfig
from utils.logging_utils import get_logger

class EmbeddingPipeline:
    """
    Simplified pipeline for generating embeddings from CSV data using MediClaimGPT API.
    
    This class provides a straightforward embedding generation process with strict
    error handling and no compromise on data quality.
    
    Attributes:
        config (PipelineConfig): Pydantic configuration object
        logger: Structured logger instance
        tokenizer: AutoTokenizer instance (required)
        expected_embedding_dim: Expected dimension of embeddings
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the embedding pipeline with configuration.
        
        Args:
            config (PipelineConfig): Validated Pydantic configuration object
            
        Raises:
            ValueError: If tokenizer path is not specified
            RuntimeError: If tokenizer cannot be loaded
        """
        self.config = config
        self.logger = get_logger("embedding_pipeline", config.logging)
        self.expected_embedding_dim = None
        
        # Initialize tokenizer (required - no fallback)
        self._initialize_tokenizer()
        
    def _initialize_tokenizer(self):
        """Initialize the tokenizer - fail if not available."""
        tokenizer_path = self.config.embedding_generation.tokenizer_path
        
        if not tokenizer_path:
            raise ValueError("Tokenizer path must be specified in config.embedding_generation.tokenizer_path")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.logger.info(f"Successfully loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer from {tokenizer_path}: {e}. "
                "Tokenizer is required for proper text processing."
            )
    
    def run(self, dataset_path: str, output_path: str, 
            model_endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the embedding generation pipeline.
        
        Args:
            dataset_path: Path to the input CSV file
            output_path: Path for the output CSV file
            model_endpoint: Override model endpoint from config
            
        Returns:
            Dictionary with results including output path and statistics
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If data validation fails
            RuntimeError: If embedding generation fails
        """
        self.logger.info(f"Starting embedding generation for dataset: {dataset_path}")
        
        # Load and validate data
        data = self._load_csv(dataset_path)
        self.logger.info(f"Loaded {len(data)} samples from dataset")
        
        # Generate embeddings (all or nothing)
        embeddings = self._generate_embeddings(
            data,
            model_endpoint or self.config.model_api.base_url
        )
        
        # Validate final count
        if len(embeddings) != len(data):
            raise RuntimeError(
                f"Embedding count mismatch: generated {len(embeddings)}, expected {len(data)}. "
                "Data integrity check failed!"
            )
        
        # Save results
        output_file = self._save_csv(embeddings, data, output_path)
        
        # Calculate basic statistics
        embedding_stats = self._calculate_embedding_stats(embeddings)
        
        return {
            'output_path': output_file,
            'n_samples': len(data),
            'embedding_dim': self.expected_embedding_dim,
            'embedding_stats': embedding_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_csv(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file with validation.
        
        Args:
            dataset_path: Path to the CSV file
            
        Returns:
            Validated DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        if path.suffix != '.csv':
            raise ValueError(f"Expected CSV file, got: {path.suffix}")
        
        # Load data
        try:
            data = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Validate required columns
        required_cols = ['claims', 'mcid', 'label']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        # After existing validation, add:
        if not pd.api.types.is_integer_dtype(data['mcid']) and not pd.api.types.is_string_dtype(data['mcid']):
            raise ValueError("'mcid' column must contain integers or strings")
        
        # Validate MCID uniqueness
        if data['mcid'].duplicated().any():
            duplicate_mcids = data[data['mcid'].duplicated()]['mcid'].unique()
            raise ValueError(
                f"Found {len(duplicate_mcids)} duplicate MCIDs. "
                f"First few examples: {list(duplicate_mcids[:5])}"
            )
        
        # Validate no null values in required columns
        for col in required_cols:
            if data[col].isnull().any():
                null_count = data[col].isnull().sum()
                raise ValueError(f"Found {null_count} null values in column '{col}'")
        
        # Validate data types
        if not pd.api.types.is_string_dtype(data['claims']):
            raise ValueError("'claims' column must contain strings")
        
        return data
    
    def _generate_embeddings(self, data: pd.DataFrame, 
                           model_endpoint: str) -> List[List[float]]:
        """
        Generate embeddings for all claims with strict validation.
        
        Args:
            data: DataFrame with claims
            model_endpoint: API endpoint base URL
            
        Returns:
            List of embeddings
            
        Raises:
            RuntimeError: If any embedding generation fails
        """
        embeddings = []
        batch_size = self.config.embedding_generation.batch_size
        endpoint = f"{model_endpoint}{self.config.model_api.endpoints['embeddings_batch']}"
        
        # Calculate total batches for progress bar
        total_batches = (len(data) + batch_size - 1) // batch_size
        
        self.logger.info(f"Generating embeddings in {total_batches} batches")
        
        # Process batches
        with tqdm(total=total_batches, desc="Generating embeddings") as pbar:
            for i in range(0, len(data), batch_size):
                batch_end = min(i + batch_size, len(data))
                batch_data = data.iloc[i:batch_end]
                batch_claims = batch_data['claims'].tolist()
                
                try:
                    # Truncate texts if needed
                    if self.config.data_processing.max_sequence_length:
                        batch_claims = self._truncate_claims(batch_claims)
                    
                    # Process batch
                    batch_embeddings = self._process_batch(
                        endpoint, batch_claims, i
                    )
                    
                    # Validate dimensions
                    self._validate_embedding_dimensions(batch_embeddings, i)
                    
                    # Add to results
                    embeddings.extend(batch_embeddings)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Critical error processing batch at index {i}: {e}")
                    raise RuntimeError(
                        f"Failed to generate embeddings for batch starting at index {i}: {e}. "
                        "All embeddings must be successfully generated."
                    )
        
        self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def _process_batch(self, endpoint: str, batch_claims: List[str], 
                      batch_start_idx: int) -> List[List[float]]:
        """
        Process a batch of claims - fail on any error.
        
        Args:
            endpoint: API endpoint URL
            batch_claims: List of claims to process
            batch_start_idx: Starting index of this batch
            
        Returns:
            List of embeddings
            
        Raises:
            RuntimeError: If batch processing fails
        """
        try:
            # Call API
            response = self._call_embedding_api(endpoint, batch_claims)
            batch_embeddings = response['embeddings']
            
            # Validate response
            if len(batch_embeddings) != len(batch_claims):
                raise ValueError(
                    f"API returned {len(batch_embeddings)} embeddings for {len(batch_claims)} claims"
                )
            
            # Validate each embedding
            for idx, emb in enumerate(batch_embeddings):
                abs_idx = batch_start_idx + idx
                
                if not emb:
                    raise ValueError(f"Empty embedding at index {abs_idx}")
                
                if not isinstance(emb, list):
                    raise ValueError(f"Embedding at index {abs_idx} is not a list")
                
                if any(not isinstance(x, (int, float)) for x in emb):
                    raise ValueError(f"Embedding at index {abs_idx} contains non-numeric values")
                
                if any(math.isnan(x) or math.isinf(x) for x in emb):
                    raise ValueError(f"Embedding at index {abs_idx} contains NaN or inf values")
                
                # Check for zero vector
                # Current code:
                if all(x == 0 for x in emb):
                    raise ValueError(f"Embedding at index {abs_idx} is a zero vector")
            
            return batch_embeddings
            
        except Exception as e:
            raise RuntimeError(f"Batch processing failed: {e}")
    
    def _validate_embedding_dimensions(self, batch_embeddings: List[List[float]], 
                                     batch_start_idx: int):
        """
        Validate embedding dimensions are consistent.
        
        Args:
            batch_embeddings: List of embeddings to validate
            batch_start_idx: Starting index of this batch
            
        Raises:
            ValueError: If dimensions are inconsistent
        """
        for idx, emb in enumerate(batch_embeddings):
            abs_idx = batch_start_idx + idx
            
            if self.expected_embedding_dim is None:
                self.expected_embedding_dim = len(emb)
                self.logger.info(f"Set expected embedding dimension: {self.expected_embedding_dim}")
            elif len(emb) != self.expected_embedding_dim:
                raise ValueError(
                    f"Dimension mismatch at index {abs_idx}: "
                    f"expected {self.expected_embedding_dim}, got {len(emb)}"
                )
    
    def _truncate_claims(self, claims: List[str]) -> List[str]:
        """
        Truncate claims using tokenizer.
        
        Args:
            claims: List of claims to truncate
            
        Returns:
            List of truncated claims
            
        Raises:
            RuntimeError: If truncation fails
        """
        max_length = self.config.data_processing.max_sequence_length
        truncated_claims = []
        
        for i, claim in enumerate(claims):
            try:
                # Tokenize and truncate
                tokens = self.tokenizer.encode(
                    claim, 
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True
                )
                
                # Decode back to text
                truncated_claim = self.tokenizer.decode(tokens, skip_special_tokens=True)
                
                # Validate truncation worked
                if not truncated_claim.strip():
                    raise ValueError(f"Truncation resulted in empty claim at index {i}")
                
                truncated_claims.append(truncated_claim)
                
            except Exception as e:
                raise RuntimeError(
                    f"Failed to truncate claim at index {i}: {e}. "
                    "Tokenizer-based truncation is required."
                )
        
        return truncated_claims
    
    def _call_embedding_api(self, endpoint: str, claims: List[str]) -> Dict[str, Any]:
        """
        Call the embedding API endpoint with retry logic.
        
        Args:
            endpoint: API endpoint URL
            claims: List of claims to embed
            
        Returns:
            API response dictionary
            
        Raises:
            RuntimeError: If API call fails after all retries
        """
        payload = {
            'claims': claims,
            'batch_size': len(claims)
        }
        
        base_delay = 1.0
        base_timeout = self.config.model_api.timeout
        
        for attempt in range(self.config.model_api.max_retries):
            # Calculate timeout for this attempt
            timeout = max(1, int(min(base_timeout * (1.5 ** attempt), base_timeout * 5)))
            
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Validate API response structure
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict response, got {type(result)}")
                
                if 'embeddings' not in result:
                    raise ValueError(f"Response missing 'embeddings' field")
                
                if not isinstance(result['embeddings'], list):
                    raise ValueError("Response 'embeddings' field is not a list")
                
                return result
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.config.model_api.max_retries}): {e}"
                )
                
                if attempt == self.config.model_api.max_retries - 1:
                    raise RuntimeError(
                        f"API call failed after {self.config.model_api.max_retries} attempts: {e}"
                    )
                
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
    
    def _save_csv(self, embeddings: List[List[float]], 
                  data: pd.DataFrame, output_path: str) -> str:
        """
        Save embeddings to CSV file.
        
        Args:
            embeddings: List of embeddings
            data: Original data DataFrame
            output_path: Output file path
            
        Returns:
            Path to saved file
            
        Raises:
            RuntimeError: If save fails
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create output DataFrame
            output_df = pd.DataFrame({
                'mcid': data['mcid'],
                'label': data['label'],
                'embedding': [json.dumps(emb) for emb in embeddings]
            })
            
            # Validate output
            if len(output_df) != len(data):
                raise ValueError("Output DataFrame size doesn't match input data")
            
            # Atomic write to prevent partial files
            temp_file = output_file.with_suffix('.tmp')
            output_df.to_csv(temp_file, index=False)
            
            # Rename to final location
            temp_file.replace(output_file)
            
            self.logger.info(f"Embeddings saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save embeddings: {e}")
    
    def _calculate_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        if not embeddings:
            return {}
        
        try:
            embeddings_array = np.array(embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1)
            
            return {
                'mean_norm': float(np.mean(norms)),
                'std_norm': float(np.std(norms)),
                'min_norm': float(np.min(norms)),
                'max_norm': float(np.max(norms)),
                'embedding_dim': embeddings_array.shape[1]
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate embedding stats: {e}")
            return {'error': str(e)}