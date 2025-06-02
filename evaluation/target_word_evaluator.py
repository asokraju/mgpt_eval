"""
Target Word Evaluator for alternative evaluation using text generation.
"""

import json
import logging
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)
import re

logger = logging.getLogger(__name__)


class TargetWordEvaluator:
    """Evaluator that classifies based on target word presence in generated text."""
    
    def __init__(self, config):
        """Initialize the target word evaluator with configuration."""
        # Handle both config objects and config paths
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif hasattr(config, 'dict'):
            # Assume it's a PipelineConfig object
            self.config = config.dict()
        else:
            # Already a dictionary
            self.config = config
        
        self.model_config = self.config['model_api']
        self.target_config = self.config['target_word_evaluation']
        self.output_config = self.config.get('output', {})
        
        # No tokenizer needed - generate API handles context windows automatically
        logger.info("Target word evaluator initialized without tokenizer (API handles context windows)")
    
    def evaluate(self, dataset_path: str, target_words: List[str],
                n_samples: int, max_tokens: int,
                model_endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate by generating text and checking for target words presence.
        
        Args:
            dataset_path: Path to the input dataset
            target_words: List of words to search for - if ANY appears, classify as positive
            n_samples: Number of generations per prompt
            max_tokens: Maximum tokens to generate
            model_endpoint: Override model endpoint from config
            
        Returns:
            Dictionary with evaluation results
        """
        # Validate target words
        if not target_words or len(target_words) == 0:
            raise ValueError(
                "Target codes must be specified when target word evaluation is enabled! "
                "Please provide a list of medical codes from your model's training data in one of these ways:\n"
                "1. Set 'target_codes' in your config file under 'target_word_evaluation' section\n"
                "2. Use --target-words argument with space-separated codes: --target-words E119 76642 N6320\n"
                "3. For end-to-end pipeline: python main.py run-all --target-words E119 76642 N6320\n"
                "Example codes: ['E119', '76642', 'N6320', 'K9289']"
            )
        
        # Clean target words and validate (preserve case for medical codes)
        clean_target_words = []
        for word in target_words:
            clean_word = str(word).strip()
            if clean_word:
                clean_target_words.append(clean_word)
        
        if not clean_target_words:
            raise ValueError(
                "No valid target words provided after cleaning. "
                "Make sure your target words are not empty strings."
            )
        
        logger.info(f"Starting target word evaluation for words: {clean_target_words}")
        target_words = clean_target_words  # Use cleaned words
        
        # Load dataset
        data = self._load_dataset(dataset_path)
        logger.info(f"Loaded {len(data)} samples from dataset")
        
        # Generate predictions
        predictions, generation_details = self._generate_predictions(
            data,
            target_words,
            n_samples,
            max_tokens,
            model_endpoint or self.model_config['base_url'],
            dataset_path
        )
        
        # Calculate metrics
        true_labels = data['label'].values
        metrics = self._calculate_metrics(true_labels, predictions)
        
        # Save detailed results
        results_path = self._save_results(
            data, predictions, generation_details, metrics, target_words
        )
        
        return {
            **metrics,
            'target_words': target_words,
            'n_samples': n_samples,
            'max_tokens': max_tokens,
            'results_path': results_path,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from file."""
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
    
    def _generate_predictions(self, data: pd.DataFrame, target_words: List[str],
                            n_samples: int, max_tokens: int,
                            model_endpoint: str, dataset_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate predictions using dictionary-based MCID tracking for optimal efficiency.
        
        ALGORITHM LOGIC:
        ===============
        1. INITIALIZATION: Create pending_mcids = {mcid: tries_completed} dictionary
        2. MAIN LOOP: While pending_mcids is not empty:
           a. Sample batch_size MCIDs from pending dictionary
           b. Generate batch text for these MCIDs  
           c. Check each result for target codes
           d. UPDATE DICTIONARY:
              - If target found OR max tries reached → Remove from pending
              - Else → Increment try count and keep in pending
        3. AUTOMATIC CLEANUP: Resolved MCIDs are automatically removed
        
        TIME COMPLEXITY: O(N × S) where N=MCIDs, S=avg samples per MCID
        SPACE COMPLEXITY: O(N × S × T) where T=avg text length
        API EFFICIENCY: ~(N × S) / batch_size total API calls (vs N × S individual calls)
        """
        
        # Use batch endpoint for efficiency  
        if model_endpoint.endswith('/generate_batch'):
            endpoint = model_endpoint
        elif model_endpoint.endswith('/generate'):
            endpoint = model_endpoint.replace('/generate', '/generate_batch')
        else:
            endpoint = f"{model_endpoint.rstrip('/')}{self.model_config['endpoints']['generate_batch']}"
        
        # STEP 1: INITIALIZE TRACKING STRUCTURES WITH CHECKPOINT SUPPORT
        # ==============================================================
        
        # Set up checkpoint directory
        checkpoint_dir = Path(self.output_config.get('checkpoint_dir', 'outputs/checkpoints')) / 'target_word_evaluation'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique checkpoint filename based on dataset and target words
        import hashlib
        checkpoint_key = hashlib.md5(f"{dataset_path}_{target_words}_{n_samples}_{max_tokens}".encode()).hexdigest()[:8]
        checkpoint_file = checkpoint_dir / f"target_eval_checkpoint_{checkpoint_key}.json"
        
        # Try to resume from checkpoint
        pending_mcids, predictions, generation_details, batch_count = self._load_checkpoint(
            checkpoint_file, data, target_words, n_samples
        )
        
        # Fast lookup for remaining operations
        mcid_to_data = {row['mcid']: row for _, row in data.iterrows()}
        
        # Get batch size from config (defaults to 32 if not specified)
        batch_size = self.model_config.get('batch_size', 32)
        
        if batch_count == 0:
            logger.info(f"Starting NEW dictionary-based processing: {len(pending_mcids)} MCIDs, "
                       f"max {n_samples} samples each, batch_size={batch_size}")
        else:
            logger.info(f"RESUMING dictionary-based processing: {len(pending_mcids)} pending MCIDs, "
                       f"{len(predictions)} already resolved, batch_count={batch_count}")
        
        # Checkpoint save frequency
        checkpoint_every = self.target_config.get('checkpoint_every', 10)  # Save every 10 batches
        
        # Initialize batch failure tracking with bounded cache to prevent memory leaks
        self._batch_failures = OrderedDict()
        max_failure_tracking = 100  # Limit failure tracking to last 100 batches
        
        # STEP 2: MAIN PROCESSING LOOP
        # ============================
        # Continue until ALL MCIDs are resolved (either found target OR exhausted tries)
        
        # Initialize progress tracking
        total_mcids = len(data)
        initial_resolved = len(predictions)
        
        # Create progress bar with meaningful description
        progress_bar = tqdm(
            total=total_mcids,
            initial=initial_resolved, 
            desc="Processing MCIDs",
            unit="MCID",
            leave=True,
            bar_format="{l_bar}{bar}| {n}/{total} MCIDs [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        try:
            while pending_mcids:
                # 2a. SAMPLE BATCH: Take first batch_size items from pending dictionary
                # Dictionary iteration order is insertion order (Python 3.7+)
                # This ensures fair processing - no MCID gets starved
                batch_items = list(pending_mcids.items())[:batch_size]
                
                if not batch_items:
                    logger.warning("No batch items found but pending_mcids not empty - breaking")
                    break
                
                # Only increment batch_count for new batches, not retries
                # Create a unique key for this batch composition to detect retries
                is_retry = False
                batch_key = str(sorted([(mcid, tries) for mcid, tries in batch_items]))
                if hasattr(self, '_last_batch_key') and self._last_batch_key == batch_key:
                    is_retry = True
                    logger.debug(f"Retrying previous batch (batch {batch_count})")
                else:
                    batch_count += 1
                    self._last_batch_key = batch_key
                    logger.debug(f"Starting new batch {batch_count}")
                    
                # Update progress bar description with current batch info
                progress_bar.set_description(f"Batch {batch_count}: {len(batch_items)} MCIDs {'(retry)' if is_retry else ''}")
                logger.debug(f"Batch {batch_count}: Processing {len(batch_items)} MCIDs, "
                           f"{len(pending_mcids)} total pending {'(retry)' if is_retry else ''}")
                
                # 2b. BUILD BATCH PROMPTS
                # For each MCID in batch, prepare its prompt for generation
                batch_prompts = []
                batch_mcid_info = []  # Track (mcid, next_try_number) for each prompt
                
                for mcid, tries_completed in batch_items:
                    row_data = mcid_to_data[mcid]
                    batch_prompts.append(row_data['claims'])  # Medical claims text
                    batch_mcid_info.append((mcid, tries_completed + 1))  # This will be try #(tries_completed + 1)
                
                # 2c. GENERATE BATCH
                # Single API call for entire batch - much more efficient than individual calls
                try:
                    generated_texts = self._generate_cross_mcid_batch(
                        endpoint, batch_prompts, max_tokens, batch_mcid_info
                    )
                    
                    # 2d. PROCESS RESULTS AND UPDATE DICTIONARY
                    # This is where the magic happens - we decide which MCIDs to keep/remove
                    newly_resolved = []  # Track MCIDs that get resolved this batch
                    
                    # Track batch-level results for potential partial retry
                    batch_results = {'success': [], 'failed': [], 'continuing': []}
                    
                    for (mcid, try_number), generated_text in zip(batch_mcid_info, generated_texts):
                        
                        # Initialize tracking for new MCIDs (first time we see them)
                        if mcid not in generation_details:
                            row_data = mcid_to_data[mcid]
                            generation_details[mcid] = {
                                'mcid': mcid,
                                'prompt': row_data['claims'],
                                'true_label': row_data['label'],
                                'predicted_label': 0,  # Default to negative, update if target found
                                'samples': [],
                                'total_tries': 0,
                                'resolved_at_try': None,
                                'resolution_reason': None
                            }
                        
                        # Check if generated text contains any target medical codes
                        word_check_result = self._check_word_presence(generated_text, target_words)
                        
                        # Add this generation attempt to tracking
                        generation_details[mcid]['samples'].append({
                            'try_number': try_number,
                            'generated_text': generated_text,
                            'any_word_found': word_check_result['any_match'],
                            'word_matches': word_check_result['word_matches']
                        })
                        generation_details[mcid]['total_tries'] = try_number
                        
                        # DECISION LOGIC: Determine if this MCID should be resolved
                        # =========================================================
                        if word_check_result['any_match']:
                            # SUCCESS CASE: Found target medical code
                            # Mark as positive and remove from pending dictionary
                            predictions[mcid] = 1
                            generation_details[mcid]['predicted_label'] = 1
                            generation_details[mcid]['resolved_at_try'] = try_number
                            generation_details[mcid]['resolution_reason'] = 'target_found'
                            newly_resolved.append(mcid)
                            batch_results['success'].append(mcid)
                            logger.debug(f"✅ MCID {mcid} resolved POSITIVE at try {try_number}")
                            
                        elif try_number >= n_samples:
                            # EXHAUSTION CASE: Reached maximum allowed tries
                            # Mark as negative and remove from pending dictionary
                            predictions[mcid] = 0
                            generation_details[mcid]['predicted_label'] = 0
                            generation_details[mcid]['resolved_at_try'] = try_number
                            generation_details[mcid]['resolution_reason'] = 'max_tries_reached'
                            newly_resolved.append(mcid)
                            batch_results['success'].append(mcid)  # Successfully exhausted (negative result)
                            logger.debug(f"❌ MCID {mcid} resolved NEGATIVE after {try_number} tries")
                            
                        else:
                            # CONTINUE CASE: Keep trying
                            # Update try count in pending dictionary
                            pending_mcids[mcid] = try_number
                            batch_results['continuing'].append(mcid)
                            logger.debug(f"🔄 MCID {mcid} continuing, {try_number}/{n_samples} tries completed")
                
                    # STEP 3: AUTOMATIC CLEANUP
                    # =========================
                    # Remove all resolved MCIDs from pending dictionary
                    # This is the key efficiency: resolved MCIDs never get processed again
                    for mcid in newly_resolved:
                        pending_mcids.pop(mcid, None)  # Remove from pending (pop with default handles missing keys)
                    
                    # Update progress bar with newly resolved MCIDs
                    progress_bar.update(len(newly_resolved))
                    
                    # Log batch-level results for detailed tracking
                    success_count = len(batch_results['success'])
                    continuing_count = len(batch_results['continuing'])
                    logger.debug(f"Batch {batch_count} completed: {success_count} resolved, "
                               f"{continuing_count} continuing, {len(pending_mcids)} still pending")
                    
                    # CHECKPOINT: Save progress periodically
                    if batch_count % checkpoint_every == 0:
                        self._save_checkpoint(
                            checkpoint_file, pending_mcids, predictions, 
                            generation_details, batch_count, target_words, n_samples
                        )
                        logger.info(f"Checkpoint saved after batch {batch_count}")
                    
                except Exception as e:
                    # IMPROVED ERROR HANDLING: More resilient approach
                    logger.warning(f"Batch {batch_count} generation failed: {e}")
                    
                    # Track which MCIDs failed in this batch for detailed logging
                    failed_mcids = [mcid for mcid, _ in batch_mcid_info]
                    logger.debug(f"Failed MCIDs in batch {batch_count}: {failed_mcids[:5]}{'...' if len(failed_mcids) > 5 else ''}")
                    
                    # Count consecutive failures for this batch with bounded cache
                    batch_failure_key = f"batch_{batch_count}_failures"
                    
                    # Update failure count
                    self._batch_failures[batch_failure_key] = self._batch_failures.get(batch_failure_key, 0) + 1
                    
                    # Implement bounded cache to prevent memory leaks
                    if len(self._batch_failures) > max_failure_tracking:
                        # Remove oldest failure records
                        oldest_key = next(iter(self._batch_failures))
                        del self._batch_failures[oldest_key]
                        logger.debug(f"Cleaned up old failure tracking for {oldest_key}")
                    
                    max_batch_retries = self.target_config.get('max_batch_retries', 3)
                    
                    if self._batch_failures[batch_failure_key] <= max_batch_retries:
                        # RETRY: Keep MCIDs in pending for another attempt
                        logger.info(f"Retrying batch {batch_count} (attempt {self._batch_failures[batch_failure_key]}/{max_batch_retries})")
                        # Don't modify pending_mcids - they'll be retried in next iteration
                        continue
                    else:
                        # FINAL FAILURE: Only now mark MCIDs as failed after exhausting retries
                        logger.error(f"Batch {batch_count} failed permanently after {max_batch_retries} retries")
                    
                    for mcid, try_number in batch_mcid_info:
                        # Track error in generation details but don't immediately mark as negative
                        if mcid not in generation_details:
                            row_data = mcid_to_data[mcid]
                            generation_details[mcid] = {
                                'mcid': mcid,
                                'prompt': row_data['claims'],
                                'true_label': row_data['label'],
                                'predicted_label': 0,
                                'samples': [],
                                'total_tries': 0,
                                'resolved_at_try': None,
                                'resolution_reason': 'max_retries_exceeded'
                            }
                        
                        # Add error sample
                        generation_details[mcid]['samples'].append({
                            'try_number': try_number,
                            'generated_text': None,
                            'any_word_found': False,
                            'word_matches': {word: False for word in target_words},
                            'error': f"Batch failed after {max_batch_retries} retries: {str(e)}"
                        })
                        generation_details[mcid]['total_tries'] = try_number
                        
                        # Check if this MCID has reached max individual tries, or if we should mark as failed due to persistent API issues
                        max_individual_fails = self.target_config.get('max_individual_api_failures', 5)
                        api_failure_count = sum(1 for sample in generation_details[mcid]['samples'] if 'error' in sample)
                        
                        if try_number >= n_samples or api_failure_count >= max_individual_fails:
                            # Mark as negative and remove from pending
                            predictions[mcid] = 0
                            generation_details[mcid]['predicted_label'] = 0
                            generation_details[mcid]['resolved_at_try'] = try_number
                            generation_details[mcid]['resolution_reason'] = 'api_error' if api_failure_count >= max_individual_fails else 'max_tries_reached'
                            pending_mcids.pop(mcid, None)
                        else:
                            # Keep trying but increment failure count
                            pending_mcids[mcid] = try_number
        
        finally:
            # Ensure progress bar is always closed
            progress_bar.close()
        
        # STEP 4: FINALIZE RESULTS
        # ========================
        # Convert from dictionary format to ordered list format expected by caller
        
        # Memory usage statistics
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
        
        logger.info(f"Dictionary-based processing complete: {batch_count} batches, "
                   f"all {len(data)} MCIDs resolved")
        
        # CLEANUP: Remove checkpoint file on successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Checkpoint file removed after successful completion")
        
        # Memory optimization: Clear large intermediate structures
        del mcid_to_data  # Clear the lookup dictionary
        if hasattr(self, '_batch_failures'):
            del self._batch_failures  # Clear batch failure tracking
        
        # Validate results before finalizing
        self._validate_results(predictions, generation_details, data)
        
        # Create ordered results matching original data order
        final_predictions = []
        final_details = []
        
        for _, row in data.iterrows():
            mcid = row['mcid']
            final_predictions.append(predictions.get(mcid, 0))  # Default to negative if missing
            
            # Add summary statistics to details
            detail = generation_details.get(mcid, {})
            if detail and 'samples' in detail:
                processed_samples = [s for s in detail['samples'] if 'error' not in s]
                detail['word_found_count'] = sum(1 for s in processed_samples if s.get('any_word_found', False))
                detail['word_found_ratio'] = (detail['word_found_count'] / len(processed_samples) 
                                            if processed_samples else 0)
                detail['samples_processed'] = len(processed_samples)
                detail['samples_total'] = n_samples
                detail['early_termination'] = detail.get('resolution_reason') == 'target_found'
            
            final_details.append(detail)
        
        return np.array(final_predictions), final_details
    
    def _generate_cross_mcid_batch(self, endpoint: str, batch_prompts: List[str], 
                                 max_tokens: int, batch_mcid_info: List[tuple]) -> List[str]:
        """Generate one sample for each prompt in the batch (cross-MCID optimization)."""
        # Create batch payload - one generation per prompt with deterministic seeds based on MCID and try
        batch_payload = {
            'prompts': batch_prompts,
            'max_new_tokens': max_tokens,
            'temperature': self.target_config.get('temperature', 0.8),
            'top_k': self.target_config.get('top_k', 50),
            # Use deterministic but unique seeds based on MCID and try number to avoid collisions on resume
            'seeds': [hash(f"{mcid}_{try_number}") % (2**32) for mcid, try_number in batch_mcid_info]
        }
        
        for attempt in range(self.model_config['max_retries']):
            try:
                response = requests.post(
                    endpoint,
                    json=batch_payload,
                    timeout=self.model_config['timeout']
                )
                response.raise_for_status()
                
                # COMPREHENSIVE API RESPONSE VALIDATION
                try:
                    result = response.json()
                except ValueError as e:
                    raise ValueError(f"Invalid JSON response from API: {e}")
                
                # Validate response structure
                if not isinstance(result, dict):
                    raise ValueError(f"Expected JSON object response, got {type(result)}")
                
                if 'generated_texts' not in result:
                    raise ValueError(f"Missing 'generated_texts' field in API response. Available fields: {list(result.keys())}")
                
                generated_texts_response = result['generated_texts']
                if not isinstance(generated_texts_response, list):
                    raise ValueError(f"Expected 'generated_texts' to be a list, got {type(generated_texts_response)}")
                
                # Validate response length matches request
                expected_count = len(batch_prompts)
                actual_count = len(generated_texts_response)
                if actual_count != expected_count:
                    raise ValueError(f"API returned {actual_count} texts but expected {expected_count}")
                
                # Extract and validate generated texts
                generated_texts = []
                for i, generated_text in enumerate(generated_texts_response):
                    if generated_text is None:
                        logger.warning(f"API returned None for prompt {i}, using empty string")
                        generated_text = ""
                    elif not isinstance(generated_text, str):
                        logger.warning(f"API returned non-string type {type(generated_text)} for prompt {i}, converting to string")
                        generated_text = str(generated_text)
                    
                    prompt = batch_prompts[i]
                    # Extract only the generated part (excluding the prompt)
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    
                    # Additional validation - warn if generation is suspiciously short or long
                    text_length = len(generated_text)
                    if text_length == 0:
                        logger.warning(f"Empty generation for prompt {i}")
                    elif text_length > max_tokens * 10:  # Rough heuristic for too long
                        logger.warning(f"Suspiciously long generation ({text_length} chars) for prompt {i}")
                    
                    generated_texts.append(generated_text)
                
                logger.debug(f"Successfully validated API response: {len(generated_texts)} texts generated")
                return generated_texts
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Cross-MCID batch API call failed (attempt {attempt + 1}): {e}")
                if attempt == self.model_config['max_retries'] - 1:
                    raise
    
    def _check_word_presence(self, text: str, target_words: List[str]) -> Dict[str, bool]:
        """
        Check if any of the target words are present in the text.
        
        Args:
            text: Generated text to search in
            target_words: List of target words to search for
            
        Returns:
            Dictionary with word presence results and overall match
        """
        if text is None:
            return {'any_match': False, 'word_matches': {word: False for word in target_words}}
        
        search_method = self.target_config.get('search_method', 'exact')
        word_matches = {}
        
        for word in target_words:
            if search_method == 'exact':
                # Exact word match with word boundaries (case-sensitive for medical codes)
                pattern = r'\b' + re.escape(word) + r'\b'
                word_matches[word] = bool(re.search(pattern, text))
            elif search_method == 'fuzzy':
                # Simple substring match (case-sensitive for medical codes)
                word_matches[word] = word in text
            else:
                raise ValueError(f"Unknown search method: {search_method}")
        
        # Return True if ANY target word is found
        any_match = any(word_matches.values())
        
        return {
            'any_match': any_match,
            'word_matches': word_matches
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return metrics
    
    def _save_results(self, data: pd.DataFrame, predictions: np.ndarray,
                     generation_details: List[Dict], metrics: Dict[str, Any],
                     target_words: List[str]) -> str:
        """Save detailed evaluation results."""
        output_dir = Path(self.output_config['metrics_dir']) / 'target_word_evaluation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary results
        summary_path = output_dir / f'target_word_eval_summary_{timestamp}.json'
        summary = {
            'target_words': target_words,
            'metrics': metrics,
            'config': self.target_config,
            'timestamp': datetime.now().isoformat()
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        details_path = output_dir / f'target_word_eval_details_{timestamp}.json'
        with open(details_path, 'w') as f:
            json.dump(generation_details, f, indent=2)
        
        # Save predictions CSV
        predictions_df = pd.DataFrame({
            'mcid': data['mcid'],
            'true_label': data['label'],
            'predicted_label': predictions,
            'correct': data['label'].values == predictions
        })
        csv_path = output_dir / f'target_word_predictions_{timestamp}.csv'
        predictions_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to: {output_dir}")
        return str(summary_path)
    
    def _load_checkpoint(self, checkpoint_file: Path, data: pd.DataFrame, 
                        target_words: List[str], n_samples: int) -> Tuple[Dict, Dict, Dict, int]:
        """
        Load checkpoint data if available and compatible.
        
        Returns:
            Tuple of (pending_mcids, predictions, generation_details, batch_count)
        """
        if not checkpoint_file.exists():
            # No checkpoint exists - start fresh
            pending_mcids = {row['mcid']: 0 for _, row in data.iterrows()}
            return pending_mcids, {}, {}, 0
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Validate checkpoint compatibility
            if (checkpoint_data.get('target_words') != target_words or
                checkpoint_data.get('n_samples') != n_samples):
                logger.warning("Checkpoint incompatible with current parameters - starting fresh")
                pending_mcids = {row['mcid']: 0 for _, row in data.iterrows()}
                return pending_mcids, {}, {}, 0
            
            # Load checkpoint data
            pending_mcids = checkpoint_data.get('pending_mcids', {})
            predictions = checkpoint_data.get('predictions', {})
            generation_details = checkpoint_data.get('generation_details', {})
            batch_count = checkpoint_data.get('batch_count', 0)
            
            # Validate that all expected MCIDs are accounted for
            all_mcids = set(row['mcid'] for _, row in data.iterrows())
            resolved_mcids = set(predictions.keys())
            pending_mcids_set = set(pending_mcids.keys())
            
            if not (resolved_mcids | pending_mcids_set) >= all_mcids:
                logger.warning("Checkpoint missing some MCIDs - starting fresh")
                pending_mcids = {row['mcid']: 0 for _, row in data.iterrows()}
                return pending_mcids, {}, {}, 0
            
            logger.info(f"Loaded checkpoint: {len(resolved_mcids)} resolved, "
                       f"{len(pending_mcids)} pending, batch {batch_count}")
            return pending_mcids, predictions, generation_details, batch_count
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e} - starting fresh")
            pending_mcids = {row['mcid']: 0 for _, row in data.iterrows()}
            return pending_mcids, {}, {}, 0
    
    def _save_checkpoint(self, checkpoint_file: Path, pending_mcids: Dict, 
                        predictions: Dict, generation_details: Dict, batch_count: int,
                        target_words: List[str], n_samples: int):
        """Save current progress to checkpoint file."""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'target_words': target_words,  # Use actual parameter, not config
                'n_samples': n_samples,  # Use actual parameter, not config
                'pending_mcids': pending_mcids,
                'predictions': predictions,
                'generation_details': generation_details,
                'batch_count': batch_count,
                'total_resolved': len(predictions),
                'total_pending': len(pending_mcids)
            }
            
            # Write atomically using temporary file
            temp_file = checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(checkpoint_file)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Don't raise - continue processing even if checkpoint fails
    
    def _validate_results(self, predictions: Dict, generation_details: Dict, data: pd.DataFrame):
        """Validate that all MCIDs have been processed correctly."""
        logger.debug("Validating processing results...")
        
        all_mcids = set(row['mcid'] for _, row in data.iterrows())
        processed_mcids = set(predictions.keys())
        
        # Check for missing MCIDs
        missing = all_mcids - processed_mcids
        if missing:
            logger.error(f"Missing predictions for {len(missing)} MCIDs: {list(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            raise ValueError(f"Processing incomplete: {len(missing)} MCIDs missing predictions")
        
        # Check for extra MCIDs (shouldn't happen but good to catch)
        extra = processed_mcids - all_mcids
        if extra:
            logger.warning(f"Extra predictions for {len(extra)} unknown MCIDs: {list(extra)[:10]}{'...' if len(extra) > 10 else ''}")
        
        # Check for consistency between predictions and generation_details
        inconsistent_count = 0
        for mcid in processed_mcids:
            if mcid in generation_details:
                detail = generation_details[mcid]
                predicted_label = detail.get('predicted_label', -1)
                if predicted_label != predictions[mcid]:
                    logger.warning(f"Inconsistent prediction for MCID {mcid}: predictions={predictions[mcid]}, details={predicted_label}")
                    inconsistent_count += 1
        
        if inconsistent_count > 0:
            logger.warning(f"Found {inconsistent_count} inconsistent predictions between main dict and details")
        
        # Validate generation details structure
        details_issues = 0
        for mcid, detail in generation_details.items():
            required_fields = ['mcid', 'predicted_label', 'samples', 'total_tries']
            missing_fields = [field for field in required_fields if field not in detail]
            if missing_fields:
                logger.warning(f"MCID {mcid} missing detail fields: {missing_fields}")
                details_issues += 1
        
        if details_issues > 0:
            logger.warning(f"Found {details_issues} MCIDs with incomplete generation details")
        
        # Summary validation log
        total_processed = len(processed_mcids)
        total_expected = len(all_mcids)
        positive_predictions = sum(1 for pred in predictions.values() if pred == 1)
        negative_predictions = total_processed - positive_predictions
        
        logger.info(f"Validation complete: {total_processed}/{total_expected} MCIDs processed")
        logger.info(f"Results: {positive_predictions} positive, {negative_predictions} negative")
        
        if total_processed != total_expected:
            raise ValueError(f"Validation failed: Expected {total_expected} MCIDs, got {total_processed}")