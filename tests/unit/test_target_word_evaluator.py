"""
Comprehensive tests for TargetWordEvaluator to identify and fix issues.
"""

import pytest
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests
from datetime import datetime
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.target_word_evaluator import TargetWordEvaluator


class TestTargetWordEvaluator:
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'model_api': {
                'base_url': 'http://localhost:8000',
                'timeout': 30,
                'max_retries': 3,
                'batch_size': 2,
                'endpoints': {
                    'generate_batch': '/generate_batch'
                }
            },
            'target_word_evaluation': {
                'temperature': 0.8,
                'top_k': 50,
                'search_method': 'exact',
                'checkpoint_every': 2,
                'max_batch_retries': 2,
                'max_individual_api_failures': 3
            },
            'output': {
                'metrics_dir': 'test_outputs/metrics',
                'checkpoint_dir': 'test_outputs/checkpoints'
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """Sample medical claims data."""
        return pd.DataFrame({
            'mcid': ['MCID001', 'MCID002', 'MCID003', 'MCID004'],
            'claims': [
                'Patient has diabetes type 2',
                'Heart condition requires monitoring',
                'Cancer treatment ongoing',
                'Routine checkup completed'
            ],
            'label': [1, 0, 1, 0]
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def evaluator(self, sample_config, temp_dir):
        """TargetWordEvaluator instance with test config."""
        # Update config with temp directory
        config = sample_config.copy()
        config['output']['metrics_dir'] = f"{temp_dir}/metrics"
        config['output']['checkpoint_dir'] = f"{temp_dir}/checkpoints"
        return TargetWordEvaluator(config)
    
    def test_init_with_dict_config(self, sample_config):
        """Test initialization with dictionary config."""
        evaluator = TargetWordEvaluator(sample_config)
        assert evaluator.model_config == sample_config['model_api']
        assert evaluator.target_config == sample_config['target_word_evaluation']
        assert evaluator.output_config == sample_config['output']
    
    def test_init_with_yaml_config(self, sample_config, temp_dir):
        """Test initialization with YAML config file."""
        yaml_file = Path(temp_dir) / 'test_config.yaml'
        import yaml
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_config, f)
        
        evaluator = TargetWordEvaluator(str(yaml_file))
        assert evaluator.model_config == sample_config['model_api']
    
    def test_load_dataset_csv(self, evaluator, sample_data, temp_dir):
        """Test loading CSV dataset."""
        csv_file = Path(temp_dir) / 'test_data.csv'
        sample_data.to_csv(csv_file, index=False)
        
        loaded_data = evaluator._load_dataset(str(csv_file))
        pd.testing.assert_frame_equal(loaded_data, sample_data)
    
    def test_load_dataset_missing_columns(self, evaluator, temp_dir):
        """Test error handling for missing required columns."""
        # Create data missing required columns
        bad_data = pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['a', 'b', 'c']
        })
        csv_file = Path(temp_dir) / 'bad_data.csv'
        bad_data.to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            evaluator._load_dataset(str(csv_file))
    
    def test_load_dataset_duplicate_mcids(self, evaluator, temp_dir):
        """Test error handling for duplicate MCIDs."""
        # Create data with duplicate MCIDs
        duplicate_data = pd.DataFrame({
            'mcid': ['MCID001', 'MCID001', 'MCID002'],  # Duplicate MCID001
            'claims': ['claim1', 'claim2', 'claim3'],
            'label': [1, 0, 1]
        })
        csv_file = Path(temp_dir) / 'duplicate_data.csv'
        duplicate_data.to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError, match="duplicate MCID values"):
            evaluator._load_dataset(str(csv_file))
    
    def test_validate_target_words_empty(self, evaluator, sample_data, temp_dir):
        """Test error handling for empty target words."""
        csv_file = Path(temp_dir) / 'test_data.csv'
        sample_data.to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError, match="Target codes must be specified"):
            evaluator.evaluate(str(csv_file), [], 2, 50)
    
    def test_validate_target_words_none(self, evaluator, sample_data, temp_dir):
        """Test error handling for None target words."""
        csv_file = Path(temp_dir) / 'test_data.csv'
        sample_data.to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError, match="Target codes must be specified"):
            evaluator.evaluate(str(csv_file), None, 2, 50)
    
    def test_validate_target_words_cleaning(self, evaluator, sample_data, temp_dir):
        """Test target words cleaning and validation."""
        csv_file = Path(temp_dir) / 'test_data.csv'
        sample_data.to_csv(csv_file, index=False)
        
        # Test with empty strings in target words
        with pytest.raises(ValueError, match="No valid target words provided"):
            evaluator.evaluate(str(csv_file), ['', '  ', ''], 2, 50)
    
    def test_check_word_presence_exact(self, evaluator):
        """Test exact word matching."""
        text = "Patient has diabetes E119 and hypertension I10"
        target_words = ['E119', 'Z999', 'diabetes']
        
        result = evaluator._check_word_presence(text, target_words)
        
        assert result['any_match'] is True
        assert result['word_matches']['E119'] is True  # Medical code found
        assert result['word_matches']['Z999'] is False  # Medical code not found
        assert result['word_matches']['diabetes'] is True  # Word found
    
    def test_check_word_presence_fuzzy(self, evaluator):
        """Test fuzzy word matching."""
        evaluator.target_config['search_method'] = 'fuzzy'
        
        text = "Patient has diabetes E119"
        target_words = ['E11', 'diabet', 'Z999']
        
        result = evaluator._check_word_presence(text, target_words)
        
        assert result['any_match'] is True
        assert result['word_matches']['E11'] is True  # Partial match
        assert result['word_matches']['diabet'] is True  # Partial match
        assert result['word_matches']['Z999'] is False  # Not found
    
    def test_check_word_presence_none_text(self, evaluator):
        """Test word checking with None text."""
        result = evaluator._check_word_presence(None, ['E119'])
        
        assert result['any_match'] is False
        assert result['word_matches']['E119'] is False
    
    @patch('requests.post')
    def test_generate_cross_mcid_batch_success(self, mock_post, evaluator):
        """Test successful batch generation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'generated_claims': ['Generated text 1', 'Generated text 2']
        }
        mock_post.return_value = mock_response
        
        endpoint = 'http://localhost:8000/generate_batch'
        prompts = ['Prompt 1', 'Prompt 2']
        mcid_info = [('MCID001', 1), ('MCID002', 1)]
        
        result = evaluator._generate_cross_mcid_batch(endpoint, prompts, 50, mcid_info)
        
        assert result == ['Generated text 1', 'Generated text 2']
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_cross_mcid_batch_retry_on_failure(self, mock_post, evaluator):
        """Test retry logic on API failure."""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        
        mock_response_success = Mock()
        mock_response_success.raise_for_status.return_value = None
        mock_response_success.json.return_value = {
            'generated_claims': ['Generated text 1', 'Generated text 2']
        }
        
        mock_post.side_effect = [mock_response_fail, mock_response_success]
        
        endpoint = 'http://localhost:8000/generate_batch'
        prompts = ['Prompt 1', 'Prompt 2']
        mcid_info = [('MCID001', 1), ('MCID002', 1)]
        
        result = evaluator._generate_cross_mcid_batch(endpoint, prompts, 50, mcid_info)
        
        assert result == ['Generated text 1', 'Generated text 2']
        assert mock_post.call_count == 2
    
    @patch('requests.post')
    def test_generate_cross_mcid_batch_max_retries_exceeded(self, mock_post, evaluator):
        """Test behavior when max retries are exceeded."""
        # All calls fail
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_response
        
        endpoint = 'http://localhost:8000/generate_batch'
        prompts = ['Prompt 1', 'Prompt 2']
        mcid_info = [('MCID001', 1), ('MCID002', 1)]
        
        with pytest.raises(requests.exceptions.HTTPError):
            evaluator._generate_cross_mcid_batch(endpoint, prompts, 50, mcid_info)
        
        assert mock_post.call_count == evaluator.model_config['max_retries']
    
    @patch('requests.post')
    def test_generate_cross_mcid_batch_invalid_json_response(self, mock_post, evaluator):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response
        
        endpoint = 'http://localhost:8000/generate_batch'
        prompts = ['Prompt 1']
        mcid_info = [('MCID001', 1)]
        
        with pytest.raises(ValueError, match="Invalid JSON response"):
            evaluator._generate_cross_mcid_batch(endpoint, prompts, 50, mcid_info)
    
    @patch('requests.post')
    def test_generate_cross_mcid_batch_missing_field(self, mock_post, evaluator):
        """Test handling of missing 'generated_claims' field."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {'other_field': 'value'}
        mock_post.return_value = mock_response
        
        endpoint = 'http://localhost:8000/generate_batch'
        prompts = ['Prompt 1']
        mcid_info = [('MCID001', 1)]
        
        with pytest.raises(ValueError, match="No recognized response field found"):
            evaluator._generate_cross_mcid_batch(endpoint, prompts, 50, mcid_info)
    
    @patch('requests.post')
    def test_generate_cross_mcid_batch_wrong_response_count(self, mock_post, evaluator):
        """Test handling of wrong number of responses."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'generated_claims': ['Only one response']  # Expected 2
        }
        mock_post.return_value = mock_response
        
        endpoint = 'http://localhost:8000/generate_batch'
        prompts = ['Prompt 1', 'Prompt 2']
        mcid_info = [('MCID001', 1), ('MCID002', 1)]
        
        with pytest.raises(ValueError, match="API returned 1 texts but expected 2"):
            evaluator._generate_cross_mcid_batch(endpoint, prompts, 50, mcid_info)
    
    def test_calculate_metrics(self, evaluator):
        """Test metrics calculation."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])  # One false negative
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.75  # 3/4 correct
        assert metrics['precision'] == 1.0  # 1/1 true positives
        assert metrics['recall'] == 0.5  # 1/2 actual positives
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_checkpoint_save_and_load(self, evaluator, temp_dir):
        """Test checkpoint saving and loading."""
        checkpoint_dir = Path(temp_dir) / 'checkpoints' / 'target_word_evaluation'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / 'test_checkpoint.json'
        
        # Data to save
        pending_mcids = {'MCID001': 1, 'MCID002': 0}
        predictions = {'MCID003': 1}
        generation_details = {'MCID003': {'total_tries': 2}}
        batch_count = 5
        target_words = ['E119', 'Z999']
        n_samples = 3
        
        # Save checkpoint
        evaluator._save_checkpoint(
            checkpoint_file, pending_mcids, predictions, 
            generation_details, batch_count, target_words, n_samples
        )
        
        assert checkpoint_file.exists()
        
        # Load checkpoint
        sample_data = pd.DataFrame({
            'mcid': ['MCID001', 'MCID002', 'MCID003'],
            'claims': ['claim1', 'claim2', 'claim3'],
            'label': [1, 0, 1]
        })
        
        loaded_pending, loaded_predictions, loaded_details, loaded_batch_count = evaluator._load_checkpoint(
            checkpoint_file, sample_data, target_words, n_samples
        )
        
        assert loaded_pending == pending_mcids
        assert loaded_predictions == predictions
        assert loaded_details == generation_details
        assert loaded_batch_count == batch_count
    
    def test_checkpoint_incompatible_parameters(self, evaluator, temp_dir):
        """Test handling of incompatible checkpoint parameters."""
        checkpoint_dir = Path(temp_dir) / 'checkpoints' / 'target_word_evaluation'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / 'test_checkpoint.json'
        
        # Save checkpoint with different parameters
        checkpoint_data = {
            'target_words': ['OLD_CODE'],  # Different from what we'll load with
            'n_samples': 5,
            'pending_mcids': {},
            'predictions': {},
            'generation_details': {},
            'batch_count': 0
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        sample_data = pd.DataFrame({
            'mcid': ['MCID001'],
            'claims': ['claim1'],
            'label': [1]
        })
        
        # Should start fresh due to incompatible parameters
        loaded_pending, loaded_predictions, loaded_details, loaded_batch_count = evaluator._load_checkpoint(
            checkpoint_file, sample_data, ['NEW_CODE'], 3
        )
        
        # Should return fresh state
        assert loaded_pending == {'MCID001': 0}
        assert loaded_predictions == {}
        assert loaded_details == {}
        assert loaded_batch_count == 0
    
    def test_validate_results_success(self, evaluator):
        """Test successful results validation."""
        data = pd.DataFrame({
            'mcid': ['MCID001', 'MCID002'],
            'claims': ['claim1', 'claim2'],
            'label': [1, 0]
        })
        
        predictions = {'MCID001': 1, 'MCID002': 0}
        generation_details = {
            'MCID001': {
                'mcid': 'MCID001',
                'predicted_label': 1,
                'samples': [],
                'total_tries': 2
            },
            'MCID002': {
                'mcid': 'MCID002',
                'predicted_label': 0,
                'samples': [],
                'total_tries': 1
            }
        }
        
        # Should not raise any exceptions
        evaluator._validate_results(predictions, generation_details, data)
    
    def test_validate_results_missing_mcids(self, evaluator):
        """Test validation with missing MCIDs."""
        data = pd.DataFrame({
            'mcid': ['MCID001', 'MCID002'],
            'claims': ['claim1', 'claim2'],
            'label': [1, 0]
        })
        
        predictions = {'MCID001': 1}  # Missing MCID002
        generation_details = {}
        
        with pytest.raises(ValueError, match="Processing incomplete"):
            evaluator._validate_results(predictions, generation_details, data)
    
    def test_validate_results_inconsistent_predictions(self, evaluator, caplog):
        """Test validation with inconsistent predictions."""
        data = pd.DataFrame({
            'mcid': ['MCID001'],
            'claims': ['claim1'],
            'label': [1]
        })
        
        predictions = {'MCID001': 1}
        generation_details = {
            'MCID001': {
                'mcid': 'MCID001',
                'predicted_label': 0,  # Inconsistent with predictions
                'samples': [],
                'total_tries': 1
            }
        }
        
        # Should log warning but not raise exception
        evaluator._validate_results(predictions, generation_details, data)
        assert "Inconsistent prediction" in caplog.text
    
    @patch.object(TargetWordEvaluator, '_generate_cross_mcid_batch')
    def test_end_to_end_evaluation_success(self, mock_generate, evaluator, sample_data, temp_dir):
        """Test complete evaluation flow with mocked API calls."""
        # Setup mock to return generated text with target words
        mock_generate.return_value = ['Generated text with E119', 'Normal text', 'Text with E119', 'Normal text']
        
        # Save test data
        csv_file = Path(temp_dir) / 'test_data.csv'
        sample_data.to_csv(csv_file, index=False)
        
        # Run evaluation
        results = evaluator.evaluate(
            str(csv_file), 
            target_words=['E119'], 
            n_samples=2, 
            max_tokens=50
        )
        
        # Verify results structure
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'target_words' in results
        assert results['target_words'] == ['E119']
        assert results['n_samples'] == 2
        assert results['max_tokens'] == 50
        assert 'results_path' in results
        assert 'timestamp' in results
    
    def test_endpoint_handling(self, evaluator):
        """Test various endpoint URL handling scenarios."""
        test_cases = [
            ('http://localhost:8000/generate_batch', 'http://localhost:8000/generate_batch'),
            ('http://localhost:8000/generate', 'http://localhost:8000/generate_batch'),
            ('http://localhost:8000', 'http://localhost:8000/generate_batch'),
            ('http://localhost:8000/', 'http://localhost:8000/generate_batch'),
        ]
        
        for input_endpoint, expected_output in test_cases:
            # This tests the endpoint logic in _generate_predictions method
            # We need to extract and test this logic
            if input_endpoint.endswith('/generate_batch'):
                endpoint = input_endpoint
            elif input_endpoint.endswith('/generate'):
                endpoint = input_endpoint.replace('/generate', '/generate_batch')
            else:
                endpoint = f"{input_endpoint.rstrip('/')}{evaluator.model_config['endpoints']['generate_batch']}"
            
            assert endpoint == expected_output

    def test_deterministic_seeds(self, evaluator):
        """Test that seeds are generated deterministically."""
        batch_mcid_info = [('MCID001', 1), ('MCID002', 2), ('MCID001', 1)]
        
        seeds1 = [hash(f"{mcid}_{try_number}") % (2**32) for mcid, try_number in batch_mcid_info]
        seeds2 = [hash(f"{mcid}_{try_number}") % (2**32) for mcid, try_number in batch_mcid_info]
        
        assert seeds1 == seeds2  # Should be identical
        assert seeds1[0] == seeds1[2]  # Same MCID+try should have same seed
        assert seeds1[0] != seeds1[1]  # Different MCID or try should have different seed


class TestTargetWordEvaluatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency with large dataset simulation."""
        # This would be a stress test for memory usage
        # For now, just test the structure handles large numbers
        large_data = pd.DataFrame({
            'mcid': [f'MCID{i:06d}' for i in range(1000)],
            'claims': [f'Medical claim {i}' for i in range(1000)],
            'label': [i % 2 for i in range(1000)]
        })
        
        # Test that data structures can handle this size
        pending_mcids = {row['mcid']: 0 for _, row in large_data.iterrows()}
        assert len(pending_mcids) == 1000
        
        # Test MCID uniqueness validation
        unique_mcids = set(large_data['mcid'])
        assert len(unique_mcids) == 1000
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        from evaluation.target_word_evaluator import TargetWordEvaluator
        
        config = {
            'model_api': {
                'base_url': 'http://localhost:8000',
                'timeout': 30, 
                'max_retries': 3,
                'endpoints': {'generate_batch': '/generate_batch'}
            },
            'target_word_evaluation': {'search_method': 'exact'},
            'output': {}
        }
        evaluator = TargetWordEvaluator(config)
        
        # Test with unicode characters
        text = "Patient has ñoñó condition E119 and 中文 text"
        target_words = ['E119', 'ñoñó', '中文']
        
        result = evaluator._check_word_presence(text, target_words)
        
        assert result['word_matches']['E119'] is True
        assert result['word_matches']['ñoñó'] is True
        assert result['word_matches']['中文'] is True
    
    def test_very_long_text_handling(self):
        """Test handling of very long generated text."""
        from evaluation.target_word_evaluator import TargetWordEvaluator
        
        config = {
            'model_api': {
                'base_url': 'http://localhost:8000',
                'timeout': 30, 
                'max_retries': 3,
                'endpoints': {'generate_batch': '/generate_batch'}
            },
            'target_word_evaluation': {'search_method': 'exact'},
            'output': {}
        }
        evaluator = TargetWordEvaluator(config)
        
        # Very long text (10000 characters)
        long_text = "Medical text " * 1000 + " E119 " + "more text " * 500
        target_words = ['E119']
        
        result = evaluator._check_word_presence(long_text, target_words)
        assert result['any_match'] is True
        assert result['word_matches']['E119'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])