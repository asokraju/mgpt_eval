"""
Tests for additional fixes found during code review
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.target_word_evaluator import TargetWordEvaluator


class MockEvaluatorSimple(TargetWordEvaluator):
    """Simple mock for testing input validation."""
    
    def _generate_cross_mcid_batch(self, endpoint, batch_prompts, max_tokens, batch_mcid_info):
        return ["mock response"] * len(batch_prompts)


@pytest.fixture
def basic_config():
    """Basic valid config for testing."""
    return {
        'model_api': {
            'base_url': 'http://localhost:8000',
            'endpoints': {'generate_batch': '/generate_batch'},
            'max_retries': 1,
            'timeout': 5
        },
        'target_word_evaluation': {
            'search_method': 'exact'
        },
        'output': {
            'metrics_dir': 'test_output'
        }
    }


class TestInputValidation:
    """Test input validation improvements."""
    
    def test_negative_n_samples(self, basic_config, tmp_path):
        """Test error for negative n_samples."""
        evaluator = MockEvaluatorSimple(basic_config)
        
        # Create minimal test data
        data = pd.DataFrame({'mcid': [1], 'claims': ['test'], 'label': [0]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            evaluator.evaluate(str(csv_file), ['target'], n_samples=-1, max_tokens=10)
    
    def test_zero_n_samples(self, basic_config, tmp_path):
        """Test error for zero n_samples."""
        evaluator = MockEvaluatorSimple(basic_config)
        
        data = pd.DataFrame({'mcid': [1], 'claims': ['test'], 'label': [0]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            evaluator.evaluate(str(csv_file), ['target'], n_samples=0, max_tokens=10)
    
    def test_negative_max_tokens(self, basic_config, tmp_path):
        """Test error for negative max_tokens."""
        evaluator = MockEvaluatorSimple(basic_config)
        
        data = pd.DataFrame({'mcid': [1], 'claims': ['test'], 'label': [0]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            evaluator.evaluate(str(csv_file), ['target'], n_samples=1, max_tokens=-10)
    
    def test_non_integer_n_samples(self, basic_config, tmp_path):
        """Test error for non-integer n_samples."""
        evaluator = MockEvaluatorSimple(basic_config)
        
        data = pd.DataFrame({'mcid': [1], 'claims': ['test'], 'label': [0]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            evaluator.evaluate(str(csv_file), ['target'], n_samples=1.5, max_tokens=10)
    
    def test_warning_for_large_n_samples(self, basic_config, tmp_path, caplog):
        """Test warning for large n_samples."""
        evaluator = MockEvaluatorSimple(basic_config)
        
        data = pd.DataFrame({'mcid': [1], 'claims': ['test'], 'label': [0]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        evaluator.evaluate(str(csv_file), ['target'], n_samples=1001, max_tokens=10)
        assert "Large n_samples (1001) may cause performance issues" in caplog.text
    
    def test_warning_for_large_max_tokens(self, basic_config, tmp_path, caplog):
        """Test warning for large max_tokens."""
        evaluator = MockEvaluatorSimple(basic_config)
        
        data = pd.DataFrame({'mcid': [1], 'claims': ['test'], 'label': [0]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        evaluator.evaluate(str(csv_file), ['target'], n_samples=1, max_tokens=5000)
        assert "Large max_tokens (5000) may cause API issues" in caplog.text


class TestEndpointHandling:
    """Test endpoint URL construction."""
    
    def test_missing_endpoints_config(self, tmp_path):
        """Test handling when endpoints config is missing."""
        config = {
            'model_api': {
                'base_url': 'http://localhost:8000',
                # Missing 'endpoints' section
                'max_retries': 1,
                'timeout': 5
            },
            'target_word_evaluation': {
                'search_method': 'exact'
            },
            'output': {
                'metrics_dir': 'test_output'
            }
        }
        
        evaluator = MockEvaluatorSimple(config)
        
        data = pd.DataFrame({'mcid': [1], 'claims': ['test'], 'label': [0]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        # Should work with default endpoint
        result = evaluator.evaluate(str(csv_file), ['target'], n_samples=1, max_tokens=10)
        assert 'accuracy' in result
    
    def test_empty_endpoints_config(self, tmp_path):
        """Test handling when endpoints config is empty."""
        config = {
            'model_api': {
                'base_url': 'http://localhost:8000',
                'endpoints': {},  # Empty endpoints
                'max_retries': 1,
                'timeout': 5
            },
            'target_word_evaluation': {
                'search_method': 'exact'
            },
            'output': {
                'metrics_dir': 'test_output'
            }
        }
        
        evaluator = MockEvaluatorSimple(config)
        
        data = pd.DataFrame({'mcid': [1], 'claims': ['test'], 'label': [0]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        # Should work with default endpoint
        result = evaluator.evaluate(str(csv_file), ['target'], n_samples=1, max_tokens=10)
        assert 'accuracy' in result


class TestTypeAnnotations:
    """Test that return types match annotations."""
    
    def test_generate_predictions_return_type(self, basic_config, tmp_path):
        """Test that _generate_predictions returns correct types."""
        evaluator = MockEvaluatorSimple(basic_config)
        
        data = pd.DataFrame({'mcid': [1, 2], 'claims': ['test1', 'test2'], 'label': [0, 1]})
        csv_file = tmp_path / 'test.csv'
        data.to_csv(csv_file, index=False)
        
        result = evaluator.evaluate(str(csv_file), ['target'], n_samples=1, max_tokens=10)
        
        # Verify the result structure indicates correct internal types
        assert isinstance(result['accuracy'], float)
        assert isinstance(result['target_words'], list)
        assert isinstance(result['n_samples'], int)
        assert isinstance(result['max_tokens'], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])