"""
Tests for newly identified bugs in TargetWordEvaluator
"""

import pytest
import tempfile
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from unittest.mock import Mock, patch
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.target_word_evaluator import TargetWordEvaluator


class MockEvaluator(TargetWordEvaluator):
    """Mock evaluator that doesn't make real API calls."""
    
    def _generate_cross_mcid_batch(self, endpoint: str, batch_prompts: List[str], 
                                 max_tokens: int, batch_mcid_info: List[tuple]) -> List[str]:
        """Mock generation that returns predictable results."""
        results = []
        for prompt, (mcid, try_num) in zip(batch_prompts, batch_mcid_info):
            # Even MCIDs get target word, odd MCIDs don't
            if int(mcid) % 2 == 0:
                results.append(f"{prompt} TARGET")
            else:
                results.append(f"{prompt} no match")
        return results


def create_temp_dataset(tmp_path, rows: List[Dict[str, Any]], suffix: str = ".csv"):
    """Creates a temporary dataset file."""
    df = pd.DataFrame(rows)
    file_path = tmp_path / f"temp_dataset{suffix}"
    if suffix == ".csv":
        df.to_csv(file_path, index=False)
    elif suffix == ".json":
        df.to_json(file_path, orient="records", lines=False)
    elif suffix == ".parquet":
        df.to_parquet(file_path, index=False)
    return file_path


@pytest.fixture
def tmp_config(tmp_path):
    """Provides a temporary configuration dictionary."""
    config = {
        "model_api": {
            "base_url": "http://dummy-endpoint",
            "endpoints": {"generate_batch": "/generate_batch"},
            "batch_size": 2,
            "max_retries": 1,
            "timeout": 5
        },
        "target_word_evaluation": {
            "search_method": "exact",
            "checkpoint_every": 1,
            "max_batch_retries": 1,
            "global_timeout_minutes": 1,
            "max_batches": 5,
            "temperature": 0.5,
            "top_k": 10
        },
        "output": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "metrics_dir": str(tmp_path / "metrics")
        }
    }
    return config


class TestBug1MixedTypes:
    """Test for Bug #1: Mixed types for generation_details"""
    
    def test_generation_details_is_dict_not_list(self, tmp_path, tmp_config):
        """Ensure generation_details is returned as dict, not list."""
        rows = [
            {"claims": "Claim A", "mcid": "1", "label": 0},
            {"claims": "Claim B", "mcid": "2", "label": 1}
        ]
        temp_file = create_temp_dataset(tmp_path, rows)
        evaluator = MockEvaluator(tmp_config)
        
        results = evaluator.evaluate(str(temp_file), ["TARGET"], n_samples=1, max_tokens=5)
        
        # Check saved details file is a dict
        details_files = list(Path(tmp_config["output"]["metrics_dir"]).glob("**/target_word_eval_details_*.json"))
        assert len(details_files) > 0
        
        with open(details_files[0], 'r') as f:
            saved_details = json.load(f)
        
        # Should be a dict keyed by mcid, not a list
        assert isinstance(saved_details, dict)
        assert "1" in saved_details
        assert "2" in saved_details
        assert isinstance(saved_details["1"], dict)
        # mcid might be stored as int or string depending on implementation
        assert str(saved_details["1"]["mcid"]) == "1"


class TestBug2MissingDefaults:
    """Test for Bug #2: Missing defaults for max_retries and timeout"""
    
    def test_missing_max_retries_has_default(self, tmp_path):
        """Test that missing max_retries doesn't cause KeyError."""
        config = {
            "model_api": {
                "base_url": "http://dummy-endpoint",
                "endpoints": {"generate_batch": "/generate_batch"},
                # Intentionally missing max_retries
                "timeout": 5
            },
            "target_word_evaluation": {
                "search_method": "exact"
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "metrics_dir": str(tmp_path / "metrics")
            }
        }
        
        evaluator = MockEvaluator(config)
        # Should not raise KeyError
        assert evaluator is not None
    
    def test_missing_timeout_has_default(self, tmp_path):
        """Test that missing timeout doesn't cause KeyError."""
        config = {
            "model_api": {
                "base_url": "http://dummy-endpoint",
                "endpoints": {"generate_batch": "/generate_batch"},
                "max_retries": 3
                # Intentionally missing timeout
            },
            "target_word_evaluation": {
                "search_method": "exact"
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "metrics_dir": str(tmp_path / "metrics")
            }
        }
        
        evaluator = MockEvaluator(config)
        # Should not raise KeyError
        assert evaluator is not None


class TestBug3CheckpointCompatibility:
    """Test for Bug #3: Checkpoint compatibility check logic"""
    
    def test_checkpoint_incompatible_max_tokens(self, tmp_path, tmp_config):
        """Test checkpoint is rejected when max_tokens differs."""
        evaluator = MockEvaluator(tmp_config)
        
        # Create checkpoint with different max_tokens
        checkpoint_dir = Path(tmp_config["output"]["checkpoint_dir"]) / 'target_word_evaluation'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "test_checkpoint.json"
        
        checkpoint_data = {
            'target_words': ["TARGET"],
            'n_samples': 2,
            'max_tokens': 10,  # Different from test value of 5
            'pending_mcids': {"1": 0},
            'predictions': {},
            'generation_details': {},
            'batch_count': 0
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Load with different max_tokens
        data = pd.DataFrame([{"claims": "Test", "mcid": "1", "label": 0}])
        pending, predictions, details, batch_count = evaluator._load_checkpoint(
            checkpoint_file, data, ["TARGET"], n_samples=2, max_tokens=5
        )
        
        # Should start fresh due to max_tokens mismatch
        assert len(pending) == 1
        assert "1" in pending
        assert pending["1"] == 0
        assert len(predictions) == 0
        assert batch_count == 0
    
    def test_checkpoint_compatible_all_params(self, tmp_path, tmp_config):
        """Test checkpoint is loaded when all params match."""
        evaluator = MockEvaluator(tmp_config)
        
        checkpoint_dir = Path(tmp_config["output"]["checkpoint_dir"]) / 'target_word_evaluation'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "test_checkpoint.json"
        
        checkpoint_data = {
            'target_words': ["TARGET"],
            'n_samples': 2,
            'max_tokens': 5,  # Matches test value
            'pending_mcids': {"1": 1},
            'predictions': {"2": 1},
            'generation_details': {"2": {"mcid": "2", "predicted_label": 1}},
            'batch_count': 3
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Load with same parameters
        data = pd.DataFrame([
            {"claims": "Test1", "mcid": "1", "label": 0},
            {"claims": "Test2", "mcid": "2", "label": 1}
        ])
        pending, predictions, details, batch_count = evaluator._load_checkpoint(
            checkpoint_file, data, ["TARGET"], n_samples=2, max_tokens=5
        )
        
        # Should load existing checkpoint
        assert pending == {"1": 1}
        assert predictions == {"2": 1}
        assert batch_count == 3


class TestBug4OutputConfigDefaults:
    """Test for Bug #4: Missing defaults for output config"""
    
    def test_missing_metrics_dir_has_default(self, tmp_path):
        """Test that missing metrics_dir doesn't cause KeyError."""
        config = {
            "model_api": {
                "base_url": "http://dummy-endpoint",
                "endpoints": {"generate_batch": "/generate_batch"},
                "max_retries": 1,
                "timeout": 5
            },
            "target_word_evaluation": {
                "search_method": "exact"
            },
            "output": {
                "checkpoint_dir": str(tmp_path / "checkpoints")
                # Intentionally missing metrics_dir
            }
        }
        
        rows = [{"claims": "Test", "mcid": "1", "label": 0}]
        temp_file = create_temp_dataset(tmp_path, rows)
        
        evaluator = MockEvaluator(config)
        # Should not raise KeyError
        results = evaluator.evaluate(str(temp_file), ["TARGET"], n_samples=1, max_tokens=5)
        assert "accuracy" in results
        
        # Check that default directory was created
        default_metrics_dir = Path("outputs/metrics/target_word_evaluation")
        assert default_metrics_dir.exists()
        
        # Cleanup
        import shutil
        if Path("outputs").exists():
            shutil.rmtree("outputs")


class TestBug5EmptyTargetWords:
    """Test for Bug #5: Empty-string prompt trimming"""
    
    def test_empty_string_target_words(self, tmp_path, tmp_config):
        """Test error message for empty target words."""
        rows = [{"claims": "Test", "mcid": "1", "label": 0}]
        temp_file = create_temp_dataset(tmp_path, rows)
        evaluator = MockEvaluator(tmp_config)
        
        with pytest.raises(ValueError) as excinfo:
            evaluator.evaluate(str(temp_file), [" ", "  ", ""], n_samples=1, max_tokens=5)
        
        assert "No valid target words provided" in str(excinfo.value)


class TestBug6ProgressBarHang:
    """Test for Bug #6: Progress bar hang with exceptions"""
    
    @patch('evaluation.target_word_evaluator.requests.post')
    def test_max_retries_prevents_infinite_loop(self, mock_post, tmp_path, tmp_config):
        """Test that max retries prevents infinite loops."""
        # Configure to always fail
        mock_post.side_effect = Exception("API Error")
        
        rows = [
            {"claims": "Test1", "mcid": "1", "label": 0},
            {"claims": "Test2", "mcid": "2", "label": 1}
        ]
        temp_file = create_temp_dataset(tmp_path, rows)
        
        # Use real evaluator to test actual error handling
        evaluator = TargetWordEvaluator(tmp_config)
        
        # Should complete without hanging (all MCIDs marked as failed)
        results = evaluator.evaluate(str(temp_file), ["TARGET"], n_samples=2, max_tokens=5)
        
        # All predictions should be 0 (negative) due to failures
        assert results["accuracy"] == 0.5  # Both predicted as 0, one is correct


class TestBug7WordBoundaryRegex:
    """Test for Bug #7: Word boundary regex behavior"""
    
    def test_word_boundary_with_medical_codes(self, tmp_path, tmp_config):
        """Test exact match with medical codes."""
        evaluator = MockEvaluator(tmp_config)
        
        # Test various cases
        test_cases = [
            ("Patient has E119 code", ["E119"], True),  # Normal case
            ("Patient has E119.", ["E119"], True),      # With punctuation
            ("Patient has E1190", ["E119"], False),     # Part of larger code
            ("E119 at start", ["E119"], True),          # At start
            ("End with E119", ["E119"], True),          # At end
            ("76642", ["76642"], True),                 # Numeric only
            ("376642", ["76642"], False),               # Part of larger number
        ]
        
        for text, target_words, expected_match in test_cases:
            result = evaluator._check_word_presence(text, target_words)
            assert result["any_match"] == expected_match, f"Failed for: {text}"
    
    def test_fuzzy_match_behavior(self, tmp_path, tmp_config):
        """Test fuzzy match doesn't use word boundaries."""
        evaluator = MockEvaluator(tmp_config)
        evaluator.target_config["search_method"] = "fuzzy"
        
        # Fuzzy should find partial matches
        result = evaluator._check_word_presence("Patient has E1190", ["E119"])
        assert result["any_match"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])