"""
Tests for target word evaluator functionality.
"""

import pytest
import json
import requests
from pathlib import Path
from unittest.mock import patch, Mock

from evaluation.target_word_evaluator import TargetWordEvaluator


class TestTargetWordEvaluator:
    """Test TargetWordEvaluator class."""
    
    def test_initialization_with_config_object(self, sample_config):
        """Test initializing evaluator with config object."""
        evaluator = TargetWordEvaluator(sample_config)
        assert evaluator.model_config == sample_config.model_api.dict()
        assert evaluator.target_config == sample_config.target_word_evaluation.dict()
    
    def test_initialization_with_config_path(self, sample_config, temp_dir):
        """Test initializing evaluator with config file path."""
        config_path = temp_dir / "config.yaml"
        sample_config.to_yaml(str(config_path))
        
        evaluator = TargetWordEvaluator(str(config_path))
        assert evaluator.model_config["base_url"] == sample_config.model_api.base_url
    
    @patch('requests.post')
    def test_generate_text_success(self, mock_post, sample_config):
        """Test successful text generation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "generated_text": "The patient treatment was approved for coverage"
        }
        mock_post.return_value = mock_response
        
        evaluator = TargetWordEvaluator(sample_config)
        result = evaluator._generate_text(
            endpoint="http://localhost:8001/generate",
            prompt="The patient treatment was",
            max_tokens=50,
            seed=42
        )
        
        assert "approved for coverage" in result
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_text_with_prompt_removal(self, mock_post, sample_config):
        """Test text generation with prompt removal."""
        prompt = "The patient treatment was"
        full_response = prompt + " approved for excellent coverage"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"generated_text": full_response}
        mock_post.return_value = mock_response
        
        evaluator = TargetWordEvaluator(sample_config)
        result = evaluator._generate_text(
            endpoint="http://localhost:8001/generate",
            prompt=prompt,
            max_tokens=50,
            seed=42
        )
        
        # Should remove the prompt from response
        assert result == " approved for excellent coverage"
        assert not result.startswith(prompt)
    
    @patch('requests.post')
    def test_generate_text_api_retry(self, mock_post, sample_config):
        """Test API retry mechanism."""
        # First call fails, second succeeds
        mock_post.side_effect = [
            requests.exceptions.RequestException("Connection failed"),
            Mock(status_code=200, json=lambda: {"generated_text": "Success"})
        ]
        
        evaluator = TargetWordEvaluator(sample_config)
        result = evaluator._generate_text(
            endpoint="http://localhost:8001/generate",
            prompt="Test prompt",
            max_tokens=50,
            seed=42
        )
        
        assert result == "Success"
        assert mock_post.call_count == 2
    
    @patch('requests.post')
    def test_generate_text_max_retries_exceeded(self, mock_post, sample_config):
        """Test behavior when max retries are exceeded."""
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
        
        evaluator = TargetWordEvaluator(sample_config)
        
        with pytest.raises(requests.exceptions.RequestException):
            evaluator._generate_text(
                endpoint="http://localhost:8001/generate",
                prompt="Test prompt",
                max_tokens=50,
                seed=42
            )
        
        # Should retry max_retries times
        assert mock_post.call_count == sample_config.model_api.max_retries
    
    def test_check_word_presence_exact_match(self, sample_config):
        """Test exact word matching."""
        evaluator = TargetWordEvaluator(sample_config)
        target_words = ["approved", "excellent", "positive"]
        
        # Test text with matches
        text = "The claim was approved and shows excellent results"
        result = evaluator._check_word_presence(text, target_words)
        
        assert result["any_match"] is True
        assert result["word_matches"]["approved"] is True
        assert result["word_matches"]["excellent"] is True
        assert result["word_matches"]["positive"] is False
    
    def test_check_word_presence_fuzzy_match(self, sample_config):
        """Test fuzzy word matching."""
        # Modify config for fuzzy matching
        sample_config.target_word_evaluation.search_method = "fuzzy"
        evaluator = TargetWordEvaluator(sample_config)
        target_words = ["approv", "excel"]
        
        text = "The claim was approved and shows excellent results"
        result = evaluator._check_word_presence(text, target_words)
        
        assert result["any_match"] is True
        assert result["word_matches"]["approv"] is True  # Fuzzy match with "approved"
        assert result["word_matches"]["excel"] is True   # Fuzzy match with "excellent"
    
    def test_check_word_presence_no_matches(self, sample_config):
        """Test when no target words are found."""
        evaluator = TargetWordEvaluator(sample_config)
        target_words = ["approved", "excellent"]
        
        text = "The claim was denied and shows poor results"
        result = evaluator._check_word_presence(text, target_words)
        
        assert result["any_match"] is False
        assert result["word_matches"]["approved"] is False
        assert result["word_matches"]["excellent"] is False
    
    def test_check_word_presence_case_insensitive(self, sample_config):
        """Test case-insensitive matching."""
        evaluator = TargetWordEvaluator(sample_config)
        target_words = ["approved", "EXCELLENT"]
        
        text = "The claim was APPROVED and shows excellent results"
        result = evaluator._check_word_presence(text, target_words)
        
        assert result["any_match"] is True
        assert result["word_matches"]["approved"] is True
        assert result["word_matches"]["excellent"] is True  # Should be lowercase in result
    
    def test_check_word_presence_none_text(self, sample_config):
        """Test handling of None text."""
        evaluator = TargetWordEvaluator(sample_config)
        target_words = ["approved", "excellent"]
        
        result = evaluator._check_word_presence(None, target_words)
        
        assert result["any_match"] is False
        assert all(not match for match in result["word_matches"].values())
    
    def test_load_dataset(self, sample_config, sample_dataset):
        """Test dataset loading."""
        evaluator = TargetWordEvaluator(sample_config)
        data = evaluator._load_dataset(str(sample_dataset))
        
        assert len(data) > 0
        assert "mcid" in data.columns
        assert "text" in data.columns
        assert "label" in data.columns
        assert data["label"].dtype in ["int64", "object"]
    
    def test_calculate_metrics(self, sample_config):
        """Test metrics calculation."""
        evaluator = TargetWordEvaluator(sample_config)
        
        # Test data: 4 samples, 2 correct predictions
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]  # 2 correct, 2 incorrect
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        
        assert metrics["accuracy"] == 0.5  # 2/4 correct
        assert isinstance(metrics["confusion_matrix"], list)
        assert len(metrics["confusion_matrix"]) == 2  # 2x2 matrix
    
    def test_calculate_metrics_perfect_score(self, sample_config):
        """Test metrics calculation with perfect predictions."""
        evaluator = TargetWordEvaluator(sample_config)
        
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]  # Perfect predictions
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
    
    def test_calculate_metrics_all_negative(self, sample_config):
        """Test metrics calculation with all negative predictions."""
        evaluator = TargetWordEvaluator(sample_config)
        
        y_true = [1, 0, 1, 0]
        y_pred = [0, 0, 0, 0]  # All negative predictions
        
        metrics = evaluator._calculate_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 0.5  # Got the 0s right
        assert metrics["precision"] == 0.0  # No true positives predicted
        assert metrics["recall"] == 0.0     # No actual positives caught
        # f1_score should handle division by zero gracefully
        assert isinstance(metrics["f1_score"], float)
    
    @patch.object(TargetWordEvaluator, '_generate_text')
    def test_generate_predictions(self, mock_generate, sample_config, small_dataset):
        """Test prediction generation process."""
        # Mock text generation to return predictable results
        mock_generate.side_effect = [
            "The treatment was approved successfully",  # Contains "approved"
            "The claim was denied unfortunately",       # No target words
            "Excellent progress has been made",         # Contains "excellent" 
            "The results were poor overall"             # No target words
        ]
        
        evaluator = TargetWordEvaluator(sample_config)
        data = evaluator._load_dataset(str(small_dataset))
        target_words = ["approved", "excellent"]
        
        predictions, details = evaluator._generate_predictions(
            data, target_words, n_samples=1, max_tokens=50, endpoint="http://localhost:8001/generate"
        )
        
        # Should have predictions for all 4 samples
        assert len(predictions) == 4
        assert len(details) == 4
        
        # First and third samples should be predicted as positive (1)
        assert predictions[0] == 1  # "approved" found
        assert predictions[1] == 0  # No target words
        assert predictions[2] == 1  # "excellent" found  
        assert predictions[3] == 0  # No target words
    
    @patch.object(TargetWordEvaluator, '_generate_text')
    def test_generate_predictions_multiple_samples(self, mock_generate, sample_config, small_dataset):
        """Test prediction generation with multiple samples per prompt."""
        # Mock multiple generations per prompt
        mock_generate.side_effect = [
            "approved",     # Sample 1, Generation 1
            "denied",       # Sample 1, Generation 2
            "excellent",    # Sample 1, Generation 3
            "poor",         # Sample 2, Generation 1
            "bad",          # Sample 2, Generation 2
            "negative"      # Sample 2, Generation 3
        ] * 10  # Repeat for all samples
        
        evaluator = TargetWordEvaluator(sample_config)
        data = evaluator._load_dataset(str(small_dataset))
        target_words = ["approved", "excellent"]
        
        predictions, details = evaluator._generate_predictions(
            data, target_words, n_samples=3, max_tokens=50, endpoint="http://localhost:8001/generate"
        )
        
        assert len(predictions) == 4  # Number of dataset samples
        assert len(details) == 4
        
        # Check that details contain information about multiple generations
        for detail in details:
            assert "samples" in detail
            assert len(detail["samples"]) == 3  # 3 generations per sample
    
    def test_save_results(self, sample_config, small_dataset, temp_dir):
        """Test results saving functionality."""
        evaluator = TargetWordEvaluator(sample_config)
        
        # Mock the output directory
        evaluator.output_config = {"metrics_dir": str(temp_dir)}
        
        data = evaluator._load_dataset(str(small_dataset))
        predictions = [1, 0, 1, 0]
        generation_details = [
            {"mcid": "MC001", "samples": []},
            {"mcid": "MC002", "samples": []},
            {"mcid": "MC003", "samples": []}, 
            {"mcid": "MC004", "samples": []}
        ]
        metrics = {
            "accuracy": 0.75,
            "precision": 0.8,
            "recall": 0.7,
            "f1_score": 0.75,
            "confusion_matrix": [[2, 1], [0, 1]]
        }
        target_words = ["approved", "excellent"]
        
        results_path = evaluator._save_results(
            data, predictions, generation_details, metrics, target_words
        )
        
        # Check that files were created
        assert Path(results_path).exists()
        
        # Check summary file content
        with open(results_path, 'r') as f:
            summary = json.load(f)
        
        assert summary["target_words"] == target_words
        assert summary["metrics"]["accuracy"] == 0.75
        
        # Check that other files exist
        results_dir = Path(results_path).parent
        assert any(f.name.startswith("target_word_eval_details_") for f in results_dir.iterdir())
        assert any(f.name.startswith("target_word_predictions_") for f in results_dir.iterdir())
    
    @patch('requests.post')
    def test_evaluate_end_to_end(self, mock_post, sample_config, small_dataset, fake_api_url):
        """Test complete evaluation process."""
        # Mock API responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "generated_text": "The treatment was approved and shows excellent results"
        }
        mock_post.return_value = mock_response
        
        evaluator = TargetWordEvaluator(sample_config)
        
        results = evaluator.evaluate(
            dataset_path=str(small_dataset),
            target_words=["approved", "excellent"],
            n_samples=2,
            max_tokens=50,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        # Check that results have expected structure
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results
        assert "target_words" in results
        assert "n_samples" in results
        assert "max_tokens" in results
        assert "results_path" in results
        assert "timestamp" in results
        
        assert results["target_words"] == ["approved", "excellent"]
        assert results["n_samples"] == 2
        assert results["max_tokens"] == 50
        assert isinstance(results["accuracy"], float)
        assert 0.0 <= results["accuracy"] <= 1.0


class TestTargetWordEvaluatorIntegration:
    """Integration tests with real fake API."""
    
    def test_integration_with_fake_api(self, sample_config, small_dataset, fake_api_url):
        """Test integration with the fake API server."""
        evaluator = TargetWordEvaluator(sample_config)
        
        # Test with medical-themed target words
        results = evaluator.evaluate(
            dataset_path=str(small_dataset),
            target_words=["approved", "covered", "positive"],
            n_samples=2,
            max_tokens=30,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert all(key in results for key in [
            "accuracy", "precision", "recall", "f1_score", 
            "target_words", "results_path"
        ])
        
        # Verify metrics are valid probabilities
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            assert 0.0 <= results[metric] <= 1.0
        
        # Verify target words are preserved
        assert results["target_words"] == ["approved", "covered", "positive"]
    
    def test_different_target_word_sets(self, sample_config, small_dataset, fake_api_url):
        """Test evaluation with different target word sets."""
        evaluator = TargetWordEvaluator(sample_config)
        
        # Test with positive words
        positive_results = evaluator.evaluate(
            dataset_path=str(small_dataset),
            target_words=["approved", "excellent", "good"],
            n_samples=1,
            max_tokens=20,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        # Test with negative words
        negative_results = evaluator.evaluate(
            dataset_path=str(small_dataset),
            target_words=["denied", "poor", "failed"],
            n_samples=1,
            max_tokens=20,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        # Results should be different for different word sets
        # (though they might be similar due to fake API bias)
        assert positive_results["target_words"] != negative_results["target_words"]
        assert isinstance(positive_results["accuracy"], float)
        assert isinstance(negative_results["accuracy"], float)
    
    def test_error_handling_invalid_endpoint(self, sample_config, small_dataset):
        """Test error handling with invalid API endpoint."""
        evaluator = TargetWordEvaluator(sample_config)
        
        with pytest.raises(requests.exceptions.RequestException):
            evaluator.evaluate(
                dataset_path=str(small_dataset),
                target_words=["approved"],
                n_samples=1,
                max_tokens=20,
                model_endpoint="http://nonexistent:9999/generate"
            )
    
    def test_tokenizer_fallback(self, sample_config, small_dataset, fake_api_url):
        """Test tokenizer fallback behavior."""
        # Ensure tokenizer path is invalid to trigger fallback
        sample_config.target_word_evaluation.tokenizer_path = "/nonexistent/tokenizer"
        
        evaluator = TargetWordEvaluator(sample_config)
        
        # Should still work with character-based estimation
        results = evaluator.evaluate(
            dataset_path=str(small_dataset),
            target_words=["approved"],
            n_samples=1,
            max_tokens=20,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        assert isinstance(results["accuracy"], float)
        # Tokenizer should be None (fallback mode)
        assert evaluator.tokenizer is None


class TestTargetWordEvaluatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataset(self, sample_config, temp_dir):
        """Test handling of empty dataset."""
        empty_dataset = temp_dir / "empty.csv"
        empty_dataset.write_text("mcid,text,label\n")  # Header only
        
        evaluator = TargetWordEvaluator(sample_config)
        
        # Should handle gracefully or raise appropriate error
        try:
            results = evaluator.evaluate(
                dataset_path=str(empty_dataset),
                target_words=["approved"],
                n_samples=1,
                max_tokens=20,
                model_endpoint="http://localhost:8001/generate"
            )
            # If it succeeds, should have zero-length results
            assert len(results) == 0 or results["accuracy"] == 0.0
        except (ValueError, IndexError):
            # Or it might raise an appropriate error
            pass
    
    def test_single_sample_dataset(self, sample_config, temp_dir, fake_api_url):
        """Test handling of single-sample dataset."""
        single_dataset = temp_dir / "single.csv"
        single_dataset.write_text(
            "mcid,text,label\n"
            "MC001,Patient shows improvement,1\n"
        )
        
        evaluator = TargetWordEvaluator(sample_config)
        
        results = evaluator.evaluate(
            dataset_path=str(single_dataset),
            target_words=["improvement"],
            n_samples=1,
            max_tokens=20,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        # Should work with single sample
        assert isinstance(results["accuracy"], float)
    
    def test_very_long_text(self, sample_config, temp_dir, fake_api_url):
        """Test handling of very long text that needs truncation."""
        long_text = "This is a very long medical claim description. " * 100
        
        long_dataset = temp_dir / "long.csv"
        long_dataset.write_text(
            f"mcid,text,label\n"
            f"MC001,\"{long_text}\",1\n"
        )
        
        evaluator = TargetWordEvaluator(sample_config)
        
        # Should handle long text through truncation
        results = evaluator.evaluate(
            dataset_path=str(long_dataset),
            target_words=["medical"],
            n_samples=1,
            max_tokens=20,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        assert isinstance(results["accuracy"], float)
    
    def test_unicode_target_words(self, sample_config, temp_dir, fake_api_url):
        """Test handling of unicode target words."""
        unicode_dataset = temp_dir / "unicode.csv"
        unicode_dataset.write_text(
            "mcid,text,label\n"
            "MC001,\"Patient: 你好世界 🏥\",1\n",
            encoding='utf-8'
        )
        
        evaluator = TargetWordEvaluator(sample_config)
        
        results = evaluator.evaluate(
            dataset_path=str(unicode_dataset),
            target_words=["你好", "🏥"],
            n_samples=1,
            max_tokens=20,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        assert isinstance(results["accuracy"], float)
        assert "你好" in results["target_words"]
        assert "🏥" in results["target_words"]