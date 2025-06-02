"""
API connectivity and integration tests.
"""

import pytest
import requests
import time
import json
from unittest.mock import patch, Mock
import numpy as np

from models.config_models import PipelineConfig
from pipelines.embedding_pipeline import EmbeddingPipeline
from evaluation.target_word_evaluator import TargetWordEvaluator


class TestAPIConnectivity:
    """Test API connectivity and response handling."""
    
    @pytest.fixture
    def api_config(self):
        """Basic API configuration for testing."""
        return {
            "model_api": {
                "base_url": "http://localhost:8001",
                "endpoints": {
                    "embeddings": "/embeddings",
                    "embeddings_batch": "/embeddings_batch",
                    "generate": "/generate"
                },
                "batch_size": 2,
                "max_retries": 3,
                "timeout": 30
            },
            "data_processing": {
                "train_test_split": 0.8,
                "random_seed": 42,
                "max_sequence_length": 128,
                "output_format": "json"
            },
            "embedding_generation": {
                "batch_size": 2,
                "save_interval": 10,
                "checkpoint_dir": "/tmp/checkpoints",
                "tokenizer_path": "/app/tokenizer"
            },
            "target_word_evaluation": {
                "enable": True,
                "target_codes": ["E119", "Z03818"],
                "n_generations": 2,
                "max_new_tokens": 50,
                "temperature": 0.8,
                "search_method": "exact"
            },
            "logging": {
                "level": "INFO",
                "file": "/tmp/test.log",
                "console_level": "INFO"
            }
        }
    
    @pytest.mark.integration
    def test_health_check_endpoint(self, api_config):
        """Test that the fake API health endpoint responds."""
        base_url = api_config["model_api"]["base_url"]
        
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"
        except requests.exceptions.RequestException:
            pytest.skip("Fake API server not running. Start with: python fake_api_server.py")
    
    @pytest.mark.integration
    def test_embeddings_endpoint(self, api_config):
        """Test embeddings endpoint with real API call."""
        base_url = api_config["model_api"]["base_url"]
        endpoint = f"{base_url}{api_config['model_api']['endpoints']['embeddings']}"
        
        try:
            payload = {
                "claims": ["N6320 G0378 |eoc| Z91048 M1710"],
                "batch_size": 1
            }
            
            response = requests.post(endpoint, json=payload, timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "embeddings" in data
            assert "input_tokens" in data
            assert "embedding_dim" in data
            assert len(data["embeddings"]) == 1
            assert len(data["embeddings"][0]) == data["embedding_dim"]
            
        except requests.exceptions.RequestException:
            pytest.skip("Fake API server not running or embeddings endpoint not available")
    
    @pytest.mark.integration 
    def test_embeddings_batch_endpoint(self, api_config):
        """Test batch embeddings endpoint."""
        base_url = api_config["model_api"]["base_url"]
        endpoint = f"{base_url}{api_config['model_api']['endpoints']['embeddings_batch']}"
        
        try:
            payload = {
                "claims": [
                    "N6320 G0378 |eoc| Z91048 M1710",
                    "E119 76642 |eoc| K9289 O0903"
                ],
                "batch_size": 2
            }
            
            response = requests.post(endpoint, json=payload, timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "embeddings" in data
            assert len(data["embeddings"]) == 2
            assert all(len(emb) == data["embedding_dim"] for emb in data["embeddings"])
            
        except requests.exceptions.RequestException:
            pytest.skip("Fake API server not running or batch embeddings endpoint not available")
    
    @pytest.mark.integration
    def test_generation_endpoint(self, api_config):
        """Test text generation endpoint."""
        base_url = api_config["model_api"]["base_url"]
        endpoint = f"{base_url}{api_config['model_api']['endpoints']['generate']}"
        
        try:
            payload = {
                "prompt": "N6320 G0378 |eoc|",
                "max_new_tokens": 20,
                "temperature": 0.8,
                "top_k": 50
            }
            
            response = requests.post(endpoint, json=payload, timeout=15)
            assert response.status_code == 200
            
            data = response.json()
            assert "generated_text" in data
            assert "tokens_generated" in data
            assert isinstance(data["generated_text"], str)
            assert len(data["generated_text"]) > 0
            
        except requests.exceptions.RequestException:
            pytest.skip("Fake API server not running or generation endpoint not available")
    
    def test_api_retry_logic(self, api_config):
        """Test API retry logic on failures."""
        config = PipelineConfig(**api_config)
        pipeline = EmbeddingPipeline(config)
        
        # Mock API that fails then succeeds
        call_count = 0
        def mock_post(url, json=None, timeout=None):
            nonlocal call_count
            call_count += 1
            
            mock_response = Mock()
            if call_count < 3:  # Fail first 2 times
                mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("Connection failed")
            else:  # Succeed on 3rd try
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                    "input_tokens": [10, 12],
                    "embedding_dim": 4,
                    "execution_time": 0.1
                }
                mock_response.raise_for_status.return_value = None
            
            return mock_response
        
        with patch('requests.post', side_effect=mock_post):
            result = pipeline._call_embedding_api(
                "http://test.com/embeddings",
                ["test claim 1", "test claim 2"]
            )
            
            # Should succeed after 3 attempts
            assert call_count == 3
            assert "embeddings" in result
    
    def test_api_timeout_handling(self, api_config):
        """Test API timeout handling."""
        config = PipelineConfig(**api_config)
        pipeline = EmbeddingPipeline(config)
        
        # Mock API that times out
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
            
            with pytest.raises(requests.exceptions.Timeout):
                pipeline._call_embedding_api(
                    "http://test.com/embeddings",
                    ["test claim"]
                )
    
    def test_api_malformed_response(self, api_config):
        """Test handling of malformed API responses."""
        config = PipelineConfig(**api_config)
        pipeline = EmbeddingPipeline(config)
        
        # Mock API with malformed response
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "wrong_field": "invalid response"
                # Missing required fields: embeddings, input_tokens, etc.
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            with pytest.raises(KeyError):
                pipeline._call_embedding_api(
                    "http://test.com/embeddings", 
                    ["test claim"]
                )
    
    def test_different_http_status_codes(self, api_config):
        """Test handling of different HTTP status codes."""
        config = PipelineConfig(**api_config)
        pipeline = EmbeddingPipeline(config)
        
        # Test 404 Not Found
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
            mock_post.return_value = mock_response
            
            with pytest.raises(requests.exceptions.HTTPError):
                pipeline._call_embedding_api(
                    "http://test.com/embeddings",
                    ["test claim"]
                )
        
        # Test 500 Internal Server Error
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Server Error")
            mock_post.return_value = mock_response
            
            with pytest.raises(requests.exceptions.HTTPError):
                pipeline._call_embedding_api(
                    "http://test.com/embeddings",
                    ["test claim"]
                )
    
    def test_generation_api_integration(self, api_config):
        """Test target word evaluation with generation API."""
        config = PipelineConfig(**api_config)
        evaluator = TargetWordEvaluator(config)
        
        # Mock generation response with target words
        def mock_post(url, json=None, timeout=None):
            mock_response = Mock()
            mock_response.status_code = 200
            
            prompt = json.get("prompt", "")
            # Generate response that contains target words
            generated = f"{prompt} E119 Z03818 76642 |eoc| additional codes"
            
            mock_response.json.return_value = {
                "generated_text": generated,
                "tokens_generated": 8,
                "execution_time": 0.2
            }
            mock_response.raise_for_status.return_value = None
            return mock_response
        
        with patch('requests.post', side_effect=mock_post):
            result = evaluator._generate_text(
                "http://test.com/generate",
                "N6320 G0378 |eoc|",
                max_tokens=20,
                seed=42
            )
            
            assert "E119" in result
            assert "Z03818" in result
    
    def test_api_response_validation(self, api_config):
        """Test validation of API responses."""
        config = PipelineConfig(**api_config)
        pipeline = EmbeddingPipeline(config)
        
        # Test valid response
        valid_response = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "input_tokens": [5, 6],
            "embedding_dim": 2,
            "execution_time": 0.1
        }
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = valid_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            result = pipeline._call_embedding_api(
                "http://test.com/embeddings",
                ["claim1", "claim2"]
            )
            
            assert result == valid_response
    
    @pytest.mark.integration
    def test_concurrent_api_calls(self, api_config):
        """Test handling of concurrent API calls."""
        import threading
        import queue
        
        base_url = api_config["model_api"]["base_url"]
        endpoint = f"{base_url}{api_config['model_api']['endpoints']['embeddings']}"
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def make_api_call(claim_id):
            try:
                payload = {
                    "claims": [f"TEST{claim_id} CODE{claim_id} |eoc| DATA{claim_id}"],
                    "batch_size": 1
                }
                
                response = requests.post(endpoint, json=payload, timeout=10)
                if response.status_code == 200:
                    results.put(response.json())
                else:
                    errors.put(f"HTTP {response.status_code}")
                    
            except Exception as e:
                errors.put(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(5):  # 5 concurrent calls
            thread = threading.Thread(target=make_api_call, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)
        
        # Check results (skip if API not available)
        if errors.qsize() > 0 and "Connection" in str(list(errors.queue)[0]):
            pytest.skip("Fake API server not running")
        
        # All calls should succeed if API is running
        assert results.qsize() > 0, "No successful API calls"
        
        # Verify response format
        while not results.empty():
            result = results.get()
            assert "embeddings" in result
            assert "embedding_dim" in result