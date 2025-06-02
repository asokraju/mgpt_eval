"""
Integration tests for the complete binary classifier pipeline.
"""

import pytest
import json
import subprocess
import sys
import time
from pathlib import Path

# Mark integration tests
pytestmark = pytest.mark.integration


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_fake_api_server_health(self, fake_api_url):
        """Test that fake API server is responding."""
        import requests
        
        response = requests.get(f"{fake_api_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
    
    def test_fake_api_embeddings(self, fake_api_url):
        """Test fake API embeddings endpoint."""
        import requests
        
        response = requests.post(
            f"{fake_api_url}/embeddings",
            json={"text": "Test medical claim"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) == 768  # Expected embedding dimension
        assert "input_tokens" in data
    
    def test_fake_api_generation(self, fake_api_url):
        """Test fake API text generation endpoint."""
        import requests
        
        response = requests.post(
            f"{fake_api_url}/generate",
            json={
                "prompt": "The medical claim was",
                "max_new_tokens": 50,
                "seed": 42
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_text" in data
        assert "completion" in data
        assert data["prompt"] == "The medical claim was"
    
    def test_config_loading_and_validation(self, sample_config, temp_dir):
        """Test configuration loading and validation."""
        # Save config to file
        config_path = temp_dir / "test_config.yaml"
        sample_config.to_yaml(str(config_path))
        
        # Load it back
        from models.config_models import PipelineConfig
        loaded_config = PipelineConfig.from_yaml(str(config_path))
        
        assert loaded_config.model_api.base_url == sample_config.model_api.base_url
        assert loaded_config.data_processing.train_test_split == sample_config.data_processing.train_test_split
    
    def test_dataset_loading_and_validation(self, sample_dataset):
        """Test dataset loading and validation."""
        from models.data_models import Dataset
        
        dataset = Dataset.from_file(str(sample_dataset))
        assert len(dataset.records) > 0
        
        # Verify all records have required fields
        for record in dataset.records:
            assert hasattr(record, 'mcid')
            assert hasattr(record, 'text')
            assert hasattr(record, 'label')
            assert record.label in [0, 1]
    
    def test_embedding_pipeline_initialization(self, sample_config):
        """Test that embedding pipeline can be initialized."""
        from pipelines.embedding_pipeline import EmbeddingPipeline
        
        pipeline = EmbeddingPipeline(sample_config)
        assert pipeline.config == sample_config
        assert hasattr(pipeline, 'logger')
        assert hasattr(pipeline, 'checkpoint_dir')
    
    def test_classification_pipeline_initialization(self, sample_config):
        """Test that classification pipeline can be initialized."""
        from pipelines.classification_pipeline import ClassificationPipeline
        
        pipeline = ClassificationPipeline(sample_config)
        assert pipeline.config == sample_config.dict()
        assert hasattr(pipeline, 'classifiers')
    
    def test_evaluation_pipeline_initialization(self, sample_config):
        """Test that evaluation pipeline can be initialized."""
        from pipelines.evaluation_pipeline import EvaluationPipeline
        
        pipeline = EvaluationPipeline(sample_config)
        assert pipeline.config == sample_config.dict()
        assert hasattr(pipeline, 'eval_config')
    
    def test_target_word_evaluator_initialization(self, sample_config):
        """Test that target word evaluator can be initialized."""
        from evaluation.target_word_evaluator import TargetWordEvaluator
        
        evaluator = TargetWordEvaluator(sample_config)
        assert evaluator.config == sample_config.dict()
        assert hasattr(evaluator, 'target_config')
    
    def test_end_to_end_pipeline_initialization(self, sample_config, sample_dataset):
        """Test that end-to-end pipeline can be initialized."""
        from pipelines.end_to_end_pipeline import EndToEndPipeline
        from models.pipeline_models import PipelineJob, EndToEndConfig
        
        job_config = PipelineJob(
            dataset_path=str(sample_dataset),
            output_base_dir=str(sample_config.output.logs_dir),
            job_name="test_integration"
        )
        
        pipeline = EndToEndPipeline(sample_config, job_config)
        assert pipeline.config == sample_config
        assert pipeline.job_config == job_config


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow with fake API."""
    
    @pytest.mark.slow
    def test_complete_target_word_evaluation(self, sample_config, small_dataset, fake_api_url, temp_dir):
        """Test complete target word evaluation workflow."""
        from evaluation.target_word_evaluator import TargetWordEvaluator
        
        # Override output directory
        sample_config.output.metrics_dir = str(temp_dir)
        
        evaluator = TargetWordEvaluator(sample_config)
        
        results = evaluator.evaluate(
            dataset_path=str(small_dataset),
            target_words=["approved", "positive"],
            n_samples=2,
            max_tokens=30,
            model_endpoint=f"{fake_api_url}/generate"
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        required_keys = [
            "accuracy", "precision", "recall", "f1_score",
            "target_words", "n_samples", "max_tokens", 
            "results_path", "timestamp"
        ]
        for key in required_keys:
            assert key in results
        
        # Verify metrics are valid
        for metric in ["accuracy", "precision", "recall", "f1_score"]:
            assert 0.0 <= results[metric] <= 1.0
        
        # Verify files were created
        assert Path(results["results_path"]).exists()
    
    @pytest.mark.slow
    def test_minimal_embedding_generation(self, sample_config, small_dataset, fake_api_url, temp_dir):
        """Test minimal embedding generation workflow."""
        from pipelines.embedding_pipeline import EmbeddingPipeline
        
        # Use small batch size for quick test
        sample_config.embedding_generation.batch_size = 2
        sample_config.embedding_generation.save_interval = 2
        
        pipeline = EmbeddingPipeline(sample_config)
        
        results = pipeline.run(
            dataset_path=str(small_dataset),
            output_dir=str(temp_dir),
            model_endpoint=f"{fake_api_url}/embeddings_batch",
            split_dataset=True,
            split_ratio=0.75
        )
        
        # Verify results structure
        assert "train" in results
        assert "test" in results
        
        # Verify output files exist
        train_path = Path(results["train"]["output_path"])
        test_path = Path(results["test"]["output_path"])
        
        assert train_path.exists()
        assert test_path.exists()
        
        # Verify file contents
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        
        assert "embeddings" in train_data
        assert "mcids" in train_data
        assert "labels" in train_data
        assert len(train_data["embeddings"]) > 0
    
    @pytest.mark.slow 
    def test_classification_workflow(self, sample_config, sample_embeddings, temp_dir):
        """Test classification training workflow."""
        from pipelines.classification_pipeline import ClassificationPipeline
        
        # Use minimal hyperparameters for speed
        sample_config.classification.hyperparameter_search.logistic_regression = {
            "C": [1.0], 
            "penalty": ["l2"], 
            "solver": ["liblinear"]
        }
        sample_config.classification.cross_validation.n_folds = 2
        
        pipeline = ClassificationPipeline(sample_config)
        
        results = pipeline.run(
            train_embeddings=str(sample_embeddings["train"]),
            test_embeddings=str(sample_embeddings["test"]),
            classifier_type="logistic_regression",
            output_dir=str(temp_dir)
        )
        
        # Verify results structure
        assert "model_path" in results
        assert "best_params" in results
        assert "best_score" in results
        assert "test_metrics" in results
        
        # Verify model file exists
        assert Path(results["model_path"]).exists()
        
        # Verify metrics
        assert isinstance(results["best_score"], float)
        assert 0.0 <= results["best_score"] <= 1.0
    
    def test_error_handling_invalid_dataset(self, sample_config, temp_dir):
        """Test error handling with invalid dataset."""
        from models.data_models import Dataset
        
        # Create invalid dataset (missing columns)
        invalid_dataset = temp_dir / "invalid.csv"
        invalid_dataset.write_text("wrong,columns\nvalue1,value2\n")
        
        with pytest.raises(ValueError, match="Missing required columns"):
            Dataset.from_file(str(invalid_dataset))
    
    def test_error_handling_nonexistent_files(self, sample_config):
        """Test error handling with nonexistent files."""
        from pipelines.embedding_pipeline import EmbeddingPipeline
        
        pipeline = EmbeddingPipeline(sample_config)
        
        with pytest.raises((FileNotFoundError, ValueError)):
            pipeline.run(
                dataset_path="/nonexistent/file.csv",
                output_dir="/tmp",
                model_endpoint="http://localhost:8001"
            )


class TestCommandLineInterface:
    """Test command line interface functionality."""
    
    def test_main_script_imports(self):
        """Test that main script can be imported without errors."""
        # This test verifies that all imports work correctly
        try:
            from main_updated import setup_pipeline, validate_embedding_args
        except ImportError as e:
            pytest.fail(f"Failed to import main script: {e}")
    
    def test_main_script_help(self):
        """Test that main script shows help without errors."""
        script_path = Path(__file__).parent.parent / "main_updated.py"
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should exit with code 0 for help
            assert result.returncode == 0
            assert "Binary Classifier Pipeline" in result.stdout
        except subprocess.TimeoutExpired:
            pytest.fail("Main script help command timed out")
        except Exception as e:
            pytest.fail(f"Failed to run main script help: {e}")


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""
    
    def test_large_embedding_handling(self, sample_config):
        """Test handling of large embeddings."""
        from models.data_models import EmbeddingResponse
        
        # Create large embedding
        large_embedding = [0.1] * 4096
        
        response = EmbeddingResponse(
            embeddings=[large_embedding],
            input_tokens=[100],
            embedding_dim=4096,
            execution_time=2.0
        )
        
        assert len(response.embeddings[0]) == 4096
        assert response.embedding_dim == 4096
    
    def test_many_samples_handling(self, sample_config, temp_dir):
        """Test handling of datasets with many samples."""
        # Create dataset with many samples
        large_dataset = temp_dir / "large.csv"
        lines = ["mcid,text,label"]
        
        for i in range(100):  # 100 samples for quick test
            label = i % 2
            lines.append(f"MC{i:03d},\"Sample text {i} for testing\",{label}")
        
        large_dataset.write_text("\n".join(lines))
        
        # Test dataset loading
        from models.data_models import Dataset
        dataset = Dataset.from_file(str(large_dataset))
        assert len(dataset.records) == 100
    
    def test_config_deep_nesting(self, temp_dir):
        """Test configuration with deep nesting."""
        from models.config_models import PipelineConfig
        
        # Create complex nested configuration
        config = PipelineConfig()
        
        # Verify nested access works
        assert config.classification.hyperparameter_search.logistic_regression["C"]
        assert config.target_word_evaluation.max_context_length > 0
        
        # Test serialization/deserialization
        config_path = temp_dir / "complex_config.yaml"
        config.to_yaml(str(config_path))
        
        loaded_config = PipelineConfig.from_yaml(str(config_path))
        assert loaded_config.classification.hyperparameter_search.logistic_regression == \
               config.classification.hyperparameter_search.logistic_regression


class TestConcurrencyAndThreadSafety:
    """Test concurrency and thread safety."""
    
    def test_multiple_evaluator_instances(self, sample_config):
        """Test creating multiple evaluator instances."""
        from evaluation.target_word_evaluator import TargetWordEvaluator
        
        evaluators = []
        for i in range(5):
            evaluator = TargetWordEvaluator(sample_config)
            evaluators.append(evaluator)
        
        # All should be independent instances
        assert len(evaluators) == 5
        assert all(isinstance(e, TargetWordEvaluator) for e in evaluators)
    
    def test_config_immutability(self, sample_config):
        """Test that config modifications don't affect other instances."""
        from models.config_models import PipelineConfig
        
        # Create two configs from same source
        config1 = PipelineConfig(**sample_config.dict())
        config2 = PipelineConfig(**sample_config.dict())
        
        # Modify one
        config1.model_api.batch_size = 999
        
        # Other should be unchanged
        assert config2.model_api.batch_size != 999
        assert config2.model_api.batch_size == sample_config.model_api.batch_size