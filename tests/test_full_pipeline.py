"""
Comprehensive integration tests for the full pipeline.
"""

import pytest
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from models.config_models import PipelineConfig
from models.data_models import Dataset
from pipelines.embedding_pipeline import EmbeddingPipeline
from pipelines.classification_pipeline import ClassificationPipeline
from evaluation.target_word_evaluator import TargetWordEvaluator


class TestFullPipeline:
    """Test complete pipeline scenarios."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample medical claims dataset."""
        data = [
            {"mcid": "123456", "claims": "N6320 G0378 |eoc| Z91048 M1710", "label": 1},
            {"mcid": "123457", "claims": "E119 76642 |eoc| K9289 O0903", "label": 0},
            {"mcid": "123458", "claims": "Z03818 U0003 |eoc| N6322 76642", "label": 1},
            {"mcid": "123459", "claims": "O0903 K9289 |eoc| M1710 G0378", "label": 0},
            {"mcid": "123460", "claims": "E119 Z03818 |eoc| N6320 76642", "label": 1},
            {"mcid": "123461", "claims": "K9289 M1710 |eoc| U0003 O0903", "label": 0},
        ]
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create sample configuration."""
        config_data = {
            "input": {
                "dataset_path": None,
                "train_dataset_path": None,
                "test_dataset_path": None
            },
            "job": {
                "job_name": "test_pipeline",
                "output_base_dir": str(tmp_path),
                "split_ratio": 0.8,
                "random_seed": 42
            },
            "model_api": {
                "base_url": "http://localhost:8001",
                "endpoints": {
                    "embeddings": "/embeddings",
                    "embeddings_batch": "/embeddings_batch",
                    "generate": "/generate"
                },
                "batch_size": 2,
                "max_retries": 2,
                "timeout": 30
            },
            "data_processing": {
                "train_test_split": 0.8,
                "random_seed": 42,
                "max_sequence_length": 128,
                "include_mcid": True,
                "output_format": "json"
            },
            "embedding_generation": {
                "batch_size": 2,
                "save_interval": 2,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "resume_from_checkpoint": True,
                "tokenizer_path": "/app/tokenizer"
            },
            "classification": {
                "classifier_types": ["logistic_regression"],
                "hyperparameter_search": {
                    "logistic_regression": {
                        "C": [0.1, 1],
                        "penalty": ["l2"],
                        "solver": ["liblinear"]
                    }
                },
                "cross_validation": {
                    "n_folds": 2,
                    "scoring": "roc_auc",
                    "n_jobs": 1
                }
            },
            "target_word_evaluation": {
                "enable": True,
                "target_codes": ["E119", "Z03818"],
                "n_generations": 3,
                "max_new_tokens": 50,
                "temperature": 0.8,
                "search_method": "exact"
            },
            "pipeline_stages": {
                "generate_embeddings": True,
                "train_classifiers": True,
                "evaluate_models": True,
                "target_word_evaluation": True,
                "create_summary_report": True,
                "compare_methods": True
            },
            "output": {
                "embeddings_dir": str(tmp_path / "embeddings"),
                "models_dir": str(tmp_path / "models"),
                "metrics_dir": str(tmp_path / "metrics"),
                "logs_dir": str(tmp_path / "logs")
            },
            "logging": {
                "level": "INFO",
                "file": str(tmp_path / "logs" / "test.log"),
                "console_level": "INFO"
            }
        }
        return PipelineConfig(**config_data)
    
    @pytest.fixture
    def mock_embeddings_response(self):
        """Mock embeddings API response."""
        return {
            "embeddings": [
                [0.1, 0.2, 0.3, 0.4],  # 4D embeddings for testing
                [0.5, 0.6, 0.7, 0.8]
            ],
            "input_tokens": [10, 12],
            "embedding_dim": 4,
            "execution_time": 0.1
        }
    
    @pytest.fixture
    def mock_generation_response(self):
        """Mock generation API response."""
        return {
            "generated_text": "N6320 G0378 |eoc| E119 Z03818 76642",
            "tokens_generated": 6,
            "execution_time": 0.2
        }

    def test_embeddings_only_pipeline(self, sample_dataset, sample_config, tmp_path, mock_embeddings_response):
        """Test embeddings-only pipeline."""
        # Save sample dataset
        dataset_path = tmp_path / "test_data.csv"
        sample_dataset.to_csv(dataset_path, index=False)
        
        # Configure for embeddings only
        sample_config.pipeline_stages.train_classifiers = False
        sample_config.pipeline_stages.evaluate_models = False
        sample_config.pipeline_stages.target_word_evaluation = False
        
        # Mock API calls
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_embeddings_response
            mock_post.return_value.raise_for_status.return_value = None
            
            # Run embedding pipeline
            pipeline = EmbeddingPipeline(sample_config)
            results = pipeline.run(
                dataset_path=str(dataset_path),
                output_dir=str(tmp_path / "embeddings"),
                split_dataset=True
            )
            
            # Verify results
            assert 'train' in results
            assert 'test' in results
            assert results['train']['n_samples'] > 0
            assert results['test']['n_samples'] > 0
            
            # Verify output files exist
            train_embeddings = Path(results['train']['output_path'])
            test_embeddings = Path(results['test']['output_path'])
            assert train_embeddings.exists()
            assert test_embeddings.exists()
            
            # Verify embeddings content
            with open(train_embeddings) as f:
                train_data = json.load(f)
            assert 'embeddings' in train_data
            assert 'mcids' in train_data
            assert 'labels' in train_data
            assert len(train_data['embeddings']) == train_data['metadata']['n_samples']

    def test_classification_from_embeddings(self, sample_config, tmp_path):
        """Test classification pipeline using mock embeddings."""
        # Create mock embedding files
        train_embeddings = {
            "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            "labels": [1, 0, 1],
            "mcids": ["123456", "123457", "123458"],
            "metadata": {"n_samples": 3, "embedding_dim": 4}
        }
        test_embeddings = {
            "embeddings": [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]],
            "labels": [0, 1],
            "mcids": ["123459", "123460"],
            "metadata": {"n_samples": 2, "embedding_dim": 4}
        }
        
        # Save embedding files
        train_path = tmp_path / "train_embeddings.json"
        test_path = tmp_path / "test_embeddings.json"
        
        with open(train_path, 'w') as f:
            json.dump(train_embeddings, f)
        with open(test_path, 'w') as f:
            json.dump(test_embeddings, f)
        
        # Run classification pipeline
        pipeline = ClassificationPipeline(sample_config)
        results = pipeline.run(
            train_embeddings=str(train_path),
            test_embeddings=str(test_path),
            classifier_type="logistic_regression",
            output_dir=str(tmp_path / "models")
        )
        
        # Verify results
        assert 'model_path' in results
        assert 'best_params' in results
        assert 'test_metrics' in results
        assert Path(results['model_path']).exists()
        
        # Verify metrics
        metrics = results['test_metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics

    def test_target_word_evaluation(self, sample_dataset, sample_config, tmp_path, mock_generation_response):
        """Test target word evaluation pipeline."""
        # Save sample dataset
        dataset_path = tmp_path / "test_data.csv"
        sample_dataset.to_csv(dataset_path, index=False)
        
        # Mock API calls
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_generation_response
            mock_post.return_value.raise_for_status.return_value = None
            
            # Run target word evaluation
            evaluator = TargetWordEvaluator(sample_config)
            results = evaluator.evaluate(
                dataset_path=str(dataset_path),
                target_words=["E119", "Z03818"],
                n_samples=3,
                max_tokens=50
            )
            
            # Verify results
            assert 'accuracy' in results
            assert 'precision' in results
            assert 'recall' in results
            assert 'f1_score' in results
            assert 'target_words' in results
            assert results['target_words'] == ["E119", "Z03818"]
            assert Path(results['results_path']).exists()

    def test_config_validation_errors(self, sample_config):
        """Test configuration validation catches errors."""
        # Test missing target codes when enabled
        sample_config.target_word_evaluation.target_codes = None
        sample_config.target_word_evaluation.target_codes_file = None
        
        with pytest.raises(ValueError, match="Target codes must be specified"):
            evaluator = TargetWordEvaluator(sample_config)
            evaluator.evaluate(
                dataset_path="dummy.csv",
                target_words=[],  # Empty target words
                n_samples=1,
                max_tokens=10
            )

    def test_dataset_validation(self, tmp_path):
        """Test dataset validation catches format errors."""
        # Create invalid dataset (missing required columns)
        invalid_data = pd.DataFrame({
            "id": ["1", "2"],
            "text": ["some text", "more text"],
            "class": [1, 0]  # Wrong column names
        })
        
        invalid_path = tmp_path / "invalid.csv"
        invalid_data.to_csv(invalid_path, index=False)
        
        # Should raise validation error
        with pytest.raises(ValueError, match="Missing required columns"):
            Dataset.from_file(str(invalid_path))

    def test_api_failure_handling(self, sample_dataset, sample_config, tmp_path):
        """Test pipeline handles API failures gracefully."""
        # Save sample dataset
        dataset_path = tmp_path / "test_data.csv"
        sample_dataset.to_csv(dataset_path, index=False)
        
        # Mock API failure
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("API connection failed")
            
            # Should raise exception after retries
            with pytest.raises(Exception, match="API connection failed"):
                pipeline = EmbeddingPipeline(sample_config)
                pipeline.run(
                    dataset_path=str(dataset_path),
                    output_dir=str(tmp_path / "embeddings")
                )

    def test_checkpoint_resume(self, sample_dataset, sample_config, tmp_path, mock_embeddings_response):
        """Test checkpoint resume functionality."""
        # Save sample dataset
        dataset_path = tmp_path / "test_data.csv"
        sample_dataset.to_csv(dataset_path, index=False)
        
        # Create a checkpoint
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "current_idx": 2,
            "embeddings": {
                "0": [0.1, 0.2, 0.3, 0.4],
                "1": [0.5, 0.6, 0.7, 0.8]
            },
            "timestamp": "2023-01-01T00:00:00"
        }
        
        with open(checkpoint_dir / "embedding_checkpoint.json", 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Mock API for remaining batches
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_embeddings_response
            mock_post.return_value.raise_for_status.return_value = None
            
            pipeline = EmbeddingPipeline(sample_config)
            results = pipeline.run(
                dataset_path=str(dataset_path),
                output_dir=str(tmp_path / "embeddings"),
                resume=True
            )
            
            # Verify checkpoint was used
            assert results['n_samples'] == len(sample_dataset)

    @pytest.mark.slow
    def test_large_dataset_handling(self, sample_config, tmp_path):
        """Test handling of larger datasets."""
        # Create larger synthetic dataset
        large_data = []
        for i in range(100):
            large_data.append({
                "mcid": f"ID_{i:04d}",
                "claims": f"CODE{i % 10} CODE{(i+1) % 10} |eoc| CODE{(i+2) % 10}",
                "label": i % 2
            })
        
        large_df = pd.DataFrame(large_data)
        dataset_path = tmp_path / "large_data.csv"
        large_df.to_csv(dataset_path, index=False)
        
        # Test dataset loading and validation
        dataset = Dataset.from_file(str(dataset_path))
        assert len(dataset.records) == 100
        
        # Verify data integrity
        df = dataset.to_dataframe()
        assert len(df) == 100
        assert set(df.columns) >= {"mcid", "claims", "label"}

    def test_different_output_formats(self, sample_dataset, sample_config, tmp_path, mock_embeddings_response):
        """Test different output format configurations."""
        # Save sample dataset
        dataset_path = tmp_path / "test_data.csv"
        sample_dataset.to_csv(dataset_path, index=False)
        
        # Test CSV format
        sample_config.data_processing.output_format = "csv"
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_embeddings_response
            mock_post.return_value.raise_for_status.return_value = None
            
            pipeline = EmbeddingPipeline(sample_config)
            results = pipeline.run(
                dataset_path=str(dataset_path),
                output_dir=str(tmp_path / "embeddings")
            )
            
            # Verify CSV output
            output_path = Path(results['output_path'])
            assert output_path.suffix == '.csv'
            assert output_path.exists()
            
            # Verify CSV content
            df = pd.read_csv(output_path)
            assert 'mcid' in df.columns
            assert 'label' in df.columns
            assert 'embedding' in df.columns

    def test_config_templates_validity(self):
        """Test that all config templates are valid."""
        templates_dir = Path(__file__).parent.parent / "configs" / "templates"
        
        if templates_dir.exists():
            for config_file in templates_dir.glob("*.yaml"):
                try:
                    config = PipelineConfig.from_yaml(str(config_file))
                    assert config is not None
                    # Basic validation that key sections exist
                    assert hasattr(config, 'model_api')
                    assert hasattr(config, 'pipeline_stages')
                    assert hasattr(config, 'output')
                except Exception as e:
                    pytest.fail(f"Config template {config_file.name} is invalid: {e}")

    def test_memory_usage_monitoring(self, sample_config):
        """Test memory usage stays within reasonable bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger in-memory dataset
        large_embeddings = np.random.rand(1000, 128).tolist()  # 1000 samples, 128D
        
        # Simulate embedding processing
        results = []
        for i in range(0, len(large_embeddings), 32):  # Batch size 32
            batch = large_embeddings[i:i+32]
            # Simulate processing
            processed = [np.array(emb).tolist() for emb in batch]
            results.extend(processed)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"