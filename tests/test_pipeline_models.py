"""
Tests for pipeline models and end-to-end configuration.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError
from datetime import datetime

from models.pipeline_models import (
    EndToEndConfig, PipelineJob, PipelineResults, ComparisonReport
)


class TestEndToEndConfig:
    """Test EndToEndConfig model."""
    
    def test_default_config(self):
        """Test default end-to-end configuration."""
        config = EndToEndConfig()
        assert config.generate_embeddings is True
        assert config.train_classifiers is True
        assert config.evaluate_models is True
        assert config.target_word_evaluation is True
        assert "logistic_regression" in config.classifier_types
        assert "positive" in config.target_words
        assert config.target_word_samples == 10
        assert config.create_summary_report is True
        assert config.compare_methods is True
    
    def test_custom_config(self):
        """Test custom end-to-end configuration."""
        config = EndToEndConfig(
            generate_embeddings=False,
            train_classifiers=True,
            classifier_types=["svm", "random_forest"],
            target_words=["approved", "denied"],
            target_word_samples=5,
            target_word_max_tokens=100
        )
        assert config.generate_embeddings is False
        assert len(config.classifier_types) == 2
        assert "svm" in config.classifier_types
        assert len(config.target_words) == 2
        assert config.target_word_samples == 5
    
    def test_invalid_classifier_types(self):
        """Test invalid classifier type validation."""
        with pytest.raises(ValidationError):
            EndToEndConfig(classifier_types=["invalid_classifier"])
    
    def test_empty_target_words(self):
        """Test empty target words validation."""
        with pytest.raises(ValidationError):
            EndToEndConfig(target_words=[])
    
    def test_target_word_cleaning(self):
        """Test target word cleaning (strip, lowercase)."""
        config = EndToEndConfig(
            target_words=["  APPROVED  ", "Positive", "", "   ", "good"]
        )
        # Should clean up and remove empty words
        assert "approved" in config.target_words
        assert "positive" in config.target_words
        assert "good" in config.target_words
        assert len(config.target_words) == 3  # Empty strings removed
    
    def test_target_word_sample_limits(self):
        """Test target word sample limits."""
        with pytest.raises(ValidationError):
            EndToEndConfig(target_word_samples=0)
        
        with pytest.raises(ValidationError):
            EndToEndConfig(target_word_samples=51)
    
    def test_target_word_token_limits(self):
        """Test target word max token limits."""
        with pytest.raises(ValidationError):
            EndToEndConfig(target_word_max_tokens=9)
        
        with pytest.raises(ValidationError):
            EndToEndConfig(target_word_max_tokens=1001)


class TestPipelineJob:
    """Test PipelineJob model."""
    
    def test_single_dataset_config(self, sample_dataset):
        """Test configuration with single dataset."""
        job = PipelineJob(
            dataset_path=str(sample_dataset),
            model_endpoint="http://localhost:8000",
            output_base_dir="/tmp/output",
            job_name="test_job"
        )
        assert job.dataset_path == str(sample_dataset)
        assert job.train_dataset_path is None
        assert job.test_dataset_path is None
        assert job.job_name == "test_job"
    
    def test_separate_datasets_config(self, temp_dir):
        """Test configuration with separate train/test datasets."""
        # Create dummy dataset files
        train_path = temp_dir / "train.csv"
        test_path = temp_dir / "test.csv"
        train_path.write_text("mcid,text,label\nMC001,text1,1")
        test_path.write_text("mcid,text,label\nMC002,text2,0")
        
        job = PipelineJob(
            train_dataset_path=str(train_path),
            test_dataset_path=str(test_path),
            model_endpoint="http://localhost:8000",
            output_base_dir="/tmp/output",
            job_name="test_job"
        )
        assert job.dataset_path is None
        assert job.train_dataset_path == str(train_path)
        assert job.test_dataset_path == str(test_path)
    
    def test_invalid_dataset_combination(self, sample_dataset, temp_dir):
        """Test invalid dataset path combinations."""
        train_path = temp_dir / "train.csv"
        train_path.write_text("mcid,text,label\nMC001,text1,1")
        
        # Cannot specify both single dataset and separate datasets
        with pytest.raises(ValidationError):
            PipelineJob(
                dataset_path=str(sample_dataset),
                train_dataset_path=str(train_path),
                output_base_dir="/tmp/output"
            )
    
    def test_missing_dataset_paths(self):
        """Test missing dataset paths validation."""
        # Must specify either single dataset OR both train/test
        with pytest.raises(ValidationError):
            PipelineJob(
                output_base_dir="/tmp/output",
                job_name="test_job"
            )
    
    def test_incomplete_separate_datasets(self, temp_dir):
        """Test incomplete separate dataset specification."""
        train_path = temp_dir / "train.csv"
        train_path.write_text("mcid,text,label\nMC001,text1,1")
        
        # Must specify both train AND test if using separate datasets
        with pytest.raises(ValidationError):
            PipelineJob(
                train_dataset_path=str(train_path),
                # Missing test_dataset_path
                output_base_dir="/tmp/output"
            )
    
    def test_nonexistent_dataset_files(self):
        """Test validation of nonexistent dataset files."""
        with pytest.raises(ValidationError):
            PipelineJob(
                dataset_path="/nonexistent/dataset.csv",
                output_base_dir="/tmp/output"
            )
    
    def test_default_values(self, sample_dataset):
        """Test default configuration values."""
        job = PipelineJob(
            dataset_path=str(sample_dataset),
            output_base_dir="/tmp/output"
        )
        assert job.model_endpoint == "http://localhost:8000"
        assert job.job_name == "pipeline_job"
        assert job.split_ratio is None
        assert isinstance(job.end_to_end, EndToEndConfig)
    
    def test_split_ratio_limits(self, sample_dataset):
        """Test split ratio validation."""
        with pytest.raises(ValidationError):
            PipelineJob(
                dataset_path=str(sample_dataset),
                output_base_dir="/tmp/output",
                split_ratio=0.05  # Too low
            )
        
        with pytest.raises(ValidationError):
            PipelineJob(
                dataset_path=str(sample_dataset),
                output_base_dir="/tmp/output",
                split_ratio=0.95  # Too high
            )
    
    def test_get_output_structure(self, sample_dataset):
        """Test output directory structure generation."""
        job = PipelineJob(
            dataset_path=str(sample_dataset),
            output_base_dir="/tmp/output",
            job_name="test_job"
        )
        
        structure = job.get_output_structure()
        
        expected_base = "/tmp/output/test_job"
        assert structure["embeddings"] == f"{expected_base}/embeddings"
        assert structure["models"] == f"{expected_base}/models"
        assert structure["metrics"] == f"{expected_base}/metrics"
        assert structure["target_word_results"] == f"{expected_base}/target_word_evaluation"
        assert structure["summary"] == f"{expected_base}/summary"
        assert structure["logs"] == f"{expected_base}/logs"
    
    def test_custom_end_to_end_config(self, sample_dataset):
        """Test custom end-to-end configuration."""
        custom_e2e = EndToEndConfig(
            generate_embeddings=False,
            classifier_types=["logistic_regression"],
            target_words=["approved"]
        )
        
        job = PipelineJob(
            dataset_path=str(sample_dataset),
            output_base_dir="/tmp/output",
            end_to_end=custom_e2e
        )
        
        assert job.end_to_end.generate_embeddings is False
        assert len(job.end_to_end.classifier_types) == 1
        assert len(job.end_to_end.target_words) == 1


class TestPipelineResults:
    """Test PipelineResults model."""
    
    def test_default_results(self):
        """Test default pipeline results."""
        results = PipelineResults(
            job_name="test_job",
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            total_duration=120.5
        )
        assert results.job_name == "test_job"
        assert results.total_duration == 120.5
        assert results.success is True
        assert len(results.errors) == 0
        assert results.embedding_results is None
        assert results.best_classifier is None
    
    def test_complete_results(self):
        """Test complete pipeline results with all data."""
        results = PipelineResults(
            job_name="test_job",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:02:00",
            total_duration=120.0,
            embedding_results={"status": "completed"},
            classification_results={
                "logistic_regression": {"best_score": 0.85},
                "svm": {"best_score": 0.82}
            },
            evaluation_results={
                "logistic_regression": {"accuracy": 0.85},
                "svm": {"accuracy": 0.82}
            },
            target_word_results={"accuracy": 0.78},
            best_classifier="logistic_regression",
            best_embedding_score=0.85,
            target_word_score=0.78,
            output_paths={
                "embeddings": "/path/to/embeddings",
                "models": "/path/to/models"
            },
            success=True,
            errors=[]
        )
        
        assert results.best_classifier == "logistic_regression"
        assert results.best_embedding_score == 0.85
        assert results.target_word_score == 0.78
        assert results.success is True
    
    def test_failed_results(self):
        """Test pipeline results with failures."""
        results = PipelineResults(
            job_name="failed_job",
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:01:00",
            total_duration=60.0,
            success=False,
            errors=["API connection failed", "Model training failed"]
        )
        
        assert results.success is False
        assert len(results.errors) == 2
        assert "API connection failed" in results.errors


class TestComparisonReport:
    """Test ComparisonReport model."""
    
    def test_valid_comparison_report(self):
        """Test creating a valid comparison report."""
        report = ComparisonReport(
            embedding_method={
                "best_classifier": "logistic_regression",
                "roc_auc": 0.85
            },
            target_word_method={"accuracy": 0.78},
            better_method="embedding",
            improvement=8.97,
            recommendations=[
                "Embedding method shows better performance",
                "Consider using trained classifier for production"
            ],
            per_classifier_comparison={
                "logistic_regression": {
                    "roc_auc": 0.85,
                    "accuracy": 0.82,
                    "f1_score": 0.80
                },
                "svm": {
                    "roc_auc": 0.83,
                    "accuracy": 0.80,
                    "f1_score": 0.78
                }
            }
        )
        
        assert report.better_method == "embedding"
        assert report.improvement == 8.97
        assert len(report.recommendations) == 2
        assert "logistic_regression" in report.per_classifier_comparison
    
    def test_minimal_comparison_report(self):
        """Test minimal comparison report."""
        report = ComparisonReport(
            embedding_method={"best_classifier": "svm", "roc_auc": 0.80},
            target_word_method={"accuracy": 0.85},
            better_method="target_word",
            improvement=6.25
        )
        
        assert report.better_method == "target_word"
        assert len(report.recommendations) == 0
        assert len(report.per_classifier_comparison) == 0


class TestPipelineModelIntegration:
    """Test integration between pipeline models."""
    
    def test_complete_pipeline_workflow(self, sample_dataset, temp_dir):
        """Test complete pipeline workflow with all models."""
        # Create end-to-end configuration
        e2e_config = EndToEndConfig(
            generate_embeddings=True,
            train_classifiers=True,
            evaluate_models=True,
            target_word_evaluation=True,
            classifier_types=["logistic_regression", "svm"],
            target_words=["approved", "positive"],
            target_word_samples=5,
            create_summary_report=True,
            compare_methods=True
        )
        
        # Create pipeline job
        job = PipelineJob(
            dataset_path=str(sample_dataset),
            model_endpoint="http://localhost:8001",
            output_base_dir=str(temp_dir),
            job_name="integration_test",
            split_ratio=0.8,
            end_to_end=e2e_config
        )
        
        # Create pipeline results
        results = PipelineResults(
            job_name=job.job_name,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            total_duration=300.0,
            embedding_results={"train": {"status": "completed"}, "test": {"status": "completed"}},
            classification_results={
                "logistic_regression": {"best_score": 0.85, "model_path": "/path/to/lr.pkl"},
                "svm": {"best_score": 0.82, "model_path": "/path/to/svm.pkl"}
            },
            evaluation_results={
                "logistic_regression": {"accuracy": 0.85, "precision": 0.83, "recall": 0.87},
                "svm": {"accuracy": 0.82, "precision": 0.80, "recall": 0.84}
            },
            target_word_results={"accuracy": 0.78, "precision": 0.76, "recall": 0.80},
            output_paths=job.get_output_structure(),
            success=True
        )
        
        # Create comparison report
        comparison = ComparisonReport(
            embedding_method={"best_classifier": "logistic_regression", "roc_auc": 0.85},
            target_word_method={"accuracy": 0.78},
            better_method="embedding",
            improvement=8.97,
            recommendations=["Use embedding method for production"],
            per_classifier_comparison={
                "logistic_regression": {"roc_auc": 0.85, "accuracy": 0.85},
                "svm": {"roc_auc": 0.82, "accuracy": 0.82}
            }
        )
        
        # Verify all components work together
        assert job.job_name == results.job_name
        assert len(job.end_to_end.classifier_types) == 2
        assert len(results.classification_results) == 2
        assert comparison.better_method == "embedding"
        assert results.success is True
    
    def test_pipeline_with_failures(self, sample_dataset, temp_dir):
        """Test pipeline handling with various failures."""
        job = PipelineJob(
            dataset_path=str(sample_dataset),
            output_base_dir=str(temp_dir),
            job_name="failure_test"
        )
        
        # Simulate partial failure
        results = PipelineResults(
            job_name=job.job_name,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            total_duration=60.0,
            embedding_results={"status": "completed"},
            classification_results={
                "logistic_regression": {"error": "Training failed"},
                "svm": {"best_score": 0.75, "model_path": "/path/to/svm.pkl"}
            },
            evaluation_results={
                "svm": {"accuracy": 0.75}
            },
            target_word_results={"error": "API timeout"},
            success=False,
            errors=["Logistic regression training failed", "Target word evaluation failed"]
        )
        
        assert results.success is False
        assert len(results.errors) == 2
        assert "error" in results.classification_results["logistic_regression"]
        assert "error" in results.target_word_results
    
    def test_edge_case_configurations(self, sample_dataset):
        """Test edge case configurations."""
        # Configuration with minimal settings
        minimal_e2e = EndToEndConfig(
            generate_embeddings=True,
            train_classifiers=False,
            evaluate_models=False,
            target_word_evaluation=False,
            classifier_types=["logistic_regression"],
            target_words=["positive"],
            target_word_samples=1,
            target_word_max_tokens=10,
            create_summary_report=False,
            compare_methods=False
        )
        
        job = PipelineJob(
            dataset_path=str(sample_dataset),
            output_base_dir="/tmp/minimal",
            job_name="minimal_test",
            split_ratio=0.5,  # Edge case split ratio
            end_to_end=minimal_e2e
        )
        
        assert job.end_to_end.train_classifiers is False
        assert job.end_to_end.target_word_samples == 1
        assert job.split_ratio == 0.5
    
    def test_unicode_handling(self, temp_dir):
        """Test handling of unicode characters in configurations."""
        # Create dataset with unicode content
        unicode_dataset = temp_dir / "unicode_data.csv"
        unicode_dataset.write_text(
            "mcid,text,label\n"
            "MC001,\"Patient: 你好世界 🏥\",1\n"
            "MC002,\"Résultat négatif\",0\n",
            encoding='utf-8'
        )
        
        job = PipelineJob(
            dataset_path=str(unicode_dataset),
            output_base_dir=str(temp_dir),
            job_name="unicode_测试",
            end_to_end=EndToEndConfig(
                target_words=["positive", "négatif", "好的"]
            )
        )
        
        assert job.job_name == "unicode_测试"
        assert "好的" in job.end_to_end.target_words