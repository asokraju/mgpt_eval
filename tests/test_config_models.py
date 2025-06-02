"""
Tests for configuration models and validation.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from pydantic import ValidationError

from models.config_models import (
    ModelAPIConfig, DataProcessingConfig, EmbeddingGenerationConfig,
    HyperparameterConfig, CrossValidationConfig, ClassificationConfig,
    VisualizationConfig, EvaluationConfig, TargetWordEvaluationConfig,
    OutputConfig, LoggingConfig, PipelineConfig
)


class TestModelAPIConfig:
    """Test ModelAPIConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelAPIConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.batch_size == 32
        assert config.max_retries == 3
        assert config.timeout == 300
        assert "/embeddings" in config.endpoints.values()
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ModelAPIConfig(
            base_url="https://api.example.com",
            batch_size=64,
            max_retries=5,
            timeout=600
        )
        assert config.base_url == "https://api.example.com"
        assert config.batch_size == 64
    
    def test_invalid_url(self):
        """Test invalid URL validation."""
        with pytest.raises(ValidationError):
            ModelAPIConfig(base_url="invalid-url")
    
    def test_url_trailing_slash_removal(self):
        """Test that trailing slashes are removed from URLs."""
        config = ModelAPIConfig(base_url="http://localhost:8000/")
        assert config.base_url == "http://localhost:8000"
    
    def test_batch_size_limits(self):
        """Test batch size validation."""
        with pytest.raises(ValidationError):
            ModelAPIConfig(batch_size=0)
        
        with pytest.raises(ValidationError):
            ModelAPIConfig(batch_size=513)
    
    def test_retry_limits(self):
        """Test retry limits validation."""
        with pytest.raises(ValidationError):
            ModelAPIConfig(max_retries=0)
        
        with pytest.raises(ValidationError):
            ModelAPIConfig(max_retries=11)
    
    def test_timeout_limits(self):
        """Test timeout validation."""
        with pytest.raises(ValidationError):
            ModelAPIConfig(timeout=29)
        
        with pytest.raises(ValidationError):
            ModelAPIConfig(timeout=3601)


class TestDataProcessingConfig:
    """Test DataProcessingConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DataProcessingConfig()
        assert config.train_test_split == 0.8
        assert config.random_seed == 42
        assert config.max_sequence_length == 512
        assert config.include_mcid is True
        assert config.output_format == "json"
    
    def test_split_ratio_limits(self):
        """Test train/test split ratio validation."""
        with pytest.raises(ValidationError):
            DataProcessingConfig(train_test_split=0.05)
        
        with pytest.raises(ValidationError):
            DataProcessingConfig(train_test_split=0.95)
    
    def test_sequence_length_limits(self):
        """Test sequence length validation."""
        with pytest.raises(ValidationError):
            DataProcessingConfig(max_sequence_length=31)
        
        with pytest.raises(ValidationError):
            DataProcessingConfig(max_sequence_length=8193)
    
    def test_random_seed_validation(self):
        """Test random seed validation."""
        # Valid seeds
        config = DataProcessingConfig(random_seed=0)
        assert config.random_seed == 0
        
        config = DataProcessingConfig(random_seed=None)
        assert config.random_seed is None
        
        # Invalid seeds
        with pytest.raises(ValidationError):
            DataProcessingConfig(random_seed=-1)
        
        with pytest.raises(ValidationError):
            DataProcessingConfig(random_seed=2**32)
    
    def test_output_format_validation(self):
        """Test output format validation."""
        config = DataProcessingConfig(output_format="csv")
        assert config.output_format == "csv"
        
        with pytest.raises(ValidationError):
            DataProcessingConfig(output_format="xml")


class TestEmbeddingGenerationConfig:
    """Test EmbeddingGenerationConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingGenerationConfig()
        assert config.batch_size == 16
        assert config.save_interval == 100
        assert config.checkpoint_dir == "outputs/checkpoints"
        assert config.resume_from_checkpoint is True
    
    def test_batch_size_limits(self):
        """Test batch size validation."""
        with pytest.raises(ValidationError):
            EmbeddingGenerationConfig(batch_size=0)
        
        with pytest.raises(ValidationError):
            EmbeddingGenerationConfig(batch_size=129)
    
    def test_save_interval_limits(self):
        """Test save interval validation."""
        with pytest.raises(ValidationError):
            EmbeddingGenerationConfig(save_interval=9)
        
        with pytest.raises(ValidationError):
            EmbeddingGenerationConfig(save_interval=1001)
    
    def test_checkpoint_dir_creation(self, temp_dir):
        """Test that checkpoint directory is created."""
        checkpoint_path = temp_dir / "test_checkpoints"
        config = EmbeddingGenerationConfig(checkpoint_dir=str(checkpoint_path))
        assert checkpoint_path.exists()


class TestHyperparameterConfig:
    """Test HyperparameterConfig model."""
    
    def test_default_config(self):
        """Test default hyperparameter configuration."""
        config = HyperparameterConfig()
        assert "C" in config.logistic_regression
        assert "kernel" in config.svm
        assert "n_estimators" in config.random_forest
    
    def test_custom_parameters(self):
        """Test custom hyperparameter values."""
        config = HyperparameterConfig(
            logistic_regression={"C": [0.1, 1.0]},
            svm={"C": [1.0], "kernel": ["linear"]},
            random_forest={"n_estimators": [100]}
        )
        assert config.logistic_regression["C"] == [0.1, 1.0]
        assert config.svm["kernel"] == ["linear"]


class TestCrossValidationConfig:
    """Test CrossValidationConfig model."""
    
    def test_default_config(self):
        """Test default cross-validation configuration."""
        config = CrossValidationConfig()
        assert config.n_folds == 5
        assert config.scoring == "roc_auc"
        assert config.n_jobs == -1
    
    def test_fold_limits(self):
        """Test number of folds validation."""
        with pytest.raises(ValidationError):
            CrossValidationConfig(n_folds=1)
        
        with pytest.raises(ValidationError):
            CrossValidationConfig(n_folds=11)


class TestClassificationConfig:
    """Test ClassificationConfig model."""
    
    def test_default_config(self):
        """Test default classification configuration."""
        config = ClassificationConfig()
        assert "logistic_regression" in config.classifier_types
        assert "svm" in config.classifier_types
        assert "random_forest" in config.classifier_types
    
    def test_invalid_classifier_type(self):
        """Test invalid classifier type validation."""
        with pytest.raises(ValidationError):
            ClassificationConfig(classifier_types=["invalid_classifier"])
    
    def test_custom_classifier_types(self):
        """Test custom classifier types."""
        config = ClassificationConfig(
            classifier_types=["logistic_regression", "svm"]
        )
        assert len(config.classifier_types) == 2
        assert "random_forest" not in config.classifier_types


class TestVisualizationConfig:
    """Test VisualizationConfig model."""
    
    def test_default_config(self):
        """Test default visualization configuration."""
        config = VisualizationConfig()
        assert config.generate_plots is True
        assert "png" in config.plot_formats
        assert config.dpi == 300
    
    def test_invalid_plot_format(self):
        """Test invalid plot format validation."""
        with pytest.raises(ValidationError):
            VisualizationConfig(plot_formats=["invalid_format"])
    
    def test_dpi_limits(self):
        """Test DPI validation."""
        with pytest.raises(ValidationError):
            VisualizationConfig(dpi=71)
        
        with pytest.raises(ValidationError):
            VisualizationConfig(dpi=601)


class TestEvaluationConfig:
    """Test EvaluationConfig model."""
    
    def test_default_config(self):
        """Test default evaluation configuration."""
        config = EvaluationConfig()
        assert "accuracy" in config.metrics
        assert "roc_auc" in config.metrics
        assert config.visualization.generate_plots is True
    
    def test_invalid_metric(self):
        """Test invalid metric validation."""
        with pytest.raises(ValidationError):
            EvaluationConfig(metrics=["invalid_metric"])
    
    def test_custom_metrics(self):
        """Test custom metrics selection."""
        config = EvaluationConfig(
            metrics=["accuracy", "precision", "recall"]
        )
        assert len(config.metrics) == 3
        assert "f1_score" not in config.metrics


class TestTargetWordEvaluationConfig:
    """Test TargetWordEvaluationConfig model."""
    
    def test_default_config(self):
        """Test default target word evaluation configuration."""
        config = TargetWordEvaluationConfig()
        assert config.n_generations == 10
        assert config.max_new_tokens == 200
        assert config.temperature == 0.8
        assert config.search_method == "exact"
    
    def test_generation_limits(self):
        """Test generation parameter limits."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(n_generations=0)
        
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(n_generations=101)
    
    def test_token_limits(self):
        """Test token limits validation."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(max_new_tokens=9)
        
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(max_new_tokens=2049)
    
    def test_temperature_limits(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(temperature=0.05)
        
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(temperature=2.1)
    
    def test_top_k_limits(self):
        """Test top_k validation."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(top_k=0)
        
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(top_k=201)
    
    def test_context_length_limits(self):
        """Test context length validation."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(max_context_length=511)
        
        with pytest.raises(ValidationError):
            TargetWordEvaluationConfig(max_context_length=16385)


class TestOutputConfig:
    """Test OutputConfig model."""
    
    def test_default_config(self):
        """Test default output configuration."""
        config = OutputConfig()
        assert config.embeddings_dir == "outputs/embeddings"
        assert config.models_dir == "outputs/models"
        assert config.save_best_model_only is False
        assert config.model_format == "pickle"
    
    def test_directory_creation(self, temp_dir):
        """Test that output directories are created."""
        config = OutputConfig(
            embeddings_dir=str(temp_dir / "embeddings"),
            models_dir=str(temp_dir / "models"),
            metrics_dir=str(temp_dir / "metrics"),
            logs_dir=str(temp_dir / "logs")
        )
        config.create_directories()
        
        assert (temp_dir / "embeddings").exists()
        assert (temp_dir / "models").exists()
        assert (temp_dir / "metrics").exists()
        assert (temp_dir / "logs").exists()


class TestLoggingConfig:
    """Test LoggingConfig model."""
    
    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.console_level == "INFO"
        assert config.backup_count == 5
    
    def test_invalid_log_level(self):
        """Test invalid log level validation."""
        with pytest.raises(ValidationError):
            LoggingConfig(level="INVALID")
    
    def test_backup_count_limits(self):
        """Test backup count validation."""
        with pytest.raises(ValidationError):
            LoggingConfig(backup_count=0)
        
        with pytest.raises(ValidationError):
            LoggingConfig(backup_count=21)
    
    def test_log_file_directory_creation(self, temp_dir):
        """Test that log file directory is created."""
        log_file = temp_dir / "logs" / "test.log"
        config = LoggingConfig(file=str(log_file))
        assert log_file.parent.exists()


class TestPipelineConfig:
    """Test PipelineConfig model."""
    
    def test_default_config(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()
        assert isinstance(config.model_api, ModelAPIConfig)
        assert isinstance(config.data_processing, DataProcessingConfig)
        assert isinstance(config.embedding_generation, EmbeddingGenerationConfig)
        assert isinstance(config.classification, ClassificationConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert isinstance(config.target_word_evaluation, TargetWordEvaluationConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_custom_config(self):
        """Test custom pipeline configuration."""
        config = PipelineConfig(
            model_api=ModelAPIConfig(base_url="http://custom:8080"),
            data_processing=DataProcessingConfig(train_test_split=0.9)
        )
        assert config.model_api.base_url == "http://custom:8080"
        assert config.data_processing.train_test_split == 0.9
    
    def test_from_yaml(self, temp_dir):
        """Test loading configuration from YAML file."""
        config_data = {
            "model_api": {"base_url": "http://test:8000"},
            "data_processing": {"train_test_split": 0.75},
            "embedding_generation": {"batch_size": 8}
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = PipelineConfig.from_yaml(str(config_file))
        assert config.model_api.base_url == "http://test:8000"
        assert config.data_processing.train_test_split == 0.75
        assert config.embedding_generation.batch_size == 8
    
    def test_to_yaml(self, temp_dir):
        """Test saving configuration to YAML file."""
        config = PipelineConfig(
            model_api=ModelAPIConfig(base_url="http://test:8000")
        )
        
        config_file = temp_dir / "output_config.yaml"
        config.to_yaml(str(config_file))
        
        assert config_file.exists()
        
        # Load back and verify
        with open(config_file, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["model_api"]["base_url"] == "http://test:8000"
    
    def test_setup_directories(self, temp_dir):
        """Test setup_directories method."""
        config = PipelineConfig(
            embedding_generation=EmbeddingGenerationConfig(
                checkpoint_dir=str(temp_dir / "checkpoints")
            ),
            output=OutputConfig(
                embeddings_dir=str(temp_dir / "embeddings"),
                models_dir=str(temp_dir / "models"),
                metrics_dir=str(temp_dir / "metrics"),
                logs_dir=str(temp_dir / "logs")
            )
        )
        
        config.setup_directories()
        
        assert (temp_dir / "checkpoints").exists()
        assert (temp_dir / "embeddings").exists()
        assert (temp_dir / "models").exists()
        assert (temp_dir / "metrics").exists()
        assert (temp_dir / "logs").exists()
    
    def test_yaml_round_trip(self, temp_dir):
        """Test YAML save and load round trip."""
        original_config = PipelineConfig(
            model_api=ModelAPIConfig(base_url="http://test:8000", batch_size=16),
            data_processing=DataProcessingConfig(train_test_split=0.75),
            embedding_generation=EmbeddingGenerationConfig(batch_size=8)
        )
        
        config_file = temp_dir / "round_trip.yaml"
        original_config.to_yaml(str(config_file))
        loaded_config = PipelineConfig.from_yaml(str(config_file))
        
        assert loaded_config.model_api.base_url == original_config.model_api.base_url
        assert loaded_config.model_api.batch_size == original_config.model_api.batch_size
        assert loaded_config.data_processing.train_test_split == original_config.data_processing.train_test_split
        assert loaded_config.embedding_generation.batch_size == original_config.embedding_generation.batch_size
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in configuration."""
        config_data = {
            "model_api": {"base_url": "http://test:8000"},
            "extra_field": "extra_value"
        }
        
        config = PipelineConfig(**config_data)
        assert hasattr(config, "extra_field")
        assert config.extra_field == "extra_value"
    
    def test_validate_assignment(self):
        """Test that assignment validation works."""
        config = PipelineConfig()
        
        # This should work
        config.model_api = ModelAPIConfig(base_url="http://test:8000")
        
        # This should fail validation
        with pytest.raises(ValidationError):
            config.data_processing = DataProcessingConfig(train_test_split=1.5)


class TestConfigurationIntegration:
    """Test configuration integration and edge cases."""
    
    def test_minimal_valid_config(self):
        """Test minimal valid configuration."""
        config = PipelineConfig()
        assert config is not None
        
        # All required components should be present
        assert config.model_api is not None
        assert config.data_processing is not None
        assert config.embedding_generation is not None
        assert config.classification is not None
        assert config.evaluation is not None
        assert config.target_word_evaluation is not None
        assert config.output is not None
        assert config.logging is not None
    
    def test_config_with_all_custom_values(self, temp_dir):
        """Test configuration with all custom values."""
        config = PipelineConfig(
            model_api=ModelAPIConfig(
                base_url="https://custom-api.example.com",
                batch_size=64,
                max_retries=5,
                timeout=600
            ),
            data_processing=DataProcessingConfig(
                train_test_split=0.75,
                random_seed=123,
                max_sequence_length=1024,
                include_mcid=False,
                output_format="csv"
            ),
            embedding_generation=EmbeddingGenerationConfig(
                batch_size=32,
                save_interval=50,
                checkpoint_dir=str(temp_dir / "custom_checkpoints"),
                resume_from_checkpoint=False,
                tokenizer_path="/custom/tokenizer"
            ),
            classification=ClassificationConfig(
                classifier_types=["logistic_regression"],
                hyperparameter_search=HyperparameterConfig(
                    logistic_regression={"C": [1.0], "penalty": ["l2"]}
                ),
                cross_validation=CrossValidationConfig(
                    n_folds=3,
                    scoring="accuracy",
                    n_jobs=2
                )
            ),
            evaluation=EvaluationConfig(
                metrics=["accuracy", "precision"],
                visualization=VisualizationConfig(
                    generate_plots=False,
                    plot_formats=["png"],
                    dpi=150
                )
            ),
            target_word_evaluation=TargetWordEvaluationConfig(
                n_generations=5,
                max_new_tokens=100,
                temperature=0.5,
                top_k=20,
                search_method="fuzzy",
                max_context_length=1024,
                tokenizer_path="/custom/tokenizer"
            ),
            output=OutputConfig(
                embeddings_dir=str(temp_dir / "custom_embeddings"),
                models_dir=str(temp_dir / "custom_models"),
                metrics_dir=str(temp_dir / "custom_metrics"),
                logs_dir=str(temp_dir / "custom_logs"),
                save_best_model_only=True,
                model_format="joblib"
            ),
            logging=LoggingConfig(
                level="DEBUG",
                format="%(message)s",
                file=str(temp_dir / "custom.log"),
                max_file_size="5MB",
                backup_count=3,
                console_level="ERROR"
            )
        )
        
        # Verify all custom values are set
        assert config.model_api.base_url == "https://custom-api.example.com"
        assert config.data_processing.train_test_split == 0.75
        assert config.embedding_generation.batch_size == 32
        assert config.classification.classifier_types == ["logistic_regression"]
        assert config.evaluation.metrics == ["accuracy", "precision"]
        assert config.target_word_evaluation.n_generations == 5
        assert config.output.save_best_model_only is True
        assert config.logging.level == "DEBUG"
    
    def test_invalid_yaml_file(self, temp_dir):
        """Test loading from invalid YAML file."""
        config_file = temp_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            PipelineConfig.from_yaml(str(config_file))
    
    def test_nonexistent_yaml_file(self):
        """Test loading from nonexistent YAML file."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/config.yaml")