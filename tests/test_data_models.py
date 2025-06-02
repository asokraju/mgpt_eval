"""
Tests for data models and validation.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from pydantic import ValidationError

from models.data_models import (
    DatasetRecord, Dataset, EmbeddingRequest, EmbeddingResponse,
    ClassificationRequest, ClassificationResult, EvaluationRequest,
    EvaluationResult, TargetWordEvaluationRequest, TargetWordEvaluationResult,
    PipelineStatus, ModelInfo, ValidationError as CustomValidationError
)


class TestDatasetRecord:
    """Test DatasetRecord model."""
    
    def test_valid_record(self):
        """Test creating a valid dataset record."""
        record = DatasetRecord(
            mcid="MC001",
            text="Patient shows improvement",
            label=1
        )
        assert record.mcid == "MC001"
        assert record.text == "Patient shows improvement"
        assert record.label == 1
    
    def test_invalid_label(self):
        """Test invalid label values."""
        with pytest.raises(ValidationError):
            DatasetRecord(mcid="MC001", text="Some text", label=2)
        
        with pytest.raises(ValidationError):
            DatasetRecord(mcid="MC001", text="Some text", label=-1)
    
    def test_empty_text(self):
        """Test empty text validation."""
        with pytest.raises(ValidationError):
            DatasetRecord(mcid="MC001", text="", label=1)
    
    def test_extra_fields(self):
        """Test that extra fields are allowed."""
        record = DatasetRecord(
            mcid="MC001",
            text="Some text",
            label=1,
            extra_field="extra_value"
        )
        assert record.extra_field == "extra_value"


class TestDataset:
    """Test Dataset model."""
    
    def test_valid_dataset(self):
        """Test creating a valid dataset."""
        records = [
            DatasetRecord(mcid="MC001", text="Text 1", label=1),
            DatasetRecord(mcid="MC002", text="Text 2", label=0)
        ]
        dataset = Dataset(records=records)
        assert len(dataset.records) == 2
    
    def test_empty_dataset(self):
        """Test empty dataset validation."""
        with pytest.raises(ValidationError):
            Dataset(records=[])
    
    def test_from_dataframe(self):
        """Test creating dataset from DataFrame."""
        df = pd.DataFrame([
            {"mcid": "MC001", "text": "Text 1", "label": 1},
            {"mcid": "MC002", "text": "Text 2", "label": 0}
        ])
        dataset = Dataset.from_dataframe(df)
        assert len(dataset.records) == 2
        assert dataset.records[0].mcid == "MC001"
    
    def test_to_dataframe(self):
        """Test converting dataset to DataFrame."""
        records = [
            DatasetRecord(mcid="MC001", text="Text 1", label=1),
            DatasetRecord(mcid="MC002", text="Text 2", label=0)
        ]
        dataset = Dataset(records=records)
        df = dataset.to_dataframe()
        
        assert len(df) == 2
        assert "mcid" in df.columns
        assert "text" in df.columns
        assert "label" in df.columns
    
    def test_from_file_csv(self, temp_dir):
        """Test loading dataset from CSV file."""
        # Create test CSV
        df = pd.DataFrame([
            {"mcid": "MC001", "text": "Text 1", "label": 1},
            {"mcid": "MC002", "text": "Text 2", label": 0}
        ])
        csv_path = temp_dir / "test.csv"
        df.to_csv(csv_path, index=False)
        
        dataset = Dataset.from_file(str(csv_path))
        assert len(dataset.records) == 2
    
    def test_from_file_missing_columns(self, temp_dir):
        """Test loading dataset with missing required columns."""
        df = pd.DataFrame([
            {"mcid": "MC001", "text": "Text 1"},  # Missing label
        ])
        csv_path = temp_dir / "test.csv"
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            Dataset.from_file(str(csv_path))
    
    def test_from_file_nonexistent(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Dataset.from_file("/nonexistent/file.csv")
    
    def test_from_file_unsupported_format(self, temp_dir):
        """Test loading from unsupported file format."""
        txt_path = temp_dir / "test.txt"
        txt_path.write_text("some content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            Dataset.from_file(str(txt_path))


class TestEmbeddingRequest:
    """Test EmbeddingRequest model."""
    
    def test_valid_request(self):
        """Test creating a valid embedding request."""
        request = EmbeddingRequest(
            texts=["Text 1", "Text 2"],
            batch_size=32
        )
        assert len(request.texts) == 2
        assert request.batch_size == 32
    
    def test_empty_texts(self):
        """Test empty texts validation."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(texts=[])
    
    def test_empty_text_in_list(self):
        """Test empty text within list."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(texts=["Valid text", ""])
    
    def test_whitespace_only_text(self):
        """Test whitespace-only text."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(texts=["Valid text", "   "])
    
    def test_batch_size_limits(self):
        """Test batch size validation."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(texts=["Text"], batch_size=0)
        
        with pytest.raises(ValidationError):
            EmbeddingRequest(texts=["Text"], batch_size=129)


class TestEmbeddingResponse:
    """Test EmbeddingResponse model."""
    
    def test_valid_response(self):
        """Test creating a valid embedding response."""
        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            input_tokens=[5, 6],
            embedding_dim=2,
            execution_time=1.5
        )
        assert len(response.embeddings) == 2
        assert response.embedding_dim == 2
    
    def test_mismatched_lengths(self):
        """Test validation of mismatched embedding and token counts."""
        with pytest.raises(ValidationError):
            EmbeddingResponse(
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                input_tokens=[5],  # Wrong length
                embedding_dim=2,
                execution_time=1.5
            )


class TestClassificationRequest:
    """Test ClassificationRequest model."""
    
    def test_valid_request(self, sample_embeddings):
        """Test creating a valid classification request."""
        request = ClassificationRequest(
            train_embeddings_path=str(sample_embeddings["train"]),
            test_embeddings_path=str(sample_embeddings["test"]),
            classifier_type="logistic_regression",
            output_dir="/tmp/output"
        )
        assert request.classifier_type == "logistic_regression"
    
    def test_invalid_classifier_type(self, sample_embeddings):
        """Test invalid classifier type."""
        with pytest.raises(ValidationError):
            ClassificationRequest(
                train_embeddings_path=str(sample_embeddings["train"]),
                test_embeddings_path=str(sample_embeddings["test"]),
                classifier_type="invalid_classifier",
                output_dir="/tmp/output"
            )
    
    def test_nonexistent_file(self):
        """Test nonexistent embedding files."""
        with pytest.raises(ValidationError):
            ClassificationRequest(
                train_embeddings_path="/nonexistent/train.json",
                test_embeddings_path="/nonexistent/test.json",
                classifier_type="logistic_regression",
                output_dir="/tmp/output"
            )


class TestEvaluationRequest:
    """Test EvaluationRequest model."""
    
    def test_valid_request(self, sample_embeddings, temp_dir):
        """Test creating a valid evaluation request."""
        # Create dummy model file
        model_path = temp_dir / "model.pkl"
        model_path.write_text("dummy model")
        
        request = EvaluationRequest(
            model_path=str(model_path),
            test_embeddings_path=str(sample_embeddings["test"]),
            output_dir=str(temp_dir)
        )
        assert Path(request.model_path).exists()
    
    def test_nonexistent_model(self, sample_embeddings):
        """Test nonexistent model file."""
        with pytest.raises(ValidationError):
            EvaluationRequest(
                model_path="/nonexistent/model.pkl",
                test_embeddings_path=str(sample_embeddings["test"]),
                output_dir="/tmp/output"
            )


class TestTargetWordEvaluationRequest:
    """Test TargetWordEvaluationRequest model."""
    
    def test_valid_request(self, sample_dataset):
        """Test creating a valid target word evaluation request."""
        request = TargetWordEvaluationRequest(
            dataset_path=str(sample_dataset),
            target_words=["approved", "positive"],
            n_samples=5,
            max_tokens=100
        )
        assert len(request.target_words) == 2
        assert "approved" in request.target_words
    
    def test_empty_target_words(self, sample_dataset):
        """Test empty target words."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationRequest(
                dataset_path=str(sample_dataset),
                target_words=[],
                n_samples=5,
                max_tokens=100
            )
    
    def test_whitespace_target_words(self, sample_dataset):
        """Test target words with only whitespace."""
        # Should clean up and remove empty words
        request = TargetWordEvaluationRequest(
            dataset_path=str(sample_dataset),
            target_words=["approved", "  ", "positive", ""],
            n_samples=5,
            max_tokens=100
        )
        assert len(request.target_words) == 2
        assert "approved" in request.target_words
        assert "positive" in request.target_words
    
    def test_target_word_cleaning(self, sample_dataset):
        """Test target word cleaning (lowercase, strip)."""
        request = TargetWordEvaluationRequest(
            dataset_path=str(sample_dataset),
            target_words=["  APPROVED  ", "Positive"],
            n_samples=5,
            max_tokens=100
        )
        assert "approved" in request.target_words
        assert "positive" in request.target_words
    
    def test_invalid_n_samples(self, sample_dataset):
        """Test invalid n_samples values."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationRequest(
                dataset_path=str(sample_dataset),
                target_words=["approved"],
                n_samples=0,
                max_tokens=100
            )
        
        with pytest.raises(ValidationError):
            TargetWordEvaluationRequest(
                dataset_path=str(sample_dataset),
                target_words=["approved"],
                n_samples=101,
                max_tokens=100
            )
    
    def test_invalid_max_tokens(self, sample_dataset):
        """Test invalid max_tokens values."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationRequest(
                dataset_path=str(sample_dataset),
                target_words=["approved"],
                n_samples=5,
                max_tokens=5
            )
        
        with pytest.raises(ValidationError):
            TargetWordEvaluationRequest(
                dataset_path=str(sample_dataset),
                target_words=["approved"],
                n_samples=5,
                max_tokens=2049
            )


class TestTargetWordEvaluationResult:
    """Test TargetWordEvaluationResult model."""
    
    def test_valid_result(self):
        """Test creating a valid target word evaluation result."""
        result = TargetWordEvaluationResult(
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1_score=0.85,
            confusion_matrix=[[10, 2], [1, 7]],
            target_words=["approved", "positive"],
            word_occurrence_stats={"approved": 15, "positive": 10},
            n_samples=10,
            max_tokens=100,
            results_path="/tmp/results.json",
            timestamp="2024-01-01T00:00:00"
        )
        assert result.accuracy == 0.85
        assert len(result.target_words) == 2
    
    def test_metric_bounds(self):
        """Test metric value bounds."""
        with pytest.raises(ValidationError):
            TargetWordEvaluationResult(
                accuracy=1.5,  # Invalid: > 1.0
                precision=0.80,
                recall=0.90,
                f1_score=0.85,
                confusion_matrix=[[10, 2], [1, 7]],
                target_words=["approved"],
                n_samples=10,
                max_tokens=100,
                results_path="/tmp/results.json",
                timestamp="2024-01-01T00:00:00"
            )


class TestPipelineStatus:
    """Test PipelineStatus model."""
    
    def test_valid_status(self):
        """Test creating a valid pipeline status."""
        status = PipelineStatus(
            stage="embedding_generation",
            status="running",
            progress=0.5,
            message="Processing batch 5 of 10"
        )
        assert status.stage == "embedding_generation"
        assert status.status == "running"
        assert status.progress == 0.5
    
    def test_invalid_status(self):
        """Test invalid status values."""
        with pytest.raises(ValidationError):
            PipelineStatus(
                stage="embedding_generation",
                status="invalid_status",
                progress=0.5
            )
    
    def test_invalid_progress(self):
        """Test invalid progress values."""
        with pytest.raises(ValidationError):
            PipelineStatus(
                stage="embedding_generation",
                status="running",
                progress=1.5  # Invalid: > 1.0
            )


class TestModelInfo:
    """Test ModelInfo model."""
    
    def test_valid_model_info(self):
        """Test creating valid model info."""
        info = ModelInfo(
            model_type="logistic_regression",
            parameters={"C": 1.0, "penalty": "l2"},
            training_data_size=1000,
            test_data_size=200,
            features=768,
            classes=[0, 1],
            created_at="2024-01-01T00:00:00"
        )
        assert info.model_type == "logistic_regression"
        assert info.features == 768


class TestCustomValidationError:
    """Test custom ValidationError model."""
    
    def test_valid_validation_error(self):
        """Test creating a valid validation error."""
        error = CustomValidationError(
            field="text",
            message="Text cannot be empty",
            invalid_value=""
        )
        assert error.field == "text"
        assert error.message == "Text cannot be empty"
        assert error.invalid_value == ""


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unicode_text(self):
        """Test handling of unicode text."""
        record = DatasetRecord(
            mcid="MC001",
            text="Patient: 你好世界 émojis: 🏥👨‍⚕️",
            label=1
        )
        assert "你好世界" in record.text
        assert "🏥" in record.text
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "A" * 10000
        record = DatasetRecord(
            mcid="MC001",
            text=long_text,
            label=1
        )
        assert len(record.text) == 10000
    
    def test_special_characters_in_mcid(self):
        """Test special characters in MCID."""
        record = DatasetRecord(
            mcid="MC-001_TEST.v2",
            text="Some text",
            label=1
        )
        assert record.mcid == "MC-001_TEST.v2"
    
    def test_large_embedding_dimensions(self):
        """Test large embedding dimensions."""
        large_embedding = [0.1] * 4096
        response = EmbeddingResponse(
            embeddings=[large_embedding],
            input_tokens=[100],
            embedding_dim=4096,
            execution_time=5.0
        )
        assert response.embedding_dim == 4096
        assert len(response.embeddings[0]) == 4096