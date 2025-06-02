"""
Pytest configuration and fixtures for binary classifier tests.
"""

import pytest
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import requests
import time
import subprocess
import sys
import threading
import atexit

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config_models import PipelineConfig
from models.data_models import Dataset, DatasetRecord


class FakeAPIManager:
    """Manages the fake API server for testing."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
        
    def start(self):
        """Start the fake API server."""
        if self.process is not None:
            return
            
        # Path to fake API server
        fake_api_path = Path(__file__).parent.parent.parent / "fake_api_server.py"
        
        # Start server in background
        self.process = subprocess.Popen([
            sys.executable, str(fake_api_path),
            "--port", str(self.port),
            "--seed", "42"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Fake API server failed to start")
    
    def stop(self):
        """Stop the fake API server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
    
    def is_running(self) -> bool:
        """Check if server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=1)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


# Global API manager
api_manager = FakeAPIManager()

def pytest_configure(config):
    """Configure pytest session."""
    # Start fake API server
    api_manager.start()
    
    # Register cleanup
    atexit.register(api_manager.stop)

def pytest_unconfigure(config):
    """Cleanup after pytest session."""
    api_manager.stop()


@pytest.fixture(scope="session")
def fake_api_url():
    """Provide the fake API URL."""
    return api_manager.base_url


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config(temp_dir: Path) -> PipelineConfig:
    """Create a sample pipeline configuration for testing."""
    config_dict = {
        "model_api": {
            "base_url": api_manager.base_url,
            "batch_size": 4,
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
            "save_interval": 5,
            "checkpoint_dir": str(temp_dir / "checkpoints"),
            "resume_from_checkpoint": True,
            "tokenizer_path": "/nonexistent/tokenizer"  # Will fallback to estimation
        },
        "classification": {
            "classifier_types": ["logistic_regression", "svm"],
            "hyperparameter_search": {
                "logistic_regression": {
                    "C": [0.1, 1.0],
                    "penalty": ["l2"],
                    "solver": ["liblinear"]
                },
                "svm": {
                    "C": [0.1, 1.0],
                    "kernel": ["rbf"],
                    "gamma": ["scale"]
                }
            },
            "cross_validation": {
                "n_folds": 3,
                "scoring": "roc_auc",
                "n_jobs": 1
            }
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            "visualization": {
                "generate_plots": True,
                "plot_formats": ["png"],
                "dpi": 150
            }
        },
        "target_word_evaluation": {
            "n_generations": 3,
            "max_new_tokens": 50,
            "temperature": 0.8,
            "top_k": 50,
            "search_method": "exact",
            "max_context_length": 512,
            "tokenizer_path": "/nonexistent/tokenizer"
        },
        "output": {
            "embeddings_dir": str(temp_dir / "embeddings"),
            "models_dir": str(temp_dir / "models"),
            "metrics_dir": str(temp_dir / "metrics"),
            "logs_dir": str(temp_dir / "logs"),
            "save_best_model_only": False,
            "model_format": "pickle"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": str(temp_dir / "logs" / "test.log"),
            "max_file_size": "1MB",
            "backup_count": 1,
            "console_level": "WARNING"  # Reduce console noise during tests
        }
    }
    
    return PipelineConfig(**config_dict)


@pytest.fixture
def sample_dataset(temp_dir: Path) -> Path:
    """Create a sample dataset for testing."""
    # Create balanced dataset with various text patterns
    data = []
    
    # Positive examples (label = 1)
    positive_texts = [
        "The medical claim has been approved for treatment coverage",
        "Patient shows excellent improvement after the prescribed treatment",
        "Insurance coverage approved for the requested procedure", 
        "Treatment was successful and patient is now stable",
        "Claim approved for full reimbursement of medical expenses",
        "Patient condition improved significantly with new medication",
        "Surgery was completed successfully with positive outcomes",
        "Medical review completed - claim approved for processing",
        "Treatment plan shows good results and patient recovery",
        "Insurance approval granted for continued care coverage"
    ]
    
    # Negative examples (label = 0)
    negative_texts = [
        "The medical claim has been denied due to insufficient documentation",
        "Patient condition deteriorated despite ongoing treatment efforts",
        "Insurance claim rejected - treatment not covered under policy",
        "No significant improvement observed in patient condition",
        "Claim denied due to policy exclusions and limitations",
        "Patient status remains poor with ongoing complications",
        "Treatment failed to produce expected therapeutic results",
        "Medical review completed - claim rejected for processing", 
        "Procedure not approved by insurance review board",
        "Patient response to treatment has been inadequate overall"
    ]
    
    # Combine data
    for i, text in enumerate(positive_texts):
        data.append({
            "mcid": f"MC{i+1:03d}",
            "text": text,
            "label": 1
        })
    
    for i, text in enumerate(negative_texts):
        data.append({
            "mcid": f"MC{i+101:03d}",
            "text": text,
            "label": 0
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    dataset_path = temp_dir / "sample_dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    return dataset_path


@pytest.fixture
def small_dataset(temp_dir: Path) -> Path:
    """Create a small dataset for quick tests."""
    data = [
        {"mcid": "MC001", "text": "Patient approved for treatment", "label": 1},
        {"mcid": "MC002", "text": "Claim denied by insurance", "label": 0},
        {"mcid": "MC003", "text": "Excellent improvement noted", "label": 1},
        {"mcid": "MC004", "text": "Treatment failed completely", "label": 0}
    ]
    
    df = pd.DataFrame(data)
    dataset_path = temp_dir / "small_dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    return dataset_path


@pytest.fixture
def imbalanced_dataset(temp_dir: Path) -> Path:
    """Create an imbalanced dataset for testing edge cases."""
    data = []
    
    # 90% positive examples
    for i in range(18):
        data.append({
            "mcid": f"MC{i+1:03d}",
            "text": f"Approved claim number {i+1} for medical treatment",
            "label": 1
        })
    
    # 10% negative examples  
    for i in range(2):
        data.append({
            "mcid": f"MC{i+101:03d}",
            "text": f"Denied claim number {i+1} for insufficient coverage",
            "label": 0
        })
    
    df = pd.DataFrame(data)
    dataset_path = temp_dir / "imbalanced_dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    return dataset_path


@pytest.fixture
def invalid_datasets(temp_dir: Path) -> Dict[str, Path]:
    """Create various invalid datasets for testing error handling."""
    datasets = {}
    
    # Missing required columns
    missing_cols_data = [
        {"mcid": "MC001", "text": "Some text"},  # Missing label
        {"mcid": "MC002", "text": "More text"}
    ]
    missing_cols_path = temp_dir / "missing_columns.csv"
    pd.DataFrame(missing_cols_data).to_csv(missing_cols_path, index=False)
    datasets["missing_columns"] = missing_cols_path
    
    # Invalid labels
    invalid_labels_data = [
        {"mcid": "MC001", "text": "Some text", "label": 2},  # Invalid label
        {"mcid": "MC002", "text": "More text", "label": "invalid"}
    ]
    invalid_labels_path = temp_dir / "invalid_labels.csv"
    pd.DataFrame(invalid_labels_data).to_csv(invalid_labels_path, index=False)
    datasets["invalid_labels"] = invalid_labels_path
    
    # Empty dataset
    empty_path = temp_dir / "empty.csv"
    pd.DataFrame().to_csv(empty_path, index=False)
    datasets["empty"] = empty_path
    
    # Empty text fields
    empty_text_data = [
        {"mcid": "MC001", "text": "", "label": 1},  # Empty text
        {"mcid": "MC002", "text": "   ", "label": 0}  # Whitespace only
    ]
    empty_text_path = temp_dir / "empty_text.csv"
    pd.DataFrame(empty_text_data).to_csv(empty_text_path, index=False)
    datasets["empty_text"] = empty_text_path
    
    return datasets


@pytest.fixture
def sample_embeddings(temp_dir: Path) -> Dict[str, Path]:
    """Create sample embedding files for testing."""
    embeddings = {}
    
    # Training embeddings
    train_data = {
        "mcids": ["MC001", "MC002", "MC003", "MC004"],
        "labels": [1, 0, 1, 0],
        "embeddings": [
            [0.1] * 768,  # Positive embedding
            [-0.1] * 768, # Negative embedding  
            [0.2] * 768,  # Positive embedding
            [-0.2] * 768  # Negative embedding
        ],
        "metadata": {
            "n_samples": 4,
            "embedding_dim": 768,
            "model": "fake-mediclaim-gpt",
            "created_at": "2024-01-01T00:00:00"
        }
    }
    
    train_path = temp_dir / "train_embeddings.json"
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    embeddings["train"] = train_path
    
    # Test embeddings
    test_data = {
        "mcids": ["MC005", "MC006"],
        "labels": [1, 0],
        "embeddings": [
            [0.15] * 768,  # Positive embedding
            [-0.15] * 768  # Negative embedding
        ],
        "metadata": {
            "n_samples": 2,
            "embedding_dim": 768,
            "model": "fake-mediclaim-gpt",
            "created_at": "2024-01-01T00:00:00"
        }
    }
    
    test_path = temp_dir / "test_embeddings.json"
    with open(test_path, 'w') as f:
        json.dump(test_data, f)
    embeddings["test"] = test_path
    
    return embeddings


@pytest.fixture  
def target_words() -> List[str]:
    """Sample target words for testing."""
    return ["approved", "positive", "excellent", "good"]


@pytest.fixture
def negative_target_words() -> List[str]:
    """Sample negative target words for testing."""
    return ["denied", "negative", "poor", "failed"]


# Utility functions for tests

def wait_for_api(url: str, timeout: int = 30) -> bool:
    """Wait for API to be available."""
    for _ in range(timeout):
        try:
            response = requests.get(f"{url}/health", timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def create_test_config_file(temp_dir: Path, config: PipelineConfig) -> Path:
    """Save config to YAML file."""
    config_path = temp_dir / "test_config.yaml"
    config.to_yaml(str(config_path))
    return config_path


def assert_valid_embeddings(embeddings_data: Dict[str, Any]):
    """Assert that embeddings data has valid structure."""
    required_keys = {"mcids", "labels", "embeddings", "metadata"}
    assert set(embeddings_data.keys()) >= required_keys
    
    assert len(embeddings_data["mcids"]) == len(embeddings_data["labels"])
    assert len(embeddings_data["mcids"]) == len(embeddings_data["embeddings"])
    
    for embedding in embeddings_data["embeddings"]:
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)


def assert_valid_metrics(metrics: Dict[str, Any]):
    """Assert that metrics have valid structure and values."""
    required_metrics = {"accuracy", "precision", "recall", "f1_score"}
    
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))
        assert 0.0 <= metrics[metric] <= 1.0