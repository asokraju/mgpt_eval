"""
Pydantic models for configuration validation and management.
"""

from typing import Dict, List, Optional, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field, validator, model_validator, root_validator
import yaml


class ModelAPIConfig(BaseModel):
    """Configuration for model API settings."""
    
    base_url: str = Field(default="http://localhost:8000", description="Base URL for model API")
    endpoints: Dict[str, str] = Field(
        default={
            "embeddings_batch": "/embeddings_batch"
        },
        description="API endpoint mappings"
    )
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    timeout: int = Field(default=300, ge=30, le=3600, description="Request timeout in seconds")
    
    @validator('base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return v.rstrip('/')


    


class EmbeddingGenerationConfig(BaseModel):
    """Configuration for embedding generation pipeline."""
    
    batch_size: int = Field(default=8, ge=1, le=128, description="Processing batch size")
    max_sequence_length: int = Field(default=1024, ge=32, le=8192, description="Maximum token length")
    padding_side: str = Field(default="left", description="Padding side for tokenization")
    truncation_side: str = Field(default="left", description="Truncation side for tokenization")
    tokenizer_path: str = Field(default="/app/tokenizer", description="Path to tokenizer")
    output_filename: str = Field(default="${job.name}_embeddings.csv", description="Output filename pattern")
    














class OutputConfig(BaseModel):
    """Configuration for output directories."""
    
    embeddings_dir: str = Field(default="${job.output_dir}/embeddings", description="Embeddings output directory")
    logs_dir: str = Field(default="${job.output_dir}/logs", description="Logs output directory")
    
    def create_directories(self, base_dir: str):
        """Create all output directories."""
        embeddings_path = self.embeddings_dir.replace("${job.output_dir}", base_dir)
        logs_path = self.logs_dir.replace("${job.output_dir}", base_dir)
        
        for dir_path in [embeddings_path, logs_path]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file: str = Field(default="${output.logs_dir}/embedding_pipeline.log", description="Log file path")
    max_file_size: str = Field(default="10MB", description="Maximum log file size")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup log files")
    


class HyperparameterConfig(BaseModel):
    """Configuration for hyperparameter search."""
    
    logistic_regression: Dict[str, List] = Field(
        default={
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"]
        }
    )
    svm: Dict[str, List] = Field(
        default={
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
    )
    random_forest: Dict[str, List] = Field(
        default={
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    )
    
    def model_post_init(self, __context):
        """Post-process to convert 'null' strings to None."""
        for classifier_name in ['logistic_regression', 'svm', 'random_forest']:
            if hasattr(self, classifier_name):
                classifier_params = getattr(self, classifier_name)
                for param_name, param_values in classifier_params.items():
                    # Convert 'null' strings to None
                    classifier_params[param_name] = [
                        None if v == 'null' else v for v in param_values
                    ]


class CrossValidationConfig(BaseModel):
    """Configuration for cross-validation."""
    
    n_folds: int = Field(default=5, ge=2, le=10, description="Number of CV folds")
    scoring: str = Field(default="roc_auc", description="Scoring metric")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs")


class ClassificationConfig(BaseModel):
    """Configuration for classification pipeline."""
    
    models: List[str] = Field(
        default=["logistic_regression", "svm", "random_forest"],
        description="Available classifier types"
    )
    hyperparameter_search: HyperparameterConfig = Field(default_factory=HyperparameterConfig)
    cross_validation: CrossValidationConfig = Field(default_factory=CrossValidationConfig)
    
    @validator('models')
    def validate_models(cls, v):
        valid_types = {"logistic_regression", "svm", "random_forest"}
        for classifier in v:
            if classifier not in valid_types:
                raise ValueError(f"Invalid classifier type: {classifier}. Must be one of {valid_types}")
        return v


class JobConfig(BaseModel):
    """Configuration for job-specific settings."""
    
    name: str = Field(default="pipeline_job", description="Job name for organizing outputs")
    output_dir: str = Field(default="outputs", description="Base output directory")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")


class ClassificationInputConfig(BaseModel):
    """Configuration for classification input data."""
    
    train_embeddings_path: str = Field(..., description="Path to training embeddings CSV")
    test_embeddings_path: str = Field(..., description="Path to test embeddings CSV")


class InputConfig(BaseModel):
    """Configuration for input data sources."""
    
    # For embedding pipeline
    dataset_path: Optional[str] = Field(default=None, description="Path to input CSV dataset")
    
    # For classification pipeline  
    train_embeddings_path: Optional[str] = Field(default=None, description="Path to training embeddings CSV")
    test_embeddings_path: Optional[str] = Field(default=None, description="Path to test embeddings CSV")





class PipelineConfig(BaseModel):
    """Main pipeline configuration combining all sub-configurations."""
    
    # Input and job configuration
    input: InputConfig = Field(default_factory=InputConfig)
    job: JobConfig = Field(default_factory=JobConfig)
    
    # Component configurations
    model_api: ModelAPIConfig = Field(default_factory=ModelAPIConfig)
    embedding_generation: EmbeddingGenerationConfig = Field(default_factory=EmbeddingGenerationConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    
    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"
        # Validate assignment to catch errors early
        validate_assignment = True
        # Use enum values for literal types
        use_enum_values = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = self.dict()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def setup_directories(self):
        """Create all required output directories."""
        self.output.create_directories(self.job.output_dir)
        
    def resolve_template_string(self, template: str) -> str:
        """Resolve template strings like ${job.name} to actual values."""
        resolved = template
        resolved = resolved.replace("${job.name}", self.job.name)
        resolved = resolved.replace("${job.output_dir}", self.job.output_dir)
        resolved = resolved.replace("${output.embeddings_dir}", self.output.embeddings_dir.replace("${job.output_dir}", self.job.output_dir))
        resolved = resolved.replace("${output.logs_dir}", self.output.logs_dir.replace("${job.output_dir}", self.job.output_dir))
        return resolved