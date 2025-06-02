"""
Pipeline models for job configuration and end-to-end workflow management.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from pathlib import Path


class EndToEndConfig(BaseModel):
    """Configuration for end-to-end pipeline execution."""
    
    generate_embeddings: bool = Field(default=True, description="Generate embeddings from text")
    train_classifiers: bool = Field(default=True, description="Train binary classifiers")
    evaluate_models: bool = Field(default=True, description="Evaluate trained models")
    target_word_evaluation: bool = Field(default=True, description="Run target word evaluation")
    create_summary_report: bool = Field(default=True, description="Create summary report")
    compare_methods: bool = Field(default=True, description="Compare different methods")
    
    # Classifier configuration
    classifier_types: List[str] = Field(
        default=["logistic_regression", "svm", "random_forest"],
        description="List of classifier types to train"
    )
    
    # Target word evaluation configuration
    target_words: List[str] = Field(
        default_factory=list,
        description="Target medical codes for evaluation"
    )
    target_word_samples: int = Field(default=10, ge=1, le=100, description="Number of generations per prompt")
    target_word_max_tokens: int = Field(default=200, ge=10, le=2048, description="Maximum tokens to generate")


class PipelineJob(BaseModel):
    """Configuration for a complete pipeline job."""
    
    # Input data configuration
    dataset_path: Optional[str] = Field(default=None, description="Single dataset path")
    train_dataset_path: Optional[str] = Field(default=None, description="Training dataset path")
    test_dataset_path: Optional[str] = Field(default=None, description="Test dataset path")
    train_embeddings_path: Optional[str] = Field(default=None, description="Pre-computed training embeddings")
    test_embeddings_path: Optional[str] = Field(default=None, description="Pre-computed test embeddings")
    
    # Pre-trained model paths (for starting from evaluation stage)
    model_paths: Optional[Dict[str, str]] = Field(default=None, description="Pre-trained model paths by classifier type")
    models_directory: Optional[str] = Field(default=None, description="Directory containing pre-trained models")
    
    # Job configuration
    job_name: str = Field(default="pipeline_job", description="Job name for organizing outputs")
    output_base_dir: str = Field(default="outputs", description="Base output directory")
    model_endpoint: str = Field(default="http://localhost:8000", description="Model API endpoint")
    split_ratio: Optional[float] = Field(default=0.8, ge=0.1, le=0.9, description="Train/test split ratio")
    
    # End-to-end configuration
    end_to_end: EndToEndConfig = Field(default_factory=EndToEndConfig)
    
    @validator('split_ratio')
    def validate_split_ratio(cls, v):
        if v is not None and (v <= 0 or v >= 1):
            raise ValueError('split_ratio must be between 0 and 1')
        return v


class PipelineResult(BaseModel):
    """Result model for complete pipeline execution."""
    
    job_name: str = Field(..., description="Job name")
    success: bool = Field(..., description="Whether pipeline completed successfully")
    total_duration: float = Field(..., description="Total execution time in seconds")
    
    # Best model information
    best_classifier: Optional[str] = Field(default=None, description="Best performing classifier type")
    best_embedding_score: Optional[float] = Field(default=None, description="Best embedding-based classifier score")
    target_word_score: Optional[float] = Field(default=None, description="Target word evaluation score")
    
    # Output paths
    output_directory: str = Field(..., description="Main output directory")
    
    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if pipeline failed")
    
    # Timestamps
    start_time: str = Field(..., description="Pipeline start time")
    end_time: str = Field(..., description="Pipeline end time")