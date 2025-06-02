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
            "embeddings": "/embeddings",
            "embeddings_batch": "/embeddings_batch", 
            "generate": "/generate",
            "generate_batch": "/generate_batch"
        },
        description="API endpoint mappings"
    )
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size for API requests")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    timeout: int = Field(default=300, ge=30, le=3600, description="Request timeout in seconds")
    
    @validator('base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return v.rstrip('/')


class DataProcessingConfig(BaseModel):
    """Configuration for data processing settings."""
    
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    max_sequence_length: int = Field(default=512, ge=32, le=8192, description="Maximum token length")
    include_mcid: bool = Field(default=True, description="Include Medical Claim IDs in output")
    output_format: Literal["json", "csv"] = Field(default="json", description="Output file format")
    train_test_split: float = Field(default=0.8, ge=0.1, le=0.9, description="Train/test split ratio")
    


class EmbeddingGenerationConfig(BaseModel):
    """Configuration for embedding generation pipeline."""
    
    batch_size: int = Field(default=16, ge=1, le=128, description="Processing batch size")
    save_interval: int = Field(default=100, ge=10, le=1000, description="Checkpoint save interval")
    checkpoint_dir: str = Field(default="outputs/checkpoints", description="Checkpoint directory")
    resume_from_checkpoint: bool = Field(default=True, description="Resume from checkpoints")
    tokenizer_path: str = Field(default="/app/tokenizer", description="Path to tokenizer")
    


class HyperparameterConfig(BaseModel):
    """Configuration for hyperparameter search."""
    
    logistic_regression: Dict[str, List] = Field(
        default={
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"]
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
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10]
        }
    )


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


class VisualizationConfig(BaseModel):
    """Configuration for evaluation visualizations."""
    
    generate_plots: bool = Field(default=True, description="Generate visualization plots")
    plot_formats: List[str] = Field(default=["png", "pdf"], description="Plot output formats")
    dpi: int = Field(default=300, ge=72, le=600, description="Plot resolution")
    
    @validator('plot_formats')
    def validate_plot_formats(cls, v):
        valid_formats = {"png", "pdf", "svg", "jpg", "eps"}
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid plot format: {fmt}. Must be one of {valid_formats}")
        return v


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    
    metrics: List[str] = Field(
        default=["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"],
        description="Metrics to calculate"
    )
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid_metrics = {
            "accuracy", "precision", "recall", "f1_score", "roc_auc", 
            "confusion_matrix", "classification_report"
        }
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")
        return v


class TargetWordEvaluationConfig(BaseModel):
    """Configuration for target word evaluation."""
    
    enable: bool = Field(default=True, description="Whether to run target code evaluation")
    target_codes: Optional[List[str]] = Field(default=None, description="List of target medical codes")
    target_codes_file: Optional[str] = Field(default=None, description="Path to file containing target codes (one per line)")
    generations_per_prompt: int = Field(default=10, ge=1, le=100, description="Number of generations per prompt")
    max_new_tokens: int = Field(default=200, ge=10, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=1, le=200, description="Top-k sampling parameter")
    search_method: Literal["exact", "fuzzy"] = Field(default="exact", description="Word search method")
    
    @model_validator(mode='before')
    @classmethod
    def validate_target_codes_configuration(cls, values):
        enable = values.get('enable', True)
        target_codes = values.get('target_codes')
        target_codes_file = values.get('target_codes_file')
        
        if not enable:
            return values
            
        # Check that at least one method is provided
        has_codes_list = target_codes and len(target_codes) > 0
        has_codes_file = target_codes_file and len(target_codes_file.strip()) > 0
        
        # Allow empty config - will be validated at runtime when actually used
        if not has_codes_list and not has_codes_file:
            # Don't raise error here - will be checked at runtime in main.py
            pass
        
        if has_codes_list and has_codes_file:
            raise ValueError(
                "Please specify target codes using only ONE method: either 'target_codes' list OR 'target_codes_file', not both."
            )
        
        # Validate file exists if specified
        if has_codes_file:
            from pathlib import Path
            if not Path(target_codes_file).exists():
                # Only warn, don't fail - file might be created later
                import warnings
                warnings.warn(f"Target codes file not found: {target_codes_file}")
        
        return values
    
    def get_target_codes(self) -> List[str]:
        """
        Get target codes from either direct list or file.
        
        Returns:
            List of target medical codes
        """
        if self.target_codes:
            return self.target_codes
        elif self.target_codes_file:
            from pathlib import Path
            codes = []
            with open(self.target_codes_file, 'r') as f:
                for line in f:
                    code = line.strip()
                    if code and not code.startswith('#'):  # Skip empty lines and comments
                        codes.append(code)
            return codes
        else:
            return []


class OutputConfig(BaseModel):
    """Configuration for output directories and formats."""
    
    embeddings_dir: str = Field(default="outputs/embeddings", description="Embeddings output directory")
    models_dir: str = Field(default="outputs/models", description="Models output directory")
    metrics_dir: str = Field(default="outputs/metrics", description="Metrics output directory")
    logs_dir: str = Field(default="outputs/logs", description="Logs output directory")
    save_best_model_only: bool = Field(default=False, description="Save only best model")
    model_format: Literal["pickle", "joblib"] = Field(default="pickle", description="Model serialization format")
    
    def create_directories(self):
        """Create all output directories."""
        for dir_path in [self.embeddings_dir, self.models_dir, self.metrics_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file: str = Field(default="outputs/logs/pipeline.log", description="Log file path")
    max_file_size: str = Field(default="10MB", description="Maximum log file size")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup log files")
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Console logging level"
    )
    


class JobConfig(BaseModel):
    """Configuration for job-specific settings."""
    
    name: str = Field(default="pipeline_job", description="Job name for organizing outputs")
    output_dir: str = Field(default="outputs", description="Base output directory")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")


class InputConfig(BaseModel):
    """Configuration for input data sources."""
    
    dataset_path: Optional[str] = Field(default=None, description="Single dataset path")
    split_ratio: Optional[float] = Field(default=0.8, ge=0.1, le=0.9, description="Train/test split ratio (only used with dataset_path)")
    train_dataset_path: Optional[str] = Field(default=None, description="Training dataset path")
    test_dataset_path: Optional[str] = Field(default=None, description="Test dataset path")
    train_embeddings_path: Optional[str] = Field(default=None, description="Pre-computed training embeddings")
    test_embeddings_path: Optional[str] = Field(default=None, description="Pre-computed test embeddings")
    
    # Pre-trained model paths (for starting from evaluation stage)
    model_paths: Optional[Dict[str, str]] = Field(default=None, description="Pre-trained model paths by classifier type")
    models_directory: Optional[str] = Field(default=None, description="Directory containing pre-trained models")
    
    @model_validator(mode='before')
    @classmethod
    def validate_input_configuration(cls, values):
        """Validate that only one input method is specified."""
        dataset_path = values.get('dataset_path')
        train_path = values.get('train_dataset_path')
        test_path = values.get('test_dataset_path')
        train_emb = values.get('train_embeddings_path')
        test_emb = values.get('test_embeddings_path')
        split_ratio = values.get('split_ratio')
        
        # Count which input methods are specified
        methods = 0
        if dataset_path:
            methods += 1
        if train_path and test_path:
            methods += 1  
        if train_emb and test_emb:
            methods += 1
        # Allow test_embeddings_path alone (for evaluation-only workflows)
        if test_emb and not train_emb and not dataset_path and not (train_path or test_path):
            methods += 1
            
        if methods == 0:
            raise ValueError("Must specify one input method: dataset_path OR (train_dataset_path + test_dataset_path) OR (train_embeddings_path + test_embeddings_path) OR test_embeddings_path (for evaluation-only)")
        if methods > 1:
            raise ValueError("Please specify only ONE input method")
            
        # split_ratio only valid with dataset_path
        if split_ratio is not None and not dataset_path:
            raise ValueError("split_ratio can only be used with dataset_path (single dataset file)")
            
        # Ensure both train and test paths are provided together (except for evaluation-only workflows)
        if (train_path and not test_path) or (test_path and not train_path):
            raise ValueError("Both train_dataset_path and test_dataset_path must be provided together")
        
        # For embeddings, allow test_emb alone for evaluation-only workflows
        if train_emb and not test_emb:
            raise ValueError("If train_embeddings_path is provided, test_embeddings_path is also required")
        # Note: test_emb without train_emb is allowed for evaluation-only workflows
            
        return values


class PipelineStagesConfig(BaseModel):
    """Configuration for pipeline workflow control."""
    
    embeddings: bool = Field(default=True, description="Generate embeddings from text")
    classification: bool = Field(default=True, description="Train binary classifiers")
    evaluation: bool = Field(default=True, description="Evaluate trained models")
    target_word_eval: bool = Field(default=True, description="Run target word evaluation")
    summary_report: bool = Field(default=True, description="Create summary report")
    method_comparison: bool = Field(default=True, description="Compare different methods")
    
    @root_validator(pre=True)
    def sync_stage_aliases(cls, values):
        """Sync old and new field names."""
        # Embeddings
        if 'embeddings' in values and 'generate_embeddings' not in values:
            values['generate_embeddings'] = values['embeddings']
        elif 'generate_embeddings' in values and 'embeddings' not in values:
            values['embeddings'] = values['generate_embeddings']
            
        # Classification
        if 'classification' in values and 'train_classifiers' not in values:
            values['train_classifiers'] = values['classification']
        elif 'train_classifiers' in values and 'classification' not in values:
            values['classification'] = values['train_classifiers']
            
        # Evaluation
        if 'evaluation' in values and 'evaluate_models' not in values:
            values['evaluate_models'] = values['evaluation']
        elif 'evaluate_models' in values and 'evaluation' not in values:
            values['evaluation'] = values['evaluate_models']
            
        # Target word evaluation
        if 'target_word_eval' in values and 'target_word_evaluation' not in values:
            values['target_word_evaluation'] = values['target_word_eval']
        elif 'target_word_evaluation' in values and 'target_word_eval' not in values:
            values['target_word_eval'] = values['target_word_evaluation']
            
        # Summary report
        if 'summary_report' in values and 'create_summary_report' not in values:
            values['create_summary_report'] = values['summary_report']
        elif 'create_summary_report' in values and 'summary_report' not in values:
            values['summary_report'] = values['create_summary_report']
            
        # Method comparison
        if 'method_comparison' in values and 'compare_methods' not in values:
            values['compare_methods'] = values['method_comparison']
        elif 'compare_methods' in values and 'method_comparison' not in values:
            values['method_comparison'] = values['compare_methods']
            
        return values


class PipelineStagesWrapper(BaseModel):
    """Wrapper for pipeline stages to support nested structure."""
    stages: PipelineStagesConfig = Field(default_factory=PipelineStagesConfig)

class PipelineConfig(BaseModel):
    """Main pipeline configuration combining all sub-configurations."""
    
    # Input and job configuration - support both old and new field names
    data: InputConfig = Field(default_factory=InputConfig)
    input: Optional[InputConfig] = Field(default=None, description="Alternative field name for data")
    job: JobConfig = Field(default_factory=JobConfig)
    pipeline: PipelineStagesWrapper = Field(default_factory=PipelineStagesWrapper)
    pipeline_stages: Optional[PipelineStagesConfig] = Field(default=None, description="Alternative field name for pipeline.stages")
    
    # Component configurations - support both old and new field names
    api: Optional[ModelAPIConfig] = Field(default=None, description="API configuration")
    model_api: ModelAPIConfig = Field(default_factory=ModelAPIConfig)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    embeddings: Optional[EmbeddingGenerationConfig] = Field(default=None, description="Embeddings configuration")
    embedding_generation: EmbeddingGenerationConfig = Field(default_factory=EmbeddingGenerationConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    target_evaluation: Optional[TargetWordEvaluationConfig] = Field(default=None, description="Target evaluation configuration")
    target_word_evaluation: TargetWordEvaluationConfig = Field(default_factory=TargetWordEvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    @root_validator(pre=True)
    def handle_field_aliases(cls, values):
        """Handle both old and new field name formats."""
        # Handle input vs data field names
        if 'input' in values and 'data' not in values:
            values['data'] = values['input']
        elif 'data' in values and 'input' not in values:
            values['input'] = values['data']
            
        # Handle pipeline_stages vs pipeline.stages 
        if 'pipeline_stages' in values and 'pipeline' not in values:
            values['pipeline'] = {'stages': values['pipeline_stages']}
        elif 'pipeline' in values and 'pipeline_stages' not in values:
            if isinstance(values['pipeline'], dict) and 'stages' in values['pipeline']:
                values['pipeline_stages'] = values['pipeline']['stages']
        
        # Handle API field aliases
        if 'api' in values and 'model_api' not in values:
            values['model_api'] = values['api']
        elif 'model_api' in values and 'api' not in values:
            values['api'] = values['model_api']
            
        # Handle embeddings field aliases  
        if 'embeddings' in values and 'embedding_generation' not in values:
            values['embedding_generation'] = values['embeddings']
        elif 'embedding_generation' in values and 'embeddings' not in values:
            values['embeddings'] = values['embedding_generation']
            
        # Handle target evaluation field aliases
        if 'target_evaluation' in values and 'target_word_evaluation' not in values:
            values['target_word_evaluation'] = values['target_evaluation']
        elif 'target_word_evaluation' in values and 'target_evaluation' not in values:
            values['target_evaluation'] = values['target_word_evaluation']
                
        return values
    
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
        self.output.create_directories()
        # Also create checkpoint directory
        Path(self.embedding_generation.checkpoint_dir).mkdir(parents=True, exist_ok=True)