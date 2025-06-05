"""
Pipeline models for job configuration and end-to-end workflow management.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from pathlib import Path






class EmbeddingResult(BaseModel):
    """Result model for embedding pipeline execution."""
    
    output_path: str = Field(..., description="Path to generated embeddings file")
    n_samples: int = Field(..., description="Number of samples processed")
    embedding_dim: int = Field(..., description="Embedding dimension")
    embedding_stats: Optional[Dict[str, Any]] = Field(default=None, description="Embedding statistics")
    timestamp: str = Field(..., description="Completion timestamp")