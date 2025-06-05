"""
Pydantic models for data validation and API requests/responses.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, root_validator
import pandas as pd
from pathlib import Path


class DatasetRecord(BaseModel):
    """Model for a single dataset record."""
    
    mcid: str = Field(..., description="Medical Claim ID")
    claims: str = Field(..., min_length=1, description="Medical claims codes")
    label: int = Field(..., ge=0, le=1, description="Binary label (0 or 1)")
    
    class Config:
        # Allow extra fields for additional metadata
        extra = "allow"


class Dataset(BaseModel):
    """Model for validating entire datasets."""
    
    records: List[DatasetRecord] = Field(..., description="List of dataset records")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Dataset metadata")
    
    @validator('records')
    def validate_non_empty(cls, v):
        if not v:
            raise ValueError("Dataset cannot be empty")
        return v
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'Dataset':
        """Create Dataset from pandas DataFrame."""
        records = []
        for _, row in df.iterrows():
            record_data = row.to_dict()
            records.append(DatasetRecord(**record_data))
        
        return cls(records=records)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert Dataset to pandas DataFrame."""
        data = [record.dict() for record in self.records]
        df = pd.DataFrame(data)
        # Ensure we're using 'claims' as the primary column name
        return df
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Dataset':
        """Load dataset from file (CSV, JSON, or Parquet)."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix == '.json':
            df = pd.read_json(path)
        elif path.suffix == '.parquet':
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Validate required columns (updated for medical claims data)
        required_cols = {'claims', 'mcid', 'label'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate MCID uniqueness
        if 'mcid' in df.columns:
            duplicate_mcids = df[df.duplicated(subset=['mcid'], keep=False)]
            if not duplicate_mcids.empty:
                duplicate_count = len(duplicate_mcids)
                unique_duplicates = df[df.duplicated(subset=['mcid'])]['mcid'].unique()
                sample_duplicates = list(unique_duplicates[:5])
                error_msg = (
                    f"Found {duplicate_count} rows with duplicate MCID values. "
                    f"MCIDs must be unique. Duplicate MCIDs include: {sample_duplicates}"
                )
                if len(unique_duplicates) > 5:
                    error_msg += f" (and {len(unique_duplicates) - 5} more)"
                raise ValueError(error_msg)
        
        # Ensure we're using 'claims' as the primary column name
        
        return cls.from_dataframe(df)


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    
    claims: List[str] = Field(..., min_items=1, description="List of medical claims to embed")
    padding_side: Optional[str] = Field(default="left", description="Padding side for tokenization")
    truncation_side: Optional[str] = Field(default="left", description="Truncation side for tokenization")
    max_length: Optional[int] = Field(default=1024, description="Maximum sequence length")
    
    @validator('claims')
    def validate_claims(cls, v):
        # Check for empty claims
        for i, claim in enumerate(v):
            if not claim.strip():
                raise ValueError(f"Claim at index {i} is empty or only whitespace")
        return v