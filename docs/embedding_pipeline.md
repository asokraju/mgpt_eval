can you create a markdownfile with mermaid chart... Give higher level over view and for lowerlevel use subgraphs


# Embedding Pipeline Architecture
## Overview

The MediClaimGPT Embedding Pipeline is a fail-fast system for generating embeddings from CSV data using a tokenizer and API endpoint. It follows strict validation principles with no compromise on data quality.

## Architecture Diagram

```mermaid
flowchart TB
    Start([Start]) --> Init[Initialize Pipeline]
    Init --> LoadData[Load CSV Data]
    LoadData --> GenerateEmb[Generate Embeddings]
    GenerateEmb --> SaveResults[Save Results]
    SaveResults --> End([End])

    %% Initialization Subgraph
    subgraph InitSub[" "]
        direction TB
        I1[Load Config]
        I2[Setup Logger]
        I3[Initialize Tokenizer]
        I1 --> I2 --> I3
        I3 -.->|Fail if not found| E1[RuntimeError]
    end

    %% Data Loading Subgraph
    subgraph LoadSub[" "]
        direction TB
        L1[Check File Exists]
        L2[Read CSV]
        L3[Validate Columns]
        L4[Check MCIDs Unique]
        L5[Validate Data Types]
        L6[Check for Nulls]
        L1 --> L2 --> L3 --> L4 --> L5 --> L6
        L1 -.->|File not found| E2[FileNotFoundError]
        L3 -.->|Missing columns| E3[ValueError]
        L4 -.->|Duplicates found| E3
        L5 -.->|Wrong types| E3
        L6 -.->|Nulls found| E3
    end

    %% Embedding Generation Subgraph
    subgraph GenSub[" "]
        direction TB
        G1[Calculate Batches]
        G2[Process Batch]
        G3{More Batches?}
        G1 --> G2 --> G3
        G3 -->|Yes| G2
        G3 -->|No| G4[Validate Count]
        
        %% Batch Processing Sub-subgraph
        subgraph BatchSub[" "]
            direction LR
            B1[Truncate Claims]
            B2[Call API]
            B3[Validate Response]
            B4[Check Dimensions]
            B1 --> B2 --> B3 --> B4
        end
        
        G2 --> BatchSub
        BatchSub -.->|Any failure| E4[RuntimeError]
    end

    %% API Call Subgraph
    subgraph APISub[" "]
        direction TB
        A1[Prepare Payload]
        A2[Send Request]
        A3{Success?}
        A4[Exponential Backoff]
        A5{Max Retries?}
        A1 --> A2 --> A3
        A3 -->|No| A4 --> A5
        A5 -->|No| A2
        A5 -->|Yes| E5[RuntimeError]
        A3 -->|Yes| A6[Validate Response]
    end

    %% Save Results Subgraph
    subgraph SaveSub[" "]
        direction TB
        S1[Create Output DF]
        S2[Calculate Stats]
        S3[Write to Temp File]
        S4[Atomic Rename]
        S1 --> S2 --> S3 --> S4
        S3 -.->|Write fails| E6[RuntimeError]
    end

    %% Connect main flow to subgraphs
    Init -.-> InitSub
    LoadData -.-> LoadSub
    GenerateEmb -.-> GenSub
    BatchSub -.-> APISub
    SaveResults -.-> SaveSub

    %% Style error nodes
    classDef errorClass fill:#ff6b6b,stroke:#c92a2a,color:#fff
    class E1,E2,E3,E4,E5,E6 errorClass

    %% Style main flow
    classDef mainClass fill:#4c6ef5,stroke:#364fc7,color:#fff
    class Init,LoadData,GenerateEmb,SaveResults mainClass

    %% Style subgraphs
    classDef subgraphClass fill:#f8f9fa,stroke:#868e96
```

## Component Details

### 1. Pipeline Initialization
- **Config Loading**: Loads PipelineConfig with all settings
- **Logger Setup**: Initializes structured logging
- **Tokenizer**: REQUIRED - fails fast if tokenizer cannot be loaded

### 2. Data Loading & Validation
**Required Columns**:
- `claims`: Text data (must be strings)
- `mcid`: Unique identifiers (integers or strings)
- `label`: Labels for the data

**Validation Steps**:
1. File existence and CSV format check
2. Column presence validation
3. MCID uniqueness enforcement
4. Data type validation
5. Null value rejection

### 3. Embedding Generation
**Process Flow**:
1. Batch calculation based on config batch size
2. Optional text truncation using tokenizer
3. API calls with exponential backoff retry
4. Strict validation of each embedding:
   - Non-empty
   - Correct data type (list of floats)
   - No NaN or infinity values
   - No zero vectors
   - Consistent dimensions

**API Integration**:
- Configurable endpoint
- Timeout increases with retries
- Jittered exponential backoff
- Comprehensive error handling

### 4. Results Storage
**Output Format**: CSV with columns:
- `mcid`: Original identifiers
- `label`: Original labels
- `embedding`: JSON-serialized embedding vectors

**Safety Features**:
- Atomic writes using temp files
- Parent directory creation
- Final validation before save

## Key Design Principles

1. **Fail-Fast Philosophy**: Any error stops the entire pipeline
2. **No Placeholders**: No zero embeddings or missing data
3. **Strict Validation**: Every step validates data integrity
4. **Required Tokenizer**: No character-based fallbacks
5. **All-or-Nothing**: Complete success or complete failure

## Error Handling

| Error Type | Cause | Impact |
|------------|-------|---------|
| FileNotFoundError | Missing input CSV | Pipeline stops |
| ValueError | Data validation failure | Pipeline stops |
| RuntimeError | Tokenizer/API/Save failure | Pipeline stops |

## Configuration Requirements

The pipeline requires a `PipelineConfig` object with:
- `embedding_generation.tokenizer_path`: Path to tokenizer (REQUIRED)
- `embedding_generation.batch_size`: Batch size for processing
- `model_api.base_url`: API endpoint base URL
- `model_api.endpoints['embeddings_batch']`: Batch endpoint path
- `model_api.timeout`: Request timeout
- `model_api.max_retries`: Retry attempts
- `data_processing.max_sequence_length`: Optional truncation length
- `logging`: Logging configuration

## Usage

```python
from models.config_models import PipelineConfig
from embedding_pipeline import EmbeddingPipeline

# Load configuration
config = PipelineConfig(...)

# Initialize pipeline
pipeline = EmbeddingPipeline(config)

# Run pipeline
results = pipeline.run(
    dataset_path="input.csv",
    output_path="output.csv",
    model_endpoint="https://api.example.com"  # Optional override
)
```

## Output Statistics

The pipeline calculates and returns:
- Mean, standard deviation, min, and max embedding norms
- Embedding dimensionality
- Processing timestamp
- Number of samples processed