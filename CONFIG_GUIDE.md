# Configuration Guide for Binary Classifier Pipeline

This guide explains all configuration options in `configs/pipeline_config.yaml`.

## Table of Contents
1. [Data Processing Configuration](#data-processing-configuration)
2. [Model API Configuration](#model-api-configuration)
3. [Embedding Generation Configuration](#embedding-generation-configuration)
4. [Classification Configuration](#classification-configuration)
5. [Evaluation Configuration](#evaluation-configuration)
6. [Target Word Evaluation Configuration](#target-word-evaluation-configuration)
7. [Output Configuration](#output-configuration)
8. [Context Window Management](#context-window-management)

## Data Processing Configuration

These settings control how your input data is processed before embedding generation.

### `train_test_split`
- **Type**: float (0.0 - 1.0)
- **Default**: 0.8
- **Purpose**: Defines the ratio for splitting data into training and test sets
- **Example**: 0.8 means 80% training, 20% test
- **When used**: Only when you provide a single dataset that needs splitting
- **Note**: If you already have separate train/test files, this is ignored

### `random_seed`
- **Type**: integer or null
- **Default**: 42
- **Purpose**: Ensures reproducible data splits and shuffling
- **Use cases**:
  - Set to specific number for reproducible experiments
  - Set to `null` for random splits each run
- **Affects**: Train/test splitting, data shuffling, cross-validation folds

### `max_sequence_length`
- **Type**: integer
- **Default**: 512
- **Purpose**: Maximum number of tokens allowed per text
- **Behavior**: Texts longer than this are truncated
- **Impact on memory**: 
  ```
  GPU Memory ∝ batch_size × max_sequence_length × model_size
  ```
- **Common values**:
  - 128: Short texts (tweets, titles)
  - 256: Medium texts (paragraphs)
  - 512: Standard (most common)
  - 1024: Long documents
  - 2048: Very long documents (requires more GPU memory)

### `include_mcid`
- **Type**: boolean
- **Default**: true
- **Purpose**: Whether to include Medical Claim IDs in outputs
- **When true**: 
  - MCIDs are saved with embeddings
  - Allows tracking predictions back to original claims
  - Required for audit trails
- **When false**: 
  - Only embeddings and labels saved
  - Smaller file sizes
  - Use when MCIDs aren't needed

### `output_format`
- **Type**: string ("json" or "csv")
- **Default**: "json"
- **Purpose**: Format for saving embeddings and results

#### JSON Format
```json
{
  "mcids": ["MC001", "MC002"],
  "labels": [0, 1],
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "metadata": {...}
}
```
**Advantages**:
- Human-readable
- Includes metadata in same file
- Easy to load in Python
- Preserves data types

**Disadvantages**:
- Larger file size
- Slower for very large datasets

#### CSV Format
```csv
mcid,label,embedding
MC001,0,"[0.1, 0.2, ...]"
MC002,1,"[0.3, 0.4, ...]"
```
**Advantages**:
- Compact file size
- Works with data analysis tools
- Streaming-friendly for large datasets

**Disadvantages**:
- Embeddings stored as JSON strings
- Metadata in separate file
- Requires parsing embedding column

## Usage Examples

### Example 1: Processing Medical Claims (Long Texts)
```yaml
data_processing:
  train_test_split: 0.8
  random_seed: 42
  max_sequence_length: 1024  # Longer for detailed claims
  include_mcid: true         # Track claim IDs
  output_format: "json"      # Easy to work with
```

### Example 2: Large-Scale Processing (Millions of Records)
```yaml
data_processing:
  train_test_split: 0.9      # More training data
  random_seed: 42
  max_sequence_length: 256   # Shorter to save memory
  include_mcid: true
  output_format: "csv"       # Efficient for large data
```

### Example 3: Quick Prototyping
```yaml
data_processing:
  train_test_split: 0.7      # Smaller training set
  random_seed: null          # Random splits OK
  max_sequence_length: 128   # Fast processing
  include_mcid: false        # Not needed for prototype
  output_format: "json"      # Easy debugging
```

## How Settings Affect Pipeline Performance

### Memory Usage
```
Memory Required ≈ batch_size × max_sequence_length × model_parameters × 4 bytes
```

For a model with 1B parameters:
- max_sequence_length=128: ~0.5GB per batch item
- max_sequence_length=512: ~2GB per batch item
- max_sequence_length=2048: ~8GB per batch item

### Processing Speed
- Shorter sequences = Faster processing
- JSON format = Slower I/O for large datasets
- CSV format = Faster I/O for large datasets

### Accuracy Considerations
- Longer sequences = More context = Potentially better embeddings
- But truncation might lose important information
- Consider your domain: Are key features at start or end of text?

## Validation Rules

The pipeline validates these settings:
1. `train_test_split` must be between 0.0 and 1.0
2. `max_sequence_length` must be positive integer
3. `output_format` must be "json" or "csv"
4. Dataset must have required columns: 'text', 'mcid', 'label'

## Best Practices

1. **Always set `random_seed`** for reproducible experiments
2. **Start with `max_sequence_length=512`** and adjust based on:
   - Your GPU memory
   - Your text lengths (check distribution)
   - Model performance
3. **Use JSON format** for datasets < 100k samples
4. **Use CSV format** for datasets > 100k samples
5. **Keep `include_mcid=true`** for production systems
6. **Monitor truncation**: Log how many texts are truncated

## Troubleshooting

### "CUDA out of memory" errors
- Reduce `max_sequence_length`
- Reduce batch size in `embedding_generation`
- Use CSV format to reduce memory footprint

### Slow processing
- Reduce `max_sequence_length` if texts are short
- Switch to CSV format for large datasets
- Increase batch size if GPU memory allows

### Poor model performance
- Increase `max_sequence_length` if truncating important info
- Check if key information is being truncated
- Consider preprocessing to put important info first