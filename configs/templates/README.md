# Configuration Templates

This directory contains clean, ready-to-use configuration templates for different MGPT-Eval workflows. Each template is designed for a specific use case with clear documentation and cost considerations.

## Quick Start Guide

1. **Copy the template** that matches your use case
2. **Update the key fields** marked with 👈 UPDATE comments
3. **Run the pipeline** with your custom config

```bash
# Copy template
cp configs/templates/01_embeddings_only.yaml my_config.yaml

# Edit the file with your settings
# Update: dataset_path, api.base_url, target_codes (if applicable)

# Run pipeline
python main.py run-all --config my_config.yaml
```

## Available Templates

### 🔄 01_embeddings_only.yaml
**Purpose**: Generate embeddings from text data and save for later use  
**Cost**: 💰💰💰 HIGH (embedding generation API calls)  
**When to use**: First-time setup, one-time embedding generation for multiple experiments  
**Outputs**: Train/test embeddings (JSON files)  
**Next step**: Use `02_from_embeddings.yaml` to train classifiers  

### 🤖 02_from_embeddings.yaml
**Purpose**: Train classifiers using pre-computed embeddings  
**Cost**: 💰 LOW (no API calls, only compute time)  
**When to use**: Multiple experiments with different classifiers or hyperparameters  
**Prerequisites**: Run `01_embeddings_only.yaml` first  
**Outputs**: Trained models, evaluation metrics, performance plots  

### 🎯 03_target_words_only.yaml
**Purpose**: Evaluate model using target medical code generation  
**Cost**: 💰💰 MEDIUM (text generation API calls)  
**When to use**: Test alternative evaluation approach without training classifiers  
**Outputs**: Target word predictions, accuracy metrics  

### 🔗 04_full_pipeline.yaml
**Purpose**: Complete workflow comparing all methods  
**Cost**: 💰💰💰 HIGH (embedding + text generation API calls)  
**When to use**: Comprehensive evaluation, method comparison  
**Outputs**: Everything - embeddings, models, evaluations, comparisons  

### 📊 05_evaluation_from_models.yaml
**Purpose**: Evaluate existing trained models  
**Cost**: 💸 MINIMAL (no API calls)  
**When to use**: Test saved models, generate additional metrics/plots  
**Prerequisites**: Have trained models and test embeddings  
**Outputs**: Evaluation metrics, performance visualizations  

## Cost-Efficient Workflow Strategy

For maximum cost efficiency, follow this sequence:

1. **Generate embeddings once** (01_embeddings_only.yaml)
   - Most expensive step, but only done once
   - Save the embedding files carefully

2. **Experiment with classifiers** (02_from_embeddings.yaml)
   - Run multiple times with different settings
   - No additional API costs

3. **Test target word approach** (03_target_words_only.yaml)
   - Compare alternative evaluation method
   - Moderate API cost

4. **Evaluate saved models** (05_evaluation_from_models.yaml)
   - No API costs, fast execution
   - Perfect for analysis and visualization

## Key Configuration Fields

Every template requires these updates:

### 📁 Data Input
```yaml
data:
  dataset_path: "data/medical_claims.csv"  # 👈 UPDATE: Your CSV file
```

### 🌐 API Endpoint
```yaml
api:
  base_url: "http://localhost:8000"        # 👈 UPDATE: Your model server
```

### 🎯 Target Codes (for target word evaluation)
```yaml
target_evaluation:
  target_codes:                            # 👈 UPDATE: Your medical codes
    - "E119"    # Diabetes
    - "76642"   # Ultrasound
    - "N6320"   # Urological
```

## File Path Examples

The templates use realistic file paths that follow the output structure:

```
outputs/
├── embeddings_generation/          # From 01_embeddings_only.yaml
│   ├── train_embeddings.json      # Used by 02_from_embeddings.yaml
│   └── test_embeddings.json       # Used by 05_evaluation_from_models.yaml
├── classification_from_embeddings/ # From 02_from_embeddings.yaml
│   ├── models/
│   │   ├── logistic_regression_model.pkl  # Used by 05_evaluation_from_models.yaml
│   │   ├── svm_model.pkl
│   │   └── random_forest_model.pkl
│   └── metrics/
└── target_word_evaluation/         # From 03_target_words_only.yaml
    └── metrics/
```

## Advanced Usage

### Custom Hyperparameters
Modify the `classification.hyperparameters` section in any template:

```yaml
classification:
  hyperparameters:
    logistic_regression:
      C: [0.1, 1, 10]           # Fewer values = faster training
      penalty: ["l2"]           # Single penalty = faster training
```

### Memory Optimization
For large datasets, adjust batch sizes:

```yaml
embeddings:
  batch_size: 8               # Reduce if memory issues
api:
  batch_size: 16              # Reduce if API timeouts
```

### Debug Mode
Enable detailed logging:

```yaml
system:
  log_level: "DEBUG"          # More detailed logs
  console_log_level: "DEBUG"  # Verbose console output
```

## Troubleshooting

**Common issues and solutions:**

1. **File not found errors**
   - Check that paths in templates match your file structure
   - Use absolute paths if needed

2. **API connection errors**
   - Verify `api.base_url` is correct
   - Check that your model server is running

3. **Memory errors**
   - Reduce `embeddings.batch_size` and `api.batch_size`
   - Set `system.memory_limit_gb` to available RAM

4. **Target codes required error**
   - Update `target_evaluation.target_codes` with your medical codes
   - Or set `target_word_eval: false` to skip this stage

## Support

For detailed configuration options, see `master_template.yaml` which contains all available settings with comprehensive documentation.