# MGPT-Eval Configuration Files

This directory contains configuration files for the MGPT-Eval pipeline.

## Quick Start

1. **For immediate testing:**
   ```bash
   python main.py run-all --config configs/default_config.yaml
   ```

2. **For production workflows:**
   Copy and customize a template from `templates/`

## Available Files

### 📋 Default Configuration
- **`default_config.yaml`** - Minimal working configuration for quick testing

### 📁 Templates Directory
- **`templates/`** - Production-ready workflow templates
  - See `templates/README.md` for detailed usage guide

## Configuration Structure

All configurations use the new standardized format:

```yaml
pipeline:
  stages:
    embeddings: true/false
    classification: true/false
    evaluation: true/false
    target_word_eval: true/false
    summary_report: true/false
    method_comparison: true/false

data:
  dataset_path: "path/to/data.csv"
  # OR train_embeddings_path + test_embeddings_path
  # OR model_paths for evaluation-only

job:
  name: "job_name"
  output_dir: "outputs"

api:
  base_url: "http://localhost:8000"
```

## Key Features

- **✅ Config-driven workflows** - Single YAML controls entire pipeline
- **✅ Path resolution priority** - Smart handling of file paths
- **✅ Cost-efficient patterns** - Reuse expensive embeddings
- **✅ Flexible stage control** - Run any combination of stages
- **✅ Field name compatibility** - Supports both old and new formats

## Usage Examples

```bash
# Full pipeline
python main.py run-all --config templates/04_full_pipeline.yaml

# Embeddings only (cost-efficient first step)
python main.py run-all --config templates/01_embeddings_only.yaml

# Train from saved embeddings (no API cost)
python main.py run-all --config templates/02_from_embeddings.yaml

# Target word evaluation only
python main.py run-all --config templates/03_target_words_only.yaml
```

## Migration from Old Configs

Old configuration files using different field names will still work due to backward compatibility, but it's recommended to migrate to the new template format for better maintainability.

## Support

For detailed configuration options and workflow guides, see:
- `templates/README.md` - Template usage guide
- `templates/master_template.yaml` - Complete configuration reference