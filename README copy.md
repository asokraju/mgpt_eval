# Binary Classifier Pipeline using MediClaimGPT Embeddings

This module implements a robust pipeline for training binary classifiers using embeddings from the MediClaimGPT model.

## Directory Structure

```
mgpt_eval/
├── data_processing/
│   ├── __init__.py
│   ├── dataset_loader.py      # Load and preprocess datasets
│   └── data_splitter.py       # Create train/test splits
├── embedding_generation/
│   ├── __init__.py
│   ├── embedding_generator.py  # Generate embeddings using model API
│   └── embedding_storage.py    # Save/load embeddings to JSON/CSV
├── finetuning/
│   ├── __init__.py
│   └── finetune_model.py      # Fine-tune MediClaimGPT (Phase 2)
├── classifiers/
│   ├── __init__.py
│   ├── logistic_classifier.py # Logistic regression implementation
│   ├── hyperparameter_tuner.py # Grid search and optimization
│   └── model_trainer.py        # Training pipeline
├── evaluation/
│   ├── __init__.py
│   ├── metrics_calculator.py   # Calculate standard metrics
│   ├── visualization.py        # ROC curves, confusion matrices
│   └── target_word_evaluator.py # Alternative evaluation framework
├── pipelines/
│   ├── __init__.py
│   ├── embedding_pipeline.py   # End-to-end embedding generation
│   ├── classification_pipeline.py # End-to-end classification
│   └── evaluation_pipeline.py   # Complete evaluation pipeline
├── configs/
│   ├── pipeline_config.yaml    # Pipeline configuration
│   └── model_config.yaml       # Model hyperparameters
├── outputs/
│   ├── embeddings/            # Generated embeddings
│   ├── models/                # Saved classifier models
│   └── metrics/               # Evaluation results
└── main.py                    # Main entry point
```

## Batch Size Configuration

The pipeline uses configurable batch sizes to optimize GPU memory usage:

1. **Configure in `configs/pipeline_config.yaml`**:
   ```yaml
   embedding_generation:
     batch_size: 16  # Adjust based on GPU memory
   ```

2. **GPU Memory Guidelines**:
   - **Small GPU (8GB)**: batch_size: 4-8
   - **Medium GPU (16GB)**: batch_size: 8-16  
   - **Large GPU (24GB+)**: batch_size: 16-32

3. **Factors affecting batch size**:
   - Model size (number of parameters)
   - Maximum sequence length in your dataset
   - Available GPU memory
   - Other processes using GPU

## Usage

### 1. Generate Embeddings

You have two options for providing input data:

#### Option A: Single Dataset (Automatic Split)
```bash
# Single dataset will be split based on config train_test_split ratio
python main.py generate-embeddings \
    --dataset-path /path/to/full_dataset.csv \
    --output-dir outputs/embeddings \
    --model-endpoint http://localhost:8000

# Override split ratio from command line
python main.py generate-embeddings \
    --dataset-path /path/to/full_dataset.csv \
    --split-ratio 0.8 \
    --output-dir outputs/embeddings
```

#### Option B: Separate Train/Test Files
```bash
# Use pre-split train and test datasets
python main.py generate-embeddings \
    --train-dataset /path/to/train.csv \
    --test-dataset /path/to/test.csv \
    --output-dir outputs/embeddings \
    --model-endpoint http://localhost:8000
```

**Output Structure:**
- Option A creates: `train_embeddings.json` and `test_embeddings.json`  
- Option B creates: `train_embeddings.json` and `test_embeddings.json`

### 2. Train Binary Classifier
```python
python main.py train-classifier \
    --embeddings-path outputs/embeddings \
    --classifier-type logistic \
    --output-dir outputs/models
```

### 3. Evaluate Model
```python
python main.py evaluate \
    --model-path outputs/models/best_model.pkl \
    --test-embeddings outputs/embeddings/test_embeddings.json \
    --output-dir outputs/metrics
```

### 4. Alternative Evaluation (Target Word)
```python
python main.py evaluate-target-word \
    --dataset-path /path/to/dataset \
    --target-word "claim" \
    --n-samples 10 \
    --max-tokens 200
```