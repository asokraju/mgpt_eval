# How to Run Different Pipeline Configurations

## **✅ Simple Config-Driven Approach (Recommended)**

Your pipeline is designed for **config-only execution**. Just update the config file and run one simple command:

```bash
python main.py run-all --config configs/templates/YOUR_CONFIG.yaml
```

## **📁 Available Config Templates**

### **1. Full End-to-End Pipeline** (`01_full_end_to_end.yaml`)
**What it does**: Complete pipeline from raw data to final evaluation
- Generates embeddings from CSV data
- Trains multiple classifiers (logistic, SVM, random forest)
- Evaluates all models
- Runs target word evaluation
- Creates comparison report

**How to use**:
1. Edit the config file:
   ```yaml
   input:
     dataset_path: "YOUR_DATA.csv"  # Update this
   model_api:
     base_url: "YOUR_API_URL"       # Update this
   target_word_evaluation:
     target_codes: ["E119", "Z03818"]  # Your codes
   ```

2. Run:
   ```bash
   python main.py run-all --config configs/templates/01_full_end_to_end.yaml
   ```

### **2. Embeddings Only** (`02_embeddings_only.yaml`)
**What it does**: Generate embeddings and stop (useful for large datasets)

**How to use**:
1. Update config with your data path and API URL
2. Run:
   ```bash
   python main.py run-all --config configs/templates/02_embeddings_only.yaml
   ```

### **3. Train from Existing Embeddings** (`03_from_existing_embeddings.yaml`)
**What it does**: Skip embedding generation, use pre-computed embeddings

**How to use**:
1. Update config with paths to your existing embeddings:
   ```yaml
   input:
     train_embeddings_path: "outputs/job1/embeddings/train_embeddings.json"
     test_embeddings_path: "outputs/job1/embeddings/test_embeddings.json"
   ```
2. Run:
   ```bash
   python main.py run-all --config configs/templates/03_from_existing_embeddings.yaml
   ```

### **4. Target Word Evaluation Only** (`04_target_word_only.yaml`)
**What it does**: Skip embeddings/classification, only test target word method

**How to use**:
1. Update config with your data and target codes
2. Run:
   ```bash
   python main.py run-all --config configs/templates/04_target_word_only.yaml
   ```

### **5. Separate Train/Test Files** (`05_separate_train_test.yaml`)
**What it does**: Use when you have pre-split train/test datasets

**How to use**:
1. Update config:
   ```yaml
   input:
     train_dataset_path: "data/train.csv"
     test_dataset_path: "data/test.csv"
   ```
2. Run:
   ```bash
   python main.py run-all --config configs/templates/05_separate_train_test.yaml
   ```

### **6. Production Config** (`06_production_config.yaml`)
**What it does**: Optimized for large-scale production evaluation

**How to use**:
1. Update config with production settings
2. Run:
   ```bash
   python main.py run-all --config configs/templates/06_production_config.yaml
   ```

## **🔧 Quick Customization Guide**

### **Data Format Requirements**
Your CSV must have these columns:
```csv
mcid,claims,label
123456,"N6320 G0378 |eoc| Z91048 M1710",1
789012,"E119 76642 |eoc| K9289 O0903",0
```

### **Target Codes Setup**
Choose ONE method:

**Method 1: Direct in config**
```yaml
target_word_evaluation:
  target_codes:
    - "E119"
    - "Z03818"
    - "N6320"
```

**Method 2: External file**
```yaml
target_word_evaluation:
  target_codes_file: "data/target_codes.txt"
```

### **API Endpoint Setup**
Update in every config:
```yaml
model_api:
  base_url: "http://your-api-server:8000"
```

## **📊 Output Structure**

All configs create organized output structure:
```
outputs/
├── job_name/
│   ├── embeddings/
│   │   ├── train_embeddings.json
│   │   └── test_embeddings.json
│   ├── models/
│   │   ├── logistic_regression_model_20231201_120000.pkl
│   │   └── svm_model_20231201_120000.pkl
│   ├── metrics/
│   │   ├── evaluation_results.json
│   │   └── target_word_evaluation/
│   └── logs/
│       └── pipeline.log
```

## **🚀 Quick Start Example**

1. **Copy a template**:
   ```bash
   cp configs/templates/01_full_end_to_end.yaml configs/my_experiment.yaml
   ```

2. **Edit the config**:
   ```yaml
   # Only change these lines:
   input:
     dataset_path: "data/my_medical_claims.csv"
   model_api:
     base_url: "http://localhost:8000"
   target_word_evaluation:
     target_codes: ["E119", "Z03818", "N6320"]
   ```

3. **Run**:
   ```bash
   python main.py run-all --config configs/my_experiment.yaml
   ```

4. **Check results**:
   ```bash
   ls outputs/my_experiment/
   ```

## **🔍 Monitoring Progress**

- **Logs**: Check `outputs/logs/pipeline.log`
- **Verbose mode**: Add `--verbose` flag
- **Resume**: Pipeline auto-resumes from checkpoints if interrupted

## **⚡ Performance Tips**

1. **Large datasets**: Use `02_embeddings_only.yaml` first, then `03_from_existing_embeddings.yaml`
2. **Fast iteration**: Use `04_target_word_only.yaml` to test target codes quickly
3. **Production**: Use `06_production_config.yaml` with optimized settings

## **❌ Troubleshooting**

- **Target codes error**: Make sure to specify target codes in config
- **API connection**: Verify your model API is running and accessible
- **Memory issues**: Reduce batch sizes in config
- **File not found**: Check all paths in config are correct

---

**🎯 The key advantage**: You only need to maintain different config files for different scenarios, not different commands!