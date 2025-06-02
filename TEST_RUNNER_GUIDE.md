# Complete Testing Guide for MGPT Eval Pipeline

## 🚀 Quick Start Testing

### **1. Start the Fake API Server**
```bash
# Terminal 1: Start fake API server
cd tests/
python fake_api_server.py --port 8001

# Should show:
# 🚀 Starting Fake MediClaimGPT API Server
#    URL: http://127.0.0.1:8001
```

### **2. Run All Tests**
```bash
# Terminal 2: Run tests
cd tests/
python run_tests.py --coverage --verbose

# Or run specific test categories
python run_tests.py --markers "not slow" --verbose
```

## 📁 Test Categories

### **Unit Tests** (Fast - Run First)
```bash
# Test data models and validation
python run_tests.py --test "test_data_models" -v

# Test configuration validation  
python run_tests.py --test "test_config_models" -v

# Test pipeline models
python run_tests.py --test "test_pipeline_models" -v
```

### **Integration Tests** (Medium - Require Fake API)
```bash
# Start fake API first!
python fake_api_server.py --port 8001 &

# Test API connectivity
python run_tests.py --test "test_api_connectivity" -v

# Test full pipeline scenarios
python run_tests.py --test "test_full_pipeline" -v

# Test target word evaluation
python run_tests.py --test "test_target_word_evaluator" -v
```

### **End-to-End Tests** (Slow - Full Pipeline)
```bash
# Test complete workflows
python run_tests.py --markers "slow" -v --timeout 600
```

## 🔧 Fake API Server Options

### **Basic Usage**
```bash
# Standard server for testing
python fake_api_server.py --port 8001

# Check if running
curl http://localhost:8001/health
```

### **Advanced Testing Scenarios**
```bash
# Add artificial delay (test timeout handling)
python fake_api_server.py --port 8001 --delay 0.5

# Inject random errors (test retry logic)  
python fake_api_server.py --port 8001 --error-rate 0.1

# Different embedding dimensions
python fake_api_server.py --port 8001 --embedding-dim 256

# Quiet mode for CI/CD
python fake_api_server.py --port 8001 --quiet
```

### **Server Endpoints Available**
- `GET /health` - Health check
- `POST /embeddings` - Single embedding generation
- `POST /embeddings_batch` - Batch embedding generation  
- `POST /generate` - Text generation
- `GET /stats` - Server statistics
- `GET/POST /config` - Runtime configuration

## 📊 Test Execution Options

### **Coverage Reports**
```bash
# Generate HTML coverage report
python run_tests.py --coverage
# Open htmlcov/index.html in browser
```

### **Parallel Testing**
```bash
# Run tests in parallel (faster)
pip install pytest-xdist
python run_tests.py --parallel 4
```

### **Specific Test Selection**
```bash
# Run tests matching pattern
python run_tests.py --test "embedding" -v

# Run tests with specific markers
python run_tests.py --markers "integration" -v
python run_tests.py --markers "not slow" -v

# Exit on first failure (for debugging)
python run_tests.py --exitfirst -v
```

### **Debug Mode**
```bash
# Show print statements and detailed output
python run_tests.py --capture no -v

# Run single test with maximum verbosity
python -m pytest tests/test_full_pipeline.py::TestFullPipeline::test_embeddings_only_pipeline -v -s
```

## 🧪 Creating Sample Test Data

### **Minimal Test Dataset**
```python
# Create test_data.csv
import pandas as pd

data = [
    {"mcid": "123456", "claims": "N6320 G0378 |eoc| Z91048 M1710", "label": 1},
    {"mcid": "123457", "claims": "E119 76642 |eoc| K9289 O0903", "label": 0},
    {"mcid": "123458", "claims": "Z03818 U0003 |eoc| N6322 76642", "label": 1},
]

df = pd.DataFrame(data)
df.to_csv("test_data.csv", index=False)
```

### **Target Codes File**
```bash
# Create target_codes.txt
cat > target_codes.txt << EOF
E119
Z03818
N6320
K9289
76642
EOF
```

## 🔍 Testing Different Scenarios

### **1. Test Config-Driven Pipeline**
```bash
# Test each config template
for config in configs/templates/*.yaml; do
    echo "Testing $config"
    python main.py run-all --config "$config" --dry-run || echo "Config $config has issues"
done
```

### **2. Test API Connectivity**
```bash
# Check API health before tests
python run_tests.py --check-api --api-port 8001

# Test with API failures
python fake_api_server.py --port 8001 --error-rate 0.2 &
python run_tests.py --test "api_failure" -v
```

### **3. Test Memory and Performance**
```bash
# Run memory usage tests
python run_tests.py --test "memory" -v

# Test with large datasets
python run_tests.py --markers "slow" --timeout 1200
```

## 📈 Continuous Integration Setup

### **CI Test Script**
```bash
#!/bin/bash
# ci_test.sh

set -e  # Exit on any error

echo "Starting CI tests for MGPT Eval Pipeline"

# Start fake API server in background
python tests/fake_api_server.py --port 8001 --quiet &
API_PID=$!

# Wait for API to start
sleep 2

# Check API health
curl -f http://localhost:8001/health || {
    echo "Fake API failed to start"
    exit 1
}

# Run tests
python tests/run_tests.py \
    --coverage \
    --markers "not slow" \
    --parallel 2 \
    --timeout 300 \
    --exitfirst

# Cleanup
kill $API_PID 2>/dev/null || true

echo "All tests passed!"
```

### **Docker Test Environment**
```dockerfile
# tests/Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_test.txt .
RUN pip install -r requirements_test.txt

COPY . .

# Run tests in container
CMD ["python", "tests/run_tests.py", "--coverage", "--quiet"]
```

## 🚨 Troubleshooting

### **Common Issues**

1. **API Connection Failed**
   ```bash
   # Check if fake API is running
   python run_tests.py --check-api
   
   # Start API if not running
   python fake_api_server.py --port 8001
   ```

2. **Tests Timeout**
   ```bash
   # Increase timeout
   python run_tests.py --timeout 600
   
   # Or run without slow tests
   python run_tests.py --markers "not slow"
   ```

3. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd /path/to/mgpt_eval
   
   # Install test dependencies
   pip install -r tests/requirements_test.txt
   ```

4. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x tests/run_tests.py
   chmod +x tests/fake_api_server.py
   ```

### **Debug Individual Components**

```bash
# Test specific pipeline component
python -c "
from models.config_models import PipelineConfig
config = PipelineConfig.from_yaml('configs/templates/01_full_end_to_end.yaml')
print('Config loaded successfully')
"

# Test data loading
python -c "
from models.data_models import Dataset
dataset = Dataset.from_file('test_data.csv')
print(f'Dataset loaded: {len(dataset.records)} records')
"

# Test API manually
curl -X POST http://localhost:8001/embeddings \
  -H "Content-Type: application/json" \
  -d '{"claims": ["N6320 G0378 |eoc| Z91048 M1710"]}'
```

## 📝 Test Results Interpretation

### **Coverage Report**
- **Target**: >80% coverage for core modules
- **Focus Areas**: models/, pipelines/, evaluation/
- **View**: Open `htmlcov/index.html` after running with `--coverage`

### **Performance Benchmarks**
- **Unit Tests**: <5 seconds total
- **Integration Tests**: <30 seconds total
- **Full Pipeline**: <5 minutes (with fake API)

### **Success Criteria**
- ✅ All unit tests pass
- ✅ Integration tests pass with fake API
- ✅ Config templates validate correctly
- ✅ No memory leaks in long-running tests
- ✅ Error handling works correctly

---

**🎯 Key Takeaway**: Your pipeline is designed to be config-driven, so most testing focuses on validating that different configurations work correctly and that the API interactions are robust.