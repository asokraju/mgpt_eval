# Testing Directory

This directory contains all testing components for the mgpt_eval pipeline.

## Directory Structure

```
tests/
├── README.md              # This file
├── configs/               # Test configuration files
├── data/                  # Test datasets
├── outputs/               # Test outputs (gitignored)
├── scripts/               # Test scripts
│   ├── test_embedding_pipeline.py    # Main embedding pipeline tests
│   ├── test_edge_cases.py            # Edge case testing
│   └── run_all_tests.py              # Test runner
├── unit/                  # Unit tests (future)
└── integration/           # Integration tests (future)
```

## Usage

### Run all embedding pipeline tests:
```bash
cd tests
python scripts/run_all_tests.py
```

### Run specific test categories:
```bash
cd tests
python scripts/test_embedding_pipeline.py    # Basic functionality
python scripts/test_edge_cases.py            # Edge cases
```

### Configuration

Test configurations are stored in `configs/` directory:
- `test_config.yaml` - Main test configuration
- `edge_case_config.yaml` - Configuration for edge case testing

### Test Data

Test datasets are generated automatically and stored in `data/` directory.
All test data files follow the required CSV format with columns: `mcid`, `claims`, `label`.

### Outputs

All test outputs (embeddings, logs, results) are stored in `outputs/` directory.
This directory is excluded from git to keep the repository clean.