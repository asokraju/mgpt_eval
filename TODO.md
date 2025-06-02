# TODO

## Completed Tasks
- [x] Analyze all occurrences of '_texts' in the codebase
- [x] Identify where '_texts' should be replaced with '_claims'
- [x] Create fix plan for API response handling
- [x] Create branch for MCID uniqueness validation (fix/mcid-uniqueness-validation)
- [x] Add MCID uniqueness validation to all data loading methods
- [x] Test validation logic with various scenarios

## MCID Uniqueness Validation

### Implementation
Added validation to ensure MCID values are unique when loading datasets. The validation was added to:

1. **models/data_models.py** - `Dataset.from_file()` method
2. **pipelines/embedding_pipeline.py** - `_load_dataset()` method
3. **evaluation/target_word_evaluator.py** - `_load_dataset()` method

### Validation Logic
- Checks for duplicate MCID values after loading the dataset
- If duplicates are found, raises a ValueError with:
  - Total count of rows with duplicate MCIDs
  - Sample of up to 5 duplicate MCID values
  - Indication if there are more duplicates beyond the sample

### Error Message Format
```
Found {duplicate_count} rows with duplicate MCID values. 
MCIDs must be unique. Duplicate MCIDs include: ['MC001', 'MC002', ...]
(and X more)
```

## Bug Fix: API Response Field Name

### Issue
The API is returning `generated_claims` but the code expects `generated_texts`.

### Location
File: `/home/kosaraju/mgpt_eval/evaluation/target_word_evaluator.py`
Method: `_generate_cross_mcid_batch` (lines 526-564)

### Required Changes
1. Line 526: Change `'generated_texts'` to `'generated_claims'`
2. Line 527: Update error message to reference `'generated_claims'`
3. Line 529: Change `result['generated_texts']` to `result['generated_claims']`
4. Line 531: Update error message to reference `'generated_claims'`

### Implementation
Replace the following lines in the `_generate_cross_mcid_batch` method:

```python
# Line 526
if 'generated_claims' not in result:
    raise ValueError(f"Missing 'generated_claims' field in API response. Available fields: {list(result.keys())}")

# Line 529
generated_texts_response = result['generated_claims']
if not isinstance(generated_texts_response, list):
    raise ValueError(f"Expected 'generated_claims' to be a list, got {type(generated_texts_response)}")
```

### Testing
After implementing the fix:
1. Test with actual API to ensure it returns `generated_claims`
2. Verify the target word evaluation pipeline works correctly
3. Check that batch processing handles the response properly

### Notes
- Internal variable names (`generated_texts_response`, `generated_texts`) don't need to change
- The fix is backward-compatible if we need to support both field names

## Token Decoding Fix

### Issue
The API returns token IDs that need to be decoded before searching for target codes.

### Implementation
Added token decoding functionality to `evaluation/target_word_evaluator.py`:

1. **Added tokenizer initialization** in `__init__` method
   - Imports `AutoTokenizer` from transformers
   - Calls `_initialize_tokenizer()` during initialization
   - Throws error if tokenizer path is not specified or loading fails

2. **Added `_initialize_tokenizer()` method**
   - Loads tokenizer from path specified in config
   - Raises `ValueError` if tokenizer path is missing or invalid
   - No fallback - tokenizer is required

3. **Updated `_generate_cross_mcid_batch()` method**
   - Handles both token IDs (list) and text (string) from API
   - Decodes token IDs using tokenizer before processing
   - Raises `RuntimeError` if API returns tokens but tokenizer unavailable
   
4. **Also fixed API response field**
   - Changed from `generated_texts` to `generated_claims`

### Configuration Required
The tokenizer path can be specified in either location:

```yaml
# Option 1: In target_word_evaluation section (takes precedence)
target_word_evaluation:
  tokenizer_path: "/path/to/tokenizer"

# Option 2: In embedding_generation section (shared)
embedding_generation:
  tokenizer_path: "/path/to/tokenizer"
```

The implementation checks `target_word_evaluation.tokenizer_path` first, then falls back to `embedding_generation.tokenizer_path`. This avoids duplication while allowing module-specific overrides if needed.