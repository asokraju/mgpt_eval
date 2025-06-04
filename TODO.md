# TODO

## Current Tasks - Population Health Analytics with VAE (June 4, 2025)

### Critical - Memory Issues
- [x] Investigate JavaScript heap out of memory error 
- [x] Identify Node.js default heap limit (4GB on 64-bit systems)
- [ ] Configure increased memory limit for Claude Code CLI
- [ ] Add memory optimization guidelines to CLAUDE.md

### High Priority
- [x] Review existing VAE notebook and Pythae integration
- [x] Define population health analytics requirements and metrics  
- [x] Design member embedding analysis framework
- [x] Create new notebook for population health analytics with Pythae VAE
- [ ] Implement Pythae-based VAE architectures (β-VAE, conditional VAE, hierarchical VAE)

### Medium Priority  
- [ ] Implement VAE-based population health insights
  - [ ] Member risk stratification using latent space positioning
  - [ ] Health phenotype identification through latent clustering
  - [ ] Care pathway analysis via latent interpolation
- [ ] Add member clustering and cohort analysis
  - [ ] Dynamic cohort segmentation based on embedding evolution
  - [ ] Chronic condition group identification
  - [ ] Utilization pattern clustering
- [ ] Implement risk stratification using embeddings
  - [ ] Reconstruction error-based outlier detection for high-risk members
  - [ ] Risk scoring from latent representations
  - [ ] Early warning systems using latent trajectories
- [ ] Build outcome prediction models from member embeddings
  - [ ] Healthcare utilization forecasting
  - [ ] Cost prediction models
  - [ ] Health deterioration prediction

### Low Priority
- [ ] Create interactive visualizations for population insights
  - [ ] Real-time population health monitoring dashboard
  - [ ] Member journey visualization in latent space
  - [ ] Risk stratification dashboards
- [ ] Update documentation with population health analytics
  - [ ] Population health methodology documentation
  - [ ] Clinical interpretation guidelines
  - [ ] Model validation and performance metrics

## Previous Tasks (Completed)

## Completed Tasks
- [x] Analyze all occurrences of '_texts' in the codebase
- [x] Identify where '_texts' should be replaced with '_claims'
- [x] Create fix plan for API response handling
- [x] Create branch for MCID uniqueness validation (fix/mcid-uniqueness-validation)
- [x] Add MCID uniqueness validation to all data loading methods
- [x] Test validation logic with various scenarios
- [x] Create comprehensive debug script for embedding pipeline

## Debug Script Implementation
- [x] Created comprehensive debug script `/home/kosaraju/mgpt_eval/debug_embedding_pipeline.py`
- [x] Created test configuration file `/home/kosaraju/mgpt_eval/test_config.yaml` 
- [x] Implemented 13 test cases covering various edge cases:
  - Normal operation with small batch size
  - Single item batching
  - Large batch testing (50 items)
  - Long claims requiring truncation
  - Missing columns (should fail)
  - Duplicate MCIDs (should fail)
  - Null values (should fail)
  - API timeout scenarios (should fail)
  - Invalid API responses (should fail)
  - Empty embeddings (should fail)
  - Zero vector embeddings (should fail)
  - Dimension mismatch (should fail)
  - Tokenizer failure (should fail)

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

## Target Word Evaluator Testing and Debugging ✅ COMPLETED

### Issues Identified and Fixed
1. **API Payload Structure Mismatch** ✅ FIXED
   - Changed from `'prompts'` to `'claims'` field to match API spec
   - Added flexible response field detection for multiple API formats

2. **Infinite Retry Loop Risk** ✅ FIXED  
   - Added global timeout and batch count safety limits
   - Implemented circuit breaker to prevent infinite retries

3. **Memory Leaks** ✅ FIXED
   - Reduced batch failure tracking cache from 100 to 50 entries
   - Enhanced memory cleanup on completion
   - Added proper exception handling for cleanup operations

4. **Error Handling Issues** ✅ FIXED
   - Simplified error handling to fail-fast on persistent API issues
   - Removed complex error state tracking that could mask real issues
   - Added global timeout to prevent stuck processes

5. **API Integration Problems** ✅ FIXED
   - Made API response parsing flexible to handle multiple field names
   - Added comprehensive response validation
   - Fixed missing time import

### Testing Completed
- ✅ Created comprehensive unit tests (30+ test cases)
- ✅ Created integration tests with real API server
- ✅ Verified all fixes work with actual API responses
- ✅ Tested error resilience and timeout handling
- ✅ Confirmed memory efficiency improvements

### Test Results
- All unit tests pass
- Integration tests pass with real API server
- Error resilience tests complete successfully
- API format correctly detected: `'generated_claims'` field
- Evaluation completes in <1 second for small datasets
- Proper cleanup and checkpoint handling verified

## Additional Bug Fixes Completed (June 3, 2025) ✅

### Critical Bugs Fixed 🔴
1. **Timeout Not Initialized on Checkpoint Resume** ✅
   - Fixed: `start_time` now initialized before any usage
   - Prevents NameError when resuming from checkpoint

2. **Attribute _last_batch_key Not Initialized** ✅
   - Fixed: Added proper initialization in `__init__` section
   - Prevents AttributeError on first batch comparison

3. **Hard-coded Batch Limit Too Restrictive** ✅
   - Fixed: Made batch limit configurable via `max_batches` parameter
   - Default increased from 1000 to 10000 batches

### Medium Severity Bugs Fixed 🟡
4. **Inefficient Batch Key Comparison** ✅
   - Fixed: Use hash instead of string comparison
   - Improves performance for large batch comparisons

5. **Config Validation Missing** ✅
   - Fixed: Added validation for required config sections
   - Better error messages for missing configuration

6. **Type Consistency in Results** ✅
   - Fixed: Ensure predictions are always int type
   - Prevents type errors in downstream processing

### Test Infrastructure Created ✅
- Created comprehensive test runner script: `/tests/scripts/run_target_evaluator_tests.py`
- 30+ unit tests covering all edge cases
- Integration tests with real API server
- All tests pass successfully

### Results Summary
- ✅ All 6 priority bugs fixed and tested
- ✅ Code is more robust with better error handling
- ✅ Performance improved with hash-based comparisons
- ✅ Memory usage optimized with bounded caches
- ✅ Configuration is now validated on initialization
- ✅ Integration tests confirm fixes work with real API

## Additional Bug Fixes Completed (Round 2 - June 3, 2025) ✅

### Fixed Bugs Summary

1. **Mixed Types for generation_details** ✅
   - Fixed: `_generate_predictions` now returns dict instead of list
   - Impact: Enables MCID-based lookups in saved JSON files
   - Test: Verified saved details are dict keyed by MCID

2. **Missing Defaults for max_retries and timeout** ✅
   - Fixed: Added defaults (max_retries=3, timeout=30)
   - Impact: No more KeyError when config missing these values
   - Test: Verified evaluator works with incomplete config

3. **Enhanced Checkpoint Compatibility** ✅
   - Fixed: Now checks max_tokens in addition to target_words and n_samples
   - Impact: Prevents loading incompatible checkpoints
   - Test: Verified checkpoint rejection on parameter mismatch

4. **Output Config Defaults** ✅
   - Fixed: Added default metrics_dir='outputs/metrics'
   - Impact: No KeyError when metrics_dir missing from config
   - Test: Verified default directory creation

5. **Empty Target Words Validation** ✅
   - Already handled correctly with proper error messages
   - Test: Verified error message clarity

6. **Progress Bar Hang Prevention** ✅
   - Already fixed with max_batches and timeout limits
   - Test: Verified no infinite loops on API failures

7. **Word Boundary Regex** ✅
   - Working correctly for medical codes
   - Test: Verified exact vs fuzzy matching behavior

### Test Results
- Created comprehensive test suite: `tests/unit/test_target_evaluator_bugs.py`
- All 10 tests pass successfully
- Covers all identified edge cases and error conditions

## Final Code Review Fixes (Round 3 - June 4, 2025) ✅

### Additional Issues Found and Fixed

1. **Incorrect Return Type Annotation** ✅
   - Fixed: `_generate_predictions` type hint now matches actual return type
   - Changed from `Tuple[np.ndarray, List[Dict]]` to `Tuple[np.ndarray, Dict[str, Dict]]`

2. **Import Organization** ✅
   - Fixed: Moved `hashlib` import to top of file
   - Improves code organization and follows Python conventions

3. **Enhanced Input Validation** ✅
   - Added validation for negative/zero n_samples and max_tokens
   - Added warnings for excessively large parameter values
   - Better error messages for invalid input types

4. **Robust Endpoint Handling** ✅
   - Fixed potential KeyError when `endpoints` config missing
   - Added safe defaults for endpoint construction
   - More resilient URL building logic

5. **Comprehensive Test Coverage** ✅
   - Created additional test suite: `test_additional_fixes.py`
   - Tests all new validation and error conditions
   - All 48 tests pass successfully

### Final Test Results
- **48 total tests** across 3 test suites
- **29 tests**: Original comprehensive test suite
- **10 tests**: Bug-specific fixes test suite  
- **9 tests**: Additional fixes test suite
- **Integration tests**: Real API verification
- **Zero failures**: All tests pass