# TODO

## Completed Tasks
- [x] Analyze all occurrences of '_texts' in the codebase
- [x] Identify where '_texts' should be replaced with '_claims'
- [x] Create fix plan for API response handling

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