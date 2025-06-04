"""
Integration tests for TargetWordEvaluator with actual API server.
Run this when the API server is available to test real integration.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import requests
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.target_word_evaluator import TargetWordEvaluator


def check_api_server(base_url="http://localhost:8000", timeout=5):
    """Check if API server is running."""
    try:
        response = requests.get(f"{base_url}/docs", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def test_api_response_format(base_url="http://localhost:8000"):
    """Test actual API response format to understand the structure."""
    print("Testing API response format...")
    
    # Test single generation
    try:
        response = requests.post(f"{base_url}/generate", json={
            "claim": "Patient has diabetes",
            "max_new_tokens": 20,
            "temperature": 0.8
        }, timeout=10)
        
        print(f"Single generate response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Single generate response fields: {list(result.keys())}")
            print(f"Sample response: {result}")
        
    except Exception as e:
        print(f"Single generate failed: {e}")
    
    # Test batch generation
    try:
        response = requests.post(f"{base_url}/generate_batch", json={
            "claims": ["Patient has diabetes", "Heart condition present"],
            "max_new_tokens": 20,
            "temperature": 0.8
        }, timeout=10)
        
        print(f"Batch generate response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Batch generate response fields: {list(result.keys())}")
            print(f"Sample batch response: {result}")
            
            # Check if it's a list and what type of data
            for field_name, field_value in result.items():
                print(f"  {field_name}: {type(field_value)} - {len(field_value) if isinstance(field_value, list) else 'not a list'}")
        
    except Exception as e:
        print(f"Batch generate failed: {e}")


def test_target_evaluator_integration():
    """Test TargetWordEvaluator with real API server."""
    base_url = "http://localhost:8000"
    
    if not check_api_server(base_url):
        print("❌ API server not available - skipping integration tests")
        print("Start the API server with: python -m mgpt_serve.main")
        return False
    
    print("✅ API server is available - running integration tests")
    
    # Test API format first
    test_api_response_format(base_url)
    
    # Create temporary test data
    temp_dir = tempfile.mkdtemp()
    try:
        # Small test dataset
        test_data = pd.DataFrame({
            'mcid': ['TEST001', 'TEST002', 'TEST003'],
            'claims': [
                'Patient presents with diabetes mellitus type 2',
                'Routine checkup shows normal vitals',
                'Patient has hypertension and requires medication'
            ],
            'label': [1, 0, 1]
        })
        
        csv_file = Path(temp_dir) / 'test_data.csv'
        test_data.to_csv(csv_file, index=False)
        
        # Configure evaluator
        config = {
            'model_api': {
                'base_url': base_url,
                'timeout': 30,
                'max_retries': 2,
                'batch_size': 2,
                'endpoints': {
                    'generate_batch': '/generate_batch'
                }
            },
            'target_word_evaluation': {
                'temperature': 0.8,
                'top_k': 50,
                'search_method': 'exact',
                'checkpoint_every': 1,
                'max_batch_retries': 2,
                'global_timeout_minutes': 5
            },
            'output': {
                'metrics_dir': f'{temp_dir}/metrics',
                'checkpoint_dir': f'{temp_dir}/checkpoints'
            }
        }
        
        evaluator = TargetWordEvaluator(config)
        
        # Test with medical codes that might appear in generated text
        target_words = ['E11', 'diabetes', 'I10', 'hypertension']
        
        print(f"\nRunning evaluation with target words: {target_words}")
        print(f"Test data shape: {test_data.shape}")
        
        start_time = time.time()
        
        try:
            results = evaluator.evaluate(
                str(csv_file),
                target_words=target_words,
                n_samples=2,  # Small number for testing
                max_tokens=30
            )
            
            end_time = time.time()
            
            print(f"\n✅ Evaluation completed successfully in {end_time - start_time:.2f} seconds")
            print(f"Results: {results}")
            
            # Verify results structure
            expected_fields = ['accuracy', 'precision', 'recall', 'f1_score', 'target_words', 'results_path']
            missing_fields = [field for field in expected_fields if field not in results]
            
            if missing_fields:
                print(f"❌ Missing result fields: {missing_fields}")
                return False
            else:
                print("✅ All expected result fields present")
            
            # Check output files
            metrics_dir = Path(temp_dir) / 'metrics' / 'target_word_evaluation'
            if metrics_dir.exists():
                output_files = list(metrics_dir.glob('*'))
                print(f"✅ Generated {len(output_files)} output files: {[f.name for f in output_files]}")
            else:
                print("❌ No output files generated")
            
            return True
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_error_resilience():
    """Test error handling with bad requests."""
    base_url = "http://localhost:8000"
    
    if not check_api_server(base_url):
        print("❌ API server not available - skipping error resilience tests")
        return
    
    print("\n🧪 Testing error resilience...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        test_data = pd.DataFrame({
            'mcid': ['ERR001'],
            'claims': ['Test claim for error testing'],
            'label': [1]
        })
        
        csv_file = Path(temp_dir) / 'error_test.csv'
        test_data.to_csv(csv_file, index=False)
        
        # Configure with aggressive timeouts to trigger errors
        config = {
            'model_api': {
                'base_url': base_url,
                'timeout': 1,  # Very short timeout
                'max_retries': 1,
                'batch_size': 1,
                'endpoints': {
                    'generate_batch': '/generate_batch'
                }
            },
            'target_word_evaluation': {
                'temperature': 0.8,
                'top_k': 50,
                'search_method': 'exact',
                'checkpoint_every': 1,
                'max_batch_retries': 1,
                'global_timeout_minutes': 1
            },
            'output': {
                'metrics_dir': f'{temp_dir}/metrics',
                'checkpoint_dir': f'{temp_dir}/checkpoints'
            }
        }
        
        evaluator = TargetWordEvaluator(config)
        
        try:
            results = evaluator.evaluate(
                str(csv_file),
                target_words=['test'],
                n_samples=1,
                max_tokens=10
            )
            
            print("✅ Error resilience test completed - errors were handled gracefully")
            
        except Exception as e:
            print(f"⚠️  Error resilience test triggered exception (this may be expected): {e}")
    
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("=== Target Word Evaluator Integration Tests ===")
    
    success = test_target_evaluator_integration()
    
    if success:
        test_error_resilience()
        print("\n🎉 All integration tests completed!")
    else:
        print("\n❌ Integration tests failed")
        exit(1)