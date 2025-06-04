"""
Test runner script for TargetWordEvaluator
Run this to execute all tests and generate a coverage report
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run the comprehensive test suite with coverage reporting."""
    
    print("=" * 60)
    print("Running TargetWordEvaluator Test Suite")
    print("=" * 60)
    
    # Ensure we have the required packages
    required_packages = ['pytest', 'pytest-cov', 'pytest-mock']
    
    print("\n1. Checking required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✓ {package} installed")
        except ImportError:
            print(f"   ✗ {package} not installed. Installing...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Set up test environment
    print("\n2. Setting up test environment...")
    
    # Create necessary directories
    test_dirs = ['test_outputs', 'test_checkpoints', 'test_data']
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✓ Created {dir_name}/")
    
    # Run tests with different configurations
    print("\n3. Running tests...")
    
    test_commands = [
        # Basic unit tests
        {
            'name': 'Unit Tests',
            'cmd': [
                sys.executable, '-m', 'pytest', 
                'tests/unit/test_target_word_evaluator.py',
                '-v',
                '--tb=short'
            ]
        },
        
        # Tests with coverage
        {
            'name': 'Tests with Coverage',
            'cmd': [
                sys.executable, '-m', 'pytest',
                'tests/unit/test_target_word_evaluator.py',
                '--cov=evaluation.target_word_evaluator',
                '--cov-report=html',
                '--cov-report=term-missing',
                '-v'
            ]
        },
        
        # Performance tests (subset)
        {
            'name': 'Performance Tests',
            'cmd': [
                sys.executable, '-m', 'pytest',
                'tests/unit/test_target_word_evaluator.py',
                '-k', 'test_end_to_end_evaluation_success',
                '--durations=10',
                '-v'
            ]
        },
        
        # Memory tests
        {
            'name': 'Memory Tests',
            'cmd': [
                sys.executable, '-m', 'pytest',
                'tests/unit/test_target_word_evaluator.py',
                '-k', 'memory',
                '-v'
            ]
        }
    ]
    
    results = []
    
    for test_config in test_commands:
        print(f"\n   Running {test_config['name']}...")
        print(f"   Command: {' '.join(test_config['cmd'])}")
        print("   " + "-" * 50)
        
        try:
            result = subprocess.run(
                test_config['cmd'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"   ✓ {test_config['name']} passed")
                results.append((test_config['name'], 'PASSED'))
            else:
                print(f"   ✗ {test_config['name']} failed")
                print(f"   Error output:\n{result.stderr}")
                results.append((test_config['name'], 'FAILED'))
                
        except Exception as e:
            print(f"   ✗ {test_config['name']} errored: {e}")
            results.append((test_config['name'], 'ERROR'))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, status in results:
        status_emoji = "✓" if status == "PASSED" else "✗"
        print(f"{status_emoji} {test_name}: {status}")
    
    # Coverage report location
    if os.path.exists('htmlcov/index.html'):
        print("\n📊 Coverage report generated at: htmlcov/index.html")
    
    # Clean up
    print("\n4. Cleaning up test artifacts...")
    for dir_name in test_dirs:
        try:
            import shutil
            shutil.rmtree(dir_name)
            print(f"   ✓ Removed {dir_name}/")
        except Exception as e:
            print(f"   ⚠ Could not remove {dir_name}/: {e}")
    
    print("\n✨ Test run complete!")
    
    # Return exit code based on results
    if all(status == 'PASSED' for _, status in results):
        return 0
    else:
        return 1


def run_specific_test(test_name: str):
    """Run a specific test by name."""
    print(f"Running specific test: {test_name}")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/unit/test_target_word_evaluator.py',
        '-k', test_name,
        '-vv',
        '--tb=long'
    ]
    
    subprocess.run(cmd)


def run_integration_test():
    """Run a quick integration test with sample data."""
    print("Running integration test with sample data...")
    
    # Create sample data
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'mcid': ['M001', 'M002', 'M003'],
        'claims': [
            'Patient diagnosed with diabetes E119',
            'Routine checkup performed',
            'X-ray 76642 completed'
        ],
        'label': [1, 0, 1]
    })
    
    sample_data.to_csv('test_integration_data.csv', index=False)
    
    # Create test config
    test_config = {
        'model_api': {
            'base_url': 'http://localhost:8000',
            'endpoints': {'generate_batch': '/generate_batch'},
            'timeout': 30,
            'max_retries': 3,
            'batch_size': 2
        },
        'target_word_evaluation': {
            'temperature': 0.8,
            'top_k': 50,
            'search_method': 'exact',
            'checkpoint_every': 5,
            'max_batch_retries': 2,
            'global_timeout_minutes': 5,
            'max_batches': 10000
        },
        'output': {
            'metrics_dir': 'test_outputs',
            'checkpoint_dir': 'test_checkpoints'
        }
    }
    
    print("Sample data created: test_integration_data.csv")
    print("To run the evaluator with this data:")
    print("  evaluator.evaluate('test_integration_data.csv', ['E119', '76642'], n_samples=3, max_tokens=50)")
    
    # Run the integration test if API is available
    import requests
    try:
        response = requests.get('http://localhost:8000/docs', timeout=2)
        if response.status_code == 200:
            print("\n✅ API server is available - running integration test...")
            subprocess.run([sys.executable, 'tests/integration/test_target_evaluator_api.py'])
    except:
        print("\n⚠️  API server not available - skipping integration test")
    
    # Clean up
    if os.path.exists('test_integration_data.csv'):
        os.remove('test_integration_data.csv')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--integration':
            run_integration_test()
        elif sys.argv[1] == '--test':
            run_specific_test(sys.argv[2] if len(sys.argv) > 2 else "")
        else:
            print("Usage:")
            print("  python run_target_evaluator_tests.py              # Run all tests")
            print("  python run_target_evaluator_tests.py --integration # Run integration test")
            print("  python run_target_evaluator_tests.py --test <name> # Run specific test")
    else:
        exit_code = run_tests()
        sys.exit(exit_code)