#!/usr/bin/env python3
"""
Comprehensive test suite for the embedding pipeline.
Tests basic functionality, batching, and performance.
"""

import os
import sys
import yaml
import pandas as pd
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.config_models import PipelineConfig
from pipelines.embedding_pipeline import EmbeddingPipeline

class EmbeddingPipelineTester:
    """Test suite for embedding pipeline functionality."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent.parent
        self.data_dir = self.test_dir / "data"
        self.config_dir = self.test_dir / "configs"
        self.output_dir = self.test_dir / "outputs"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        
        self.results = []
        
    def create_test_data(self):
        """Create test datasets for various scenarios."""
        print("📁 Creating test datasets...")
        
        # Test 1: Normal dataset (6 items)
        normal_data = pd.DataFrame({
            'mcid': [1, 2, 3, 4, 5, 6],
            'claims': [
                "Patients with diabetes should monitor blood glucose levels regularly",
                "Regular exercise helps prevent cardiovascular disease", 
                "Smoking increases lung cancer risk significantly",
                "Vitamin C supplements may boost immune system function",
                "Antibiotics treat bacterial infections effectively",
                "Sleep deprivation negatively affects cognitive performance"
            ],
            'label': [1, 1, 1, 0, 1, 1]
        })
        normal_data.to_csv(self.data_dir / "test_normal.csv", index=False)
        
        # Test 2: Single item
        single_data = pd.DataFrame({
            'mcid': [1],
            'claims': ["Single test claim for pipeline validation"],
            'label': [1]
        })
        single_data.to_csv(self.data_dir / "test_single.csv", index=False)
        
        # Test 3: Odd number of items (5 items)
        odd_data = pd.DataFrame({
            'mcid': [1, 2, 3, 4, 5],
            'claims': [
                "First medical claim for testing",
                "Second claim about treatment efficacy", 
                "Third claim regarding diagnostic accuracy",
                "Fourth claim about preventive measures",
                "Fifth claim on therapeutic interventions"
            ],
            'label': [1, 0, 1, 0, 1]
        })
        odd_data.to_csv(self.data_dir / "test_odd.csv", index=False)
        
        # Test 4: Large dataset (20 items)
        large_claims = [f"Medical claim number {i} discussing treatment protocols and patient outcomes" for i in range(1, 21)]
        large_data = pd.DataFrame({
            'mcid': list(range(1, 21)),
            'claims': large_claims,
            'label': [i % 2 for i in range(20)]  # Alternating labels
        })
        large_data.to_csv(self.data_dir / "test_large.csv", index=False)
        
        # Test 5: Long claims requiring truncation
        long_claim = "This is an extremely long medical claim that contains " + "detailed information about patient symptoms, diagnostic procedures, treatment protocols, medication dosages, follow-up schedules, and expected outcomes " * 50 + "to test the tokenizer truncation functionality."
        long_data = pd.DataFrame({
            'mcid': [1, 2],
            'claims': [
                long_claim,
                "Standard length medical claim for comparison purposes"
            ],
            'label': [1, 1]
        })
        long_data.to_csv(self.data_dir / "test_long_claims.csv", index=False)
        
        print("✅ Test datasets created successfully")
        
    def load_config(self, config_name="test_config.yaml"):
        """Load test configuration."""
        config_path = self.config_dir / config_name
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return config_data
    
    def run_test(self, test_name, dataset_file, batch_size=2, expected_to_pass=True):
        """Run a single test case."""
        print(f"\n{'='*60}")
        print(f"🧪 Test: {test_name}")
        print(f"📊 Dataset: {dataset_file}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🎯 Expected: {'PASS' if expected_to_pass else 'FAIL'}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Load and customize config
            config_data = self.load_config()
            config_data['input']['dataset_path'] = f"data/{dataset_file}"
            config_data['embedding_generation']['batch_size'] = batch_size
            config_data['job']['name'] = f"test_{test_name.lower().replace(' ', '_')}"
            
            # Initialize pipeline
            config = PipelineConfig(**config_data)
            pipeline = EmbeddingPipeline(config)
            
            # Run pipeline
            dataset_path = self.data_dir / dataset_file
            output_path = self.output_dir / "embeddings" / f"{test_name.lower().replace(' ', '_')}_embeddings.csv"
            
            results = pipeline.run(str(dataset_path), str(output_path))
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            throughput = results['n_samples'] / elapsed_time if elapsed_time > 0 else 0
            
            if expected_to_pass:
                print(f"✅ SUCCESS: {test_name}")
                print(f"   📈 Samples: {results['n_samples']}")
                print(f"   📏 Embedding dim: {results['embedding_dim']}")
                print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                print(f"   🚀 Throughput: {throughput:.1f} samples/sec")
                
                if 'embedding_stats' in results:
                    stats = results['embedding_stats']
                    print(f"   📊 Mean norm: {stats.get('mean_norm', 0):.3f}")
                    print(f"   📊 Std norm: {stats.get('std_norm', 0):.3f}")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'PASS',
                    'expected': 'PASS',
                    'samples': results['n_samples'],
                    'embedding_dim': results['embedding_dim'],
                    'time': elapsed_time,
                    'throughput': throughput,
                    'batch_size': batch_size,
                    'error': None
                })
                return True
            else:
                print(f"❌ UNEXPECTED SUCCESS: {test_name}")
                print(f"   Expected this test to fail, but it passed")
                self.results.append({
                    'test_name': test_name,
                    'status': 'FAIL',
                    'expected': 'FAIL',
                    'samples': results['n_samples'],
                    'embedding_dim': results['embedding_dim'],
                    'time': elapsed_time,
                    'throughput': throughput,
                    'batch_size': batch_size,
                    'error': 'Unexpected success'
                })
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_type = type(e).__name__
            
            if expected_to_pass:
                print(f"❌ UNEXPECTED FAILURE: {test_name}")
                print(f"   🚫 Error: {error_type}")
                print(f"   📝 Message: {str(e)[:100]}...")
                print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'FAIL',
                    'expected': 'PASS',
                    'samples': 0,
                    'embedding_dim': 0,
                    'time': elapsed_time,
                    'throughput': 0,
                    'batch_size': batch_size,
                    'error': f"{error_type}: {str(e)[:100]}"
                })
                return False
            else:
                print(f"✅ EXPECTED FAILURE: {test_name}")
                print(f"   🚫 Error: {error_type}")
                print(f"   📝 Message: {str(e)[:100]}...")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'PASS',
                    'expected': 'FAIL',
                    'samples': 0,
                    'embedding_dim': 0,
                    'time': elapsed_time,
                    'throughput': 0,
                    'batch_size': batch_size,
                    'error': f"{error_type}: {str(e)[:100]}"
                })
                return True
    
    def run_all_tests(self):
        """Run comprehensive test suite."""
        print("🚀 EMBEDDING PIPELINE COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        # Create test data
        self.create_test_data()
        
        # Define test cases
        test_cases = [
            # Basic functionality tests
            ("Normal Operation", "test_normal.csv", 2, True),
            ("Single Item", "test_single.csv", 1, True),
            ("Odd Number Items", "test_odd.csv", 2, True),
            ("Large Dataset", "test_large.csv", 4, True),
            ("Long Claims", "test_long_claims.csv", 1, True),
            
            # Batching tests
            ("Small Batch", "test_normal.csv", 1, True),
            ("Large Batch", "test_normal.csv", 10, True),
            ("Perfect Batch", "test_normal.csv", 6, True),  # Exact match
            
            # Performance tests
            ("High Throughput", "test_large.csv", 8, True),
        ]
        
        # Run tests
        passed = 0
        total = len(test_cases)
        
        for test_name, dataset, batch_size, expected_pass in test_cases:
            success = self.run_test(test_name, dataset, batch_size, expected_pass)
            if success:
                passed += 1
        
        # Print summary
        self.print_summary(passed, total)
        
        # Save detailed results
        self.save_results()
        
        return passed == total
    
    def print_summary(self, passed, total):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY REPORT")
        print(f"{'='*60}")
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        print(f"📈 Tests passed: {passed}/{total}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        if passed == total:
            print("🎉 All tests passed! Embedding pipeline is working correctly.")
        else:
            print(f"⚠️  {total - passed} tests failed. Review the errors above.")
        
        # Performance summary
        successful_tests = [r for r in self.results if r['status'] == 'PASS' and r['expected'] == 'PASS']
        if successful_tests:
            avg_throughput = sum(r['throughput'] for r in successful_tests) / len(successful_tests)
            max_throughput = max(r['throughput'] for r in successful_tests)
            print(f"\n📊 Performance Metrics:")
            print(f"   🚀 Average throughput: {avg_throughput:.1f} samples/sec")
            print(f"   ⚡ Peak throughput: {max_throughput:.1f} samples/sec")
        
        print("\n📋 Detailed Results:")
        for result in self.results:
            status_icon = "✅" if result['status'] == result['expected'] else "❌"
            print(f"   {status_icon} {result['test_name']:25} | "
                  f"Batch: {result['batch_size']:2} | "
                  f"Time: {result['time']:5.2f}s | "
                  f"Throughput: {result['throughput']:6.1f}/s")
    
    def save_results(self):
        """Save detailed results to file."""
        results_file = self.output_dir / "test_results.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total_tests': len(self.results),
                    'passed': len([r for r in self.results if r['status'] == r['expected']]),
                    'failed': len([r for r in self.results if r['status'] != r['expected']])
                },
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")

def main():
    """Main test runner."""
    tester = EmbeddingPipelineTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())