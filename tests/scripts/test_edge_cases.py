#!/usr/bin/env python3
"""
Edge case testing for the embedding pipeline.
Tests error handling, data validation, and failure scenarios.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.config_models import PipelineConfig
from pipelines.embedding_pipeline import EmbeddingPipeline

class EdgeCaseTester:
    """Test suite for edge cases and error handling."""
    
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
        
    def create_edge_case_data(self):
        """Create test datasets for edge cases."""
        print("📁 Creating edge case datasets...")
        
        # Edge Case 1: Duplicate MCIDs (should fail)
        duplicate_data = pd.DataFrame({
            'mcid': [1, 2, 2, 3, 4],  # Duplicate MCID
            'claims': [
                "First unique claim",
                "Second claim with unique MCID", 
                "Third claim with DUPLICATE MCID",
                "Fourth claim with unique MCID",
                "Fifth claim with unique MCID"
            ],
            'label': [1, 0, 1, 0, 1]
        })
        duplicate_data.to_csv(self.data_dir / "edge_duplicate_mcids.csv", index=False)
        
        # Edge Case 2: Null values in claims (should fail)
        null_data = pd.DataFrame({
            'mcid': [1, 2, 3, 4],
            'claims': ["Valid claim", None, "Another valid claim", "Fourth claim"],
            'label': [1, 0, 1, 0]
        })
        null_data.to_csv(self.data_dir / "edge_null_claims.csv", index=False)
        
        # Edge Case 3: Null values in labels (should fail)
        null_labels_data = pd.DataFrame({
            'mcid': [1, 2, 3, 4],
            'claims': ["First claim", "Second claim", "Third claim", "Fourth claim"],
            'label': [1, None, 1, 0]
        })
        null_labels_data.to_csv(self.data_dir / "edge_null_labels.csv", index=False)
        
        # Edge Case 4: Missing required columns (should fail)
        missing_cols_data = pd.DataFrame({
            'mcid': [1, 2, 3],
            'claims': ["Claim 1", "Claim 2", "Claim 3"]
            # Missing 'label' column
        })
        missing_cols_data.to_csv(self.data_dir / "edge_missing_columns.csv", index=False)
        
        # Edge Case 5: Empty claims (should fail)
        empty_claims_data = pd.DataFrame({
            'mcid': [1, 2, 3, 4],
            'claims': ["", "Valid claim", "   ", "Another valid claim"],  # Empty and whitespace
            'label': [1, 0, 1, 0]
        })
        empty_claims_data.to_csv(self.data_dir / "edge_empty_claims.csv", index=False)
        
        # Edge Case 6: Very long claims (should work with truncation)
        super_long_claim = "This is an extremely comprehensive medical claim that contains " + "detailed information about patient medical history, symptoms, diagnostic procedures, laboratory results, imaging studies, treatment protocols, medication prescriptions, dosage instructions, side effects monitoring, follow-up schedules, prognosis assessment, and expected clinical outcomes " * 100 + " designed to test the tokenizer truncation capabilities."
        long_claims_data = pd.DataFrame({
            'mcid': [1, 2],
            'claims': [
                super_long_claim,
                "Standard length medical claim for baseline comparison"
            ],
            'label': [1, 1]
        })
        long_claims_data.to_csv(self.data_dir / "edge_super_long_claims.csv", index=False)
        
        # Edge Case 7: Special characters and Unicode (should work)
        special_chars_data = pd.DataFrame({
            'mcid': [1, 2, 3, 4, 5],
            'claims': [
                "Claim with émojis 😷💊🏥 and spëcial charactërs",
                "Claim with numbers: 123.45mg dosage, 98.6°F temperature",
                "Claim with symbols: <>&%$#@!()[]{}",
                "日本語の医療用語を含む主張についてのテスト",  # Japanese
                "Claim with mixed: Français, Español, Deutsch text"
            ],
            'label': [1, 1, 0, 1, 0]
        })
        special_chars_data.to_csv(self.data_dir / "edge_special_characters.csv", index=False)
        
        # Edge Case 8: Mixed data types (should pass - strings are allowed for MCIDs)
        mixed_types_data = pd.DataFrame({
            'mcid': ["string_mcid", 2, 3],  # String and int MCIDs are both allowed
            'claims': ["First claim", "Second claim", "Third claim"],
            'label': [1, 0, 1]
        })
        mixed_types_data.to_csv(self.data_dir / "edge_wrong_types.csv", index=False)
        
        # Edge Case 9: Minimum viable dataset (1 row, should work)
        minimal_data = pd.DataFrame({
            'mcid': [1],
            'claims': ["Single minimal test claim"],
            'label': [1]
        })
        minimal_data.to_csv(self.data_dir / "edge_minimal.csv", index=False)
        
        print("✅ Edge case datasets created successfully")
        
    def load_config(self, config_name="edge_case_config.yaml"):
        """Load edge case test configuration."""
        config_path = self.config_dir / config_name
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return config_data
    
    def run_edge_case_test(self, test_name, dataset_file, expected_to_fail=True, expected_error_type=None):
        """Run a single edge case test."""
        print(f"\n{'='*60}")
        print(f"🧪 Edge Case: {test_name}")
        print(f"📊 Dataset: {dataset_file}")
        print(f"🎯 Expected: {'FAIL' if expected_to_fail else 'PASS'}")
        if expected_error_type:
            print(f"🚫 Expected error: {expected_error_type}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Load and customize config
            config_data = self.load_config()
            config_data['input']['dataset_path'] = f"data/{dataset_file}"
            config_data['job']['name'] = f"edge_case_{test_name.lower().replace(' ', '_')}"
            
            # Initialize pipeline
            config = PipelineConfig(**config_data)
            pipeline = EmbeddingPipeline(config)
            
            # Run pipeline
            dataset_path = self.data_dir / dataset_file
            output_path = self.output_dir / "embeddings" / f"edge_{test_name.lower().replace(' ', '_')}_embeddings.csv"
            
            results = pipeline.run(str(dataset_path), str(output_path))
            
            elapsed_time = time.time() - start_time
            
            if expected_to_fail:
                print(f"❌ UNEXPECTED SUCCESS: {test_name}")
                print(f"   Expected this test to fail, but it succeeded")
                print(f"   📈 Generated {results['n_samples']} embeddings")
                print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'UNEXPECTED_SUCCESS',
                    'expected': 'FAIL',
                    'samples': results['n_samples'],
                    'time': elapsed_time,
                    'error': None,
                    'passed': False
                })
                return False
            else:
                print(f"✅ SUCCESS: {test_name}")
                print(f"   📈 Generated {results['n_samples']} embeddings")
                print(f"   📏 Embedding dim: {results['embedding_dim']}")
                print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'SUCCESS',
                    'expected': 'PASS',
                    'samples': results['n_samples'],
                    'time': elapsed_time,
                    'error': None,
                    'passed': True
                })
                return True
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_type = type(e).__name__
            error_message = str(e)
            
            if expected_to_fail:
                # Check if error type matches expectation
                if expected_error_type and expected_error_type not in error_type:
                    print(f"⚠️  WRONG ERROR TYPE: {test_name}")
                    print(f"   🎯 Expected: {expected_error_type}")
                    print(f"   🚫 Got: {error_type}")
                    print(f"   📝 Message: {error_message[:150]}...")
                    print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                    
                    self.results.append({
                        'test_name': test_name,
                        'status': 'WRONG_ERROR',
                        'expected': 'FAIL',
                        'samples': 0,
                        'time': elapsed_time,
                        'error': f"{error_type}: {error_message[:100]}",
                        'passed': False
                    })
                    return False
                else:
                    print(f"✅ EXPECTED FAILURE: {test_name}")
                    print(f"   🚫 Error type: {error_type}")
                    print(f"   📝 Message: {error_message[:150]}...")
                    print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                    
                    self.results.append({
                        'test_name': test_name,
                        'status': 'EXPECTED_FAILURE',
                        'expected': 'FAIL',
                        'samples': 0,
                        'time': elapsed_time,
                        'error': f"{error_type}: {error_message[:100]}",
                        'passed': True
                    })
                    return True
            else:
                print(f"❌ UNEXPECTED FAILURE: {test_name}")
                print(f"   🚫 Error type: {error_type}")
                print(f"   📝 Message: {error_message[:150]}...")
                print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'UNEXPECTED_FAILURE',
                    'expected': 'PASS',
                    'samples': 0,
                    'time': elapsed_time,
                    'error': f"{error_type}: {error_message[:100]}",
                    'passed': False
                })
                return False
    
    def run_all_edge_cases(self):
        """Run comprehensive edge case test suite."""
        print("🚀 EMBEDDING PIPELINE EDGE CASE TEST SUITE")
        print("="*60)
        
        # Create edge case data
        self.create_edge_case_data()
        
        # Define edge case tests
        edge_cases = [
            # Error cases (should fail)
            ("Duplicate MCIDs", "edge_duplicate_mcids.csv", True, "ValueError"),
            ("Null Claims", "edge_null_claims.csv", True, "ValueError"),
            ("Null Labels", "edge_null_labels.csv", True, "ValueError"),
            ("Missing Columns", "edge_missing_columns.csv", True, "ValueError"),
            ("Empty Claims", "edge_empty_claims.csv", True, "ValueError"),
            ("Mixed Data Types", "edge_wrong_types.csv", False, None),  # String MCIDs are allowed
            
            # Success cases (should pass)
            ("Super Long Claims", "edge_super_long_claims.csv", False, None),
            ("Special Characters", "edge_special_characters.csv", False, None),
            ("Minimal Dataset", "edge_minimal.csv", False, None),
        ]
        
        # Run edge case tests
        passed = 0
        total = len(edge_cases)
        
        for test_name, dataset, should_fail, error_type in edge_cases:
            success = self.run_edge_case_test(test_name, dataset, should_fail, error_type)
            if success:
                passed += 1
        
        # Print summary
        self.print_summary(passed, total)
        
        # Save results
        self.save_results()
        
        return passed == total
    
    def print_summary(self, passed, total):
        """Print edge case test summary."""
        print(f"\n{'='*60}")
        print("📊 EDGE CASE TEST SUMMARY")
        print(f"{'='*60}")
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        print(f"📈 Tests passed: {passed}/{total}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        if passed == total:
            print("🎉 All edge case tests passed! Error handling is robust.")
        else:
            print(f"⚠️  {total - passed} edge case tests failed. Review the errors above.")
        
        # Categorize results
        expected_failures = len([r for r in self.results if r['status'] == 'EXPECTED_FAILURE'])
        unexpected_failures = len([r for r in self.results if r['status'] == 'UNEXPECTED_FAILURE'])
        unexpected_successes = len([r for r in self.results if r['status'] == 'UNEXPECTED_SUCCESS'])
        wrong_errors = len([r for r in self.results if r['status'] == 'WRONG_ERROR'])
        successes = len([r for r in self.results if r['status'] == 'SUCCESS'])
        
        print(f"\n📋 Result Breakdown:")
        print(f"   ✅ Expected failures: {expected_failures}")
        print(f"   ✅ Expected successes: {successes}")
        print(f"   ❌ Unexpected failures: {unexpected_failures}")
        print(f"   ❌ Unexpected successes: {unexpected_successes}")
        print(f"   ⚠️  Wrong error types: {wrong_errors}")
        
        print(f"\n📋 Detailed Results:")
        for result in self.results:
            status_map = {
                'EXPECTED_FAILURE': '✅ PASS',
                'SUCCESS': '✅ PASS',
                'UNEXPECTED_FAILURE': '❌ FAIL',
                'UNEXPECTED_SUCCESS': '❌ FAIL',
                'WRONG_ERROR': '⚠️  WARN'
            }
            status_icon = status_map.get(result['status'], '❓')
            print(f"   {status_icon} {result['test_name']:25} | "
                  f"Expected: {result['expected']:4} | "
                  f"Time: {result['time']:5.2f}s")
    
    def save_results(self):
        """Save detailed edge case results."""
        results_file = self.output_dir / "edge_case_results.json"
        
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'edge_cases',
                'summary': {
                    'total_tests': len(self.results),
                    'passed': len([r for r in self.results if r['passed']]),
                    'failed': len([r for r in self.results if not r['passed']])
                },
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📄 Edge case results saved to: {results_file}")

def main():
    """Main edge case test runner."""
    tester = EdgeCaseTester()
    success = tester.run_all_edge_cases()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())