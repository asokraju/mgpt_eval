#!/usr/bin/env python3
"""
Comprehensive test suite for the classification pipeline.
Tests binary classification using embeddings from the embedding pipeline.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.config_models import PipelineConfig
from pipelines.classification_pipeline import ClassificationPipeline

class ClassificationPipelineTester:
    """Test suite for classification pipeline functionality."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent.parent
        self.data_dir = self.test_dir / "data"
        self.config_dir = self.test_dir / "configs"
        self.output_dir = self.test_dir / "outputs"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.models_dir = self.output_dir / "models"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        
        self.results = []
        
    def prepare_classification_data(self):
        """Prepare train/test split from existing embedding files."""
        print("📁 Preparing classification datasets from embeddings...")
        
        # Use the normal operation embeddings as our base dataset
        source_file = self.embeddings_dir / "normal_operation_embeddings.csv"
        if not source_file.exists():
            raise FileNotFoundError(f"Source embeddings not found: {source_file}")
        
        # Load source data
        df = pd.read_csv(source_file)
        print(f"Loaded {len(df)} samples from {source_file.name}")
        
        # Create train/test split (80/20)
        np.random.seed(42)  # For reproducible splits
        train_indices = np.random.choice(len(df), size=int(0.8 * len(df)), replace=False)
        test_indices = np.setdiff1d(np.arange(len(df)), train_indices)
        
        train_df = df.iloc[train_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        # Save train/test splits
        train_file = self.data_dir / "classification_train.csv"
        test_file = self.data_dir / "classification_test.csv"
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"✅ Created train set: {len(train_df)} samples -> {train_file}")
        print(f"✅ Created test set: {len(test_df)} samples -> {test_file}")
        
        # Create larger dataset for more comprehensive testing
        large_source = self.embeddings_dir / "large_dataset_embeddings.csv"
        if large_source.exists():
            large_df = pd.read_csv(large_source)
            print(f"Found large dataset: {len(large_df)} samples")
            
            # Create larger train/test split
            large_train_indices = np.random.choice(len(large_df), size=int(0.8 * len(large_df)), replace=False)
            large_test_indices = np.setdiff1d(np.arange(len(large_df)), large_train_indices)
            
            large_train_df = large_df.iloc[large_train_indices].copy()
            large_test_df = large_df.iloc[large_test_indices].copy()
            
            large_train_file = self.data_dir / "classification_large_train.csv"
            large_test_file = self.data_dir / "classification_large_test.csv"
            
            large_train_df.to_csv(large_train_file, index=False)
            large_test_df.to_csv(large_test_file, index=False)
            
            print(f"✅ Created large train set: {len(large_train_df)} samples -> {large_train_file}")
            print(f"✅ Created large test set: {len(large_test_df)} samples -> {large_test_file}")
        
        return train_file, test_file
        
    def create_edge_case_data(self):
        """Create edge case datasets for classification testing."""
        print("📁 Creating edge case classification datasets...")
        
        # Load normal embeddings as template
        source_file = self.embeddings_dir / "normal_operation_embeddings.csv"
        df = pd.read_csv(source_file)
        
        # Edge Case 1: Single class train data (should fail)
        single_class_df = df[df['label'] == 1].copy()
        if len(single_class_df) > 0:
            single_class_file = self.data_dir / "classification_single_class.csv"
            single_class_df.to_csv(single_class_file, index=False)
            print(f"✅ Created single class dataset: {len(single_class_df)} samples")
        
        # Edge Case 2: Imbalanced dataset (90% class 0, 10% class 1)
        n_samples = len(df)
        n_class_0 = int(0.9 * n_samples)
        n_class_1 = n_samples - n_class_0
        
        imbalanced_df = df.copy()
        imbalanced_df['label'] = [0] * n_class_0 + [1] * n_class_1
        np.random.shuffle(imbalanced_df['label'].values)
        
        imbalanced_file = self.data_dir / "classification_imbalanced.csv"
        imbalanced_df.to_csv(imbalanced_file, index=False)
        print(f"✅ Created imbalanced dataset: {n_class_0} class 0, {n_class_1} class 1")
        
        # Edge Case 3: Minimal dataset (just enough for CV)
        minimal_df = df.head(10).copy()  # Take up to 10 samples for 5-fold CV
        n_minimal = len(minimal_df)
        # Create balanced labels based on actual number of samples
        minimal_labels = [i % 2 for i in range(n_minimal)]
        minimal_df['label'] = minimal_labels
        
        minimal_file = self.data_dir / "classification_minimal.csv"
        minimal_df.to_csv(minimal_file, index=False)
        print(f"✅ Created minimal dataset: {len(minimal_df)} samples")
        
        return {
            'single_class': single_class_file if len(single_class_df) > 0 else None,
            'imbalanced': imbalanced_file,
            'minimal': minimal_file
        }
    
    def load_config(self, config_name="test_config.yaml"):
        """Load test configuration."""
        config_path = self.config_dir / config_name
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return config_data
    
    def run_classification_test(self, test_name, train_file, test_file, classifier_type="logistic_regression", expected_to_pass=True):
        """Run a single classification test."""
        print(f"\n{'='*60}")
        print(f"🧪 Classification Test: {test_name}")
        print(f"📊 Train file: {train_file.name}")
        print(f"📊 Test file: {test_file.name}")
        print(f"🤖 Classifier: {classifier_type}")
        print(f"🎯 Expected: {'PASS' if expected_to_pass else 'FAIL'}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Load config
            config_data = self.load_config()
            
            # Initialize pipeline
            pipeline = ClassificationPipeline(config_data)
            
            # Run classification
            output_dir = self.models_dir / f"{test_name.lower().replace(' ', '_')}"
            output_dir.mkdir(exist_ok=True)
            
            results = pipeline.run(
                train_embeddings=str(train_file),
                test_embeddings=str(test_file),
                classifier_type=classifier_type,
                output_dir=str(output_dir)
            )
            
            elapsed_time = time.time() - start_time
            
            if expected_to_pass:
                print(f"✅ SUCCESS: {test_name}")
                print(f"   📈 Model saved: {Path(results['model_path']).name}")
                print(f"   🎯 Best CV score: {results['best_cv_score']:.4f}")
                print(f"   📊 Test accuracy: {results['test_metrics']['accuracy']:.4f}")
                print(f"   📊 Test F1: {results['test_metrics']['f1_score']:.4f}")
                if results['test_metrics']['roc_auc']:
                    print(f"   📊 Test AUC: {results['test_metrics']['roc_auc']:.4f}")
                print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'PASS',
                    'expected': 'PASS',
                    'classifier': classifier_type,
                    'cv_score': results['best_cv_score'],
                    'test_accuracy': results['test_metrics']['accuracy'],
                    'test_f1': results['test_metrics']['f1_score'],
                    'test_auc': results['test_metrics'].get('roc_auc'),
                    'time': elapsed_time,
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
                    'classifier': classifier_type,
                    'cv_score': results['best_cv_score'],
                    'test_accuracy': results['test_metrics']['accuracy'],
                    'test_f1': results['test_metrics']['f1_score'],
                    'test_auc': results['test_metrics'].get('roc_auc'),
                    'time': elapsed_time,
                    'error': 'Unexpected success'
                })
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_type = type(e).__name__
            
            if expected_to_pass:
                print(f"❌ UNEXPECTED FAILURE: {test_name}")
                print(f"   🚫 Error: {error_type}")
                print(f"   📝 Message: {str(e)[:150]}...")
                print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'FAIL',
                    'expected': 'PASS',
                    'classifier': classifier_type,
                    'cv_score': None,
                    'test_accuracy': None,
                    'test_f1': None,
                    'test_auc': None,
                    'time': elapsed_time,
                    'error': f"{error_type}: {str(e)[:100]}"
                })
                return False
            else:
                print(f"✅ EXPECTED FAILURE: {test_name}")
                print(f"   🚫 Error: {error_type}")
                print(f"   📝 Message: {str(e)[:150]}...")
                print(f"   ⏱️  Time: {elapsed_time:.2f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'PASS',
                    'expected': 'FAIL',
                    'classifier': classifier_type,
                    'cv_score': None,
                    'test_accuracy': None,
                    'test_f1': None,
                    'test_auc': None,
                    'time': elapsed_time,
                    'error': f"{error_type}: {str(e)[:100]}"
                })
                return True
    
    def run_all_tests(self):
        """Run comprehensive classification test suite."""
        print("🚀 CLASSIFICATION PIPELINE COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        # Prepare data
        train_file, test_file = self.prepare_classification_data()
        edge_case_files = self.create_edge_case_data()
        
        # Basic functionality tests
        test_cases = [
            # Basic classifier tests
            ("Logistic Regression", train_file, test_file, "logistic_regression", True),
            ("SVM", train_file, test_file, "svm", True),
            ("Random Forest", train_file, test_file, "random_forest", True),
        ]
        
        # Add large dataset tests if available
        large_train = self.data_dir / "classification_large_train.csv"
        large_test = self.data_dir / "classification_large_test.csv"
        if large_train.exists() and large_test.exists():
            test_cases.extend([
                ("Large Dataset - LR", large_train, large_test, "logistic_regression", True),
                ("Large Dataset - SVM", large_train, large_test, "svm", True),
            ])
        
        # Edge case tests
        if edge_case_files['single_class']:
            # Single class should fail (not enough classes for binary classification)
            test_cases.append(
                ("Single Class", edge_case_files['single_class'], test_file, "logistic_regression", False)
            )
        
        test_cases.extend([
            ("Imbalanced Dataset", edge_case_files['imbalanced'], test_file, "logistic_regression", True),
            ("Minimal Dataset", edge_case_files['minimal'], edge_case_files['minimal'], "logistic_regression", True),
        ])
        
        # Run tests
        passed = 0
        total = len(test_cases)
        
        for test_name, train_f, test_f, classifier, expected_pass in test_cases:
            success = self.run_classification_test(test_name, train_f, test_f, classifier, expected_pass)
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
        print("📊 CLASSIFICATION TEST SUMMARY REPORT")
        print(f"{'='*60}")
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        print(f"📈 Tests passed: {passed}/{total}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        if passed == total:
            print("🎉 All classification tests passed! Pipeline is working correctly.")
        else:
            print(f"⚠️  {total - passed} tests failed. Review the errors above.")
        
        # Performance summary
        successful_tests = [r for r in self.results if r['status'] == 'PASS' and r['expected'] == 'PASS']
        if successful_tests:
            avg_accuracy = np.mean([r['test_accuracy'] for r in successful_tests if r['test_accuracy']])
            avg_f1 = np.mean([r['test_f1'] for r in successful_tests if r['test_f1']])
            valid_aucs = [r['test_auc'] for r in successful_tests if r['test_auc'] is not None]
            avg_auc = np.mean(valid_aucs) if valid_aucs else None
            
            print(f"\n📊 Performance Metrics:")
            print(f"   🎯 Average accuracy: {avg_accuracy:.3f}")
            print(f"   📈 Average F1-score: {avg_f1:.3f}")
            if avg_auc:
                print(f"   📊 Average AUC: {avg_auc:.3f}")
        
        print("\n📋 Detailed Results:")
        for result in self.results:
            status_icon = "✅" if result['status'] == result['expected'] else "❌"
            classifier_short = result['classifier'][:3].upper() if result['classifier'] else "N/A"
            accuracy = f"{result['test_accuracy']:.3f}" if result['test_accuracy'] else "N/A"
            f1 = f"{result['test_f1']:.3f}" if result['test_f1'] else "N/A"
            
            print(f"   {status_icon} {result['test_name']:25} | "
                  f"{classifier_short:3} | "
                  f"Acc: {accuracy:5} | "
                  f"F1: {f1:5} | "
                  f"Time: {result['time']:5.1f}s")
    
    def save_results(self):
        """Save detailed results to file."""
        results_file = self.output_dir / "classification_test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'classification_pipeline',
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
    tester = ClassificationPipelineTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())