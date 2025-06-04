#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for the complete MGPT-Eval Pipeline.
Tests embedding generation → classification → evaluation in integrated workflows.
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
from pipelines.embedding_pipeline import EmbeddingPipeline
from pipelines.classification_pipeline import ClassificationPipeline

class EndToEndPipelineTester:
    """Comprehensive end-to-end test suite for the complete pipeline."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent.parent
        self.data_dir = self.test_dir / "data"
        self.config_dir = self.test_dir / "configs"
        self.output_dir = self.test_dir / "outputs"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.models_dir = self.output_dir / "models"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        self.results = []
        
    def create_test_datasets(self):
        """Create comprehensive test datasets for end-to-end testing."""
        print("📁 Creating comprehensive test datasets...")
        
        datasets = {}
        
        # Dataset 1: Balanced binary classification (20 samples)
        balanced_data = pd.DataFrame({
            'mcid': list(range(1, 21)),
            'claims': [
                # Class 1 (positive) - 10 samples
                "Patients with diabetes should monitor blood glucose levels regularly",
                "Regular exercise helps prevent cardiovascular disease effectively", 
                "Smoking significantly increases lung cancer risk in patients",
                "Proper medication adherence improves treatment outcomes substantially",
                "Vaccination prevents serious infectious diseases in populations",
                "Healthy diet reduces risk of chronic diseases significantly",
                "Sleep quality affects cognitive performance and mental health",
                "Stress management techniques improve overall patient wellbeing",
                "Physical therapy accelerates recovery from musculoskeletal injuries",
                "Preventive care reduces long-term healthcare costs effectively",
                # Class 0 (negative) - 10 samples  
                "Homeopathic remedies cure all types of cancer completely",
                "Vaccines cause autism and other developmental disorders",
                "Essential oils can replace all prescription medications safely",
                "Detox diets remove all toxins from the body permanently",
                "Supplements alone prevent all nutritional deficiencies completely",
                "Alternative medicine works better than evidence-based treatments",
                "Miracle cures exist for all chronic diseases naturally",
                "Vitamin megadoses prevent all infectious diseases effectively",
                "Herbal remedies have no side effects or contraindications",
                "Natural treatments always work better than pharmaceutical drugs"
            ],
            'label': [1]*10 + [0]*10
        })
        
        balanced_file = self.data_dir / "e2e_balanced_dataset.csv"
        balanced_data.to_csv(balanced_file, index=False)
        datasets['balanced'] = balanced_file
        print(f"✅ Created balanced dataset: 20 samples (10 per class)")
        
        # Dataset 2: Larger dataset (50 samples)
        large_claims = []
        large_labels = []
        
        # Generate positive claims (evidence-based medical statements)
        positive_templates = [
            "Evidence shows that {} improves patient outcomes significantly",
            "Clinical trials demonstrate {} reduces disease progression effectively", 
            "Research confirms {} prevents complications in high-risk patients",
            "Studies prove {} enhances quality of life for chronic conditions",
            "Data indicates {} decreases mortality rates in target populations"
        ]
        
        treatments = [
            "statins for cardiovascular disease", "insulin therapy for diabetes",
            "antibiotics for bacterial infections", "chemotherapy for cancer treatment",
            "physical therapy for rehabilitation", "behavioral therapy for mental health",
            "surgical intervention for acute conditions", "immunotherapy for autoimmune diseases",
            "radiation therapy for cancer treatment", "hormone replacement therapy",
            "antihypertensive medications", "antidepressant medications",
            "anticoagulation therapy", "bronchodilators for asthma",
            "corticosteroids for inflammation", "analgesics for pain management",
            "antiretroviral therapy for HIV", "beta-blockers for heart conditions",
            "calcium channel blockers", "ACE inhibitors for hypertension",
            "proton pump inhibitors", "antihistamines for allergies",
            "antifungal medications", "muscle relaxants for spasticity",
            "diuretics for fluid retention"
        ]
        
        for i in range(25):
            template = positive_templates[i % len(positive_templates)]
            treatment = treatments[i % len(treatments)]
            large_claims.append(template.format(treatment))
            large_labels.append(1)
        
        # Generate negative claims (pseudoscientific statements)
        negative_templates = [
            "Miracle cure {} eliminates all diseases without side effects",
            "Natural remedy {} works better than all conventional treatments",
            "Alternative therapy {} cures cancer using only natural ingredients",
            "Secret formula {} reverses aging and prevents all illnesses",
            "Herbal supplement {} replaces all prescription medications safely"
        ]
        
        fake_treatments = [
            "magnetic therapy", "crystal healing", "alkaline water", "detox teas",
            "oxygen therapy", "colloidal silver", "enzyme supplements", "chelation therapy",
            "ozone therapy", "frequency healing", "energy drinks", "miracle minerals",
            "superfoods", "ionized water", "electromagnetic fields", "sound therapy",
            "color therapy", "aromatherapy", "reflexology", "acupuncture claims",
            "chiropractic adjustments", "homeopathic dilutions", "psychic healing",
            "quantum medicine", "bio-resonance therapy"
        ]
        
        for i in range(25):
            template = negative_templates[i % len(negative_templates)]
            fake_treatment = fake_treatments[i % len(fake_treatments)]
            large_claims.append(template.format(fake_treatment))
            large_labels.append(0)
        
        large_data = pd.DataFrame({
            'mcid': list(range(1, 51)),
            'claims': large_claims,
            'label': large_labels
        })
        
        # Shuffle the dataset
        large_data = large_data.sample(frac=1, random_state=42).reset_index(drop=True)
        large_data['mcid'] = range(1, 51)  # Reassign sequential MCIDs
        
        large_file = self.data_dir / "e2e_large_dataset.csv"
        large_data.to_csv(large_file, index=False)
        datasets['large'] = large_file
        print(f"✅ Created large dataset: 50 samples (25 per class)")
        
        # Dataset 3: Imbalanced dataset (30 samples: 20 negative, 10 positive)
        imbalanced_data = pd.DataFrame({
            'mcid': list(range(1, 31)),
            'claims': large_claims[:20] + large_claims[25:35],  # 20 negative + 10 positive
            'label': [0]*20 + [1]*10
        })
        
        imbalanced_file = self.data_dir / "e2e_imbalanced_dataset.csv"
        imbalanced_data.to_csv(imbalanced_file, index=False)
        datasets['imbalanced'] = imbalanced_file
        print(f"✅ Created imbalanced dataset: 30 samples (20 class 0, 10 class 1)")
        
        return datasets
    
    def load_config(self, config_name="test_config.yaml"):
        """Load and customize test configuration."""
        config_path = self.config_dir / config_name
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Customize for end-to-end testing
        config_data['job']['name'] = 'e2e_pipeline_test'
        config_data['embedding_generation']['batch_size'] = 4  # Small batches for testing
        
        return config_data
    
    def run_embedding_phase(self, dataset_file, test_name):
        """Run embedding generation phase."""
        print(f"🔧 Phase 1: Generating embeddings for {test_name}...")
        
        try:
            # Load config
            config_data = self.load_config()
            config_data['input']['dataset_path'] = str(dataset_file)
            
            # Initialize embedding pipeline
            config = PipelineConfig(**config_data)
            embedding_pipeline = EmbeddingPipeline(config)
            
            # Generate embeddings
            output_path = self.embeddings_dir / f"e2e_{test_name}_embeddings.csv"
            embedding_results = embedding_pipeline.run(str(dataset_file), str(output_path))
            
            print(f"   ✅ Generated {embedding_results['n_samples']} embeddings")
            print(f"   📏 Embedding dimension: {embedding_results['embedding_dim']}")
            
            return str(output_path), embedding_results
            
        except Exception as e:
            print(f"   ❌ Embedding phase failed: {str(e)[:100]}...")
            raise
    
    def prepare_train_test_split(self, embeddings_file, test_name, split_ratio=0.8):
        """Create train/test split from embeddings."""
        print(f"🔧 Phase 2: Creating train/test split for {test_name}...")
        
        try:
            # Load embeddings
            df = pd.read_csv(embeddings_file)
            
            # Create stratified split to maintain class balance
            np.random.seed(42)
            
            # Separate by class
            class_0 = df[df['label'] == 0]
            class_1 = df[df['label'] == 1]
            
            # Split each class
            n_train_0 = int(len(class_0) * split_ratio)
            n_train_1 = int(len(class_1) * split_ratio)
            
            train_0 = class_0.sample(n=n_train_0, random_state=42)
            test_0 = class_0.drop(train_0.index)
            
            train_1 = class_1.sample(n=n_train_1, random_state=42)
            test_1 = class_1.drop(train_1.index)
            
            # Combine and shuffle
            train_df = pd.concat([train_0, train_1]).sample(frac=1, random_state=42)
            test_df = pd.concat([test_0, test_1]).sample(frac=1, random_state=42)
            
            # Save splits
            train_file = self.data_dir / f"e2e_{test_name}_train.csv"
            test_file = self.data_dir / f"e2e_{test_name}_test.csv"
            
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            
            print(f"   ✅ Train set: {len(train_df)} samples (class 0: {sum(train_df['label']==0)}, class 1: {sum(train_df['label']==1)})")
            print(f"   ✅ Test set: {len(test_df)} samples (class 0: {sum(test_df['label']==0)}, class 1: {sum(test_df['label']==1)})")
            
            return str(train_file), str(test_file)
            
        except Exception as e:
            print(f"   ❌ Train/test split failed: {str(e)[:100]}...")
            raise
    
    def run_classification_phase(self, train_file, test_file, test_name, classifier_type="logistic_regression"):
        """Run classification training and evaluation phase."""
        print(f"🔧 Phase 3: Training {classifier_type} classifier for {test_name}...")
        
        try:
            # Load config
            config_data = self.load_config()
            
            # Initialize classification pipeline
            classification_pipeline = ClassificationPipeline(config_data)
            
            # Create output directory
            output_dir = self.models_dir / f"e2e_{test_name}_{classifier_type}"
            output_dir.mkdir(exist_ok=True)
            
            # Run classification
            classification_results = classification_pipeline.run(
                train_embeddings=train_file,
                test_embeddings=test_file,
                classifier_type=classifier_type,
                output_dir=str(output_dir)
            )
            
            metrics = classification_results['test_metrics']
            print(f"   ✅ Model trained successfully")
            print(f"   🎯 CV Score: {classification_results['best_cv_score']:.4f}")
            print(f"   📊 Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"   📊 Test F1-Score: {metrics['f1_score']:.4f}")
            if metrics['roc_auc']:
                print(f"   📊 Test AUC: {metrics['roc_auc']:.4f}")
            
            return classification_results
            
        except Exception as e:
            print(f"   ❌ Classification phase failed: {str(e)[:100]}...")
            raise
    
    def run_end_to_end_test(self, test_name, dataset_file, classifier_types=None, expected_to_pass=True):
        """Run complete end-to-end test."""
        if classifier_types is None:
            classifier_types = ["logistic_regression"]
        
        print(f"\n{'='*80}")
        print(f"🚀 END-TO-END TEST: {test_name}")
        print(f"📊 Dataset: {dataset_file.name}")
        print(f"🤖 Classifiers: {', '.join(classifier_types)}")
        print(f"🎯 Expected: {'PASS' if expected_to_pass else 'FAIL'}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Phase 1: Generate embeddings
            embeddings_file, embedding_results = self.run_embedding_phase(dataset_file, test_name)
            
            # Phase 2: Create train/test split
            train_file, test_file = self.prepare_train_test_split(embeddings_file, test_name)
            
            # Phase 3: Train and evaluate classifiers
            classifier_results = {}
            for classifier_type in classifier_types:
                classification_results = self.run_classification_phase(
                    train_file, test_file, test_name, classifier_type
                )
                classifier_results[classifier_type] = classification_results
            
            elapsed_time = time.time() - start_time
            
            if expected_to_pass:
                print(f"\n✅ END-TO-END SUCCESS: {test_name}")
                print(f"   ⏱️  Total time: {elapsed_time:.1f}s")
                print(f"   📈 Embedding samples: {embedding_results['n_samples']}")
                print(f"   🧠 Embedding dimension: {embedding_results['embedding_dim']}")
                
                # Best classifier performance
                best_classifier = max(classifier_results.keys(), 
                                    key=lambda k: classifier_results[k]['test_metrics']['f1_score'])
                best_f1 = classifier_results[best_classifier]['test_metrics']['f1_score']
                print(f"   🏆 Best F1-Score: {best_f1:.4f} ({best_classifier})")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'PASS',
                    'expected': 'PASS',
                    'embedding_samples': embedding_results['n_samples'],
                    'embedding_dim': embedding_results['embedding_dim'],
                    'classifiers': list(classifier_types),
                    'best_classifier': best_classifier,
                    'best_f1': best_f1,
                    'time': elapsed_time,
                    'error': None
                })
                return True
            else:
                print(f"\n❌ UNEXPECTED SUCCESS: {test_name}")
                print(f"   Expected this test to fail, but it passed")
                self.results.append({
                    'test_name': test_name,
                    'status': 'FAIL',
                    'expected': 'FAIL',
                    'embedding_samples': embedding_results['n_samples'],
                    'embedding_dim': embedding_results['embedding_dim'],
                    'classifiers': list(classifier_types),
                    'best_classifier': None,
                    'best_f1': None,
                    'time': elapsed_time,
                    'error': 'Unexpected success'
                })
                return False
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_type = type(e).__name__
            
            if expected_to_pass:
                print(f"\n❌ END-TO-END FAILURE: {test_name}")
                print(f"   🚫 Error: {error_type}")
                print(f"   📝 Message: {str(e)[:200]}...")
                print(f"   ⏱️  Time: {elapsed_time:.1f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'FAIL',
                    'expected': 'PASS',
                    'embedding_samples': None,
                    'embedding_dim': None,
                    'classifiers': list(classifier_types),
                    'best_classifier': None,
                    'best_f1': None,
                    'time': elapsed_time,
                    'error': f"{error_type}: {str(e)[:150]}"
                })
                return False
            else:
                print(f"\n✅ EXPECTED FAILURE: {test_name}")
                print(f"   🚫 Error: {error_type}")
                print(f"   📝 Message: {str(e)[:200]}...")
                print(f"   ⏱️  Time: {elapsed_time:.1f}s")
                
                self.results.append({
                    'test_name': test_name,
                    'status': 'PASS',
                    'expected': 'FAIL',
                    'embedding_samples': None,
                    'embedding_dim': None,
                    'classifiers': list(classifier_types),
                    'best_classifier': None,
                    'best_f1': None,
                    'time': elapsed_time,
                    'error': f"{error_type}: {str(e)[:150]}"
                })
                return True
    
    def run_all_tests(self):
        """Run comprehensive end-to-end test suite."""
        print("🚀 COMPREHENSIVE END-TO-END PIPELINE TEST SUITE")
        print("="*80)
        print("Testing complete workflow: Data → Embeddings → Classification → Evaluation")
        print("="*80)
        
        # Create test datasets
        datasets = self.create_test_datasets()
        
        # Define comprehensive test cases
        test_cases = [
            # Basic end-to-end tests
            ("Balanced Dataset - Single Classifier", datasets['balanced'], ["logistic_regression"], True),
            ("Balanced Dataset - Multiple Classifiers", datasets['balanced'], ["logistic_regression", "svm", "random_forest"], True),
            ("Large Dataset - LR", datasets['large'], ["logistic_regression"], True),
            ("Large Dataset - SVM", datasets['large'], ["svm"], True),
            ("Imbalanced Dataset", datasets['imbalanced'], ["logistic_regression"], True),
        ]
        
        # Run tests
        passed = 0
        total = len(test_cases)
        
        for test_name, dataset_file, classifiers, expected_pass in test_cases:
            success = self.run_end_to_end_test(test_name, dataset_file, classifiers, expected_pass)
            if success:
                passed += 1
        
        # Print comprehensive summary
        self.print_comprehensive_summary(passed, total)
        
        # Save detailed results
        self.save_comprehensive_results()
        
        return passed == total
    
    def print_comprehensive_summary(self, passed, total):
        """Print comprehensive test summary."""
        print(f"\n{'='*80}")
        print("🏆 COMPREHENSIVE END-TO-END TEST SUMMARY")
        print(f"{'='*80}")
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        print(f"📈 Tests passed: {passed}/{total}")
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        if passed == total:
            print("🎉 ALL END-TO-END TESTS PASSED!")
            print("   The complete MGPT-Eval pipeline is working correctly:")
            print("   ✅ Embedding generation is robust")
            print("   ✅ Classification training is successful") 
            print("   ✅ Model evaluation produces valid metrics")
            print("   ✅ End-to-end workflow is seamless")
        else:
            print(f"⚠️  {total - passed} end-to-end tests failed.")
            print("   Please review the detailed results above.")
        
        # Performance analysis
        successful_tests = [r for r in self.results if r['status'] == 'PASS' and r['expected'] == 'PASS']
        if successful_tests:
            avg_f1 = np.mean([r['best_f1'] for r in successful_tests if r['best_f1']])
            avg_time = np.mean([r['time'] for r in successful_tests])
            total_samples = sum([r['embedding_samples'] for r in successful_tests if r['embedding_samples']])
            
            print(f"\n📊 Pipeline Performance Analysis:")
            print(f"   🎯 Average F1-Score: {avg_f1:.3f}")
            print(f"   ⏱️  Average test time: {avg_time:.1f}s")
            print(f"   📈 Total samples processed: {total_samples}")
            print(f"   🚀 Processing rate: {total_samples/sum([r['time'] for r in successful_tests]):.1f} samples/sec")
        
        # Classifier comparison
        classifier_performance = {}
        for result in successful_tests:
            for classifier in result['classifiers']:
                if classifier not in classifier_performance:
                    classifier_performance[classifier] = []
                if result['best_f1']:
                    classifier_performance[classifier].append(result['best_f1'])
        
        if classifier_performance:
            print(f"\n🤖 Classifier Performance Comparison:")
            for classifier, scores in classifier_performance.items():
                avg_score = np.mean(scores)
                print(f"   {classifier:20}: {avg_score:.3f} (avg F1 over {len(scores)} tests)")
        
        print(f"\n📋 Detailed Test Results:")
        for result in self.results:
            status_icon = "✅" if result['status'] == result['expected'] else "❌"
            f1_score = f"{result['best_f1']:.3f}" if result['best_f1'] else "N/A"
            samples = result['embedding_samples'] if result['embedding_samples'] else "N/A"
            
            print(f"   {status_icon} {result['test_name']:35} | "
                  f"Samples: {samples:3} | "
                  f"F1: {f1_score:5} | "
                  f"Time: {result['time']:5.1f}s")
    
    def save_comprehensive_results(self):
        """Save comprehensive test results."""
        results_file = self.output_dir / "end_to_end_test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'end_to_end_pipeline',
                'pipeline_components': ['embedding_generation', 'classification', 'evaluation'],
                'summary': {
                    'total_tests': len(self.results),
                    'passed': len([r for r in self.results if r['status'] == r['expected']]),
                    'failed': len([r for r in self.results if r['status'] != r['expected']])
                },
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📄 Comprehensive results saved to: {results_file}")

def main():
    """Main end-to-end test runner."""
    tester = EndToEndPipelineTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())