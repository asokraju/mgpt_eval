#!/usr/bin/env python3
"""
Quick test summary script to verify binary classifier pipeline functionality.
Run this after fixing bugs to ensure everything works correctly.
"""

import sys
import tempfile
import pandas as pd
import json
import requests
from pathlib import Path

def test_fake_api():
    """Test fake API server connectivity."""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Fake API server is running")
            return True
        else:
            print(f"❌ Fake API server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to fake API server: {e}")
        return False

def test_imports():
    """Test all critical imports."""
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / "mgpt_eval"))
        
        from models.config_models import PipelineConfig
        from models.data_models import Dataset
        from models.pipeline_models import PipelineJob
        from pipelines.embedding_pipeline import EmbeddingPipeline
        from pipelines.classification_pipeline import ClassificationPipeline
        from pipelines.evaluation_pipeline import EvaluationPipeline
        from pipelines.end_to_end_pipeline import EndToEndPipeline
        from evaluation.target_word_evaluator import TargetWordEvaluator
        from utils.logging_utils import get_logger, LogContext
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with sample data."""
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / "mgpt_eval"))
        
        # Create test dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            data = [
                {'mcid': 'MC001', 'text': 'Patient shows excellent improvement', 'label': 1},
                {'mcid': 'MC002', 'text': 'Treatment was denied by insurance', 'label': 0},
                {'mcid': 'MC003', 'text': 'Claim approved for coverage', 'label': 1},
                {'mcid': 'MC004', 'text': 'Poor results from treatment', 'label': 0}
            ]
            
            dataset_path = temp_path / 'test_data.csv'
            pd.DataFrame(data).to_csv(dataset_path, index=False)
            
            # Test dataset loading
            from models.data_models import Dataset
            dataset = Dataset.from_file(str(dataset_path))
            assert len(dataset.records) == 4
            print("✅ Dataset loading works")
            
            # Test configuration
            from models.config_models import PipelineConfig
            config = PipelineConfig(
                model_api={'base_url': 'http://localhost:8001'},
                target_word_evaluation={'n_generations': 1, 'max_new_tokens': 20},
                logging={'file': str(temp_path / 'test.log')}
            )
            # Ensure directories are set up
            config.setup_directories()
            print("✅ Configuration creation works")
            
            # Test pipeline initialization
            from pipelines.embedding_pipeline import EmbeddingPipeline
            from pipelines.classification_pipeline import ClassificationPipeline
            from evaluation.target_word_evaluator import TargetWordEvaluator
            
            embedding_pipeline = EmbeddingPipeline(config)
            classification_pipeline = ClassificationPipeline(config)
            evaluator = TargetWordEvaluator(config)
            print("✅ Pipeline initialization works")
            
            # Test target word evaluation
            results = evaluator.evaluate(
                dataset_path=str(dataset_path),
                target_words=['approved', 'excellent'],
                n_samples=1,
                max_tokens=20,
                model_endpoint='http://localhost:8001/generate'
            )
            
            assert 'accuracy' in results
            assert 'target_words' in results
            assert results['target_words'] == ['approved', 'excellent']
            print(f"✅ Target word evaluation works (accuracy: {results['accuracy']:.3f})")
            
            return True
            
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / "mgpt_eval"))
        
        # Test unicode handling
        from models.data_models import DatasetRecord
        unicode_record = DatasetRecord(
            mcid="MC测试",
            text="Patient: 你好世界 🏥👨‍⚕️",
            label=1
        )
        assert "你好世界" in unicode_record.text
        print("✅ Unicode handling works")
        
        # Test invalid configurations
        from models.config_models import PipelineConfig
        from pydantic import ValidationError
        
        try:
            PipelineConfig(model_api={'base_url': 'invalid-url'})
            print("❌ URL validation should have failed")
            return False
        except ValidationError:
            print("✅ Configuration validation works")
        
        # Test empty dataset handling
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / 'empty.csv'
            empty_path.write_text("mcid,text,label\n")
            
            from models.data_models import Dataset
            try:
                Dataset.from_file(str(empty_path))
                print("❌ Empty dataset validation should have failed")
                return False
            except ValueError:
                print("✅ Empty dataset validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test error: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("Binary Classifier Pipeline - Test Summary")
    print("=" * 50)
    
    tests = [
        ("Fake API Server", test_fake_api),
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Edge Cases", test_edge_cases)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready for use.")
        print("\nTo get started:")
        print("1. Ensure fake API server is running: python fake_api_server.py --port 8001")
        print("2. Run the pipeline: python mgpt_eval/main_updated.py run-all --help")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())