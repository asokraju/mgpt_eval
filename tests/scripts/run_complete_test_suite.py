#!/usr/bin/env python3
"""
Master Test Runner for Complete MGPT-Eval Pipeline Testing.
Runs all test suites in sequence: Embedding → Classification → End-to-End → Summary.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all test modules
from test_embedding_pipeline import EmbeddingPipelineTester
from test_edge_cases import EdgeCaseTester
from test_classification_pipeline import ClassificationPipelineTester
from test_end_to_end_pipeline import EndToEndPipelineTester

class CompletePipelineTestRunner:
    """Master test runner for the complete MGPT-Eval pipeline."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent.parent
        self.output_dir = self.test_dir / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        
        self.all_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_suites': {},
            'summary': {
                'total_suites': 0,
                'passed_suites': 0,
                'total_tests': 0,
                'passed_tests': 0,
                'total_time': 0
            }
        }
    
    def run_embedding_tests(self):
        """Run embedding pipeline functionality tests."""
        print("🔧 Running Embedding Pipeline Tests...")
        print("="*80)
        
        start_time = time.time()
        tester = EmbeddingPipelineTester()
        success = tester.run_all_tests()
        elapsed_time = time.time() - start_time
        
        # Collect results
        passed_tests = len([r for r in tester.results if r['status'] == r['expected']])
        total_tests = len(tester.results)
        
        self.all_results['test_suites']['embedding_functionality'] = {
            'name': 'Embedding Pipeline Tests',
            'passed': success,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'time': elapsed_time,
            'results': tester.results
        }
        
        self._update_summary(success, total_tests, passed_tests, elapsed_time)
        return success
    
    def run_edge_case_tests(self):
        """Run embedding pipeline edge case tests."""
        print("\n🧪 Running Edge Case Tests...")
        print("="*80)
        
        start_time = time.time()
        tester = EdgeCaseTester()
        success = tester.run_all_edge_cases()
        elapsed_time = time.time() - start_time
        
        # Collect results
        passed_tests = len([r for r in tester.results if r['passed']])
        total_tests = len(tester.results)
        
        self.all_results['test_suites']['edge_cases'] = {
            'name': 'Edge Case Tests',
            'passed': success,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'time': elapsed_time,
            'results': tester.results
        }
        
        self._update_summary(success, total_tests, passed_tests, elapsed_time)
        return success
    
    def run_classification_tests(self):
        """Run classification pipeline tests."""
        print("\n🤖 Running Classification Pipeline Tests...")
        print("="*80)
        
        start_time = time.time()
        tester = ClassificationPipelineTester()
        success = tester.run_all_tests()
        elapsed_time = time.time() - start_time
        
        # Collect results
        passed_tests = len([r for r in tester.results if r['status'] == r['expected']])
        total_tests = len(tester.results)
        
        self.all_results['test_suites']['classification'] = {
            'name': 'Classification Pipeline Tests',
            'passed': success,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'time': elapsed_time,
            'results': tester.results
        }
        
        self._update_summary(success, total_tests, passed_tests, elapsed_time)
        return success
    
    def run_end_to_end_tests(self):
        """Run comprehensive end-to-end tests."""
        print("\n🚀 Running End-to-End Pipeline Tests...")
        print("="*80)
        
        start_time = time.time()
        tester = EndToEndPipelineTester()
        success = tester.run_all_tests()
        elapsed_time = time.time() - start_time
        
        # Collect results
        passed_tests = len([r for r in tester.results if r['status'] == r['expected']])
        total_tests = len(tester.results)
        
        self.all_results['test_suites']['end_to_end'] = {
            'name': 'End-to-End Pipeline Tests',
            'passed': success,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'time': elapsed_time,
            'results': tester.results
        }
        
        self._update_summary(success, total_tests, passed_tests, elapsed_time)
        return success
    
    def _update_summary(self, suite_success, total_tests, passed_tests, elapsed_time):
        """Update overall summary statistics."""
        self.all_results['summary']['total_suites'] += 1
        self.all_results['summary']['total_tests'] += total_tests
        self.all_results['summary']['passed_tests'] += passed_tests
        self.all_results['summary']['total_time'] += elapsed_time
        
        if suite_success:
            self.all_results['summary']['passed_suites'] += 1
    
    def print_final_comprehensive_summary(self):
        """Print the ultimate comprehensive test summary."""
        print(f"\n{'='*100}")
        print("🏆 ULTIMATE MGPT-EVAL PIPELINE TEST RESULTS")
        print(f"{'='*100}")
        
        summary = self.all_results['summary']
        overall_success_rate = (summary['passed_tests'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        suite_success_rate = (summary['passed_suites'] / summary['total_suites'] * 100) if summary['total_suites'] > 0 else 0
        
        print(f"📊 Overall Pipeline Health:")
        print(f"   🧪 Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed ({suite_success_rate:.1f}%)")
        print(f"   🔬 Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed ({overall_success_rate:.1f}%)")
        print(f"   ⏱️  Total execution time: {summary['total_time']:.1f} seconds")
        print(f"   🚀 Average processing rate: {summary['total_tests']/summary['total_time']:.1f} tests/second")
        
        print(f"\n📋 Test Suite Breakdown:")
        suite_order = ['embedding_functionality', 'edge_cases', 'classification', 'end_to_end']
        for suite_key in suite_order:
            if suite_key in self.all_results['test_suites']:
                suite_data = self.all_results['test_suites'][suite_key]
                status = "✅ PASS" if suite_data['passed'] else "❌ FAIL"
                print(f"   {status} {suite_data['name']:30} | "
                      f"{suite_data['passed_tests']:2}/{suite_data['total_tests']:2} tests | "
                      f"{suite_data['success_rate']:5.1f}% | "
                      f"{suite_data['time']:6.1f}s")
        
        # Component health analysis
        print(f"\n🔍 Component Health Analysis:")
        
        # Embedding pipeline health
        embedding_suites = ['embedding_functionality', 'edge_cases']
        embedding_tests = sum([self.all_results['test_suites'][s]['total_tests'] for s in embedding_suites if s in self.all_results['test_suites']])
        embedding_passed = sum([self.all_results['test_suites'][s]['passed_tests'] for s in embedding_suites if s in self.all_results['test_suites']])
        embedding_health = (embedding_passed / embedding_tests * 100) if embedding_tests > 0 else 0
        
        print(f"   🔧 Embedding Pipeline: {embedding_health:.1f}% healthy ({embedding_passed}/{embedding_tests} tests)")
        
        # Classification pipeline health
        if 'classification' in self.all_results['test_suites']:
            class_data = self.all_results['test_suites']['classification']
            class_health = class_data['success_rate']
            print(f"   🤖 Classification Pipeline: {class_health:.1f}% healthy ({class_data['passed_tests']}/{class_data['total_tests']} tests)")
        
        # End-to-end integration health
        if 'end_to_end' in self.all_results['test_suites']:
            e2e_data = self.all_results['test_suites']['end_to_end']
            e2e_health = e2e_data['success_rate']
            print(f"   🚀 End-to-End Integration: {e2e_health:.1f}% healthy ({e2e_data['passed_tests']}/{e2e_data['total_tests']} tests)")
        
        # Performance insights
        successful_suites = [s for s in self.all_results['test_suites'].values() if s['passed']]
        if successful_suites:
            avg_suite_time = sum(s['time'] for s in successful_suites) / len(successful_suites)
            print(f"\n⚡ Performance Insights:")
            print(f"   📈 Average suite execution time: {avg_suite_time:.1f}s")
            print(f"   🎯 Fastest suite: {min(s['time'] for s in successful_suites):.1f}s")
            print(f"   📊 Slowest suite: {max(s['time'] for s in successful_suites):.1f}s")
        
        # Final verdict
        if summary['passed_suites'] == summary['total_suites']:
            print(f"\n🎉 ULTIMATE SUCCESS: ALL PIPELINE COMPONENTS WORKING PERFECTLY!")
            print(f"   ✅ The MGPT-Eval pipeline is production-ready")
            print(f"   ✅ All components are robust and well-tested")
            print(f"   ✅ Edge cases are properly handled")
            print(f"   ✅ End-to-end workflows are seamless")
            print(f"   ✅ Performance is satisfactory")
        else:
            failed_suites = summary['total_suites'] - summary['passed_suites']
            print(f"\n⚠️  ATTENTION REQUIRED: {failed_suites} test suite(s) failed!")
            print(f"   Please review the detailed results above to identify issues.")
            
            # List failed suites
            for suite_key, suite_data in self.all_results['test_suites'].items():
                if not suite_data['passed']:
                    failed_tests = suite_data['total_tests'] - suite_data['passed_tests']
                    print(f"   ❌ {suite_data['name']} - {failed_tests} failed tests")
    
    def save_ultimate_results(self):
        """Save comprehensive results from all test suites."""
        # Save the complete results
        results_file = self.output_dir / "ultimate_pipeline_test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"\n📄 Ultimate test results saved to: {results_file}")
        
        # Create executive summary
        summary_file = self.output_dir / "executive_test_summary.txt"
        summary = self.all_results['summary']
        
        with open(summary_file, 'w') as f:
            f.write("MGPT-EVAL PIPELINE - EXECUTIVE TEST SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Test Date: {self.all_results['timestamp']}\n\n")
            
            f.write(f"OVERALL RESULTS:\n")
            f.write(f"- Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed\n")
            f.write(f"- Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed\n")
            f.write(f"- Success Rate: {(summary['passed_tests']/summary['total_tests']*100):.1f}%\n")
            f.write(f"- Execution Time: {summary['total_time']:.1f} seconds\n\n")
            
            f.write("COMPONENT STATUS:\n")
            for suite_key, suite_data in self.all_results['test_suites'].items():
                status = "PASS" if suite_data['passed'] else "FAIL"
                f.write(f"- {suite_data['name']}: {status} "
                       f"({suite_data['passed_tests']}/{suite_data['total_tests']} tests)\n")
            
            f.write(f"\nRECOMMENDATION:\n")
            if summary['passed_suites'] == summary['total_suites']:
                f.write("✅ PRODUCTION READY - All systems operational\n")
            else:
                f.write("⚠️  REVIEW REQUIRED - Some components need attention\n")
        
        print(f"📄 Executive summary saved to: {summary_file}")
    
    def run_complete_test_suite(self):
        """Run the complete test suite in proper order."""
        print("🚀 MGPT-EVAL COMPLETE PIPELINE TEST SUITE")
        print("="*100)
        print("Testing all components: Embeddings → Classification → End-to-End Integration")
        print(f"🕐 Starting comprehensive tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
        
        suite_start_time = time.time()
        
        # Run all test suites in logical order
        embedding_success = self.run_embedding_tests()
        edge_case_success = self.run_edge_case_tests()
        classification_success = self.run_classification_tests()
        end_to_end_success = self.run_end_to_end_tests()
        
        # Calculate total time
        total_suite_time = time.time() - suite_start_time
        self.all_results['summary']['total_time'] = total_suite_time
        
        # Print ultimate summary
        self.print_final_comprehensive_summary()
        
        print(f"\n🕐 Total test suite execution time: {total_suite_time:.1f} seconds")
        
        # Save comprehensive results
        self.save_ultimate_results()
        
        # Return overall success
        overall_success = embedding_success and edge_case_success and classification_success and end_to_end_success
        
        if overall_success:
            print(f"\n🎯 COMPLETE SUCCESS: All pipeline components tested and verified!")
            return 0
        else:
            print(f"\n💥 Some components failed testing. Check detailed results above.")
            return 1

def main():
    """Main complete test runner entry point."""
    runner = CompletePipelineTestRunner()
    return runner.run_complete_test_suite()

if __name__ == "__main__":
    sys.exit(main())