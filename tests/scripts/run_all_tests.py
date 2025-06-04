#!/usr/bin/env python3
"""
Master test runner for the embedding pipeline.
Runs all test suites and provides comprehensive reporting.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from test_embedding_pipeline import EmbeddingPipelineTester
from test_edge_cases import EdgeCaseTester

class MasterTestRunner:
    """Master test runner for all embedding pipeline tests."""
    
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
                'passed_tests': 0
            }
        }
    
    def run_functionality_tests(self):
        """Run basic functionality tests."""
        print("🚀 Running Functionality Tests...")
        print("="*60)
        
        tester = EmbeddingPipelineTester()
        success = tester.run_all_tests()
        
        # Collect results
        passed_tests = len([r for r in tester.results if r['status'] == r['expected']])
        total_tests = len(tester.results)
        
        self.all_results['test_suites']['functionality'] = {
            'name': 'Functionality Tests',
            'passed': success,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'results': tester.results
        }
        
        self.all_results['summary']['total_suites'] += 1
        self.all_results['summary']['total_tests'] += total_tests
        self.all_results['summary']['passed_tests'] += passed_tests
        if success:
            self.all_results['summary']['passed_suites'] += 1
        
        return success
    
    def run_edge_case_tests(self):
        """Run edge case tests."""
        print("\n🧪 Running Edge Case Tests...")
        print("="*60)
        
        tester = EdgeCaseTester()
        success = tester.run_all_edge_cases()
        
        # Collect results
        passed_tests = len([r for r in tester.results if r['passed']])
        total_tests = len(tester.results)
        
        self.all_results['test_suites']['edge_cases'] = {
            'name': 'Edge Case Tests',
            'passed': success,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'results': tester.results
        }
        
        self.all_results['summary']['total_suites'] += 1
        self.all_results['summary']['total_tests'] += total_tests
        self.all_results['summary']['passed_tests'] += passed_tests
        if success:
            self.all_results['summary']['passed_suites'] += 1
        
        return success
    
    def print_final_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*80)
        
        summary = self.all_results['summary']
        overall_success_rate = (summary['passed_tests'] / summary['total_tests'] * 100) if summary['total_tests'] > 0 else 0
        
        print(f"📊 Overall Results:")
        print(f"   🧪 Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed")
        print(f"   🔬 Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"   📈 Overall Success Rate: {overall_success_rate:.1f}%")
        
        print(f"\n📋 Test Suite Breakdown:")
        for suite_key, suite_data in self.all_results['test_suites'].items():
            status = "✅ PASS" if suite_data['passed'] else "❌ FAIL"
            print(f"   {status} {suite_data['name']:20} | "
                  f"{suite_data['passed_tests']:2}/{suite_data['total_tests']:2} tests | "
                  f"{suite_data['success_rate']:5.1f}% success")
        
        # Performance insights
        functionality_results = self.all_results['test_suites'].get('functionality', {}).get('results', [])
        successful_perf_tests = [r for r in functionality_results 
                               if r['status'] == 'PASS' and r['expected'] == 'PASS' and r.get('throughput', 0) > 0]
        
        if successful_perf_tests:
            avg_throughput = sum(r['throughput'] for r in successful_perf_tests) / len(successful_perf_tests)
            max_throughput = max(r['throughput'] for r in successful_perf_tests)
            print(f"\n🚀 Performance Insights:")
            print(f"   📊 Average throughput: {avg_throughput:.1f} samples/second")
            print(f"   ⚡ Peak throughput: {max_throughput:.1f} samples/second")
        
        # Final verdict
        if summary['passed_suites'] == summary['total_suites']:
            print(f"\n🎉 SUCCESS: All test suites passed!")
            print(f"   The embedding pipeline is working correctly and handles all edge cases properly.")
            print(f"   ✅ Batching logic is robust")
            print(f"   ✅ Error handling is comprehensive")
            print(f"   ✅ Performance is satisfactory")
        else:
            failed_suites = summary['total_suites'] - summary['passed_suites']
            print(f"\n⚠️  WARNING: {failed_suites} test suite(s) failed!")
            print(f"   Please review the detailed results above to identify issues.")
            
            # List failed suites
            for suite_key, suite_data in self.all_results['test_suites'].items():
                if not suite_data['passed']:
                    print(f"   ❌ {suite_data['name']} - {suite_data['total_tests'] - suite_data['passed_tests']} failed tests")
    
    def save_comprehensive_results(self):
        """Save comprehensive test results."""
        results_file = self.output_dir / "comprehensive_test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"\n📄 Comprehensive results saved to: {results_file}")
        
        # Also create a summary report
        summary_file = self.output_dir / "test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("EMBEDDING PIPELINE TEST SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Timestamp: {self.all_results['timestamp']}\n\n")
            
            summary = self.all_results['summary']
            f.write(f"Overall Results:\n")
            f.write(f"- Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed\n")
            f.write(f"- Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed\n")
            f.write(f"- Success Rate: {(summary['passed_tests']/summary['total_tests']*100):.1f}%\n\n")
            
            f.write("Test Suite Results:\n")
            for suite_key, suite_data in self.all_results['test_suites'].items():
                status = "PASS" if suite_data['passed'] else "FAIL"
                f.write(f"- {suite_data['name']}: {status} "
                       f"({suite_data['passed_tests']}/{suite_data['total_tests']} tests)\n")
        
        print(f"📄 Summary report saved to: {summary_file}")
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🚀 EMBEDDING PIPELINE COMPREHENSIVE TEST RUNNER")
        print("="*80)
        print(f"🕐 Starting tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        start_time = time.time()
        
        # Run test suites
        functionality_success = self.run_functionality_tests()
        edge_case_success = self.run_edge_case_tests()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Print final summary
        self.print_final_summary()
        
        print(f"\n⏱️  Total test time: {total_time:.1f} seconds")
        
        # Save results
        self.save_comprehensive_results()
        
        # Return overall success
        return functionality_success and edge_case_success

def main():
    """Main test runner entry point."""
    runner = MasterTestRunner()
    success = runner.run_all_tests()
    
    if success:
        print(f"\n🎯 All tests completed successfully!")
        return 0
    else:
        print(f"\n💥 Some tests failed. Check the detailed results above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())