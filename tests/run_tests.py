#!/usr/bin/env python3
"""
Test runner script for binary classifier pipeline.

This script runs comprehensive tests and provides detailed reporting.
It also manages the fake API server lifecycle during testing.
"""

import sys
import subprocess
import argparse
import time
import requests
from pathlib import Path

def check_fake_api(port=8001, timeout=30):
    """Check if fake API server is running."""
    url = f"http://localhost:{port}/health"
    for _ in range(timeout):
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False

def run_tests(args):
    """Run the test suite."""
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=models",
            "--cov=pipelines", 
            "--cov=evaluation",
            "--cov=utils",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    
    # Add specific test selection
    if args.test:
        cmd.extend(["-k", args.test])
    
    # Add markers
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    # Add timeout
    if args.timeout:
        cmd.extend(["--timeout", str(args.timeout)])
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add other options
    if args.exitfirst:
        cmd.append("-x")
    
    if args.capture == "no":
        cmd.append("-s")
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=test_dir.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run binary classifier pipeline tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with coverage
  python run_tests.py --coverage
  
  # Run specific test
  python run_tests.py --test test_data_models
  
  # Run tests with specific markers
  python run_tests.py --markers "not slow"
  
  # Run tests in parallel
  python run_tests.py --parallel 4
  
  # Quick smoke test
  python run_tests.py --test "test_default" --quiet
        """
    )
    
    # Test selection
    parser.add_argument(
        "--test", "-k",
        help="Run tests matching the given substring expression"
    )
    parser.add_argument(
        "--markers", "-m",
        help="Run tests matching given mark expression"
    )
    
    # Output control
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Quiet output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--capture",
        choices=["yes", "no"],
        default="yes",
        help="Capture stdout/stderr"
    )
    
    # Execution control
    parser.add_argument(
        "--exitfirst", "-x",
        action="store_true", 
        help="Exit on first test failure"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for each test in seconds"
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    # API management
    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Only check if fake API is running"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8001,
        help="Fake API server port"
    )
    
    args = parser.parse_args()
    
    # Check if just checking API
    if args.check_api:
        if check_fake_api(args.api_port):
            print(f"✓ Fake API server is running on port {args.api_port}")
            return 0
        else:
            print(f"✗ Fake API server is not running on port {args.api_port}")
            return 1
    
    # Check that fake API is running before tests
    print("Checking fake API server...")
    if not check_fake_api(args.api_port):
        print(f"Warning: Fake API server not detected on port {args.api_port}")
        print("Some integration tests may fail.")
        print("To start the fake API server:")
        print(f"  python ../fake_api_server.py --port {args.api_port}")
        print()
    else:
        print(f"✓ Fake API server is running on port {args.api_port}")
        print()
    
    # Run tests
    return run_tests(args)

if __name__ == "__main__":
    sys.exit(main())