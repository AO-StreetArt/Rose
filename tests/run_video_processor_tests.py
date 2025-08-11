#!/usr/bin/env python3
"""
Test runner for VideoProcessor tests.

This script runs the VideoProcessor unit tests and provides a summary of results.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_video_processor_tests():
    """Run the VideoProcessor tests and return results."""
    print("Running VideoProcessor Tests")
    print("=" * 50)

    # Get the project root
    project_root = Path(__file__).parent.parent

    # Run the tests
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/test_video_processor.py",
            "-v", "--tb=short"
        ], cwd=project_root, capture_output=True, text=True)

        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"Running specific test: {test_name}")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            f"tests/test_video_processor.py::{test_name}",
            "-v"
        ], cwd=project_root, capture_output=True, text=True)

        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"Error running test: {e}")
        return False

def list_available_tests():
    """List all available VideoProcessor tests."""
    print("Available VideoProcessor Tests:")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/test_video_processor.py",
            "--collect-only", "-q"
        ], cwd=project_root, capture_output=True, text=True)

        print(result.stdout)

    except Exception as e:
        print(f"Error listing tests: {e}")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            list_available_tests()
        elif command == "test":
            if len(sys.argv) > 2:
                test_name = sys.argv[2]
                success = run_specific_test(test_name)
                sys.exit(0 if success else 1)
            else:
                print("Usage: python run_video_processor_tests.py test <test_name>")
                print("Example: python run_video_processor_tests.py test TestVideoProcessor::test_video_processor_init_default")
                sys.exit(1)
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  list - List all available tests")
            print("  test <test_name> - Run a specific test")
            print("  (no args) - Run all VideoProcessor tests")
            sys.exit(1)
    else:
        # Run all tests
        success = run_video_processor_tests()

        print("\n" + "=" * 50)
        if success:
            print("✅ All VideoProcessor tests passed!")
        else:
            print("❌ Some VideoProcessor tests failed!")

        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()