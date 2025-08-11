#!/usr/bin/env python3
"""
Test runner for depth estimation module.
"""

import unittest
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_depth_estimator import TestDepthEstimator, TestDepthEstimatorIntegration


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add unit tests
    test_suite.addTest(unittest.makeSuite(TestDepthEstimator))

    # Add integration tests (optional - can be skipped)
    # test_suite.addTest(unittest.makeSuite(TestDepthEstimatorIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    print("Running depth estimation tests...")
    exit_code = run_tests()
    sys.exit(exit_code)