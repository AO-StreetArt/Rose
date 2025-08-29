import unittest
import cv2
import numpy as np
import os
from rose.processing.feature_detector import FeatureDetector


class TestFeatureDetector(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.detector = FeatureDetector()

    def test_orb_feature_detection_on_synthetic_image(self):
        """Test ORB feature detection on a synthetic test image."""
        # Create a synthetic test image with clear features
        image = np.zeros((100, 100), dtype=np.uint8)
        # Add some geometric shapes that will produce features
        cv2.rectangle(image, (20, 20), (80, 80), 255, 2)
        cv2.circle(image, (50, 50), 15, 255, 2)
        cv2.line(image, (10, 10), (90, 90), 255, 2)

        keypoints, descriptors = self.detector.detect_and_compute(image)

        self.assertIsNotNone(keypoints, "Keypoints should not be None.")
        self.assertGreater(len(keypoints), 0, "No features detected in the test image.")
        self.assertIsNotNone(descriptors, "Descriptors should not be None.")

    def test_feature_detection_with_preprocessing_methods(self):
        """Test that feature detection works correctly with the new preprocessing methods."""
        # Create a synthetic color test image
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add some geometric shapes that will produce features
        cv2.rectangle(color_image, (20, 20), (80, 80), (255, 255, 255), 2)
        cv2.circle(color_image, (50, 50), 15, (255, 255, 255), 2)
        cv2.line(color_image, (10, 10), (90, 90), (255, 255, 255), 2)

        keypoints, descriptors = self.detector.detect_and_compute(color_image)

        self.assertIsNotNone(keypoints, "Keypoints should not be None.")
        self.assertGreater(len(keypoints), 0, "No features detected in the color test image.")
        self.assertIsNotNone(descriptors, "Descriptors should not be None.")

    def test_feature_detection_grayscale_vs_color(self):
        """Test that feature detection produces similar results for grayscale and color inputs."""
        # Create a synthetic test image
        base_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(base_image, (20, 20), (80, 80), 255, 2)
        cv2.circle(base_image, (50, 50), 15, 255, 2)

        # Create color version
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:, :, 0] = base_image  # Blue channel
        color_image[:, :, 1] = base_image  # Green channel
        color_image[:, :, 2] = base_image  # Red channel

        # Test with grayscale
        gray_keypoints, gray_descriptors = self.detector.detect_and_compute(base_image)

        # Test with color (should be converted to grayscale)
        color_keypoints, color_descriptors = self.detector.detect_and_compute(color_image)

        # Both should produce results
        self.assertIsNotNone(gray_keypoints)
        self.assertIsNotNone(color_keypoints)
        self.assertIsNotNone(gray_descriptors)
        self.assertIsNotNone(color_descriptors)

        # Should have similar number of keypoints (allowing for some variation)
        self.assertLessEqual(abs(len(gray_keypoints) - len(color_keypoints)), 5)

    def test_orb_feature_detection_on_square_image(self):
        """Test ORB feature detection on the square test image if it exists."""
        # Path to the test image
        img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')

        if not os.path.exists(img_path):
            self.skipTest(f"Test image not found at {img_path}")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.assertIsNotNone(image, f"Failed to load image at {img_path}")

        keypoints, descriptors = self.detector.detect_and_compute(image)

        self.assertIsNotNone(keypoints, "Keypoints should not be None.")
        self.assertGreater(len(keypoints), 0, "No features detected in the test image.")
        self.assertIsNotNone(descriptors, "Descriptors should not be None.")
