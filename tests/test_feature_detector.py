import cv2
import numpy as np
import os
import pytest
from rose.processing.feature_detector import FeatureDetector


def test_orb_feature_detection_on_synthetic_image():
    """Test ORB feature detection on a synthetic test image."""
    # Create a synthetic test image with clear features
    image = np.zeros((100, 100), dtype=np.uint8)
    # Add some geometric shapes that will produce features
    cv2.rectangle(image, (20, 20), (80, 80), 255, 2)
    cv2.circle(image, (50, 50), 15, 255, 2)
    cv2.line(image, (10, 10), (90, 90), 255, 2)

    detector = FeatureDetector()
    keypoints, descriptors = detector.detect_and_compute(image)

    assert keypoints is not None, "Keypoints should not be None."
    assert len(keypoints) > 0, "No features detected in the test image."
    assert descriptors is not None, "Descriptors should not be None."


def test_feature_detection_with_preprocessing_methods():
    """Test that feature detection works correctly with the new preprocessing methods."""
    # Create a synthetic color test image
    color_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add some geometric shapes that will produce features
    cv2.rectangle(color_image, (20, 20), (80, 80), (255, 255, 255), 2)
    cv2.circle(color_image, (50, 50), 15, (255, 255, 255), 2)
    cv2.line(color_image, (10, 10), (90, 90), (255, 255, 255), 2)

    detector = FeatureDetector()
    keypoints, descriptors = detector.detect_and_compute(color_image)

    assert keypoints is not None, "Keypoints should not be None."
    assert len(keypoints) > 0, "No features detected in the color test image."
    assert descriptors is not None, "Descriptors should not be None."


def test_feature_detection_grayscale_vs_color():
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

    detector = FeatureDetector()

    # Test with grayscale
    gray_keypoints, gray_descriptors = detector.detect_and_compute(base_image)

    # Test with color (should be converted to grayscale)
    color_keypoints, color_descriptors = detector.detect_and_compute(color_image)

    # Both should produce results
    assert gray_keypoints is not None
    assert color_keypoints is not None
    assert gray_descriptors is not None
    assert color_descriptors is not None

    # Should have similar number of keypoints (allowing for some variation)
    assert abs(len(gray_keypoints) - len(color_keypoints)) <= 5


def test_orb_feature_detection_on_square_image():
    """Test ORB feature detection on the square test image if it exists."""
    # Path to the test image
    img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found at {img_path}")

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert image is not None, f"Failed to load image at {img_path}"

    detector = FeatureDetector()
    keypoints, descriptors = detector.detect_and_compute(image)

    assert keypoints is not None, "Keypoints should not be None."
    assert len(keypoints) > 0, "No features detected in the test image."
    assert descriptors is not None, "Descriptors should not be None."
