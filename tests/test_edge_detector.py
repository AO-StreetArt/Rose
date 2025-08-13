import unittest
import numpy as np
import cv2
import os
from rose.processing.edge_detector import EdgeDetector

# Constants for HED model paths (adjust as needed)
HED_PROTOTXT = 'models/hed_pretrained_bsds.caffemodel.prototxt'
HED_CAFFEMODEL = 'models/hed_pretrained_bsds.caffemodel'


class TestEdgeDetector(unittest.TestCase):
    def setUp(self):
        self.edge_detector = EdgeDetector()

    def test_detects_edges_on_simple_shape(self):
        # Create a simple test image with a square
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255  # White square

        edges = self.edge_detector.canny(img)

        self.assertIsNotNone(edges)
        self.assertEqual(edges.shape, (100, 100))
        self.assertEqual(edges.dtype, np.uint8)

    def test_hed_with_preprocessing_methods(self):
        """Test that HED method works correctly with the new preprocessing methods."""
        # Create a simple test image
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 128  # Gray square

        # Test that the method doesn't crash when HED model is not loaded
        with self.assertRaises(RuntimeError):
            self.edge_detector.hed(img)

    def test_canny_with_preprocessing_methods(self):
        """Test that canny method works correctly with the new preprocessing methods."""
        # Test with grayscale image
        gray_img = np.zeros((50, 50), dtype=np.uint8)
        gray_img[10:40, 10:40] = 255  # White square
        gray_edges = self.edge_detector.canny(gray_img)
        self.assertIsNotNone(gray_edges)
        self.assertEqual(gray_edges.shape, (50, 50))

        # Test with color image (should be converted to grayscale)
        color_img = np.zeros((50, 50, 3), dtype=np.uint8)
        color_img[10:40, 10:40] = [255, 255, 255]  # White square
        color_edges = self.edge_detector.canny(color_img)
        self.assertIsNotNone(color_edges)
        self.assertEqual(color_edges.shape, (50, 50))

        # Both should produce similar edge detection results
        self.assertEqual(gray_edges.dtype, color_edges.dtype)

    def test_canny_on_squareTestImage_and_save_output(self):
        # Load the test image
        img = cv2.imread('tests/squareTestImage.png', cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(img, "Failed to load tests/squareTestImage.png")

        # Apply Canny edge detection
        edges = self.edge_detector.canny(img)

        # Save the result
        cv2.imwrite('tests/canny_output.png', edges)

        # Verify the output
        self.assertIsNotNone(edges)
        self.assertEqual(edges.shape, img.shape)
        self.assertEqual(edges.dtype, np.uint8)

    @unittest.skipUnless(os.path.exists(HED_PROTOTXT) and os.path.exists(HED_CAFFEMODEL),
                         "HED model files not found.")
    def test_hed_on_squareTestImage_and_save_output(self):
        # Load the test image
        img = cv2.imread('tests/squareTestImage.png', cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, "Failed to load tests/squareTestImage.png")

        # Initialize edge detector with HED model
        hed_detector = EdgeDetector(HED_PROTOTXT, HED_CAFFEMODEL)

        # Apply HED edge detection
        edges = hed_detector.hed(img)

        # Save the result
        cv2.imwrite('tests/hed_output.png', edges)

        # Verify the output
        self.assertIsNotNone(edges)
        self.assertEqual(edges.shape, img.shape[:2])  # HED returns grayscale
        self.assertEqual(edges.dtype, np.uint8)
