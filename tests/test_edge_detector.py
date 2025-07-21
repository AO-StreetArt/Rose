import numpy as np
import cv2
import unittest
import os
from rose.processing.edge_detector import EdgeDetector

HED_PROTOTXT = 'models/deploy.prototxt'
HED_CAFFEMODEL = 'models/hed_pretrained_bsds.caffemodel'

class TestEdgeDetector(unittest.TestCase):
    def setUp(self):
        self.detector = EdgeDetector()

    def test_detects_edges_on_simple_shape(self):
        # Create a black image with a white square in the center
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
        # Run Canny edge detection
        edges = self.detector.canny(img, threshold1=50, threshold2=150)
        # There should be nonzero edges around the square
        self.assertTrue(np.count_nonzero(edges) > 0)
        # The corners should have strong edge responses
        self.assertTrue(edges[30, 30] > 0)
        self.assertTrue(edges[30, 70] > 0)
        self.assertTrue(edges[70, 30] > 0)
        self.assertTrue(edges[70, 70] > 0)

    def test_canny_on_squareTestImage_and_save_output(self):
        # Load the test image
        img = cv2.imread('tests/squareTestImage.png', cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(img, "Failed to load tests/squareTestImage.png")
        # Run Canny edge detection
        edges = self.detector.canny(img, threshold1=50, threshold2=150)
        # Save the output for manual review
        output_path = 'tests/edge_output_squareTestImage.png'
        cv2.imwrite(output_path, edges)
        # Check that the file was saved
        self.assertTrue(cv2.imread(output_path, cv2.IMREAD_GRAYSCALE) is not None)

    @unittest.skipUnless(os.path.exists(HED_PROTOTXT) and os.path.exists(HED_CAFFEMODEL),
                         "HED model files not found.")
    def test_hed_on_squareTestImage_and_save_output(self):
        # Load the test image
        img = cv2.imread('tests/squareTestImage.png', cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, "Failed to load tests/squareTestImage.png")
        # Initialize EdgeDetector with HED model
        detector = EdgeDetector(HED_PROTOTXT, HED_CAFFEMODEL)
        # Run HED edge detection
        edges = detector.hed(img)
        # Save the output for manual review
        output_path = 'tests/hed_output_squareTestImage.png'
        cv2.imwrite(output_path, edges)
        # Check that the file was saved
        self.assertTrue(cv2.imread(output_path, cv2.IMREAD_GRAYSCALE) is not None)

if __name__ == "__main__":
    unittest.main() 