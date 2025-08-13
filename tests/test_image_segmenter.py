import unittest
import os
import numpy as np
from PIL import Image
from rose.processing.image_segmenter import ImageSegmenter


class TestImageSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = ImageSegmenter(device="cpu")

    def test_image_segmenter_on_square_image(self):
        # Path to the test image
        img_path = os.path.join(os.path.dirname(__file__), "squareTestImage.png")
        self.assertTrue(os.path.exists(img_path), f"Test image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")

        prompts = ["square"]
        masks = self.segmenter.segment(image, prompts)

        # Check output shape and type
        self.assertIsInstance(masks, np.ndarray, "Output should be a numpy array")
        self.assertEqual(masks.shape[0], 1, f"Expected 1 mask, got {masks.shape[0]}")
        self.assertEqual(masks.shape[1], image.height, f"Mask height {masks.shape[1]} does not match image height {image.height}")
        self.assertEqual(masks.shape[2], image.width, f"Mask width {masks.shape[2]} does not match image width {image.width}")
        self.assertTrue((0.0 <= masks).all() and (masks <= 1.0).all(), "Mask values should be in [0, 1]")
