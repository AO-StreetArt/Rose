import unittest
import numpy as np
import tempfile
from PIL import Image
import os
import cv2
from rose.preprocessing.image_utils import ImagePreprocessor


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
        self.image_preprocessor = ImagePreprocessor()

    def test_image_preprocessor_init(self):
        self.assertIsNotNone(self.image_preprocessor)

    def test_load_and_preprocess_image(self):
        # Create a temporary grayscale image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('L', (300, 300), color=128)
            img.save(tmp.name)
            tmp_path = tmp.name
        try:
            arr = ImagePreprocessor.load_and_preprocess_image(tmp_path)
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, (1, 224, 224, 3))
        finally:
            os.remove(tmp_path)

    def test_ensure_rgb_pil_image(self):
        # Grayscale numpy array
        gray_np = np.ones((100, 100), dtype=np.uint8) * 128
        rgb_img = ImagePreprocessor.ensure_rgb_pil_image(gray_np)
        self.assertIsInstance(rgb_img, Image.Image)
        self.assertEqual(rgb_img.mode, 'RGB')

        # Grayscale PIL Image
        gray_pil = Image.new('L', (100, 100), color=128)
        rgb_img2 = ImagePreprocessor.ensure_rgb_pil_image(gray_pil)
        self.assertIsInstance(rgb_img2, Image.Image)
        self.assertEqual(rgb_img2.mode, 'RGB')

        # Already RGB PIL Image
        rgb_pil = Image.new('RGB', (100, 100), color=128)
        rgb_img3 = ImagePreprocessor.ensure_rgb_pil_image(rgb_pil)
        self.assertIsInstance(rgb_img3, Image.Image)
        self.assertEqual(rgb_img3.mode, 'RGB')

    def test_ensure_bgr_image(self):
        """Test the ensure_bgr_image method with different input types."""
        # Test with grayscale image (2D array)
        gray_image = np.ones((100, 100), dtype=np.uint8) * 128
        bgr_image = ImagePreprocessor.ensure_bgr_image(gray_image)
        self.assertIsInstance(bgr_image, np.ndarray)
        self.assertEqual(bgr_image.shape, (100, 100, 3))
        self.assertEqual(bgr_image.dtype, np.uint8)

        # Test with already BGR image (3D array)
        bgr_input = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bgr_output = ImagePreprocessor.ensure_bgr_image(bgr_input)
        self.assertIsInstance(bgr_output, np.ndarray)
        self.assertEqual(bgr_output.shape, (100, 100, 3))
        self.assertEqual(bgr_output.dtype, np.uint8)
        # Should be the same array (no conversion needed)
        self.assertTrue(np.array_equal(bgr_input, bgr_output))

    def test_create_blob_for_hed(self):
        """Test the create_blob_for_hed method."""
        # Create a test BGR image
        bgr_image = np.ones((100, 150, 3), dtype=np.uint8) * 128

        # Create blob
        blob = ImagePreprocessor.create_blob_for_hed(bgr_image)

        # Verify blob properties
        self.assertIsInstance(blob, np.ndarray)
        self.assertEqual(blob.shape, (1, 3, 100, 150))  # (batch, channels, height, width)
        self.assertEqual(blob.dtype, np.float32)

        # Test with different image dimensions
        bgr_image_2 = np.ones((200, 300, 3), dtype=np.uint8) * 64
        blob_2 = ImagePreprocessor.create_blob_for_hed(bgr_image_2)
        self.assertEqual(blob_2.shape, (1, 3, 200, 300))

    def test_create_blob_for_hed_with_mean_subtraction(self):
        """Test that the blob creation properly applies mean subtraction."""
        # Create a test BGR image with known values
        bgr_image = np.ones((50, 50, 3), dtype=np.uint8) * 128

        # Create blob
        blob = ImagePreprocessor.create_blob_for_hed(bgr_image)

        # The mean values used in the method are (104.00698793, 116.66876762, 122.67891434)
        # So the expected values should be approximately 128 - mean for each channel
        expected_b = 128 - 104.00698793
        expected_g = 128 - 116.66876762
        expected_r = 128 - 122.67891434

        # Check that the blob values are close to expected (allowing for floating point precision)
        self.assertAlmostEqual(blob[0, 0, 0, 0], expected_b, places=5)
        self.assertAlmostEqual(blob[0, 1, 0, 0], expected_g, places=5)
        self.assertAlmostEqual(blob[0, 2, 0, 0], expected_r, places=5)

    def test_ensure_bgr_image_edge_cases(self):
        """Test edge cases for ensure_bgr_image method."""
        # Test with very small image
        small_image = np.ones((1, 1), dtype=np.uint8) * 255
        bgr_small = ImagePreprocessor.ensure_bgr_image(small_image)
        self.assertEqual(bgr_small.shape, (1, 1, 3))
        self.assertEqual(bgr_small.dtype, np.uint8)

        # Test with very large image
        large_image = np.ones((1000, 1000), dtype=np.uint8) * 100
        bgr_large = ImagePreprocessor.ensure_bgr_image(large_image)
        self.assertEqual(bgr_large.shape, (1000, 1000, 3))
        self.assertEqual(bgr_large.dtype, np.uint8)

    def test_ensure_rgb_pil_image_edge_cases(self):
        """Test edge cases for ensure_rgb_pil_image method."""
        # Test with very small image
        small_image = np.ones((1, 1), dtype=np.uint8) * 255
        rgb_small = ImagePreprocessor.ensure_rgb_pil_image(small_image)
        self.assertEqual(rgb_small.size, (1, 1))
        self.assertEqual(rgb_small.mode, 'RGB')

        # Test with very large image
        large_image = np.ones((1000, 1000), dtype=np.uint8) * 100
        rgb_large = ImagePreprocessor.ensure_rgb_pil_image(large_image)
        self.assertEqual(rgb_large.size, (1000, 1000))
        self.assertEqual(rgb_large.mode, 'RGB')

    def test_load_and_preprocess_image_edge_cases(self):
        """Test edge cases for load_and_preprocess_image method."""
        # Test with very small image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('L', (10, 10), color=128)
            img.save(tmp.name)
            tmp_path = tmp.name
        try:
            arr = ImagePreprocessor.load_and_preprocess_image(tmp_path)
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, (1, 224, 224, 3))  # Should still be resized to 224x224
        finally:
            os.remove(tmp_path)

        # Test with very large image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.new('L', (1000, 1000), color=128)
            img.save(tmp.name)
            tmp_path = tmp.name
        try:
            arr = ImagePreprocessor.load_and_preprocess_image(tmp_path)
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, (1, 224, 224, 3))  # Should be resized to 224x224
        finally:
            os.remove(tmp_path)

    def test_ensure_bgr_image_with_different_dtypes(self):
        """Test ensure_bgr_image with different numpy dtypes."""
        # Test with float32
        float_image = np.ones((50, 50), dtype=np.float32) * 0.5
        bgr_float = ImagePreprocessor.ensure_bgr_image(float_image)
        self.assertEqual(bgr_float.dtype, np.uint8)  # Should be converted to uint8

        # Test with int16
        int16_image = np.ones((50, 50), dtype=np.int16) * 128
        bgr_int16 = ImagePreprocessor.ensure_bgr_image(int16_image)
        self.assertEqual(bgr_int16.dtype, np.uint8)  # Should be converted to uint8

        # Test with bool
        bool_image = np.ones((50, 50), dtype=bool)
        bgr_bool = ImagePreprocessor.ensure_bgr_image(bool_image)
        self.assertEqual(bgr_bool.dtype, np.uint8)  # Should be converted to uint8

    def test_ensure_rgb_pil_image_with_different_dtypes(self):
        """Test ensure_rgb_pil_image with different numpy dtypes."""
        # Test with float32
        float_image = np.ones((50, 50), dtype=np.float32) * 0.5
        rgb_float = ImagePreprocessor.ensure_rgb_pil_image(float_image)
        self.assertEqual(rgb_float.mode, 'RGB')
        self.assertEqual(rgb_float.size, (50, 50))

        # Test with int16
        int16_image = np.ones((50, 50), dtype=np.int16) * 128
        rgb_int16 = ImagePreprocessor.ensure_rgb_pil_image(int16_image)
        self.assertEqual(rgb_int16.mode, 'RGB')
        self.assertEqual(rgb_int16.size, (50, 50))

        # Test with bool
        bool_image = np.ones((50, 50), dtype=bool)
        rgb_bool = ImagePreprocessor.ensure_rgb_pil_image(bool_image)
        self.assertEqual(rgb_bool.mode, 'RGB')
        self.assertEqual(rgb_bool.size, (50, 50))
