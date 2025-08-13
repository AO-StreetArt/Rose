import unittest
import numpy as np
import cv2
import os
from unittest.mock import Mock, patch

from rose.processing.image_comparator import ImageComparator
from rose.processing.feature_extractor import FeatureExtractor


class TestImageComparator(unittest.TestCase):
    """Test cases for the ImageComparator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparator = ImageComparator()
        self.test_image_path = "tests/temp_test_image.png"

        # Create a simple test image for testing
        self._create_test_image()

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove the temporary test image
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    def _create_test_image(self):
        """Create a simple test image for testing."""
        # Create a 100x100 red square
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # Red color
        cv2.imwrite(self.test_image_path, img)

    def test_init(self):
        """Test ImageComparator initialization."""
        # Test with default feature extractor
        comparator = ImageComparator()
        self.assertIsNotNone(comparator.feature_extractor)
        self.assertIsInstance(comparator.feature_extractor, FeatureExtractor)

        # Test with custom feature extractor
        custom_extractor = FeatureExtractor()
        comparator = ImageComparator(custom_extractor)
        self.assertEqual(comparator.feature_extractor, custom_extractor)

    def test_load_and_preprocess_image_from_path(self):
        """Test loading and preprocessing image from file path."""
        img_array = self.comparator._load_and_preprocess_image(self.test_image_path)

        self.assertEqual(img_array.shape, (1, 224, 224, 3))
        self.assertEqual(img_array.dtype, np.float32)
        self.assertGreaterEqual(np.min(img_array), 0.0)
        self.assertLessEqual(np.max(img_array), 1.0)

    def test_load_and_preprocess_image_from_array(self):
        """Test loading and preprocessing image from numpy array."""
        # Create a test image array
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_array = self.comparator._load_and_preprocess_image(test_img)

        self.assertEqual(img_array.shape, (1, 224, 224, 3))
        self.assertEqual(img_array.dtype, np.float32)
        self.assertGreaterEqual(np.min(img_array), 0.0)
        self.assertLessEqual(np.max(img_array), 1.0)

    def test_load_and_preprocess_image_invalid_path(self):
        """Test loading image from invalid path raises error."""
        with self.assertRaises(ValueError):
            self.comparator._load_and_preprocess_image("nonexistent_image.png")

    @patch('rose.processing.feature_extractor.FeatureExtractor.extract_features')
    def test_extract_features_vgg16(self, mock_extract):
        """Test feature extraction with VGG16 method."""
        # Mock the feature extraction
        mock_features = np.random.rand(1, 1000)
        mock_extract.return_value = mock_features

        img_array = np.random.rand(1, 224, 224, 3)
        features = self.comparator._extract_features(img_array, 'vgg16')

        self.assertEqual(features.shape, (1000,))
        mock_extract.assert_called_once_with(img_array)

    @patch('rose.processing.feature_extractor.FeatureExtractor.extract_features_vit')
    def test_extract_features_vit(self, mock_extract):
        """Test feature extraction with ViT method."""
        # Mock the ViT outputs
        mock_outputs = Mock()
        # Create a tensor-like object that supports mean() method
        mock_tensor = Mock()
        mock_tensor.mean.return_value.squeeze.return_value.numpy.return_value = np.random.rand(768)
        mock_outputs.last_hidden_state = mock_tensor
        mock_extract.return_value = mock_outputs

        img_array = np.random.rand(1, 224, 224, 3)
        features = self.comparator._extract_features(img_array, 'vit')

        self.assertEqual(features.shape, (768,))
        mock_extract.assert_called_once_with(img_array)

    def test_extract_features_invalid_method(self):
        """Test feature extraction with invalid method raises error."""
        img_array = np.random.rand(1, 224, 224, 3)

        with self.assertRaises(ValueError):
            self.comparator._extract_features(img_array, 'invalid_method')

    def test_compare_images_same_image(self):
        """Test comparing an image with itself."""
        # Load the test image
        img_array = self.comparator._load_and_preprocess_image(self.test_image_path)
        
        # Compare with itself
        similarity = self.comparator.compare_images(img_array, img_array, method='vgg16')
        
        # Should be very similar (close to 1.0)
        self.assertGreater(similarity, 0.95)

    def test_compare_images_different_images(self):
        """Test comparing different images."""
        # Create two different test images
        img1 = np.random.rand(1, 224, 224, 3)
        img2 = np.random.rand(1, 224, 224, 3)
        
        # Compare different images
        similarity = self.comparator.compare_images(img1, img2, method='vgg16')
        
        # Should be less similar than same image
        self.assertLess(similarity, 0.95)

    def test_compare_images_with_different_methods(self):
        """Test comparing images with different feature extraction methods."""
        img1 = np.random.rand(1, 224, 224, 3)
        img2 = np.random.rand(1, 224, 224, 3)
        
        # Test VGG16 method
        similarity_vgg16 = self.comparator.compare_images(img1, img2, method='vgg16')
        self.assertIsInstance(similarity_vgg16, float)
        self.assertGreaterEqual(similarity_vgg16, 0.0)
        self.assertLessEqual(similarity_vgg16, 1.0)
        
        # Test ViT method
        similarity_vit = self.comparator.compare_images(img1, img2, method='vit')
        self.assertIsInstance(similarity_vit, float)
        self.assertGreaterEqual(similarity_vit, 0.0)
        self.assertLessEqual(similarity_vit, 1.0)

    def test_compare_images_from_paths(self):
        """Test comparing images from file paths."""
        # Create a second test image
        img2_path = "tests/temp_test_image2.png"
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2[:, :] = [0, 255, 0]  # Green color
        cv2.imwrite(img2_path, img2)
        
        try:
            # Compare images from paths
            similarity = self.comparator.compare_images_from_paths(
                self.test_image_path, img2_path, method='vgg16'
            )
            
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
        finally:
            # Clean up
            if os.path.exists(img2_path):
                os.remove(img2_path)

    def test_compare_images_from_paths_invalid_path(self):
        """Test comparing images with invalid path raises error."""
        with self.assertRaises(ValueError):
            self.comparator.compare_images_from_paths(
                "nonexistent1.png", "nonexistent2.png", method='vgg16'
            )

    def test_compare_images_batch_processing(self):
        """Test comparing multiple images in batch."""
        # Create multiple test images
        images = [np.random.rand(1, 224, 224, 3) for _ in range(3)]
        
        # Compare all pairs
        similarities = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                similarity = self.comparator.compare_images(images[i], images[j], method='vgg16')
                similarities.append(similarity)
                self.assertIsInstance(similarity, float)
                self.assertGreaterEqual(similarity, 0.0)
                self.assertLessEqual(similarity, 1.0)
        
        # Should have compared 3 pairs
        self.assertEqual(len(similarities), 3)

    def test_compare_images_edge_cases(self):
        """Test edge cases for image comparison."""
        # Test with very small images
        small_img1 = np.random.rand(1, 10, 10, 3)
        small_img2 = np.random.rand(1, 10, 10, 3)
        
        similarity = self.comparator.compare_images(small_img1, small_img2, method='vgg16')
        self.assertIsInstance(similarity, float)
        
        # Test with very large images
        large_img1 = np.random.rand(1, 512, 512, 3)
        large_img2 = np.random.rand(1, 512, 512, 3)
        
        similarity = self.comparator.compare_images(large_img1, large_img2, method='vgg16')
        self.assertIsInstance(similarity, float)

    def test_compare_images_performance(self):
        """Test performance of image comparison."""
        import time
        
        # Create test images
        img1 = np.random.rand(1, 224, 224, 3)
        img2 = np.random.rand(1, 224, 224, 3)
        
        # Measure time for VGG16 comparison
        start_time = time.time()
        similarity_vgg16 = self.comparator.compare_images(img1, img2, method='vgg16')
        vgg16_time = time.time() - start_time
        
        # Measure time for ViT comparison
        start_time = time.time()
        similarity_vit = self.comparator.compare_images(img1, img2, method='vit')
        vit_time = time.time() - start_time
        
        # Both should complete in reasonable time (less than 10 seconds)
        self.assertLess(vgg16_time, 10.0)
        self.assertLess(vit_time, 10.0)
        
        # Both should return valid similarity scores
        self.assertIsInstance(similarity_vgg16, float)
        self.assertIsInstance(similarity_vit, float)
