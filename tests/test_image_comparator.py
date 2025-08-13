import pytest
import numpy as np
import cv2
import os
from unittest.mock import Mock, patch

from rose.processing.image_comparator import ImageComparator
from rose.processing.feature_extractor import FeatureExtractor


class TestImageComparator:
    """Test cases for the ImageComparator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = ImageComparator()
        self.test_image_path = "tests/temp_test_image.png"

        # Create a simple test image for testing
        self._create_test_image()

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
        assert comparator.feature_extractor is not None
        assert isinstance(comparator.feature_extractor, FeatureExtractor)

        # Test with custom feature extractor
        custom_extractor = FeatureExtractor()
        comparator = ImageComparator(custom_extractor)
        assert comparator.feature_extractor is custom_extractor

    def test_load_and_preprocess_image_from_path(self):
        """Test loading and preprocessing image from file path."""
        img_array = self.comparator._load_and_preprocess_image(self.test_image_path)

        assert img_array.shape == (1, 224, 224, 3)
        assert img_array.dtype == np.float32
        assert np.min(img_array) >= 0.0
        assert np.max(img_array) <= 1.0

    def test_load_and_preprocess_image_from_array(self):
        """Test loading and preprocessing image from numpy array."""
        # Create a test image array
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_array = self.comparator._load_and_preprocess_image(test_img)

        assert img_array.shape == (1, 224, 224, 3)
        assert img_array.dtype == np.float32
        assert np.min(img_array) >= 0.0
        assert np.max(img_array) <= 1.0

    def test_load_and_preprocess_image_invalid_path(self):
        """Test loading image from invalid path raises error."""
        with pytest.raises(ValueError, match="Could not load image from path"):
            self.comparator._load_and_preprocess_image("nonexistent_image.png")

    @patch('rose.processing.feature_extractor.FeatureExtractor.extract_features')
    def test_extract_features_vgg16(self, mock_extract):
        """Test feature extraction with VGG16 method."""
        # Mock the feature extraction
        mock_features = np.random.rand(1, 1000)
        mock_extract.return_value = mock_features

        img_array = np.random.rand(1, 224, 224, 3)
        features = self.comparator._extract_features(img_array, 'vgg16')

        assert features.shape == (1000,)
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

        assert features.shape == (768,)
        mock_extract.assert_called_once_with(img_array)

    def test_extract_features_invalid_method(self):
        """Test feature extraction with invalid method raises error."""
        img_array = np.random.rand(1, 224, 224, 3)

        with pytest.raises(ValueError, match="Unsupported method"):
            self.comparator._extract_features(img_array, 'invalid_method')

    def test_calculate_similarity(self):
        """Test cosine similarity calculation."""
        # Test with identical vectors
        features1 = np.array([1, 2, 3, 4, 5])
        features2 = np.array([1, 2, 3, 4, 5])

        similarity = self.comparator._calculate_similarity(features1, features2)
        assert similarity == pytest.approx(1.0, abs=1e-6)

        # Test with orthogonal vectors
        features1 = np.array([1, 0, 0])
        features2 = np.array([0, 1, 0])

        similarity = self.comparator._calculate_similarity(features1, features2)
        assert similarity == pytest.approx(0.0, abs=1e-6)

        # Test with opposite vectors
        features1 = np.array([1, 2, 3])
        features2 = np.array([-1, -2, -3])

        similarity = self.comparator._calculate_similarity(features1, features2)
        assert similarity == pytest.approx(-1.0, abs=1e-6)

    def test_calculate_similarity_without_normalization(self):
        """Test cosine similarity calculation without normalization."""
        features1 = np.array([1, 2, 3])
        features2 = np.array([4, 5, 6])  # Different vector (not scaled)

        similarity = self.comparator._calculate_similarity(features1, features2, normalize=False)
        # Should not be exactly 1.0 for different vectors
        assert similarity != pytest.approx(1.0, abs=1e-6)

    @patch('rose.processing.image_comparator.ImageComparator._extract_features')
    def test_compare_images(self, mock_extract):
        """Test comparing two images."""
        # Mock feature extraction
        mock_extract.side_effect = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5])
        ]

        result = self.comparator.compare_images(
            self.test_image_path,
            self.test_image_path,
            method='vgg16'
        )

        assert 'similarity_score' in result
        assert 'method' in result
        assert 'normalized' in result
        assert result['method'] == 'vgg16'
        assert result['normalized'] == True
        assert result['similarity_score'] == pytest.approx(1.0, abs=1e-6)

    @patch('rose.processing.image_comparator.ImageComparator._extract_features')
    def test_compare_multiple_images(self, mock_extract):
        """Test comparing multiple images."""
        # Mock feature extraction to return different features for each image
        mock_extract.side_effect = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]

        images = [self.test_image_path] * 3
        result = self.comparator.compare_multiple_images(images, method='vgg16')

        assert 'similarity_matrix' in result
        assert 'num_images' in result
        assert result['num_images'] == 3
        assert result['similarity_matrix'].shape == (3, 3)

        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(
            np.diag(result['similarity_matrix']),
            np.ones(3),
            decimal=6
        )

    @patch('rose.processing.image_comparator.ImageComparator._extract_features')
    def test_find_most_similar(self, mock_extract):
        """Test finding most similar images."""
        # Mock feature extraction
        query_features = np.array([1, 2, 3])
        candidate_features = [
            np.array([1, 2, 3]),  # Identical to query
            np.array([4, 5, 6]),  # Different
            np.array([1, 2, 3])   # Identical to query
        ]

        mock_extract.side_effect = [query_features] + candidate_features

        query_image = self.test_image_path
        candidate_images = [self.test_image_path] * 3

        result = self.comparator.find_most_similar(
            query_image,
            candidate_images,
            method='vgg16',
            top_k=2
        )

        assert len(result) == 2
        assert result[0]['similarity_score'] == pytest.approx(1.0, abs=1e-6)
        assert result[1]['similarity_score'] == pytest.approx(1.0, abs=1e-6)
        assert result[0]['index'] in [0, 2]  # Should be one of the identical images
        assert result[1]['index'] in [0, 2]  # Should be the other identical image

    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary test image if it was created
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
