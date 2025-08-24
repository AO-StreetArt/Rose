import pytest
import numpy as np
import cv2
import os
from unittest.mock import Mock, patch

from rose.processing.image_comparator import ImageComparator
from rose.processing.feature_extractor import FeatureExtractor
from rose.processing.feature_detector import FeatureDetector


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
        assert comparator.feature_detector is not None
        assert isinstance(comparator.feature_detector, FeatureDetector)
        assert comparator.bf_matcher is not None

        # Test with custom feature extractor
        custom_extractor = FeatureExtractor()
        custom_detector = FeatureDetector(n_features=1000)
        comparator = ImageComparator(custom_extractor, custom_detector)
        assert comparator.feature_extractor is custom_extractor
        assert comparator.feature_detector is custom_detector

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
        assert result['normalized'] is True
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

    def test_compare_images_orb_successful_comparison(self):
        """Test successful ORB-based image comparison."""
        # Create more feature-rich test images that ORB can detect
        # Create a checkerboard pattern for more features
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    img1[i:i+32, j:j+32] = [255, 0, 0]  # Red squares
                else:
                    img1[i:i+32, j:j+32] = [0, 0, 255]  # Blue squares
        
        # Create a different pattern
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 16):
            for j in range(0, 256, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    img2[i:i+16, j:j+16] = [0, 255, 0]  # Green squares
                else:
                    img2[i:i+16, j:j+16] = [255, 255, 0]  # Yellow squares
        
        # Save images temporarily
        img1_path = "tests/temp_test_image1.png"
        img2_path = "tests/temp_test_image2.png"
        cv2.imwrite(img1_path, img1)
        cv2.imwrite(img2_path, img2)
        
        try:
            result = self.comparator.compare_images_orb(img1_path, img2_path)
            
            # Check that all expected keys are present
            expected_keys = [
                'similarity_score', 'method', 'normalized', 'keypoints_1', 
                'keypoints_2', 'descriptors_1', 'descriptors_2', 
                'matches_count', 'match_ratio', 'total_possible_matches'
            ]
            for key in expected_keys:
                assert key in result
            
            # Check method is correct
            assert result['method'] == 'orb'
            assert result['normalized'] is True
            
            # Check that we got some keypoints and descriptors
            assert result['keypoints_1'] > 0
            assert result['keypoints_2'] > 0
            assert result['descriptors_1'] is not None
            assert result['descriptors_2'] is not None
            
            # Check similarity score is in valid range
            assert 0.0 <= result['similarity_score'] <= 1.0
            
            # Check match ratio is in valid range
            assert 0.0 <= result['match_ratio'] <= 1.0
            
        finally:
            # Clean up temporary files
            if os.path.exists(img1_path):
                os.remove(img1_path)
            if os.path.exists(img2_path):
                os.remove(img2_path)

    def test_compare_images_orb_simple_images_no_features(self):
        """Test ORB comparison with simple images that have no detectable features."""
        # Create simple solid color images that ORB won't detect features in
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1[:, :] = [255, 0, 0]  # Red square
        
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2[:, :] = [0, 255, 0]  # Green square
        
        # Save images temporarily
        img1_path = "tests/temp_test_image1.png"
        img2_path = "tests/temp_test_image2.png"
        cv2.imwrite(img1_path, img1)
        cv2.imwrite(img2_path, img2)
        
        try:
            result = self.comparator.compare_images_orb(img1_path, img2_path)
            
            # For simple images, we expect no descriptors to be found
            assert result['method'] == 'orb'
            assert result['similarity_score'] == 0.0
            assert result['keypoints_1'] == 0
            assert result['keypoints_2'] == 0
            assert result['descriptors_1'] is None
            assert result['descriptors_2'] is None
            assert result['matches_count'] == 0
            assert result['match_ratio'] == 0.0
            assert 'error' in result
            assert 'No descriptors found' in result['error']
            
        finally:
            # Clean up temporary files
            if os.path.exists(img1_path):
                os.remove(img1_path)
            if os.path.exists(img2_path):
                os.remove(img2_path)

    def test_compare_images_orb_identical_images(self):
        """Test ORB comparison with identical images."""
        # Create a feature-rich image that ORB can detect
        feature_rich_img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create a pattern with edges and corners
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    feature_rich_img[i:i+32, j:j+32] = [255, 0, 0]  # Red squares
                else:
                    feature_rich_img[i:i+32, j:j+32] = [0, 0, 255]  # Blue squares
        
        # Save temporarily
        temp_path = "tests/temp_feature_rich.png"
        cv2.imwrite(temp_path, feature_rich_img)
        
        try:
            result = self.comparator.compare_images_orb(temp_path, temp_path)
            
            # Identical images should have high similarity
            assert result['method'] == 'orb'
            assert result['keypoints_1'] == result['keypoints_2']
            assert result['descriptors_1'] == result['descriptors_2']
            assert result['matches_count'] > 0
            assert result['similarity_score'] > 0.0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_compare_images_orb_with_numpy_arrays(self):
        """Test ORB comparison with numpy arrays instead of file paths."""
        # Create test images as numpy arrays with more features
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create a complex pattern with corners and edges
        for i in range(0, 256, 16):
            for j in range(0, 256, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    img1[i:i+16, j:j+16] = [255, 0, 0]  # Red squares
                else:
                    img1[i:i+16, j:j+16] = [0, 0, 255]  # Blue squares
        
        # Add some diagonal lines for more features
        for k in range(0, 256, 4):
            img1[k:k+2, k:k+2] = [255, 255, 255]  # White dots
        
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create a different complex pattern
        for i in range(0, 256, 12):
            for j in range(0, 256, 12):
                if (i // 12 + j // 12) % 2 == 0:
                    img2[i:i+12, j:j+12] = [0, 255, 0]  # Green squares
                else:
                    img2[i:i+12, j:j+12] = [255, 255, 0]  # Yellow squares
        
        # Add some horizontal lines for more features
        for k in range(0, 256, 8):
            img2[k:k+2, :] = [255, 255, 255]  # White lines
        
        result = self.comparator.compare_images_orb(img1, img2)
        
        assert result['method'] == 'orb'
        assert result['keypoints_1'] > 0
        assert result['keypoints_2'] > 0
        assert result['descriptors_1'] is not None
        assert result['descriptors_2'] is not None
        assert 0.0 <= result['similarity_score'] <= 1.0

    def test_compare_images_orb_no_descriptors_found(self):
        """Test ORB comparison when no descriptors are found."""
        # Create a very small image that might not generate descriptors
        tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Mock the feature detector to return no descriptors
        with patch.object(self.comparator.feature_detector, 'detect_and_compute') as mock_detect:
            mock_detect.return_value = ([], None)  # No keypoints, no descriptors
            
            result = self.comparator.compare_images_orb(tiny_img, tiny_img)
            
            assert result['method'] == 'orb'
            assert result['similarity_score'] == 0.0
            assert result['keypoints_1'] == 0
            assert result['keypoints_2'] == 0
            assert result['descriptors_1'] is None
            assert result['descriptors_2'] is None
            assert result['matches_count'] == 0
            assert result['match_ratio'] == 0.0
            assert 'error' in result

    def test_compare_images_orb_one_image_no_descriptors(self):
        """Test ORB comparison when only one image has no descriptors."""
        # Create a very small image that ORB won't detect features in
        tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Create a feature-rich image that ORB will detect features in
        feature_rich_img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(0, 256, 16):
            for j in range(0, 256, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    feature_rich_img[i:i+16, j:j+16] = [255, 0, 0]  # Red squares
                else:
                    feature_rich_img[i:i+16, j:j+16] = [0, 0, 255]  # Blue squares
        
        # Save the feature-rich image temporarily
        temp_path = "tests/temp_feature_rich_for_test.png"
        cv2.imwrite(temp_path, feature_rich_img)
        
        try:
            # Test with tiny image first (should fail)
            result1 = self.comparator.compare_images_orb(tiny_img, tiny_img)
            assert result1['method'] == 'orb'
            assert result1['similarity_score'] == 0.0
            assert result1['keypoints_1'] == 0
            assert result1['keypoints_2'] == 0
            assert result1['descriptors_1'] is None
            assert result1['descriptors_2'] is None
            assert result1['matches_count'] == 0
            assert result1['match_ratio'] == 0.0
            assert 'error' in result1
            
            # Test with feature-rich image (should succeed)
            result2 = self.comparator.compare_images_orb(temp_path, temp_path)
            assert result2['method'] == 'orb'
            assert result2['keypoints_1'] > 0
            assert result2['keypoints_2'] > 0
            assert result2['descriptors_1'] is not None
            assert result2['descriptors_2'] is not None
            assert result2['matches_count'] > 0
            assert result2['similarity_score'] > 0.0
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_compare_images_orb_parameter_consistency(self):
        """Test that ORB method maintains same parameter interface as compare_images."""
        # Test that the method accepts the same parameters (even if not used)
        result = self.comparator.compare_images_orb(
            self.test_image_path, 
            self.test_image_path,
            method='vgg16',  # This should be ignored but not cause errors
            normalize=False   # This should be ignored but not cause errors
        )
        
        assert result['method'] == 'orb'  # Should always be 'orb'
        assert result['normalized'] is False  # Should preserve the parameter value
        assert 'similarity_score' in result

    def test_compare_images_orb_feature_detector_integration(self):
        """Test that ORB method properly integrates with FeatureDetector."""
        # Create a mock feature detector
        mock_detector = Mock()
        mock_detector.detect_and_compute.side_effect = [
            ([Mock()], np.random.rand(20, 32).astype(np.uint8)),  # First image
            ([Mock()], np.random.rand(15, 32).astype(np.uint8))   # Second image
        ]
        
        # Create comparator with mock detector
        comparator = ImageComparator(feature_detector=mock_detector)
        
        # Test images
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = comparator.compare_images_orb(img1, img2)
        
        # Verify detector was called twice (once for each image)
        assert mock_detector.detect_and_compute.call_count == 2
        
        # Verify result structure
        assert result['method'] == 'orb'
        assert result['keypoints_1'] == 1  # Mock returns 1 keypoint
        assert result['keypoints_2'] == 1  # Mock returns 1 keypoint
        assert result['descriptors_1'] == (20, 32)
        assert result['descriptors_2'] == (15, 32)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary test image if it was created
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
