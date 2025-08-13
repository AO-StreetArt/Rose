# Add the project root to the path to import from rose modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import tempfile
import os
import cv2
from PIL import Image
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from rose.postprocessing.image_creator import ImageCreator


class TestImageCreator:
    """Test suite for the ImageCreator class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a test depth map
        self.test_depth_map = np.random.rand(100, 150).astype(np.float32)

        # Create a test image
        self.test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        # Create test keypoints
        self.test_keypoints = [
            cv2.KeyPoint(x=50, y=50, size=10),
            cv2.KeyPoint(x=75, y=75, size=15),
            cv2.KeyPoint(x=25, y=25, size=8)
        ]

        # Create test descriptors
        self.test_descriptors = np.random.rand(3, 128).astype(np.float32)

        # Create test masks and prompts
        self.test_masks = np.random.rand(2, 100, 150).astype(np.float32)
        self.test_prompts = ["person", "background"]

    def test_normalize_depth_map_basic(self):
        """Test basic depth map normalization."""
        # Create a depth map with known values
        depth_map = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        normalized = ImageCreator.normalize_depth_map(depth_map)

        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.uint8
        assert normalized.shape == depth_map.shape
        assert np.min(normalized) == 0
        assert np.max(normalized) == 255

    def test_normalize_depth_map_constant(self):
        """Test normalization of a constant depth map."""
        # Create a depth map with all same values
        depth_map = np.ones((50, 50), dtype=np.float32) * 5.0

        normalized = ImageCreator.normalize_depth_map(depth_map)

        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.uint8
        assert normalized.shape == depth_map.shape
        # For constant values, should result in all zeros or all 255s
        assert np.all(normalized == 0) or np.all(normalized == 255)

    def test_normalize_depth_map_edge_cases(self):
        """Test normalization with edge cases."""
        # Test with single value
        single_value = np.array([[1.5]], dtype=np.float32)
        normalized = ImageCreator.normalize_depth_map(single_value)
        assert normalized.shape == (1, 1)
        assert normalized.dtype == np.uint8

        # Test with very small range
        small_range = np.array([[1.0, 1.001]], dtype=np.float32)
        normalized = ImageCreator.normalize_depth_map(small_range)
        assert normalized.shape == (1, 2)
        assert normalized.dtype == np.uint8

    def test_save_depth_map_as_image(self):
        """Test saving depth map as image with colormap."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.save_depth_map_as_image(self.test_depth_map, output_path, 'viridis')

            # Check that file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Try to load the saved image
            saved_image = Image.open(output_path)
            assert saved_image.mode == 'RGBA'  # PNG with colormap

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_save_depth_map_as_image_different_colormaps(self):
        """Test saving depth map with different colormaps."""
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray']

        for colormap in colormaps:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                output_path = tmp.name

            try:
                ImageCreator.save_depth_map_as_image(self.test_depth_map, output_path, colormap)
                assert os.path.exists(output_path)
                assert os.path.getsize(output_path) > 0
            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)

    def test_save_depth_map_raw(self):
        """Test saving depth map as raw grayscale image."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.save_depth_map_raw(self.test_depth_map, output_path)

            # Check that file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Try to load the saved image
            saved_image = Image.open(output_path)
            assert saved_image.mode == 'L'  # Grayscale

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_features_basic(self):
        """Test basic feature visualization."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_features(self.test_image, self.test_keypoints, self.test_descriptors, output_path)

            # Check that file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Try to load the saved image
            saved_image = Image.open(output_path)
            assert saved_image.mode in ['RGB', 'RGBA']  # matplotlib can save as either

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_features_grayscale_image(self):
        """Test feature visualization with grayscale image."""
        grayscale_image = np.random.randint(0, 255, (100, 150), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_features(grayscale_image, self.test_keypoints, self.test_descriptors, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_features_no_keypoints(self):
        """Test feature visualization with no keypoints."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_features(self.test_image, [], None, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_features_bgr_image(self):
        """Test feature visualization with BGR image."""
        bgr_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_features(bgr_image, self.test_keypoints, self.test_descriptors, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_segmentation_basic(self):
        """Test basic segmentation visualization."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_segmentation(self.test_image, self.test_masks, self.test_prompts, output_path)

            # Check that file was created
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

            # Try to load the saved image
            saved_image = Image.open(output_path)
            assert saved_image.mode in ['RGB', 'RGBA']  # matplotlib can save as either

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_segmentation_bgr_image(self):
        """Test segmentation visualization with BGR image."""
        bgr_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_segmentation(bgr_image, self.test_masks, self.test_prompts, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_segmentation_single_mask(self):
        """Test segmentation visualization with single mask."""
        single_mask = np.random.rand(1, 100, 150).astype(np.float32)
        single_prompt = ["person"]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_segmentation(self.test_image, single_mask, single_prompt, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_segmentation_multiple_masks(self):
        """Test segmentation visualization with multiple masks."""
        multiple_masks = np.random.rand(5, 100, 150).astype(np.float32)
        multiple_prompts = ["person", "background", "object", "face", "clothing"]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_segmentation(self.test_image, multiple_masks, multiple_prompts, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_segmentation_grayscale_image(self):
        """Test segmentation visualization with grayscale image."""
        grayscale_image = np.random.randint(0, 255, (100, 150), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_segmentation(grayscale_image, self.test_masks, self.test_prompts, output_path)

            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_depth_map_as_image_matplotlib_calls(self, mock_close, mock_savefig):
        """Test that matplotlib functions are called correctly."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.save_depth_map_as_image(self.test_depth_map, output_path, 'viridis')

            # Check that savefig was called with correct parameters
            mock_savefig.assert_called_once()
            call_args = mock_savefig.call_args
            assert call_args[0][0] == output_path
            assert call_args[1]['dpi'] == 300
            assert call_args[1]['bbox_inches'] == 'tight'

            # Check that close was called
            mock_close.assert_called_once()

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_normalize_depth_map_invalid_input(self):
        """Test normalize_depth_map with invalid input."""
        # Test with empty array
        with pytest.raises(ValueError):
            ImageCreator.normalize_depth_map(np.array([]))

        # Test with None
        with pytest.raises(ValueError):
            ImageCreator.normalize_depth_map(None)

    def test_save_depth_map_invalid_path(self):
        """Test saving depth map with invalid path."""
        # Test with directory that doesn't exist
        invalid_path = "/nonexistent/directory/test.png"

        with pytest.raises(FileNotFoundError):
            ImageCreator.save_depth_map_raw(self.test_depth_map, invalid_path)

    def test_visualize_features_invalid_input(self):
        """Test visualize_features with invalid input."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            # Test with None image
            with pytest.raises(AttributeError):
                ImageCreator.visualize_features(None, self.test_keypoints, self.test_descriptors, output_path)

            # Test with None keypoints
            with pytest.raises(TypeError):
                ImageCreator.visualize_features(self.test_image, None, self.test_descriptors, output_path)

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_visualize_segmentation_invalid_input(self):
        """Test visualize_segmentation with invalid input."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            # Test with mismatched masks and prompts
            mismatched_masks = np.random.rand(3, 100, 150).astype(np.float32)
            mismatched_prompts = ["person", "background"]  # Only 2 prompts for 3 masks

            with pytest.raises(ValueError):
                ImageCreator.visualize_segmentation(self.test_image, mismatched_masks, mismatched_prompts, output_path)

        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_all_methods_static(self):
        """Test that all methods are static methods."""
        # Verify that all methods can be called without instantiation
        depth_map = np.random.rand(10, 10).astype(np.float32)

        # Test normalize_depth_map
        result = ImageCreator.normalize_depth_map(depth_map)
        assert isinstance(result, np.ndarray)

        # Test save_depth_map_raw
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.save_depth_map_raw(depth_map, output_path)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

        # Test save_depth_map_as_image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.save_depth_map_as_image(depth_map, output_path)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

        # Test visualize_features
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_features(self.test_image, self.test_keypoints, self.test_descriptors, output_path)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

        # Test visualize_segmentation
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.visualize_segmentation(self.test_image, self.test_masks, self.test_prompts, output_path)
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == "__main__":
    pytest.main([__file__])
