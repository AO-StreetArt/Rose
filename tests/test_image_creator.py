# Add the project root to the path to import from rose modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from rose.postprocessing.image_creator import ImageCreator


class TestImageCreator(unittest.TestCase):
    """Test suite for the ImageCreator class."""

    def setUp(self):
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

        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, depth_map.shape)
        self.assertEqual(np.min(normalized), 0)
        self.assertEqual(np.max(normalized), 255)

    def test_normalize_depth_map_constant(self):
        """Test normalization of a constant depth map."""
        # Create a depth map with all same values
        depth_map = np.ones((50, 50), dtype=np.float32) * 5.0

        normalized = ImageCreator.normalize_depth_map(depth_map)

        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, depth_map.shape)
        # For constant values, should result in all zeros or all 255s
        self.assertTrue(np.all(normalized == 0) or np.all(normalized == 255))

    def test_normalize_depth_map_edge_cases(self):
        """Test normalization with edge cases."""
        # Test with single value
        single_value = np.array([[1.5]], dtype=np.float32)
        normalized = ImageCreator.normalize_depth_map(single_value)
        self.assertEqual(normalized.shape, (1, 1))
        self.assertEqual(normalized.dtype, np.uint8)

        # Test with very small range
        small_range = np.array([[1.0, 1.001]], dtype=np.float32)
        normalized = ImageCreator.normalize_depth_map(small_range)
        self.assertEqual(normalized.shape, (1, 2))
        self.assertEqual(normalized.dtype, np.uint8)

    def test_save_depth_map_as_image(self):
        """Test saving depth map as image with colormap."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.save_depth_map_as_image(self.test_depth_map, output_path, 'viridis')

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            # Try to load the saved image
            saved_image = Image.open(output_path)
            self.assertEqual(saved_image.mode, 'RGBA')  # PNG with colormap

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_depth_map_as_image_different_colormaps(self):
        """Test saving depth map with different colormaps."""
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
        
        for colormap in colormaps:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                output_path = tmp.name

            try:
                ImageCreator.save_depth_map_as_image(self.test_depth_map, output_path, colormap)
                
                # Check that file was created
                self.assertTrue(os.path.exists(output_path))
                self.assertGreater(os.path.getsize(output_path), 0)
                
                # Try to load the saved image
                saved_image = Image.open(output_path)
                self.assertEqual(saved_image.mode, 'RGBA')

            finally:
                # Clean up
                if os.path.exists(output_path):
                    os.unlink(output_path)

    def test_create_feature_visualization(self):
        """Test creating feature visualization."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.create_feature_visualization(
                self.test_image, 
                self.test_keypoints, 
                self.test_descriptors, 
                output_path
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            # Try to load the saved image
            saved_image = Image.open(output_path)
            self.assertEqual(saved_image.mode, 'RGB')

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_create_segmentation_visualization(self):
        """Test creating segmentation visualization."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.create_segmentation_visualization(
                self.test_image,
                self.test_masks,
                self.test_prompts,
                output_path
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            # Try to load the saved image
            saved_image = Image.open(output_path)
            self.assertEqual(saved_image.mode, 'RGB')

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_create_comparison_visualization(self):
        """Test creating comparison visualization."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.create_comparison_visualization(
                [self.test_image, self.test_image],  # Two identical images for comparison
                ["Original", "Processed"],
                output_path
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            # Try to load the saved image
            saved_image = Image.open(output_path)
            self.assertEqual(saved_image.mode, 'RGB')

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_create_depth_visualization(self):
        """Test creating depth visualization."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.create_depth_visualization(
                self.test_image,
                self.test_depth_map,
                output_path
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            # Try to load the saved image
            saved_image = Image.open(output_path)
            self.assertEqual(saved_image.mode, 'RGB')

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_create_combined_visualization(self):
        """Test creating combined visualization."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.create_combined_visualization(
                self.test_image,
                self.test_depth_map,
                self.test_masks[0],  # Use first mask
                output_path
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            # Try to load the saved image
            saved_image = Image.open(output_path)
            self.assertEqual(saved_image.mode, 'RGB')

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_normalize_depth_map_with_nan_values(self):
        """Test normalization with NaN values."""
        # Create depth map with NaN values
        depth_map = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        
        normalized = ImageCreator.normalize_depth_map(depth_map)
        
        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, depth_map.shape)
        
        # NaN values should be handled gracefully
        self.assertFalse(np.any(np.isnan(normalized)))

    def test_normalize_depth_map_with_inf_values(self):
        """Test normalization with infinite values."""
        # Create depth map with infinite values
        depth_map = np.array([[1.0, np.inf], [3.0, -np.inf]], dtype=np.float32)
        
        normalized = ImageCreator.normalize_depth_map(depth_map)
        
        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertEqual(normalized.shape, depth_map.shape)
        
        # Infinite values should be handled gracefully
        self.assertFalse(np.any(np.isinf(normalized)))

    def test_create_feature_visualization_with_empty_keypoints(self):
        """Test feature visualization with empty keypoints."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.create_feature_visualization(
                self.test_image, 
                [],  # Empty keypoints
                np.array([]),  # Empty descriptors
                output_path
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_create_segmentation_visualization_with_single_mask(self):
        """Test segmentation visualization with single mask."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            # Use single mask instead of multiple
            single_mask = self.test_masks[0]
            
            ImageCreator.create_segmentation_visualization(
                self.test_image,
                single_mask[np.newaxis, ...],  # Add batch dimension
                ["single_object"],
                output_path
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_create_comparison_visualization_with_different_sizes(self):
        """Test comparison visualization with images of different sizes."""
        # Create images of different sizes
        small_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        large_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            ImageCreator.create_comparison_visualization(
                [small_image, large_image],
                ["Small", "Large"],
                output_path
            )

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_create_depth_visualization_with_different_depths(self):
        """Test depth visualization with different depth ranges."""
        # Create depth maps with different ranges
        shallow_depth = np.random.rand(100, 150).astype(np.float32) * 0.1  # Shallow
        deep_depth = np.random.rand(100, 150).astype(np.float32) * 10.0 + 5.0  # Deep
        
        for depth_map, name in [(shallow_depth, "shallow"), (deep_depth, "deep")]:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                output_path = tmp.name

            try:
                ImageCreator.create_depth_visualization(
                    self.test_image,
                    depth_map,
                    output_path
                )

                # Check that file was created
                self.assertTrue(os.path.exists(output_path))
                self.assertGreater(os.path.getsize(output_path), 0)

            finally:
                # Clean up
                if os.path.exists(output_path):
                    os.unlink(output_path)

    def test_error_handling_invalid_output_path(self):
        """Test error handling with invalid output paths."""
        # Test with directory that doesn't exist
        invalid_path = "/nonexistent/directory/test.png"
        
        with self.assertRaises(Exception):
            ImageCreator.save_depth_map_as_image(self.test_depth_map, invalid_path, 'viridis')

    def test_error_handling_invalid_colormap(self):
        """Test error handling with invalid colormap."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            # Test with invalid colormap
            with self.assertRaises(Exception):
                ImageCreator.save_depth_map_as_image(self.test_depth_map, output_path, 'invalid_colormap')

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_performance_with_large_images(self):
        """Test performance with large images."""
        # Create large test images
        large_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        large_depth = np.random.rand(1024, 1024).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            import time
            
            # Measure time for large image processing
            start_time = time.time()
            ImageCreator.create_depth_visualization(
                large_image,
                large_depth,
                output_path
            )
            processing_time = time.time() - start_time
            
            # Should complete in reasonable time (less than 30 seconds)
            self.assertLess(processing_time, 30.0)
            
            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)
