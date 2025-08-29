import unittest
import numpy as np
import os
from PIL import Image
from rose.processing.depth_estimator import DepthEstimator
from rose.preprocessing.image_utils import ImagePreprocessor


class TestDepthEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.depth_estimator = DepthEstimator()

    def test_depth_estimator_init(self):
        self.assertIsNotNone(self.depth_estimator)

    def test_estimate_depth_on_square_image(self):
        img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
        img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
        # Remove batch dimension and convert to uint8 for PIL
        img_array = img_array[0].astype(np.uint8)
        img = Image.fromarray(img_array)

        depth_map = self.depth_estimator.estimate_depth(img)
        self.assertIsInstance(depth_map, np.ndarray)
        self.assertEqual(depth_map.ndim, 2)  # Should be a 2D depth map
        self.assertGreater(depth_map.shape[0], 0)
        self.assertGreater(depth_map.shape[1], 0)

        # Normalize depth map for visualization
        norm_depth = (255 * (depth_map - np.min(depth_map)) / (np.ptp(depth_map) + 1e-8)).astype(np.uint8)
        depth_img = Image.fromarray(norm_depth)
        save_path = os.path.join(os.path.dirname(__file__), 'depth_map_squareTestImage.png')
        depth_img.save(save_path)

    def test_estimate_depth_zoedepth_on_square_image(self):
        img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
        img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
        # Remove batch dimension and convert to uint8 for PIL
        img_array = img_array[0].astype(np.uint8)
        img = Image.fromarray(img_array)

        depth_map = self.depth_estimator.estimate_depth_zoedepth(img)
        self.assertIsInstance(depth_map, np.ndarray)
        self.assertEqual(depth_map.ndim, 2)  # Should be a 2D depth map
        self.assertGreater(depth_map.shape[0], 0)
        self.assertGreater(depth_map.shape[1], 0)

        # Normalize depth map for visualization
        norm_depth = (255 * (depth_map - np.min(depth_map)) / (np.ptp(depth_map) + 1e-8)).astype(np.uint8)
        depth_img = Image.fromarray(norm_depth)
        save_path = os.path.join(os.path.dirname(__file__), 'zoedepth_map_squareTestImage.png')
        depth_img.save(save_path)
