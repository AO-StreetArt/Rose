import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..preprocessing.image_utils import ImagePreprocessor

class FeatureDetector:
    """
    FeatureDetector uses the ORB algorithm to detect and compute features in an image.
    """
    def __init__(self, n_features: int = 500):
        self.orb = cv2.ORB_create(nfeatures=n_features)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Detects keypoints and computes descriptors using ORB.

        Args:
            image (np.ndarray): Input image (grayscale or color).

        Returns:
            keypoints (list): Detected keypoints.
            descriptors (np.ndarray): Feature descriptors.
        """
        # Use preprocessing method for grayscale conversion
        gray = ImagePreprocessor.ensure_grayscale_image(image)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors 