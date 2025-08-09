import cv2
import numpy as np
import os
from ..preprocessing.image_utils import ImagePreprocessor

class EdgeDetector:
    def __init__(self, hed_prototxt: str = None, hed_caffemodel: str = None):
        """
        Initialize the EdgeDetector. If HED model paths are provided, load the HED model.
        """
        self.hed_net = None
        if hed_prototxt and hed_caffemodel:
            if os.path.exists(hed_prototxt) and os.path.exists(hed_caffemodel):
                self.hed_net = cv2.dnn.readNetFromCaffe(hed_prototxt, hed_caffemodel)
            else:
                raise FileNotFoundError("HED model files not found.")

    def canny(self, image: np.ndarray, threshold1: float = 100, threshold2: float = 200) -> np.ndarray:
        """
        Perform Canny edge detection on an input image.
        """
        # Use preprocessing method for grayscale conversion
        gray = ImagePreprocessor.ensure_grayscale_image(image)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges

    def hed(self, image: np.ndarray) -> np.ndarray:
        """
        Perform HED edge detection using a pre-trained HED model.
        """
        if self.hed_net is None:
            raise RuntimeError("HED model is not loaded. Please provide model files during initialization.")
        
        # Use preprocessing methods for color correction and blob creation
        bgr_image = ImagePreprocessor.ensure_bgr_image(image)
        blob = ImagePreprocessor.create_blob_for_hed(bgr_image)
        
        self.hed_net.setInput(blob)
        hed_edges = self.hed_net.forward()
        hed_edges = hed_edges[0, 0]
        hed_edges = cv2.resize(hed_edges, (image.shape[1], image.shape[0]))
        hed_edges = (255 * hed_edges).astype(np.uint8)
        return hed_edges 