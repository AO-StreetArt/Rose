from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
from typing import Optional


class DepthEstimator:
    """
    Estimates depth information for elements in images, using monocular or stereo depth estimation models.
    """
    def __init__(self):
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        # Load ZoeDepth model and processor at instantiation
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.zoedepth_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        self.zoedepth_model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")

    def estimate_depth(self, img: Image.Image) -> np.ndarray:
        """
        Accepts a PIL Image object and returns a depth map using the Intel/dpt-large model.
        Args:
            img (PIL.Image.Image): Input image.
        Returns:
            np.ndarray: Depth map of the image.
        """
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        return prediction

    def estimate_depth_zoedepth(self, img: Image.Image) -> Optional[np.ndarray]:
        """
        Accepts a PIL Image object and returns a depth map using the Intel/zoedepth-nyu-kitti model.
        Args:
            img (PIL.Image.Image): Input image.
        Returns:
            np.ndarray: Depth map of the image, or None if estimation fails.
        """
        try:
            inputs = self.zoedepth_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.zoedepth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            depth_map = predicted_depth.squeeze().cpu().numpy()
            return depth_map
        except Exception as e:
            print(f"Error during ZoeDepth estimation: {e}")
            return None
