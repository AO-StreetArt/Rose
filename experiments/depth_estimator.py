import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")


class DepthEstimator:
    """
    A modular class for performing monocular depth estimation using pre-trained models.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the depth estimator.

        Args:
            model_name: Optional model name to use. If None, will try multiple models.
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self) -> bool:
        """
        Load the depth estimation model.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self._is_loaded:
            return True

        # Try multiple model options if no specific model is provided
        model_options = [
            self.model_name,
            "LiheYoung/depth_anything_vitl14",
            "facebook/dpt-large",
            "Intel/dpt-large"
        ]

        # Remove None values
        model_options = [opt for opt in model_options if opt is not None]

        for model_name in model_options:
            try:
                print(f"Trying to load model: {model_name}")
                self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForDepthEstimation.from_pretrained(model_name, trust_remote_code=True)
                self.model_name = model_name
                self._is_loaded = True
                print(f"Successfully loaded model: {model_name}")
                return True
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue

        print("Failed to load any depth estimation model")
        return False

    def preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and preprocess an image for depth estimation.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object or None if loading fails
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def estimate_depth(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Perform depth estimation on a preprocessed image.

        Args:
            image: PIL Image object

        Returns:
            Depth map as numpy array or None if estimation fails
        """
        if not self._is_loaded and not self.load_model():
            return None

        try:
            # Prepare inputs for the model
            inputs = self.processor(images=image, return_tensors="pt")

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Post-process the depth map
            depth_map = predicted_depth.squeeze().cpu().numpy()
            return depth_map

        except Exception as e:
            print(f"Error during depth estimation: {e}")
            return None

    def normalize_depth_map(self, depth_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Normalize depth map for visualization.

        Args:
            depth_map: Raw depth map

        Returns:
            Normalized depth map
        """
        if depth_map is None:
            return None

        if depth_map.size == 0:
            return depth_map

        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max > depth_min:
            return (depth_map - depth_min) / (depth_max - depth_min)
        else:
            return depth_map

    def create_depth_visualization(self, depth_map: Optional[np.ndarray],
                                 colormap: int = cv2.COLORMAP_MAGMA) -> Optional[np.ndarray]:
        """
        Create a colored visualization of the depth map.

        Args:
            depth_map: Normalized depth map
            colormap: OpenCV colormap to use

        Returns:
            Colored depth map as RGB array
        """
        if depth_map is None:
            return None

        # Convert to uint8 for OpenCV compatibility
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)

        # Apply colormap
        depth_map_colored = cv2.applyColorMap(depth_map_uint8, colormap)

        # Convert BGR to RGB for matplotlib
        return cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2RGB)

    def estimate_depth_from_image(self, image_path: str,
                                output_path: Optional[str] = None,
                                show_result: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Complete pipeline for depth estimation from image file.

        Args:
            image_path: Path to the input image file
            output_path: Optional path to save the depth map visualization
            show_result: Whether to display the result using matplotlib

        Returns:
            Tuple of (depth_map, original_image) where depth_map is a numpy array
        """
        # Load and preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            return None, None

        # Perform depth estimation
        depth_map = self.estimate_depth(image)
        if depth_map is None:
            return None, None

        # Normalize depth map for visualization
        depth_map_normalized = self.normalize_depth_map(depth_map)

        # Create colored visualization
        depth_map_colored = self.create_depth_visualization(depth_map_normalized)

        # Save result if output path is provided
        if output_path and depth_map_colored is not None:
            print(f"Saving depth map to: {output_path}")
            plt.imsave(output_path, depth_map_colored)

        # Display result if requested
        if show_result and depth_map_colored is not None:
            self._display_results(image, depth_map_colored)

        print("Depth estimation completed!")
        return depth_map, np.array(image)

    def _display_results(self, original_image: Image.Image, depth_map_colored: np.ndarray):
        """
        Display the original image and depth map side by side.

        Args:
            original_image: Original input image
            depth_map_colored: Colored depth map
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Depth map
        axes[1].imshow(depth_map_colored)
        axes[1].set_title('Depth Map')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    def get_depth_statistics(self, depth_map: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Get statistics about the depth map.

        Args:
            depth_map: Depth map array

        Returns:
            Dictionary containing depth statistics
        """
        if depth_map is None:
            return {}

        return {
            'min_depth': float(depth_map.min()),
            'max_depth': float(depth_map.max()),
            'mean_depth': float(depth_map.mean()),
            'std_depth': float(depth_map.std()),
            'shape': depth_map.shape
        }