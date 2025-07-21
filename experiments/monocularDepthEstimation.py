"""
Monocular Depth Estimation using Depth Anything v2 and other pre-trained models.

This module provides a simple interface for depth estimation from images.
"""

from depth_estimator import DepthEstimator
from pathlib import Path


def estimate_depth_from_image(image_path: str, output_path: str | None = None, show_result: bool = False):
    """
    Estimate depth from a single image using Depth Anything v2 model.
    
    Args:
        image_path (str): Path to the input image file
        output_path (str, optional): Path to save the depth map visualization
        show_result (bool): Whether to display the result using matplotlib
    
    Returns:
        tuple: (depth_map, original_image) where depth_map is a numpy array
    """
    estimator = DepthEstimator()
    return estimator.estimate_depth_from_image(image_path, output_path, show_result)


def get_depth_statistics(depth_map):
    """
    Get statistics about the depth map.
    
    Args:
        depth_map (np.ndarray): Depth map array
    
    Returns:
        dict: Dictionary containing depth statistics
    """
    estimator = DepthEstimator()
    return estimator.get_depth_statistics(depth_map)


# Example usage
if __name__ == "__main__":
    # Example usage of the depth estimation function
    image_path = "experiments/GhostGirl.png"  # Using the image in your experiments folder
    
    if Path(image_path).exists():
        print("Running depth estimation example...")
        depth_map, original_image = estimate_depth_from_image(
            image_path=image_path,
            output_path="experiments/depth_result.png",
            show_result=True
        )
        
        if depth_map is not None:
            stats = get_depth_statistics(depth_map)
            print("\nDepth Map Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
    else:
        print(f"Example image not found: {image_path}")
        print("Please provide a valid image path to test the function.")
