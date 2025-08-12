import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List


class ImageCreator:
    """
    Handles image creation tasks such as saving depth maps and other visualizations.
    """

    @staticmethod
    def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize depth map values to 0-255 range for visualization.

        Args:
            depth_map (np.ndarray): Raw depth map from the model

        Returns:
            np.ndarray: Normalized depth map (0-255)
        """
        if depth_map is None:
            raise ValueError("Depth map cannot be None")

        if depth_map.size == 0:
            raise ValueError("Depth map cannot be empty")

        # Normalize to 0-1 range
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)

        # Handle case where min == max (constant depth map)
        if depth_max == depth_min:
            return np.zeros_like(depth_map, dtype=np.uint8)

        normalized = (depth_map - depth_min) / (depth_max - depth_min)

        # Convert to 0-255 range
        return (normalized * 255).astype(np.uint8)

    @staticmethod
    def save_depth_map_as_image(depth_map: np.ndarray, output_path: str,
                               colormap: str = 'viridis') -> None:
        """
        Save depth map as an image with optional colormap.

        Args:
            depth_map (np.ndarray): Depth map array
            output_path (str): Path to save the output image
            colormap (str): Matplotlib colormap name (default: 'viridis')
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Display depth map with colormap
        im = ax.imshow(depth_map, cmap=colormap)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Depth', rotation=270, labelpad=15)

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Depth Map')

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_depth_map_raw(depth_map: np.ndarray, output_path: str) -> None:
        """
        Save depth map as a raw grayscale image.

        Args:
            depth_map (np.ndarray): Depth map array
            output_path (str): Path to save the output image
        """
        # Normalize depth map
        normalized_depth = ImageCreator.normalize_depth_map(depth_map)

        # Convert to PIL Image and save
        depth_image = Image.fromarray(normalized_depth)
        depth_image.save(output_path)

    @staticmethod
    def visualize_features(image: np.ndarray, keypoints: List, descriptors: np.ndarray, output_path: str) -> None:
        """
        Visualize detected features by drawing keypoints and adding descriptor information.

        Args:
            image (np.ndarray): Original image
            keypoints (List): Detected keypoints
            descriptors (np.ndarray): Feature descriptors
            output_path (str): Path to save the visualization
        """
        # Convert image to BGR for OpenCV drawing functions
        if len(image.shape) == 3:
            # If RGB, convert to BGR
            if image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
        else:
            # If grayscale, convert to BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw keypoints in red
        image_with_keypoints = cv2.drawKeypoints(
            image_bgr,
            keypoints,
            None,
            color=(0, 0, 255),  # Red color in BGR
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Add text overlay with descriptor information
        num_keypoints = len(keypoints)
        descriptor_shape = descriptors.shape if descriptors is not None else "None"

        # Add text information
        cv2.putText(
            image_with_keypoints,
            f"Keypoints: {num_keypoints}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),  # Red text
            2
        )

        cv2.putText(
            image_with_keypoints,
            f"Descriptors: {descriptor_shape}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),  # Red text
            2
        )

        # Convert back to RGB for saving
        image_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

        # Save the image
        Image.fromarray(image_rgb).save(output_path)

    @staticmethod
    def visualize_segmentation(image: np.ndarray, masks: np.ndarray, prompts: List[str], output_path: str) -> None:
        """
        Visualize segmentation masks by overlaying them on the original image.

        Args:
            image (np.ndarray): Original image
            masks (np.ndarray): Segmentation masks (num_prompts, H, W)
            prompts (List[str]): List of text prompts used for segmentation
            output_path (str): Path to save the visualization
        """
        # Validate inputs
        if len(masks) != len(prompts):
            raise ValueError(f"Number of masks ({len(masks)}) must match number of prompts ({len(prompts)})")

        # Convert image to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # If BGR, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # If grayscale, convert to RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image

        # Create figure with subplots
        num_prompts = len(prompts)
        fig, axes = plt.subplots(1, num_prompts + 1, figsize=(5 * (num_prompts + 1), 5))

        # Plot original image
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot each segmentation mask
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

        for i, (mask, prompt) in enumerate(zip(masks, prompts)):
            color = colors[i % len(colors)]

            # Create colored mask overlay
            colored_mask = np.zeros_like(image_rgb)
            colored_mask[:, :, 0] = mask * 255 if color in ['red', 'yellow', 'orange', 'magenta'] else 0
            colored_mask[:, :, 1] = mask * 255 if color in ['green', 'yellow', 'cyan'] else 0
            colored_mask[:, :, 2] = mask * 255 if color in ['blue', 'purple', 'cyan', 'magenta'] else 0

            # Overlay mask on original image
            alpha = 0.5
            overlay = image_rgb * (1 - alpha) + colored_mask * alpha
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(f'Segment: "{prompt}"')
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def create_depth_visualization(depth_map: np.ndarray, original_image: np.ndarray,
                                  colormap: str = 'viridis') -> np.ndarray:
        """
        Create depth visualization as a numpy array for real-time display.

        Args:
            depth_map (np.ndarray): Depth map array
            original_image (np.ndarray): Original image for reference
            colormap (str): Matplotlib colormap name (default: 'viridis')

        Returns:
            np.ndarray: Depth visualization as BGR image for OpenCV display
        """
        # Normalize depth map to 0-1 range
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)

        if depth_max == depth_min:
            normalized_depth = np.zeros_like(depth_map)
        else:
            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored_depth = cmap(normalized_depth)

        # Convert to 0-255 range and BGR format for OpenCV
        colored_depth_uint8 = (colored_depth[:, :, :3] * 255).astype(np.uint8)
        colored_depth_bgr = cv2.cvtColor(colored_depth_uint8, cv2.COLOR_RGB2BGR)

        # Resize to match original image dimensions
        if colored_depth_bgr.shape[:2] != original_image.shape[:2]:
            colored_depth_bgr = cv2.resize(colored_depth_bgr,
                                          (original_image.shape[1], original_image.shape[0]))

        return colored_depth_bgr

    @staticmethod
    def create_segmentation_visualization(original_image: np.ndarray, masks: np.ndarray,
                                        prompts: List[str]) -> np.ndarray:
        """
        Create segmentation visualization as a numpy array for real-time display.

        Args:
            original_image (np.ndarray): Original image
            masks (np.ndarray): Segmentation masks (num_prompts, H, W)
            prompts (List[str]): List of text prompts used for segmentation

        Returns:
            np.ndarray: Segmentation visualization as BGR image for OpenCV display
        """
        # Ensure original image is in BGR format
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            # If RGB, convert to BGR
            image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            # If grayscale, convert to BGR
            if len(original_image.shape) == 2:
                image_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = original_image

        # Create overlay image
        overlay = image_bgr.copy()

        # Define colors for different segments
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0)   # Orange
        ]

        # Overlay each mask
        for i, (mask, prompt) in enumerate(zip(masks, prompts)):
            color = colors[i % len(colors)]

            # Create colored mask
            colored_mask = np.zeros_like(image_bgr)
            colored_mask[:, :, 0] = mask * color[0]  # B
            colored_mask[:, :, 1] = mask * color[1]  # G
            colored_mask[:, :, 2] = mask * color[2]  # R

            # Overlay with alpha blending
            alpha = 0.5
            overlay = overlay * (1 - alpha * mask[:, :, np.newaxis]) + colored_mask * alpha * mask[:, :, np.newaxis]

        # Ensure values are in valid range
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return overlay
