import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image, ImageDraw, ImageFont
try:
    # Try newer Pillow versions first
    from PIL.Image import Resampling
    LANCZOS = Resampling.LANCZOS
    NEAREST = Resampling.NEAREST
except ImportError:
    # Fall back to older Pillow versions
    LANCZOS = Image.LANCZOS
    NEAREST = Image.NEAREST
import os
from typing import List, Tuple, Optional, Dict, Any


class VGG16FeatureExtractor:
    """Extract features from images using pre-trained VGG16 model."""

    def __init__(self):
        self.model = None
        self.feature_model = None
        self._load_model()

    def _load_model(self):
        """Load and initialize the VGG16 model."""
        self.model = VGG16(weights='imagenet', include_top=False)
        self._create_feature_model()

    def _create_feature_model(self):
        """Create a model that outputs feature maps from convolutional layers."""
        layer_outputs = []
        layer_names = []

        for layer in self.model.layers:
            if 'conv' in layer.name:
                layer_outputs.append(layer.output)
                layer_names.append(layer.name)

        self.feature_model = Model(inputs=self.model.input, outputs=layer_outputs)
        self.layer_names = layer_names

    def extract_features(self, image_path: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Extract features from an image.

        Args:
            image_path: Path to the image file

        Returns:
            Tuple of (feature_maps, layer_names)
        """
        img_array = self._preprocess_image(image_path)
        features = self.feature_model.predict(img_array)

        # Remove batch dimension from all feature maps
        features = [feature[0] for feature in features]

        return features, self.layer_names

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for VGG16 model.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image array
        """
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)


class FeatureMapAnalyzer:
    """Analyze feature maps and extract central points."""

    @staticmethod
    def get_central_point(feature_map: np.ndarray) -> Tuple[int, int]:
        """
        Get central point coordinates of a feature map.

        Args:
            feature_map: Feature map array (height, width, channels)

        Returns:
            Tuple of (center_h, center_w)
        """
        height, width = feature_map.shape[:2]
        return height // 2, width // 2

    @staticmethod
    def get_central_values(feature_map: np.ndarray) -> np.ndarray:
        """
        Get values at the central point across all channels.

        Args:
            feature_map: Feature map array (height, width, channels)

        Returns:
            Array of central point values for each channel
        """
        center_h, center_w = FeatureMapAnalyzer.get_central_point(feature_map)
        return feature_map[center_h, center_w, :]

    @staticmethod
    def get_feature_statistics(feature_map: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics for feature map central values.

        Args:
            feature_map: Feature map array

        Returns:
            Dictionary with mean, std, min, max statistics
        """
        central_values = FeatureMapAnalyzer.get_central_values(feature_map)
        return {
            'mean': float(np.mean(central_values)),
            'std': float(np.std(central_values)),
            'min': float(np.min(central_values)),
            'max': float(np.max(central_values))
        }


class FeatureMapVisualizer:
    """Create visualizations of feature maps with central points."""

    def __init__(self, canvas_width: int = 1200, canvas_height: int = 800):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.font = self._load_font()
        self.title_font = self._load_title_font()

    def _load_font(self) -> ImageFont.ImageFont:
        """Load font for text rendering."""
        try:
            return ImageFont.truetype("arial.ttf", 12)
        except:
            return ImageFont.load_default()

    def _load_title_font(self) -> ImageFont.ImageFont:
        """Load title font for text rendering."""
        try:
            return ImageFont.truetype("arial.ttf", 16)
        except:
            return ImageFont.load_default()

    def create_canvas(self, original_img: Image.Image, image_path: str) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        """
        Create the main canvas for visualization.

        Args:
            original_img: Original input image
            image_path: Path to the original image

        Returns:
            Tuple of (canvas, draw_context)
        """
        canvas_width = max(self.canvas_width, original_img.width * 2)
        canvas_height = max(self.canvas_height, original_img.height + 100)
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # Add original image
        display_size = 400
        original_resized = original_img.resize((display_size, display_size), LANCZOS)
        canvas.paste(original_resized, (50, 50))

        # Add title and info
        draw = ImageDraw.Draw(canvas)
        draw.text((50, 10), "VGG16 Feature Map Central Points", fill='black', font=self.title_font)
        draw.text((50, 30), f"Image: {os.path.basename(image_path)}", fill='black', font=self.font)

        return canvas, draw

    def normalize_feature_map(self, feature_map: np.ndarray) -> np.ndarray:
        """
        Normalize feature map for visualization.

        Args:
            feature_map: Feature map array

        Returns:
            Normalized feature map (0-255 range)
        """
        feature_avg = np.mean(feature_map, axis=2)
        feature_norm = ((feature_avg - feature_avg.min()) /
                       (feature_avg.max() - feature_avg.min() + 1e-8) * 255).astype(np.uint8)
        return feature_norm

    def calculate_visualization_params(self, feature_map: np.ndarray, index: int) -> Dict[str, Any]:
        """
        Calculate parameters for feature map visualization.

        Args:
            feature_map: Feature map array
            index: Index of the feature map

        Returns:
            Dictionary with visualization parameters
        """
        height, width, channels = feature_map.shape

        # Layout parameters
        cols = 3
        col_width = (self.canvas_width - 550) // cols
        row_height = 150

        row = index // cols
        col = index % cols
        x_pos = 550 + col * col_width
        y_pos = 50 + row * row_height

        # Scale factor for visualization
        scale_factor = min(100 // max(height, width), 4)
        if scale_factor < 1:
            scale_factor = 1

        vis_width = width * scale_factor
        vis_height = height * scale_factor

        return {
            'x_pos': x_pos,
            'y_pos': y_pos,
            'scale_factor': scale_factor,
            'vis_width': vis_width,
            'vis_height': vis_height
        }

    def draw_feature_map(self, canvas: Image.Image, draw: ImageDraw.ImageDraw,
                        feature_map: np.ndarray, layer_name: str,
                        index: int) -> None:
        """
        Draw a single feature map visualization.

        Args:
            canvas: Canvas image
            draw: Drawing context
            feature_map: Feature map array
            layer_name: Name of the layer
            index: Index of the feature map
        """
        params = self.calculate_visualization_params(feature_map, index)

        # Create and paste feature map visualization
        feature_norm = self.normalize_feature_map(feature_map)
        feature_img = Image.fromarray(feature_norm, mode='L')
        feature_img_resized = feature_img.resize((params['vis_width'], params['vis_height']),
                                                NEAREST)
        feature_img_rgb = feature_img_resized.convert('RGB')
        canvas.paste(feature_img_rgb, (params['x_pos'], params['y_pos']))

        # Draw central point square
        self._draw_central_point_square(draw, feature_map, params, index)

        # Add text information
        self._draw_feature_info(draw, feature_map, layer_name, params, index)

    def _draw_central_point_square(self, draw: ImageDraw.ImageDraw,
                                  feature_map: np.ndarray,
                                  params: Dict[str, Any],
                                  index: int) -> None:
        """Draw square around central point."""
        center_h, center_w = FeatureMapAnalyzer.get_central_point(feature_map)
        color = self.colors[index % len(self.colors)]

        vis_center_x = params['x_pos'] + center_w * params['scale_factor']
        vis_center_y = params['y_pos'] + center_h * params['scale_factor']

        square_size = max(3, params['scale_factor'])
        draw.rectangle([
            vis_center_x - square_size, vis_center_y - square_size,
            vis_center_x + square_size, vis_center_y + square_size
        ], outline=color, width=2)

    def _draw_feature_info(self, draw: ImageDraw.ImageDraw,
                          feature_map: np.ndarray,
                          layer_name: str,
                          params: Dict[str, Any],
                          index: int) -> None:
        """Draw feature map information text."""
        height, width, channels = feature_map.shape
        center_h, center_w = FeatureMapAnalyzer.get_central_point(feature_map)
        color = self.colors[index % len(self.colors)]

        text_y = params['y_pos'] + params['vis_height'] + 5
        draw.text((params['x_pos'], text_y), layer_name, fill='black', font=self.font)
        draw.text((params['x_pos'], text_y + 15), f"{height}x{width}x{channels}", fill='gray', font=self.font)
        draw.text((params['x_pos'], text_y + 30), f"Center: ({center_h},{center_w})", fill=color, font=self.font)

    def add_legend(self, draw: ImageDraw.ImageDraw) -> None:
        """Add legend to the visualization."""
        legend_x = 50
        legend_y = 470
        draw.text((legend_x, legend_y), "Legend:", fill='black', font=self.title_font)
        draw.text((legend_x, legend_y + 20), "- Grayscale patches show average feature activation", fill='black', font=self.font)
        draw.text((legend_x, legend_y + 35), "- Colored squares mark central points of each feature map", fill='black', font=self.font)
        draw.text((legend_x, legend_y + 50), "- Numbers show feature map dimensions and central coordinates", fill='black', font=self.font)


def extract_features_and_draw_squares(image_path: str, output_path: str = "feature_visualization.jpg") -> Optional[str]:
    """
    Main function to extract features and create visualization.

    Args:
        image_path: Path to the input image file
        output_path: Path for the output visualization image

    Returns:
        Path to the saved visualization or None if failed
    """
    try:
        # Initialize components
        extractor = VGG16FeatureExtractor()
        analyzer = FeatureMapAnalyzer()
        visualizer = FeatureMapVisualizer()

        # Extract features
        features, layer_names = extractor.extract_features(image_path)

        # Load original image
        original_img = Image.open(image_path)

        # Create canvas
        canvas, draw = visualizer.create_canvas(original_img, image_path)

        print(f"Processing {len(features)} feature maps...")

        # Visualize each feature map
        for i, (feature_map, layer_name) in enumerate(zip(features, layer_names)):
            visualizer.draw_feature_map(canvas, draw, feature_map, layer_name, i)

            # Print statistics
            stats = analyzer.get_feature_statistics(feature_map)
            center_h, center_w = analyzer.get_central_point(feature_map)
            print(f"Layer {layer_name}: shape={feature_map.shape}, "
                  f"center=({center_h},{center_w}), "
                  f"central_mean={stats['mean']:.3f}")

        # Add legend
        visualizer.add_legend(draw)

        # Save the visualization
        canvas.save(output_path, quality=95)
        print(f"\nVisualization saved to: {output_path}")
        print(f"Canvas size: {canvas.size}")

        return output_path

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


# Example usage and testing functions
def test_feature_extraction():
    """Test the feature extraction functionality."""
    extractor = VGG16FeatureExtractor()
    print(f"Model loaded successfully")
    print(f"Number of convolutional layers: {len(extractor.layer_names)}")
    print(f"Layer names: {extractor.layer_names}")

def test_feature_analysis():
    """Test the feature analysis functionality."""
    # Create a dummy feature map for testing
    dummy_feature = np.random.rand(14, 14, 512)
    analyzer = FeatureMapAnalyzer()

    center = analyzer.get_central_point(dummy_feature)
    central_values = analyzer.get_central_values(dummy_feature)
    stats = analyzer.get_feature_statistics(dummy_feature)

    print(f"Central point: {center}")
    print(f"Central values shape: {central_values.shape}")
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    # Test individual components
    test_feature_extraction()
    test_feature_analysis()

    # Main usage
    image_path = "GhostGirl.png"
    output_path = "vgg16_feature_visualization.jpg"

    result = extract_features_and_draw_squares(image_path, output_path)
    if result:
        print(f"Successfully created visualization: {result}")
    else:
        print("Failed to create visualization")