#!/usr/bin/env python3
"""
Command-line interface for processing single images using Rose modules.

Supports depth estimation and edge detection on input images.

Usage:
    python process_single_image.py <input_image_path> [output_image_path]

Example:
    python process_single_image.py input.jpg depth_result.png
"""

# Add the project root to the path to import from rose modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# flake8: noqa: E402
import argparse
from typing import Optional

import numpy as np
from PIL import Image

from rose.processing.depth_estimator import DepthEstimator
from rose.processing.edge_detector import EdgeDetector
from rose.processing.feature_detector import FeatureDetector
from rose.processing.image_segmenter import ImageSegmenter
from rose.processing.object_detector import ObjectDetector
from rose.preprocessing.image_utils import ImagePreprocessor
from rose.postprocessing.image_creator import ImageCreator
from rose.postprocessing.terminal_output import TerminalOutput


def process_image(input_path: str, output_path: Optional[str] = None,
                 use_colormap: bool = True,
                 colormap: str = 'viridis',
                 use_zoedepth: bool = False,
                 use_edge_detection: bool = False,
                 edge_method: str = 'canny',
                 hed_prototxt: Optional[str] = None,
                 hed_caffemodel: Optional[str] = None,
                 use_feature_detection: bool = False,
                 n_features: int = 500,
                 use_segmentation: bool = False,
                 segmentation_prompts: Optional[list] = None,
                 use_object_detection: bool = False,
                 object_model: str = 'faster_rcnn',
                 object_confidence: float = 0.5) -> None:
    """
    Estimate depth from an input image and save the result.

    Args:
        input_path (str): Path to the input image
        output_path (Optional[str]): Path for the output image (auto-generated if None)
        use_colormap (bool): Whether to use colormap visualization
        colormap (str): Colormap name for visualization
        use_zoedepth (bool): Whether to use ZoeDepth model instead of DPT
        use_edge_detection (bool): Whether to perform edge detection on the depth map
        edge_method (str): Edge detection method ('canny' or 'hed')
        hed_prototxt (Optional[str]): Path to HED prototxt file (required for HED method)
        hed_caffemodel (Optional[str]): Path to HED caffemodel file (required for HED method)
        use_feature_detection (bool): Whether to perform feature detection on the original image
        n_features (int): Number of features to detect (default: 500)
        use_segmentation (bool): Whether to perform image segmentation
        segmentation_prompts (Optional[list]): List of text prompts for segmentation
        use_object_detection (bool): Whether to perform object detection on the original image
        object_model (str): Object detection model to use ('faster_rcnn' or 'ssd')
        object_confidence (float): Minimum confidence threshold for object detection
    """
    # Validate input file
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Generate output paths if not provided
    if output_path is None:
        input_file = Path(input_path)
        model_suffix = "_zoedepth" if use_zoedepth else "_depth"

        # Initialize output paths
        depth_output_path = str(input_file.parent / f"{input_file.stem}{model_suffix}{input_file.suffix}")
        edge_output_path = None
        feature_output_path = None

        if use_edge_detection:
            edge_output_path = str(input_file.parent / f"{input_file.stem}_edges_{edge_method}{input_file.suffix}")

        if use_feature_detection:
            feature_output_path = str(input_file.parent / f"{input_file.stem}_features{input_file.suffix}")

        if use_segmentation:
            segmentation_output_path = str(input_file.parent / f"{input_file.stem}_segmentation{input_file.suffix}")
    else:
        # Use provided output path for depth, generate other paths if needed
        depth_output_path = output_path
        input_file = Path(output_path)
        edge_output_path = None
        feature_output_path = None
        segmentation_output_path = None

        if use_edge_detection:
            edge_output_path = str(input_file.parent / f"{input_file.stem}_edges_{edge_method}{input_file.suffix}")

        if use_feature_detection:
            feature_output_path = str(input_file.parent / f"{input_file.stem}_features{input_file.suffix}")

        if use_segmentation:
            segmentation_output_path = str(input_file.parent / f"{input_file.stem}_segmentation{input_file.suffix}")

    print(f"Processing image: {input_path}")
    print(f"Using {'ZoeDepth' if use_zoedepth else 'DPT'} model...")
    if use_edge_detection:
        print(f"Applying {edge_method.upper()} edge detection...")
    if use_feature_detection:
        print(f"Applying feature detection with {n_features} features...")
    if use_segmentation:
        print(f"Applying image segmentation with prompts: {segmentation_prompts}")
    if use_object_detection:
        print(f"Applying object detection using {object_model.upper()} model (confidence: {object_confidence})")

    try:
        # Load and preprocess image
        input_image = Image.open(input_path)
        input_image = ImagePreprocessor.ensure_rgb_pil_image(input_image)

        # Convert PIL image to numpy array for processing
        original_image_array = np.array(input_image)

        # Initialize depth estimator
        depth_estimator = DepthEstimator()

        # Estimate depth
        print("Estimating depth...")
        if use_zoedepth:
            depth_map = depth_estimator.estimate_depth_zoedepth(input_image)
            if depth_map is None:
                raise RuntimeError("ZoeDepth estimation failed")
        else:
            depth_map = depth_estimator.estimate_depth(input_image)

        # Save depth map result
        print(f"Saving depth map to: {depth_output_path}")
        if use_colormap:
            ImageCreator.save_depth_map_as_image(depth_map, depth_output_path, colormap)
        else:
            ImageCreator.save_depth_map_raw(depth_map, depth_output_path)

        # Apply edge detection if requested
        if use_edge_detection:
            print("Performing edge detection on original image...")

            # Initialize edge detector
            if edge_method == 'hed':
                if hed_prototxt is None or hed_caffemodel is None:
                    raise ValueError("HED method requires both --hed-prototxt and --hed-caffemodel arguments")
                edge_detector = EdgeDetector(hed_prototxt, hed_caffemodel)
                edges = edge_detector.hed(original_image_array)
            else:  # canny
                edge_detector = EdgeDetector()
                edges = edge_detector.canny(original_image_array)

            # Save edge detection result
            print(f"Saving edge detection result to: {edge_output_path}")
            if use_colormap:
                ImageCreator.save_depth_map_as_image(edges, edge_output_path, colormap)
            else:
                ImageCreator.save_depth_map_raw(edges, edge_output_path)

        # Apply feature detection if requested
        if use_feature_detection:
            print("Performing feature detection on original image...")

            # Initialize feature detector
            feature_detector = FeatureDetector(n_features=n_features)

            # Detect features
            keypoints, descriptors = feature_detector.detect_and_compute(original_image_array)

            # Visualize and save feature detection result
            print(f"Saving feature detection result to: {feature_output_path}")
            ImageCreator.visualize_features(original_image_array, keypoints, descriptors, feature_output_path)

        # Apply image segmentation if requested
        if use_segmentation:
            print("Performing image segmentation...")

            # Initialize image segmenter
            segmenter = ImageSegmenter()

            # Perform segmentation
            masks = segmenter.segment(input_image, segmentation_prompts)

            # Visualize and save segmentation result
            print(f"Saving segmentation result to: {segmentation_output_path}")
            ImageCreator.visualize_segmentation(original_image_array, masks, segmentation_prompts, segmentation_output_path)

        # Apply object detection if requested
        if use_object_detection:
            print(f"Performing object detection using {object_model.upper()} model...")

            # Initialize object detector
            object_detector = ObjectDetector(
                model_type=object_model,
                confidence_threshold=object_confidence
            )

            # Detect objects
            detections = object_detector.detect_objects(original_image_array)

            # Get detection summary
            summary = object_detector.get_detection_summary(detections)

            # Print results to terminal
            TerminalOutput.print_object_detection_results(detections, summary)

        print("Processing completed successfully!")

    except Exception as e:
        print(f"Error during depth estimation: {e}")
        sys.exit(1)


def main() -> None:
    """Main function to handle command-line arguments and execute depth estimation."""
    parser = argparse.ArgumentParser(
        description="Process a single image using Rose modules (depth estimation and edge detection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_single_image.py input.jpg
  python process_single_image.py input.jpg output_depth.png
  python process_single_image.py input.jpg --no-colormap
  python process_single_image.py input.jpg --colormap plasma
  python process_single_image.py input.jpg --zoedepth
  python process_single_image.py input.jpg --zoedepth --colormap inferno
  python process_single_image.py input.jpg --edges
  python process_single_image.py input.jpg --edges --edge-method canny
  python process_single_image.py input.jpg --edges --edge-method hed --hed-prototxt model.prototxt --hed-caffemodel model.caffemodel
  python process_single_image.py input.jpg --features
  python process_single_image.py input.jpg --features --n-features 1000
  python process_single_image.py input.jpg --edges --features --n-features 750
  python process_single_image.py input.jpg --segment --prompts "person" "background"
  python process_single_image.py input.jpg --segment --prompts "face" "hair" "clothing"
  python process_single_image.py input.jpg --edges --features --segment --prompts "person" "object"
  python process_single_image.py input.jpg --objects
  python process_single_image.py input.jpg --objects --object-model ssd
  python process_single_image.py input.jpg --objects --object-confidence 0.7
  python process_single_image.py input.jpg --objects --object-model ssd --object-confidence 0.8
  python process_single_image.py input.jpg --edges --features --objects --segment --prompts "person" "object"
        """
    )

    parser.add_argument(
        "input_path",
        help="Path to the input image file"
    )

    parser.add_argument(
        "output_path",
        nargs="?",
        help="Path for the output depth map image (auto-generated if not provided)"
    )

    parser.add_argument(
        "--no-colormap",
        action="store_true",
        help="Save as raw grayscale image instead of colormap visualization"
    )

    parser.add_argument(
        "--colormap",
        default="viridis",
        choices=["viridis", "plasma", "inferno", "magma", "cividis", "gray"],
        help="Colormap for visualization (default: viridis)"
    )

    parser.add_argument(
        "--zoedepth",
        action="store_true",
        help="Use ZoeDepth model instead of DPT for depth estimation"
    )

    parser.add_argument(
        "--edges",
        action="store_true",
        help="Apply edge detection to the depth map"
    )

    parser.add_argument(
        "--edge-method",
        choices=["canny", "hed"],
        default="canny",
        help="Edge detection method to use (default: canny)"
    )

    parser.add_argument(
        "--hed-prototxt",
        help="Path to HED prototxt file (required for HED edge detection)"
    )

    parser.add_argument(
        "--hed-caffemodel",
        help="Path to HED caffemodel file (required for HED edge detection)"
    )

    parser.add_argument(
        "--features",
        action="store_true",
        help="Apply feature detection to the original image"
    )

    parser.add_argument(
        "--n-features",
        type=int,
        default=500,
        help="Number of features to detect (default: 500)"
    )

    parser.add_argument(
        "--segment",
        action="store_true",
        help="Apply image segmentation to the original image"
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Text prompts for segmentation (e.g., 'person' 'background' 'object')"
    )

    parser.add_argument(
        "--objects",
        action="store_true",
        help="Apply object detection to the original image"
    )

    parser.add_argument(
        "--object-model",
        choices=["faster_rcnn", "ssd"],
        default="faster_rcnn",
        help="Object detection model to use (default: faster_rcnn)"
    )

    parser.add_argument(
        "--object-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for object detection (default: 0.5)"
    )

    args = parser.parse_args()

    try:
        process_image(
            input_path=args.input_path,
            output_path=args.output_path,
            use_colormap=not args.no_colormap,
            colormap=args.colormap,
            use_zoedepth=args.zoedepth,
            use_edge_detection=args.edges,
            edge_method=args.edge_method,
            hed_prototxt=args.hed_prototxt,
            hed_caffemodel=args.hed_caffemodel,
            use_feature_detection=args.features,
            n_features=args.n_features,
            use_segmentation=args.segment,
            segmentation_prompts=args.prompts,
            use_object_detection=args.objects,
            object_model=args.object_model,
            object_confidence=args.object_confidence
        )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
