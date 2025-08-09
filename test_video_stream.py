#!/usr/bin/env python3
"""
Simple test script for video stream processing.
This script tests the basic functionality without requiring a webcam.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rose.exec.process_video_stream import VideoProcessor


def create_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a test frame with some simple shapes."""
    # Create a colored test image
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add a blue rectangle
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), -1)
    
    # Add a green circle
    cv2.circle(frame, (450, 200), 80, (0, 255, 0), -1)
    
    # Add a red triangle
    pts = np.array([[400, 350], [500, 350], [450, 450]], np.int32)
    cv2.fillPoly(frame, [pts], (0, 0, 255))
    
    # Add some text
    cv2.putText(frame, "Test Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def test_video_processor():
    """Test the VideoProcessor class with a synthetic frame."""
    print("Testing VideoProcessor...")
    
    # Initialize processor
    processor = VideoProcessor(
        use_zoedepth=False,  # Use DPT for faster testing
        object_confidence=0.3,  # Lower threshold for testing
        max_objects_for_segmentation=3
    )
    
    # Create test frame
    print("Creating test frame...")
    test_frame = create_test_frame()
    
    # Process the frame
    print("Processing test frame...")
    result = processor.process_frame(test_frame)
    
    # Check results
    print(f"Depth map shape: {result['depth_map'].shape if result['depth_map'] is not None else 'None'}")
    print(f"Number of detections: {len(result['detections'])}")
    print(f"Object prompts: {result['object_prompts']}")
    print(f"Segmentation masks shape: {result['segmentation_masks'].shape if result['segmentation_masks'] is not None else 'None'}")
    
    # Test visualization
    print("Testing visualizations...")
    original_frame, depth_vis, seg_vis = processor.create_visualization(result)
    
    print(f"Original frame shape: {original_frame.shape}")
    print(f"Depth visualization shape: {depth_vis.shape}")
    print(f"Segmentation visualization shape: {seg_vis.shape}")
    
    # Test detection drawing
    frame_with_detections = processor.draw_detections(original_frame, result['detections'])
    print(f"Frame with detections shape: {frame_with_detections.shape}")
    
    print("VideoProcessor test completed successfully!")


if __name__ == "__main__":
    # Import cv2 here to avoid issues if not available
    try:
        import cv2
        test_video_processor()
    except ImportError:
        print("OpenCV not available. Skipping test.")
    except Exception as e:
        print(f"Test failed: {e}") 