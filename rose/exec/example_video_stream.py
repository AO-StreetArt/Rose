#!/usr/bin/env python3
"""
Example script demonstrating video stream processing using Rose modules.

This script shows how to use the VideoProcessor class programmatically
and can be used as a starting point for custom video processing applications.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rose.exec.process_video_stream import VideoProcessor


def example_basic_processing():
    """Example of basic video processing setup."""
    print("=== Basic Video Processing Example ===")
    
    # Initialize processor with default settings
    processor = VideoProcessor()
    
    print("VideoProcessor initialized with default settings:")
    print(f"- Depth model: DPT")
    print(f"- Object detection: Faster R-CNN")
    print(f"- Object confidence: 0.5")
    print(f"- Colormap: viridis")
    print(f"- Max objects for segmentation: 5")
    
    return processor


def example_custom_processing():
    """Example of custom video processing setup."""
    print("\n=== Custom Video Processing Example ===")
    
    # Initialize processor with custom settings
    processor = VideoProcessor(
        use_zoedepth=True,           # Use ZoeDepth for better depth estimation
        object_confidence=0.7,       # Higher confidence threshold
        object_model='ssd',          # Use SSD for faster detection
        colormap='plasma',           # Different colormap
        max_objects_for_segmentation=3  # Limit segmentation objects
    )
    
    print("VideoProcessor initialized with custom settings:")
    print(f"- Depth model: ZoeDepth")
    print(f"- Object detection: SSD")
    print(f"- Object confidence: 0.7")
    print(f"- Colormap: plasma")
    print(f"- Max objects for segmentation: 3")
    
    return processor


def example_frame_processing(processor: VideoProcessor):
    """Example of processing a single frame."""
    print("\n=== Frame Processing Example ===")
    
    # Create a simple test frame (you would normally get this from a camera)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Processing test frame...")
    start_time = time.time()
    
    # Process the frame
    result = processor.process_frame(test_frame)
    
    processing_time = time.time() - start_time
    print(f"Frame processed in {processing_time:.3f} seconds")
    
    # Display results
    print(f"Depth map shape: {result['depth_map'].shape if result['depth_map'] is not None else 'None'}")
    print(f"Number of detections: {len(result['detections'])}")
    print(f"Object prompts: {result['object_prompts']}")
    print(f"Segmentation masks shape: {result['segmentation_masks'].shape if result['segmentation_masks'] is not None else 'None'}")
    
    # Show detection details
    if result['detections']:
        print("\nDetected objects:")
        for i, detection in enumerate(result['detections']):
            print(f"  {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.3f})")
    
    return result


def example_visualization(processor: VideoProcessor, result: dict):
    """Example of creating visualizations."""
    print("\n=== Visualization Example ===")
    
    # Create visualizations
    print("Creating visualizations...")
    original_frame, depth_vis, seg_vis = processor.create_visualization(result)
    
    print(f"Original frame shape: {original_frame.shape}")
    print(f"Depth visualization shape: {depth_vis.shape}")
    print(f"Segmentation visualization shape: {seg_vis.shape}")
    
    # Draw detections on original frame
    frame_with_detections = processor.draw_detections(original_frame, result['detections'])
    print(f"Frame with detections shape: {frame_with_detections.shape}")
    
    print("Visualizations created successfully!")
    
    return original_frame, depth_vis, seg_vis, frame_with_detections


def example_integration():
    """Example of integrating video processing into a larger application."""
    print("\n=== Integration Example ===")
    
    # This shows how you might integrate the video processor into a larger application
    class VideoAnalysisApp:
        def __init__(self):
            self.processor = VideoProcessor(
                use_zoedepth=False,  # Use DPT for faster processing
                object_confidence=0.6,
                max_objects_for_segmentation=4
            )
            self.frame_count = 0
            self.total_processing_time = 0
        
        def process_frame(self, frame: np.ndarray):
            """Process a single frame and return analysis results."""
            start_time = time.time()
            
            # Process frame
            result = self.processor.process_frame(frame)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.frame_count += 1
            
            # Create visualizations
            original, depth, seg, detections = self.processor.create_visualization(result)
            
            # Return analysis results
            analysis = {
                'frame_number': self.frame_count,
                'processing_time': processing_time,
                'avg_processing_time': self.total_processing_time / self.frame_count,
                'detections': result['detections'],
                'object_count': len(result['detections']),
                'depth_available': result['depth_map'] is not None,
                'segmentation_available': result['segmentation_masks'] is not None,
                'visualizations': {
                    'original': original,
                    'depth': depth,
                    'segmentation': seg,
                    'detections': detections
                }
            }
            
            return analysis
        
        def get_statistics(self):
            """Get processing statistics."""
            return {
                'total_frames': self.frame_count,
                'total_time': self.total_processing_time,
                'avg_fps': self.frame_count / self.total_processing_time if self.total_processing_time > 0 else 0
            }
    
    # Create and test the application
    app = VideoAnalysisApp()
    
    # Process a few test frames
    for i in range(3):
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        analysis = app.process_frame(test_frame)
        
        print(f"Frame {analysis['frame_number']}:")
        print(f"  Processing time: {analysis['processing_time']:.3f}s")
        print(f"  Objects detected: {analysis['object_count']}")
        print(f"  Depth available: {analysis['depth_available']}")
        print(f"  Segmentation available: {analysis['segmentation_available']}")
    
    # Show final statistics
    stats = app.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Total processing time: {stats['total_time']:.3f}s")
    print(f"  Average FPS: {stats['avg_fps']:.2f}")


def main():
    """Run all examples."""
    print("Video Stream Processing Examples")
    print("=" * 50)
    
    try:
        # Example 1: Basic processing
        processor1 = example_basic_processing()
        
        # Example 2: Custom processing
        processor2 = example_custom_processing()
        
        # Example 3: Frame processing
        result = example_frame_processing(processor1)
        
        # Example 4: Visualization
        example_visualization(processor1, result)
        
        # Example 5: Integration
        example_integration()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 