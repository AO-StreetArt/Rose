#!/usr/bin/env python3
"""
Command-line interface for processing video streams using Rose modules.

Opens a webcam connection and processes each frame with:
- Depth estimation
- Object detection
- Image segmentation using detected objects as prompts

Displays results in separate windows as video streams.

Usage:
    python process_video_stream.py [--camera CAMERA] [--fps FPS] [--zoedepth] [--object-confidence CONF]

Example:
    python process_video_stream.py --camera 0 --fps 30 --zoedepth
"""

# Add the project root to the path to import from rose modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# flake8: noqa: E402
import argparse
import os
import time
from typing import Optional, List, Dict, Tuple, Any
import threading
import queue
from concurrent.futures import ThreadPoolExecutor


import numpy as np
from PIL import Image
import cv2

from rose.processing.depth_estimator import DepthEstimator
from rose.processing.object_detector import ObjectDetector
from rose.processing.image_segmenter import ImageSegmenter
from rose.preprocessing.image_utils import ImagePreprocessor
from rose.postprocessing.image_creator import ImageCreator
from rose.storage.memory_image_storage import MemoryImageStorage
from rose.storage.redis_image_storage import RedisImageStorage
from rose.processing.image_comparator import ImageComparator
from rose.processing.velocity_calculator import VelocityCalculator


class VideoProcessor:
    """
    Processes video streams with depth estimation, object detection, and image segmentation.
    """

    def __init__(self,
                 use_zoedepth: bool = False,
                 object_confidence: float = 0.5,
                 object_model: str = 'faster_rcnn',
                 colormap: str = 'viridis',
                 max_objects_for_segmentation: int = 5,
                 use_redis: bool = True):
        """
        Initialize the video processor.

        Args:
            use_zoedepth (bool): Whether to use ZoeDepth model instead of DPT
            object_confidence (float): Minimum confidence threshold for object detection
            object_model (str): Object detection model to use
            colormap (str): Colormap for depth visualization
            max_objects_for_segmentation (int): Maximum number of objects to use as segmentation prompts
            use_redis (bool): Whether to use Redis storage
        """
        self.use_zoedepth = use_zoedepth
        self.object_confidence = object_confidence
        self.object_model = object_model
        self.colormap = colormap
        self.max_objects_for_segmentation = max_objects_for_segmentation

        # Initialize processing modules
        print("Initializing depth estimator...")
        self.depth_estimator = DepthEstimator()

        print("Initializing object detector...")
        self.object_detector = ObjectDetector(
            model_type=object_model,
            confidence_threshold=object_confidence
        )

        print("Initializing image segmenter...")
        self.image_segmenter = ImageSegmenter()

        # Initialize storage systems
        print("Initializing memory image storage...")
        self.memory_storage = MemoryImageStorage(max_memory_mb=512, ttl_hours=1)
        
        # Initialize Redis storage only if requested
        self.redis_storage = None
        if use_redis:
            print("Initializing Redis image storage...")
            try:
                self.redis_storage = RedisImageStorage(host="localhost", port=6379, ttl_hours=24)
                print("Redis connection established successfully")
            except Exception as e:
                print(f"Warning: Redis connection failed: {e}")
                print("Falling back to memory-only storage")
                self.redis_storage = None
        else:
            print("Redis storage disabled, using memory-only storage")

        # Initialize image comparator
        print("Initializing image comparator...")
        self.image_comparator = ImageComparator()

        # Initialize velocity calculator
        print("Initializing velocity calculator...")
        self.velocity_calculator = VelocityCalculator(
            memory_storage=self.memory_storage,
            redis_storage=self.redis_storage,
            image_comparator=self.image_comparator
        )

        # Processing queues for threading
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)

        # Thread pool for parallel processing
        # Use optimal number of workers: 2 for depth+detection + 1 for segmentation
        optimal_workers = min(3, os.cpu_count() or 2)
        self.executor = ThreadPoolExecutor(max_workers=optimal_workers)
        print(f"Initialized thread pool with {optimal_workers} workers for parallel processing")

        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.frame_count = 0  # Add frame counter

        # Performance monitoring
        self.processing_times = []
        self.last_processing_start = None

    def extract_object_prompts(self, detections: List[Dict]) -> List[str]:
        """
        Extract object class names from detections to use as segmentation prompts.

        Args:
            detections (List[Dict]): Object detection results

        Returns:
            List[str]: List of object class names to use as prompts
        """
        if not detections:
            return []

        # Extract unique object classes from detections
        object_classes = []
        seen_classes = set()

        for detection in detections:
            class_name = detection.get('class_name', '').lower()
            if class_name and class_name not in seen_classes:
                object_classes.append(class_name)
                seen_classes.add(class_name)

                # Limit the number of objects for segmentation
                if len(object_classes) >= self.max_objects_for_segmentation:
                    break

        return object_classes

    def _estimate_depth(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        """
        Estimate depth for an image. This method runs in parallel.

        Args:
            pil_image (Image.Image): PIL Image to process

        Returns:
            Optional[np.ndarray]: Depth map or None if failed
        """
        try:
            print("Estimating depth...")
            if self.use_zoedepth:
                depth_map = self.depth_estimator.estimate_depth_zoedepth(pil_image)
                if depth_map is None:
                    print("ZoeDepth estimation failed, using DPT...")
                    depth_map = self.depth_estimator.estimate_depth(pil_image)
            else:
                depth_map = self.depth_estimator.estimate_depth(pil_image)
            return depth_map
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return None

    def _detect_objects(self, frame_rgb: np.ndarray) -> List[Dict]:
        """
        Detect objects in an image. This method runs in parallel.

        Args:
            frame_rgb (np.ndarray): RGB image array

        Returns:
            List[Dict]: List of detected objects
        """
        try:
            print("Detecting objects...")
            return self.object_detector.detect_objects(frame_rgb)
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame with depth estimation, object detection, and segmentation.
        Uses parallel processing for independent operations.

        Args:
            frame (np.ndarray): Input frame as numpy array (BGR format from OpenCV)

        Returns:
            Dict: Processing results containing depth map, detections, and segmentation masks
        """
        try:
            # Start performance monitoring
            start_time = time.time()

            # Convert BGR to RGB for processing
            frame_rgb = ImagePreprocessor.convertBGRtoRGB(frame)

            # Convert to PIL Image for processing
            pil_image = Image.fromarray(frame_rgb)

            # Start parallel processing of independent operations
            print("Starting parallel processing...")

            # 1. Submit depth estimation and object detection tasks simultaneously
            depth_future = self.executor.submit(self._estimate_depth, pil_image)
            detection_future = self.executor.submit(self._detect_objects, frame_rgb)

            # 2. Wait for both results (they run in parallel)
            depth_map = depth_future.result()
            detections = detection_future.result()

            # 3. Extract object prompts for segmentation
            object_prompts = self.extract_object_prompts(detections)

            # 4. Image Segmentation (if objects detected) - runs after detection completes
            segmentation_masks = None
            if object_prompts:
                print(f"Performing segmentation with prompts: {object_prompts}")
                segmentation_masks = self.image_segmenter.segment(pil_image, object_prompts)
            else:
                print("No objects detected for segmentation")

            # Record processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Keep only last 100 processing times for rolling average
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            avg_time = sum(self.processing_times) / len(self.processing_times)
            print(f"Frame processed in {processing_time:.3f}s (avg: {avg_time:.3f}s)")

            # Increment frame counter
            self.frame_count += 1

            # Update velocity calculator frame count
            self.velocity_calculator.set_frame_count(self.frame_count)

            # Start velocity identification on a separate thread (non-blocking)
            if detections:
                self.executor.submit(self.velocity_calculator.identify_objects_with_velocity, detections, depth_map, segmentation_masks, frame)

            return {
                'depth_map': depth_map,
                'detections': detections,
                'object_prompts': object_prompts,
                'segmentation_masks': segmentation_masks,
                'original_frame': frame
            }

        except Exception as e:
            print(f"Error processing frame: {e}")
            return {
                'depth_map': None,
                'detections': [],
                'object_prompts': [],
                'segmentation_masks': None,
                'original_frame': frame
            }

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for both memory and Redis storage.
        
        Returns:
            Dict[str, Any]: Storage statistics
        """
        try:
            memory_stats = self.memory_storage.get_storage_stats()
            redis_stats = {}
            
            if self.redis_storage is not None:
                redis_stats = self.redis_storage.get_storage_stats()
            
            return {
                'memory_storage': memory_stats,
                'redis_storage': redis_stats
            }
        except Exception as e:
            print(f"Error getting storage stats: {e}")
            return {}

    def get_velocity_stats(self) -> Dict[str, Any]:
        """
        Get velocity tracking statistics.
        
        Returns:
            Dict[str, Any]: Velocity tracking statistics
        """
        try:
            return self.velocity_calculator.get_velocity_stats()
        except Exception as e:
            print(f"Error getting velocity stats: {e}")
            return {}

    def processing_worker(self):
        """Worker thread for processing frames."""
        while self.is_processing:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.1)

                # Process the frame
                result = self.process_frame(frame)

                # Put result in queue
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Remove old result and put new one
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.result_queue.put_nowait(result)

                self.frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing worker: {e}")
                continue

    def start_processing(self):
        """Start the processing thread."""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("Processing thread started")

    def stop_processing(self):
        """Stop the processing thread and cleanup resources."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

        # Shutdown the thread pool executor
        self.executor.shutdown(wait=True)
        
        # Close Redis connection if it exists
        if hasattr(self, 'redis_storage') and self.redis_storage is not None:
            try:
                self.redis_storage.close()
                print("Redis connection closed")
            except Exception as e:
                print(f"Error closing Redis connection: {e}")
        
        print("Processing thread stopped and thread pool shutdown")

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the current processing session.

        Returns:
            Dict[str, float]: Performance metrics
        """
        if not self.processing_times:
            return {}

        return {
            'total_frames_processed': len(self.processing_times),
            'average_processing_time': sum(self.processing_times) / len(self.processing_times),
            'min_processing_time': min(self.processing_times),
            'max_processing_time': max(self.processing_times),
            'estimated_fps': 1.0 / (sum(self.processing_times) / len(self.processing_times))
        }

    def create_visualization(self, result: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create visualizations for the processing results.

        Args:
            result (Dict): Processing results

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Original frame, depth visualization, segmentation visualization
        """
        original_frame = result['original_frame']

        # Create depth visualization
        depth_vis = original_frame.copy()
        if result['depth_map'] is not None:
            depth_vis = ImageCreator.create_depth_visualization(
                result['depth_map'],
                original_frame,
                self.colormap
            )

        # Create segmentation visualization
        seg_vis = original_frame.copy()
        if result['segmentation_masks'] is not None and result['object_prompts']:
            seg_vis = ImageCreator.create_segmentation_visualization(
                original_frame,
                result['segmentation_masks'],
                result['object_prompts']
            )

        return original_frame, depth_vis, seg_vis

def process_video_stream(camera_id: int = 0,
                        fps: int = 30,
                        object_confidence: float = 0.5,
                        object_model: str = 'faster_rcnn',
                        colormap: str = 'viridis',
                        max_objects_for_segmentation: int = 5,
                        frame_skip: int = 5,
                        use_redis: bool = True) -> None:
    """
    Process video stream from webcam with real-time analysis.

    Args:
        camera_id (int): Camera device ID
        fps (int): Target frames per second
        object_confidence (float): Minimum confidence for object detection
        object_model (str): Object detection model to use
        colormap (str): Colormap for depth visualization
        max_objects_for_segmentation (int): Maximum objects to use for segmentation
        frame_skip (int): Process every Nth frame (higher values = faster processing)
    """
    # Initialize video processor
    processor = VideoProcessor(
        use_zoedepth=False, # Turning off ZoeDepth to standardize using relative depth models instead of absolute
        object_confidence=object_confidence,
        object_model=object_model,
        colormap=colormap,
        max_objects_for_segmentation=max_objects_for_segmentation,
        use_redis=use_redis
    )

    # Open camera
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Get actual camera properties
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Camera opened successfully!")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps}")
    print(f"Object detection model: {object_model.upper()}")
    print(f"Object confidence threshold: {object_confidence}")
    print(f"Colormap: {colormap}")
    print(f"Frame skip: Every {frame_skip}th frame will be processed")
    print(f"Velocity tracking: {'Enabled' if use_redis else 'Memory-only mode'}")
    print("Press 'q' to quit, 's' to save current frame")

    # Start processing thread
    processor.start_processing()

    # Create windows
    cv2.namedWindow('Original + Detections', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)

    # Resize windows
    cv2.resizeWindow('Original + Detections', 640, 480)
    cv2.resizeWindow('Depth Map', 640, 480)
    cv2.resizeWindow('Segmentation', 640, 480)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Frame skipping logic - only process every Nth frame
            if frame_count % frame_skip != 0:
                # Handle key presses for non-processed frames
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    cv2.imwrite(f"frame_{timestamp}.jpg", frame)
                    print(f"Saved frame_{timestamp}.jpg")

                frame_count += 1
                continue

            # Add frame to processing queue
            try:
                processor.frame_queue.put_nowait(frame)
            except queue.Full:
                # Skip frame if queue is full
                continue

            # Get processed result
            try:
                result = processor.result_queue.get_nowait()

                # Create visualizations
                original_frame, depth_vis, seg_vis = processor.create_visualization(result)

                # Draw detections on original frame
                original_with_detections = ImageCreator.draw_detections(original_frame, result['detections'])

                # Display frames
                cv2.imshow('Original + Detections', original_with_detections)
                cv2.imshow('Depth Map', depth_vis)
                cv2.imshow('Segmentation', seg_vis)

                # Print detection info
                if result['detections']:
                    print(f"Detected objects: {[d.get('class_name', 'Unknown') for d in result['detections']]}")

            except queue.Empty:
                # No processed result available yet, show original frame
                cv2.imshow('Original + Detections', frame)
                cv2.imshow('Depth Map', frame)
                cv2.imshow('Segmentation', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested")
                break
            elif key == ord('s'):
                # Save current frame and results
                timestamp = int(time.time())
                cv2.imwrite(f"frame_{timestamp}.jpg", frame)
                if 'result' in locals() and result['depth_map'] is not None:
                    depth_filename = f"depth_{timestamp}.png"
                    ImageCreator.save_depth_map_as_image(result['depth_map'], depth_filename, colormap)
                    print(f"Saved frame_{timestamp}.jpg and {depth_filename}")
                else:
                    print(f"Saved frame_{timestamp}.jpg")

            frame_count += 1

            # Print FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time
                print(f"Current FPS: {current_fps:.2f}")
                
                # Print storage and velocity statistics every 30 frames
                try:
                    storage_stats = processor.get_storage_stats()
                    velocity_stats = processor.get_velocity_stats()
                    
                    if storage_stats:
                        memory_count = storage_stats.get('memory_storage', {}).get('total_images', 0)
                        redis_count = storage_stats.get('redis_storage', {}).get('total_images', 0)
                        print(f"Storage: Memory: {memory_count} images, Redis: {redis_count} images")
                    
                    if velocity_stats:
                        tracked_objects = velocity_stats.get('total_tracked_objects', 0)
                        avg_velocity = velocity_stats.get('average_velocity', 0.0)
                        print(f"Velocity Tracking: {tracked_objects} objects, Avg: {avg_velocity:.2f} m/s")
                except Exception as e:
                    print(f"Error getting statistics: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        print("Cleaning up...")
        processor.stop_processing()
        cap.release()
        cv2.destroyAllWindows()
        print("Video processing stopped")
        sys.exit(1)


def main() -> None:
    """Main function to handle command-line arguments and execute video processing."""
    parser = argparse.ArgumentParser(
        description="Process video stream from webcam with depth estimation, object detection, segmentation, and velocity tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_video_stream.py
  python process_video_stream.py --camera 1 --fps 15
  python process_video_stream.py --zoedepth --object-confidence 0.7
  python process_video_stream.py --object-model ssd --colormap plasma
  python process_video_stream.py --max-objects 3
  python process_video_stream.py --frame-skip 10
  python process_video_stream.py --frame-skip 2 --fps 60
  python process_video_stream.py --no-redis  # Use memory-only storage
        """
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second (default: 30)"
    )

    parser.add_argument(
        "--object-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for object detection (default: 0.5)"
    )

    parser.add_argument(
        "--object-model",
        choices=["faster_rcnn", "ssd"],
        default="faster_rcnn",
        help="Object detection model to use (default: faster_rcnn)"
    )

    parser.add_argument(
        "--colormap",
        default="viridis",
        choices=["viridis", "plasma", "inferno", "magma", "cividis", "gray"],
        help="Colormap for depth visualization (default: viridis)"
    )

    parser.add_argument(
        "--max-objects",
        type=int,
        default=5,
        help="Maximum number of objects to use for segmentation prompts (default: 5)"
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every Nth frame (default: 5, higher values = faster processing)"
    )

    parser.add_argument(
        "--no-redis",
        action="store_true",
        help="Disable Redis storage and use memory-only storage"
    )

    args = parser.parse_args()

    process_video_stream(
        camera_id=args.camera,
        fps=args.fps,
        object_confidence=args.object_confidence,
        object_model=args.object_model,
        colormap=args.colormap,
        max_objects_for_segmentation=args.max_objects,
        frame_skip=args.frame_skip,
        use_redis=not args.no_redis
    )


if __name__ == "__main__":
    main()
