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
import uuid
import math

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

            # Start velocity identification on a separate thread (non-blocking)
            if detections:
                threading.Thread(
                    target=self.identify_objects_with_velocity,
                    args=(detections, depth_map, segmentation_masks, frame),
                    daemon=True
                ).start()

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

    def identify_objects_with_velocity(self, 
                                     detections: List[Dict], 
                                     depth_map: np.ndarray, 
                                     segmentation_masks: Optional[np.ndarray],
                                     original_frame: np.ndarray) -> None:
        """
        Identify objects with velocity tracking using memory storage and image comparison.
        
        This method runs on a separate thread and processes the results of object detection,
        depth estimation, and image segmentation to track object movement and calculate velocity.
        
        Args:
            detections: List of detected objects with bounding boxes
            depth_map: Depth estimation results
            segmentation_masks: Image segmentation masks
            original_frame: Original frame for processing
        """
        try:
            print(f"Starting object velocity identification for {len(detections)} detections...")
            
            if not detections:
                print("No detections to process")
                return
            
            # If memory storage is empty, populate it with current detections
            if len(self.memory_storage) == 0:
                print("Memory storage is empty, populating with current detections...")
                self._populate_initial_detections(detections, depth_map, segmentation_masks, original_frame)
                return
            
            # Process each detection for velocity tracking
            processed_count = 0
            for detection in detections:
                try:
                    self._process_detection_for_velocity(detection, depth_map, segmentation_masks, original_frame)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing detection {detection.get('class_name', 'unknown')}: {e}")
                    continue
            
            print(f"Processed {processed_count} detections for velocity tracking")
            
            # Sync data to Redis and clear memory storage
            if self.redis_storage is not None:
                print("Syncing data to Redis and clearing memory storage...")
                try:
                    transferred_count = self.memory_storage.transfer_to_redis(self.redis_storage)
                    cleared_count = self.memory_storage.clear_all()
                    print(f"Transferred {transferred_count} images to Redis, cleared {cleared_count} from memory")
                except Exception as e:
                    print(f"Error syncing to Redis: {e}")
            else:
                print("Redis storage not available, keeping data in memory")
            
        except Exception as e:
            print(f"Error in identify_objects_with_velocity: {e}")
            import traceback
            traceback.print_exc()

    def _populate_initial_detections(self, 
                                   detections: List[Dict], 
                                   depth_map: np.ndarray, 
                                   segmentation_masks: Optional[np.ndarray],
                                   original_frame: np.ndarray) -> None:
        """
        Populate memory storage with initial detections.
        
        Args:
            detections: List of detected objects
            depth_map: Depth estimation results
            segmentation_masks: Image segmentation masks
            original_frame: Original frame
        """
        try:
            for detection in detections:
                bbox = detection.get('bbox', [])
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Extract object region from original frame
                    object_region = original_frame[y1:y2, x1:x2]
                    
                    # Calculate depth for the object region using segmentation
                    object_depth = self._calculate_object_depth(bbox, depth_map, segmentation_masks)
                    
                    # Create metadata
                    metadata = {
                        'bbox': bbox,
                        'class_name': class_name,
                        'confidence': confidence,
                        'depth': object_depth,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2,
                        'timestamp': time.time(),
                        'frame_number': getattr(self, 'frame_count', 0)
                    }
                    
                    # Store in memory with object name as tag
                    tags = [class_name.lower()]
                    key = self.memory_storage.store_image(
                        image_data=object_region,
                        metadata=metadata,
                        tags=tags
                    )
                    
                    print(f"Stored initial detection: {class_name} (key: {key}) with depth: {object_depth:.2f}m")
                    
        except Exception as e:
            print(f"Error populating initial detections: {e}")

    def _process_detection_for_velocity(self, 
                                      detection: Dict, 
                                      depth_map: np.ndarray, 
                                      segmentation_masks: Optional[np.ndarray],
                                      original_frame: np.ndarray) -> None:
        """
        Process a single detection for velocity tracking.
        
        Args:
            detection: Single detection result
            depth_map: Depth estimation results
            segmentation_masks: Image segmentation masks
            original_frame: Original frame
        """
        try:
            bbox = detection.get('bbox', [])
            class_name = detection.get('class_name', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            if len(bbox) != 4:
                return
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extract current object region
            current_object_region = original_frame[y1:y2, x1:x2]
            
            # Calculate current depth
            current_depth = self._calculate_object_depth(bbox, depth_map, segmentation_masks)
            
            # Search for matching objects in memory storage
            matching_objects = self.memory_storage.search_by_tags([class_name.lower()], operator="OR")
            
            if matching_objects:
                # Find the best match using image comparison
                best_match = self._find_best_image_match(current_object_region, matching_objects)
                
                if best_match:
                    # Generate match ID
                    match_id = str(uuid.uuid4())
                    
                    # Calculate velocity using 3D distance
                    velocity = self._calculate_3d_velocity(best_match, bbox, current_depth)
                    
                    # Create new metadata with velocity and match ID
                    new_metadata = {
                        'bbox': bbox,
                        'class_name': class_name,
                        'confidence': confidence,
                        'depth': current_depth,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2,
                        'timestamp': time.time(),
                        'frame_number': getattr(self, 'frame_count', 0),
                        'match_id': match_id,
                        'velocity_mps': velocity,
                        'previous_key': best_match['key']
                    }
                    
                    # Store new detection with match ID tag
                    tags = [class_name.lower(), f"match_{match_id}"]
                    key = self.memory_storage.store_image(
                        image_data=current_object_region,
                        metadata=new_metadata,
                        tags=tags
                    )
                    
                    # Update previous detection with match ID
                    self._update_metadata_with_match_id(best_match['key'], match_id)
                    
                    print(f"Object {class_name} matched! Match ID: {match_id}, Velocity: {velocity:.2f} m/s")
                else:
                    # No good match found, store as new detection
                    self._store_new_detection(current_object_region, detection, current_depth)
            else:
                # No previous detections of this class, store as new
                self._store_new_detection(current_object_region, detection, current_depth)
                
        except Exception as e:
            print(f"Error processing detection for velocity: {e}")

    def _calculate_object_depth(self, bbox: List[int], depth_map: np.ndarray, 
                               segmentation_masks: Optional[np.ndarray]) -> float:
        """
        Calculate the depth for an object using its bounding box and segmentation.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth_map: Depth estimation results
            segmentation_masks: Image segmentation masks
            
        Returns:
            float: Average depth in meters
        """
        try:
            if depth_map is None:
                return 0.0
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within bounds
            height, width = depth_map.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Extract depth values for the object region
            object_depth_region = depth_map[y1:y2, x1:x2]
            
            # If segmentation masks are available, use them to refine depth calculation
            if segmentation_masks is not None:
                # For simplicity, use the center point of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center_x = max(0, min(center_x, width - 1))
                center_y = max(0, min(center_y, height - 1))
                return float(depth_map[center_y, center_x])
            else:
                # Use average depth of the bounding box region
                valid_depths = object_depth_region[object_depth_region > 0]
                if len(valid_depths) > 0:
                    return float(np.mean(valid_depths))
                else:
                    return 0.0
                    
        except Exception as e:
            print(f"Error calculating object depth: {e}")
            return 0.0

    def _find_best_image_match(self, current_object_region: np.ndarray, 
                              matching_objects: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching image from stored objects.
        
        Args:
            current_object_region: Current object image region
            matching_objects: List of potential matches from storage
            
        Returns:
            Optional[Dict]: Best match or None
        """
        try:
            best_match = None
            best_similarity = 0.0
            similarity_threshold = 0.6  # Lowered threshold for better matching
            
            if not matching_objects:
                return None
                
            for match_obj in matching_objects:
                # Retrieve the stored image
                stored_image = self.memory_storage.retrieve_image(match_obj['key'], format="numpy")
                
                if stored_image is not None:
                    # Ensure both images have the same dimensions for comparison
                    try:
                        # Resize current region to match stored image dimensions
                        stored_height, stored_width = stored_image.shape[:2]
                        current_height, current_width = current_object_region.shape[:2]
                        
                        if current_height != stored_height or current_width != stored_width:
                            # Resize current region to match stored image
                            current_resized = cv2.resize(current_object_region, (stored_width, stored_height))
                        else:
                            current_resized = current_object_region
                        
                        # Compare images using the image comparator
                        comparison_result = self.image_comparator.compare_images(
                            current_resized, 
                            stored_image, 
                            method='vgg16', 
                            normalize=True
                        )
                        
                        similarity_score = comparison_result.get('similarity_score', 0.0)
                        
                        if similarity_score > best_similarity and similarity_score > similarity_threshold:
                            best_similarity = similarity_score
                            best_match = match_obj
                            
                    except Exception as e:
                        print(f"Error comparing images: {e}")
                        continue
            
            if best_match:
                print(f"Best match found with similarity: {best_similarity:.3f}")
            else:
                print(f"No matches found above threshold {similarity_threshold}")
                
            return best_match
            
        except Exception as e:
            print(f"Error finding best image match: {e}")
            return None

    def _calculate_3d_velocity(self, previous_match: Dict, current_bbox: List[int], 
                              current_depth: float) -> float:
        """
        Calculate 3D velocity between previous and current object positions.
        
        Args:
            previous_match: Previous object match from storage
            current_bbox: Current bounding box
            current_depth: Current depth estimate
            
        Returns:
            float: Velocity in meters per second
        """
        try:
            # Get previous position and timestamp
            prev_metadata = previous_match.get('metadata', {})
            prev_center_x = prev_metadata.get('center_x', 0)
            prev_center_y = prev_metadata.get('center_y', 0)
            prev_depth = prev_metadata.get('depth', 0)
            prev_timestamp = prev_metadata.get('timestamp', time.time())
            
            # Calculate current position
            x1, y1, x2, y2 = map(int, current_bbox)
            current_center_x = (x1 + x2) / 2
            current_center_y = (y1 + y2) / 2
            
            # Calculate time difference
            current_timestamp = time.time()
            time_diff = current_timestamp - prev_timestamp
            
            if time_diff <= 0.1:  # Minimum 100ms threshold to avoid division by very small numbers
                return 0.0
            
            # Calculate 3D distance
            # Convert pixel coordinates to meters (assuming camera calibration)
            # For simplicity, we'll use a rough approximation
            pixel_to_meter_ratio = 0.001  # Adjust based on your camera setup
            
            dx = (current_center_x - prev_center_x) * pixel_to_meter_ratio
            dy = (current_center_y - prev_center_y) * pixel_to_meter_ratio
            dz = current_depth - prev_depth
            
            # Calculate 3D distance
            distance_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Calculate velocity (distance / time)
            velocity = distance_3d / time_diff
            
            # Cap velocity to reasonable limits (e.g., 100 m/s)
            velocity = min(velocity, 100.0)
            
            return velocity
            
        except Exception as e:
            print(f"Error calculating 3D velocity: {e}")
            return 0.0

    def _update_metadata_with_match_id(self, key: str, match_id: str) -> bool:
        """
        Update stored metadata with match ID.
        
        Args:
            key: Storage key
            match_id: Generated match ID
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            success = self.memory_storage.update_metadata(key, {'match_id': match_id})
            if success:
                print(f"Updated metadata for key {key} with match ID {match_id}")
            else:
                print(f"Failed to update metadata for key {key}")
            return success
        except Exception as e:
            print(f"Error updating metadata with match ID: {e}")
            return False

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

    def _store_new_detection(self, object_region: np.ndarray, detection: Dict, depth: float) -> None:
        """
        Store a new detection in memory storage.
        
        Args:
            object_region: Object image region
            detection: Detection result
            depth: Calculated depth
        """
        try:
            bbox = detection.get('bbox', [])
            class_name = detection.get('class_name', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                metadata = {
                    'bbox': bbox,
                    'class_name': class_name,
                    'confidence': confidence,
                    'depth': depth,
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2,
                    'timestamp': time.time(),
                    'frame_number': getattr(self, 'frame_count', 0)
                }
                
                tags = [class_name.lower()]
                key = self.memory_storage.store_image(
                    image_data=object_region,
                    metadata=metadata,
                    tags=tags
                )
                
                print(f"Stored new detection: {class_name} (key: {key}) with depth: {depth:.2f}m")
                
        except Exception as e:
            print(f"Error storing new detection: {e}")

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

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw object detection bounding boxes on the frame.

        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict]): Object detection results

        Returns:
            np.ndarray: Frame with detection boxes drawn
        """
        # Handle None frame
        if frame is None:
            return None

        frame_with_boxes = frame.copy()

        # Handle None detections
        if detections is None:
            return frame_with_boxes

        for detection in detections:
            bbox = detection.get('bbox', [])
            class_name = detection.get('class_name', 'Unknown')
            confidence = detection.get('confidence', 0.0)

            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)

                # Draw bounding box
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame_with_boxes, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame_with_boxes, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame_with_boxes


def process_video_stream(camera_id: int = 0,
                        fps: int = 30,
                        use_zoedepth: bool = False,
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
        use_zoedepth (bool): Whether to use ZoeDepth model
        object_confidence (float): Minimum confidence for object detection
        object_model (str): Object detection model to use
        colormap (str): Colormap for depth visualization
        max_objects_for_segmentation (int): Maximum objects to use for segmentation
        frame_skip (int): Process every Nth frame (higher values = faster processing)
    """
    # Initialize video processor
    processor = VideoProcessor(
        use_zoedepth=use_zoedepth,
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
    print(f"Using {'ZoeDepth' if use_zoedepth else 'DPT'} for depth estimation")
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
                original_with_detections = processor.draw_detections(original_frame, result['detections'])

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
                
                # Print storage statistics every 30 frames
                try:
                    storage_stats = processor.get_storage_stats()
                    if storage_stats:
                        memory_count = storage_stats.get('memory_storage', {}).get('total_images', 0)
                        redis_count = storage_stats.get('redis_storage', {}).get('total_images', 0)
                        print(f"Storage: Memory: {memory_count} images, Redis: {redis_count} images")
                except Exception as e:
                    print(f"Error getting storage stats: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        print("Cleaning up...")
        processor.stop_processing()
        cap.release()
        cv2.destroyAllWindows()
        print("Video processing stopped")


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
        "--zoedepth",
        action="store_true",
        help="Use ZoeDepth model instead of DPT for depth estimation"
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

    try:
        process_video_stream(
            camera_id=args.camera,
            fps=args.fps,
            use_zoedepth=args.zoedepth,
            object_confidence=args.object_confidence,
            object_model=args.object_model,
            colormap=args.colormap,
            max_objects_for_segmentation=args.max_objects,
            frame_skip=args.frame_skip,
            use_redis=not args.no_redis
        )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
