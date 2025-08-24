#!/usr/bin/env python3
"""
Velocity calculation module for tracking object movement and calculating 3D velocity.

This module provides functionality to:
- Track objects across frames using image comparison
- Calculate 3D velocity based on position and depth changes
- Manage object matching and velocity history
"""

import time
import uuid
import math
from typing import List, Dict, Optional, Any
import numpy as np
import cv2

from rose.storage.memory_image_storage import MemoryImageStorage
from rose.storage.redis_image_storage import RedisImageStorage
from rose.processing.image_comparator import ImageComparator


class VelocityCalculator:
    """
    Calculates and tracks object velocity using image comparison and depth estimation.
    
    This class handles the identification of objects across frames and calculates
    their 3D velocity based on position changes and depth information.
    """

    def __init__(self, 
                 memory_storage: MemoryImageStorage,
                 redis_storage: Optional[RedisImageStorage] = None,
                 image_comparator: Optional[ImageComparator] = None):
        """
        Initialize the velocity calculator.

        Args:
            memory_storage (MemoryImageStorage): Memory storage for object tracking
            redis_storage (Optional[RedisImageStorage]): Redis storage for persistence
            image_comparator (Optional[ImageComparator]): Image comparison utility
        """
        self.memory_storage = memory_storage
        self.redis_storage = redis_storage
        self.image_comparator = image_comparator or ImageComparator()
        self.frame_count = 0

    def identify_objects_with_velocity(self, 
                                     detections: List[Dict], 
                                     depth_map: np.ndarray, 
                                     segmentation_masks: Optional[np.ndarray],
                                     original_frame: np.ndarray) -> None:
        """
        Identify objects with velocity tracking using memory storage and image comparison.
        
        This method processes the results of object detection, depth estimation,
        and image segmentation to track object movement and calculate velocity.
        
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
            if len(self.memory_storage.list_all_images()) == 0:
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
            
            dx = (current_center_x - prev_center_x)
            dy = (current_center_y - prev_center_y)
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

    def set_frame_count(self, frame_count: int) -> None:
        """
        Set the current frame count for tracking.
        
        Args:
            frame_count: Current frame number
        """
        self.frame_count = frame_count

    def get_velocity_stats(self) -> Dict[str, Any]:
        """
        Get velocity tracking statistics.
        
        Returns:
            Dict[str, Any]: Velocity tracking statistics
        """
        try:
            # Get all objects and filter for those with velocity data
            all_objects = self.memory_storage.list_all_images()
            
            velocities = []
            for obj in all_objects:
                metadata = obj.get('metadata', {})
                if 'velocity_mps' in metadata:
                    velocities.append(metadata['velocity_mps'])
            
            if not velocities:
                return {'total_tracked_objects': 0, 'average_velocity': 0.0}
            
            return {
                'total_tracked_objects': len(velocities),
                'average_velocity': sum(velocities) / len(velocities),
                'min_velocity': min(velocities),
                'max_velocity': max(velocities),
                'velocity_samples': len(velocities)
            }
        except Exception as e:
            print(f"Error getting velocity stats: {e}")
            return {}

