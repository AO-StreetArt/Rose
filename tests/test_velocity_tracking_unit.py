#!/usr/bin/env python3
"""
Unit tests for velocity tracking functionality in VelocityCalculator.

This module provides comprehensive testing of the velocity tracking system
including object detection, depth calculation, image matching, and velocity computation.
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import time
from pathlib import Path

# Add the project root to the path to import from rose modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rose.storage.memory_image_storage import MemoryImageStorage
from rose.processing.velocity_calculator import VelocityCalculator


class TestVelocityTrackingMethods(unittest.TestCase):
    """Unit tests for velocity tracking methods in VelocityCalculator."""
    
    def setUp(self):
        """Set up test fixtures for each test method."""
        # Mock the required dependencies
        self.mock_image_comparator = Mock()
        
        # Create a real memory storage for testing
        self.memory_storage = MemoryImageStorage(max_memory_mb=100, ttl_hours=1)
        
        # Mock Redis storage
        self.mock_redis_storage = Mock()
        
        # Create VelocityCalculator instance
        self.velocity_calculator = VelocityCalculator(
            memory_storage=self.memory_storage,
            redis_storage=self.mock_redis_storage,
            image_comparator=self.mock_image_comparator
        )
        
        # Set frame count
        self.velocity_calculator.frame_count = 0
    
    def tearDown(self):
        """Clean up after each test method."""
        self.memory_storage.clear_all()
    
    def test_populate_initial_detections_success(self):
        """Test successful population of initial detections."""
        # Create test data
        detections = [
            {
                'bbox': [10, 10, 50, 50],
                'class_name': 'person',
                'confidence': 0.9
            },
            {
                'bbox': [100, 100, 150, 150],
                'class_name': 'car',
                'confidence': 0.8
            }
        ]
        
        depth_map = np.ones((200, 200)) * 5.0  # 5 meters depth
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Call the method
        self.velocity_calculator._populate_initial_detections(
            detections, depth_map, segmentation_masks, original_frame
        )
        
        # Verify storage was populated
        self.assertEqual(len(self.memory_storage), 2)
        
        # Check that objects were stored with correct tags
        person_objects = self.memory_storage.search_by_tags(['person'], operator="OR")
        car_objects = self.memory_storage.search_by_tags(['car'], operator="OR")
        
        self.assertEqual(len(person_objects), 1)
        self.assertEqual(len(car_objects), 1)
        
        # Verify metadata for person object
        person_metadata = person_objects[0]['metadata']
        self.assertEqual(person_metadata['class_name'], 'person')
        self.assertEqual(person_metadata['confidence'], 0.9)
        self.assertEqual(person_metadata['depth'], 5.0)
        self.assertEqual(person_metadata['center_x'], 30.0)
        self.assertEqual(person_metadata['center_y'], 30.0)
        self.assertIn('stored_at', person_metadata)
        self.assertIn('image_size_bytes', person_metadata)
        self.assertIn('tags', person_metadata)
        
        # Verify metadata for car object
        car_metadata = car_objects[0]['metadata']
        self.assertEqual(car_metadata['class_name'], 'car')
        self.assertEqual(car_metadata['confidence'], 0.8)
        self.assertEqual(car_metadata['depth'], 5.0)
        self.assertEqual(car_metadata['center_x'], 125.0)
        self.assertEqual(car_metadata['center_y'], 125.0)
    
    def test_populate_initial_detections_empty_list(self):
        """Test population with empty detections list."""
        detections = []
        depth_map = np.ones((100, 100)) * 3.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Call the method
        self.velocity_calculator._populate_initial_detections(
            detections, depth_map, segmentation_masks, original_frame
        )
        
        # Verify no objects were stored
        self.assertEqual(len(self.memory_storage), 0)
    
    def test_populate_initial_detections_invalid_bbox(self):
        """Test population with invalid bounding box."""
        detections = [
            {
                'bbox': [10, 10],  # Invalid: only 2 coordinates
                'class_name': 'person',
                'confidence': 0.9
            }
        ]
        
        depth_map = np.ones((100, 100)) * 3.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Call the method
        self.velocity_calculator._populate_initial_detections(
            detections, depth_map, segmentation_masks, original_frame
        )
        
        # Verify no objects were stored due to invalid bbox
        self.assertEqual(len(self.memory_storage), 0)
    
    def test_calculate_object_depth_normal_case(self):
        """Test depth calculation for normal bounding box."""
        bbox = [10, 10, 50, 50]
        depth_map = np.ones((100, 100)) * 3.0  # 3 meters depth
        segmentation_masks = None
        
        # Test depth calculation
        depth = self.velocity_calculator._calculate_object_depth(bbox, depth_map, segmentation_masks)
        
        # Should return the depth at the center point (30, 30)
        self.assertEqual(depth, 3.0)
    
    def test_calculate_object_depth_none_depth_map(self):
        """Test depth calculation with None depth map."""
        bbox = [10, 10, 50, 50]
        depth_map = None
        segmentation_masks = None
        
        depth = self.velocity_calculator._calculate_object_depth(bbox, depth_map, segmentation_masks)
        self.assertEqual(depth, 0.0)
    
    def test_calculate_object_depth_out_of_bounds_coordinates(self):
        """Test depth calculation with out-of-bounds coordinates."""
        bbox = [95, 95, 105, 105]  # Partially out of bounds
        depth_map = np.ones((100, 100)) * 3.0
        segmentation_masks = None
        
        depth = self.velocity_calculator._calculate_object_depth(bbox, depth_map, segmentation_masks)
        # Should clamp to valid coordinates and return depth
        self.assertEqual(depth, 3.0)
    
    def test_calculate_object_depth_with_segmentation_masks(self):
        """Test depth calculation when segmentation masks are available."""
        bbox = [10, 10, 50, 50]
        depth_map = np.ones((100, 100)) * 3.0
        segmentation_masks = np.ones((100, 100), dtype=bool)  # Mock segmentation mask
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        depth = self.velocity_calculator._calculate_object_depth(bbox, depth_map, segmentation_masks)
        # Should use center point when segmentation masks are available
        self.assertEqual(depth, 3.0)
    
    def test_calculate_object_depth_edge_coordinates(self):
        """Test depth calculation at edge coordinates."""
        bbox = [0, 0, 10, 10]  # At the edge
        depth_map = np.ones((100, 100)) * 2.5
        segmentation_masks = None
        
        depth = self.velocity_calculator._calculate_object_depth(bbox, depth_map, segmentation_masks)
        # Should handle edge coordinates correctly
        self.assertEqual(depth, 2.5)
    
    def test_find_best_image_match_success(self):
        """Test successful finding of best image match."""
        # Create test data
        current_object_region = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Mock the image comparator to return a high similarity score
        self.velocity_calculator.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        # Create mock matching objects
        matching_objects = [
            {
                'key': 'test_key_1',
                'metadata': {'class_name': 'person'}
            }
        ]
        
        # Mock the memory storage to return a test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.memory_storage.store_image(test_image, {'test': True}, ['test'])
        
        # Patch the retrieve_image method to return our test image
        with patch.object(self.memory_storage, 'retrieve_image', return_value=test_image):
            # Call the method
            result = self.velocity_calculator._find_best_image_match(current_object_region, matching_objects)
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertEqual(result['key'], 'test_key_1')
            
            # Verify image comparator was called
            self.velocity_calculator.image_comparator.compare_images.assert_called_once()
    
    def test_find_best_image_match_no_matches(self):
        """Test finding best match when no objects match."""
        current_object_region = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Mock the image comparator to return a low similarity score
        self.velocity_calculator.image_comparator.compare_images.return_value = {
            'similarity_score': 0.3  # Below threshold
        }
        
        matching_objects = [
            {
                'key': 'test_key_1',
                'metadata': {'class_name': 'person'}
            }
        ]
        
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.memory_storage.store_image(test_image, {'test': True}, ['test'])
        
        with patch.object(self.memory_storage, 'retrieve_image', return_value=test_image):
            result = self.velocity_calculator._find_best_image_match(current_object_region, matching_objects)
            
            # Should return None when no matches above threshold
            self.assertIsNone(result)
    
    def test_find_best_image_match_empty_matching_objects(self):
        """Test finding best match with empty matching objects list."""
        current_object_region = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        matching_objects = []
        
        result = self.velocity_calculator._find_best_image_match(current_object_region, matching_objects)
        
        # Should return None for empty list
        self.assertIsNone(result)
    
    def test_find_best_image_match_different_dimensions(self):
        """Test finding best match with images of different dimensions."""
        current_object_region = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        # Mock the image comparator
        self.velocity_calculator.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        matching_objects = [
            {
                'key': 'test_key_1',
                'metadata': {'class_name': 'person'}
            }
        ]
        
        # Stored image has different dimensions
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.memory_storage.store_image(test_image, {'test': True}, ['test'])
        
        with patch.object(self.memory_storage, 'retrieve_image', return_value=test_image):
            result = self.velocity_calculator._find_best_image_match(current_object_region, matching_objects)
            
            # Should handle dimension differences and still find match
            self.assertIsNotNone(result)
    
    def test_calculate_3d_velocity_normal_case(self):
        """Test 3D velocity calculation for normal case."""
        # Create test data
        previous_match = {
            'metadata': {
                'center_x': 10.0,
                'center_y': 20.0,
                'depth': 5.0,
                'timestamp': time.time() - 1.0  # 1 second ago
            }
        }
        current_bbox = [15, 25, 55, 65]  # 5 pixels offset
        current_depth = 6.0  # 1 meter closer
        
        # Call the method
        velocity = self.velocity_calculator._calculate_3d_velocity(previous_match, current_bbox, current_depth)
        
        # Verify velocity is calculated (should be positive)
        self.assertGreater(velocity, 0.0)
        self.assertLessEqual(velocity, 100.0)  # Should be capped at 100 m/s
    
    def test_calculate_3d_velocity_small_time_difference(self):
        """Test velocity calculation with very small time difference."""
        previous_match = {
            'metadata': {
                'center_x': 10.0,
                'center_y': 20.0,
                'depth': 5.0,
                'timestamp': time.time() - 0.05  # 50ms ago
            }
        }
        current_bbox = [15, 25, 55, 65]
        current_depth = 6.0
        
        velocity = self.velocity_calculator._calculate_3d_velocity(previous_match, current_bbox, current_depth)
        
        # Should return 0 for very small time differences
        self.assertEqual(velocity, 0.0)
    
    def test_calculate_3d_velocity_zero_time_difference(self):
        """Test velocity calculation with zero time difference."""
        previous_match = {
            'metadata': {
                'center_x': 10.0,
                'center_y': 20.0,
                'depth': 5.0,
                'timestamp': time.time()  # Same time
            }
        }
        current_bbox = [15, 25, 55, 65]
        current_depth = 6.0
        
        velocity = self.velocity_calculator._calculate_3d_velocity(previous_match, current_bbox, current_depth)
        
        # Should return 0 for zero time difference
        self.assertEqual(velocity, 0.0)
    
    def test_calculate_3d_velocity_negative_time_difference(self):
        """Test velocity calculation with negative time difference."""
        previous_match = {
            'metadata': {
                'center_x': 10.0,
                'center_y': 20.0,
                'depth': 5.0,
                'timestamp': time.time() + 1.0  # Future time
            }
        }
        current_bbox = [15, 25, 55, 65]
        current_depth = 6.0
        
        velocity = self.velocity_calculator._calculate_3d_velocity(previous_match, current_bbox, current_depth)
        
        # Should return 0 for negative time difference
        self.assertEqual(velocity, 0.0)
    
    def test_update_metadata_with_match_id_success(self):
        """Test successful metadata update with match ID."""
        # Store a test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        key = self.memory_storage.store_image(test_image, {'test': True}, ['test'])
        
        # Test updating metadata
        match_id = "test-match-123"
        success = self.velocity_calculator._update_metadata_with_match_id(key, match_id)
        
        # Verify update was successful
        self.assertTrue(success)
        
        # Verify metadata was updated
        updated_metadata = self.memory_storage.get_metadata(key)
        self.assertEqual(updated_metadata['match_id'], match_id)
        self.assertIn('updated_at', updated_metadata)
    
    def test_update_metadata_with_match_id_nonexistent_key(self):
        """Test metadata update with nonexistent key."""
        match_id = "test-match-123"
        success = self.velocity_calculator._update_metadata_with_match_id("nonexistent_key", match_id)
        
        # Should return False for nonexistent key
        self.assertFalse(success)
    
    def test_store_new_detection_success(self):
        """Test successful storage of new detection."""
        # Create test data
        object_region = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        detection = {
            'bbox': [10, 10, 50, 50],
            'class_name': 'bicycle',
            'confidence': 0.7
        }
        depth = 4.0
        
        # Call the method
        self.velocity_calculator._store_new_detection(object_region, detection, depth)
        
        # Verify object was stored
        bicycle_objects = self.memory_storage.search_by_tags(['bicycle'], operator="OR")
        self.assertEqual(len(bicycle_objects), 1)
        
        # Verify metadata
        metadata = bicycle_objects[0]['metadata']
        self.assertEqual(metadata['class_name'], 'bicycle')
        self.assertEqual(metadata['confidence'], 0.7)
        self.assertEqual(metadata['depth'], 4.0)
        self.assertEqual(metadata['center_x'], 30.0)
        self.assertEqual(metadata['center_y'], 30.0)
        self.assertIn('stored_at', metadata)
        self.assertIn('image_size_bytes', metadata)
    
    def test_store_new_detection_invalid_bbox(self):
        """Test storing new detection with invalid bounding box."""
        object_region = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        detection = {
            'bbox': [10, 10],  # Invalid: only 2 coordinates
            'class_name': 'bicycle',
            'confidence': 0.7
        }
        depth = 4.0
        
        # Call the method
        self.velocity_calculator._store_new_detection(object_region, detection, depth)
        
        # Verify no object was stored due to invalid bbox
        self.assertEqual(len(self.memory_storage), 0)
    
    def test_process_detection_for_velocity_with_match(self):
        """Test processing detection for velocity when a match is found."""
        # First populate storage with an initial detection
        initial_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.memory_storage.store_image(
            initial_image, 
            {'class_name': 'person', 'bbox': [10, 10, 50, 50]}, 
            ['person']
        )
        
        # Create current detection
        current_detection = {
            'bbox': [15, 15, 55, 55],
            'class_name': 'person',
            'confidence': 0.9
        }
        
        depth_map = np.ones((100, 100)) * 4.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock the image comparison to return a match
        self.velocity_calculator.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        # Patch the find_best_image_match method to return a match
        with patch.object(self.velocity_calculator, '_find_best_image_match') as mock_find_match:
            mock_find_match.return_value = {
                'key': 'initial_key',
                'metadata': {
                    'center_x': 30.0,
                    'center_y': 30.0,
                    'depth': 5.0,
                    'timestamp': time.time() - 1.0
                }
            }
            
            # Call the method
            self.velocity_calculator._process_detection_for_velocity(
                current_detection, depth_map, segmentation_masks, original_frame
            )
            
            # Verify the method was called
            mock_find_match.assert_called_once()
    
    def test_process_detection_for_velocity_no_match(self):
        """Test processing detection for velocity when no match is found."""
        # Create current detection
        current_detection = {
            'bbox': [15, 15, 55, 55],
            'class_name': 'person',
            'confidence': 0.9
        }
        
        depth_map = np.ones((100, 100)) * 4.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock the find_best_image_match method to return None (no match)
        with patch.object(self.velocity_calculator, '_find_best_image_match', return_value=None):
            # Call the method
            self.velocity_calculator._process_detection_for_velocity(
                current_detection, depth_map, segmentation_masks, original_frame
            )
            
            # Should store as new detection
            person_objects = self.memory_storage.search_by_tags(['person'], operator="OR")
            self.assertEqual(len(person_objects), 1)
    
    def test_identify_objects_with_velocity_empty_storage(self):
        """Test velocity identification when storage is empty."""
        # Create test data
        detections = [
            {
                'bbox': [10, 10, 50, 50],
                'class_name': 'person',
                'confidence': 0.9
            }
        ]
        
        depth_map = np.ones((100, 100)) * 3.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Verify storage is empty
        self.assertEqual(len(self.memory_storage), 0)
        
        # Call the method
        self.velocity_calculator.identify_objects_with_velocity(
            detections, depth_map, segmentation_masks, original_frame
        )
        
        # Verify storage was populated
        self.assertEqual(len(self.memory_storage), 1)
        
        # Verify object was stored with correct tags
        person_objects = self.memory_storage.search_by_tags(['person'], operator="OR")
        self.assertEqual(len(person_objects), 1)
    
    def test_identify_objects_with_velocity_with_existing_data(self):
        """Test velocity identification with existing data in storage."""
        # First populate storage
        initial_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.memory_storage.store_image(
            initial_image, 
            {'class_name': 'person', 'bbox': [10, 10, 50, 50]}, 
            ['person']
        )
        
        # Create current detections
        detections = [
            {
                'bbox': [15, 15, 55, 55],
                'class_name': 'person',
                'confidence': 0.9
            }
        ]
        
        depth_map = np.ones((100, 100)) * 4.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock the image comparison to return a match
        self.velocity_calculator.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        # Patch the find_best_image_match method to return a match
        with patch.object(self.velocity_calculator, '_find_best_image_match') as mock_find_match:
            mock_find_match.return_value = {
                'key': 'initial_key',
                'metadata': {
                    'center_x': 30.0,
                    'center_y': 30.0,
                    'depth': 5.0,
                    'timestamp': time.time() - 1.0
                }
            }
            
            # Call the method
            self.velocity_calculator.identify_objects_with_velocity(
                detections, depth_map, segmentation_masks, original_frame
            )
            
            # Verify the method was called
            mock_find_match.assert_called_once()
    
    def test_identify_objects_with_velocity_no_detections(self):
        """Test velocity identification with no detections."""
        # Call the method with empty detections
        self.velocity_calculator.identify_objects_with_velocity(
            [], None, None, None
        )
        
        # Should return early without processing
        self.assertEqual(len(self.memory_storage), 0)
    
    def test_identify_objects_with_velocity_redis_sync(self):
        """Test Redis synchronization during velocity identification."""
        # Enable Redis for this test
        self.velocity_calculator.redis_storage = self.mock_redis_storage
        
        # Mock the transfer and clear methods
        self.memory_storage.transfer_to_redis = Mock(return_value=2)
        self.memory_storage.clear_all = Mock(return_value=2)
        
        # First populate storage with some data so Redis sync will be called
        initial_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.memory_storage.store_image(
            initial_image, 
            {'class_name': 'person', 'bbox': [10, 10, 50, 50]}, 
            ['person']
        )
        
        # Create test data
        detections = [
            {
                'bbox': [15, 15, 55, 55],
                'class_name': 'person',
                'confidence': 0.9
            }
        ]
        
        depth_map = np.ones((100, 100)) * 4.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock the image comparison to return a match
        self.velocity_calculator.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        # Patch the find_best_image_match method to return a match
        with patch.object(self.velocity_calculator, '_find_best_image_match') as mock_find_match:
            mock_find_match.return_value = {
                'key': 'initial_key',
                'metadata': {
                    'center_x': 30.0,
                    'center_y': 30.0,
                    'depth': 5.0,
                    'timestamp': time.time() - 1.0
                }
            }
            
            # Call the method
            self.velocity_calculator.identify_objects_with_velocity(
                detections, depth_map, segmentation_masks, original_frame
            )
            
            # Verify Redis sync was attempted
            self.memory_storage.transfer_to_redis.assert_called_once_with(self.mock_redis_storage)
            self.memory_storage.clear_all.assert_called_once()
    
    def test_identify_objects_with_velocity_redis_sync_failure(self):
        """Test Redis synchronization failure handling."""
        # Enable Redis for this test
        self.velocity_calculator.redis_storage = self.mock_redis_storage
        
        # Mock the transfer method to raise an exception
        self.memory_storage.transfer_to_redis = Mock(side_effect=Exception("Redis sync failed"))
        self.memory_storage.clear_all = Mock(return_value=0)
        
        # First populate storage with some data so Redis sync will be called
        initial_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.memory_storage.store_image(
            initial_image, 
            {'class_name': 'person', 'bbox': [10, 10, 50, 50]}, 
            ['person']
        )
        
        # Create test data
        detections = [
            {
                'bbox': [15, 15, 55, 55],
                'class_name': 'person',
                'confidence': 0.9
            }
        ]
        
        depth_map = np.ones((100, 100)) * 4.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock the image comparison to return a match
        self.velocity_calculator.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        # Patch the find_best_image_match method to return a match
        with patch.object(self.velocity_calculator, '_find_best_image_match') as mock_find_match:
            mock_find_match.return_value = {
                'key': 'initial_key',
                'metadata': {
                    'center_x': 30.0,
                    'center_y': 30.0,
                    'depth': 5.0,
                    'timestamp': time.time() - 1.0
                }
            }
            
            # Call the method - should handle Redis sync failure gracefully
            self.velocity_calculator.identify_objects_with_velocity(
                detections, depth_map, segmentation_masks, original_frame
            )
            
            # Should not crash and should log the error
            self.memory_storage.transfer_to_redis.assert_called_once()
    
    def test_error_handling_in_velocity_identification(self):
        """Test error handling in velocity identification."""
        # Create test data that will cause an error
        detections = [
            {
                'bbox': [10, 10, 50, 50],
                'class_name': 'person',
                'confidence': 0.9
            }
        ]
        
        # Mock the _populate_initial_detections method to raise an exception
        with patch.object(self.velocity_calculator, '_populate_initial_detections', side_effect=Exception("Test error")):
            # Call the method - should handle the error gracefully
            self.velocity_calculator.identify_objects_with_velocity(
                detections, None, None, None
            )
            
            # Should not crash and should log the error
            # The method should complete without raising an exception
    
    def test_error_handling_in_individual_detection_processing(self):
        """Test error handling when processing individual detections fails."""
        # First populate storage
        initial_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        self.memory_storage.store_image(
            initial_image, 
            {'class_name': 'person', 'bbox': [10, 10, 50, 50]}, 
            ['person']
        )
        
        # Create detections where one will cause an error
        detections = [
            {
                'bbox': [15, 15, 55, 55],
                'class_name': 'person',
                'confidence': 0.9
            },
            {
                'bbox': [100, 100, 150, 150],
                'class_name': 'car',
                'confidence': 0.8
            }
        ]
        
        depth_map = np.ones((200, 200)) * 4.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Mock the _process_detection_for_velocity method to raise an exception for the first detection
        with patch.object(self.velocity_calculator, '_process_detection_for_velocity') as mock_process:
            mock_process.side_effect = [Exception("Processing error"), None]
            
            # Call the method - should handle individual errors gracefully
            self.velocity_calculator.identify_objects_with_velocity(
                detections, depth_map, segmentation_masks, original_frame
            )
            
            # Should process both detections despite the error in the first one
            self.assertEqual(mock_process.call_count, 2)
    
    def test_set_frame_count(self):
        """Test setting frame count."""
        self.velocity_calculator.set_frame_count(42)
        self.assertEqual(self.velocity_calculator.frame_count, 42)
    
    def test_get_velocity_stats(self):
        """Test getting velocity statistics."""
        # Test with empty storage
        stats = self.velocity_calculator.get_velocity_stats()
        self.assertEqual(stats['total_tracked_objects'], 0)
        self.assertEqual(stats['average_velocity'], 0.0)
        
        # Test with some velocity data - create objects with match tags and velocity
        initial_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Store an object with velocity data and match tag
        metadata_with_velocity = {
            'class_name': 'person',
            'bbox': [10, 10, 50, 50],
            'velocity_mps': 2.5,
            'match_id': 'test-match-123'
        }
        tags_with_velocity = ['person', 'match_test-match-123']
        self.memory_storage.store_image(initial_image, metadata_with_velocity, tags_with_velocity)
        
        # Store another object with velocity data
        metadata_with_velocity2 = {
            'class_name': 'car',
            'bbox': [100, 100, 150, 150],
            'velocity_mps': 5.0,
            'match_id': 'test-match-456'
        }
        tags_with_velocity2 = ['car', 'match_test-match-456']
        self.memory_storage.store_image(initial_image, metadata_with_velocity2, tags_with_velocity2)
        
        # Store an object without velocity data (should not be counted)
        metadata_no_velocity = {
            'class_name': 'bicycle',
            'bbox': [200, 200, 250, 250]
        }
        tags_no_velocity = ['bicycle']
        self.memory_storage.store_image(initial_image, metadata_no_velocity, tags_no_velocity)
        
        stats = self.velocity_calculator.get_velocity_stats()
        self.assertEqual(stats['total_tracked_objects'], 2)  # Only objects with velocity
        self.assertEqual(stats['average_velocity'], 3.75)  # (2.5 + 5.0) / 2
        self.assertEqual(stats['min_velocity'], 2.5)
        self.assertEqual(stats['max_velocity'], 5.0)
        self.assertEqual(stats['velocity_samples'], 2)


def run_tests():
    """Run all unit tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVelocityTrackingMethods)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
