#!/usr/bin/env python3
"""
Test script for velocity tracking functionality.

This script tests the basic functionality of the velocity tracking system
without requiring a camera or video stream.
"""

import sys
from pathlib import Path
# Add the project root to the path to import from rose modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from rose.storage.memory_image_storage import MemoryImageStorage
from rose.storage.redis_image_storage import RedisImageStorage
from rose.processing.image_comparator import ImageComparator
from rose.exec.process_video_stream import VideoProcessor


def test_memory_storage():
    """Test basic memory storage functionality."""
    print("Testing MemoryImageStorage...")
    
    # Create memory storage
    storage = MemoryImageStorage(max_memory_mb=100, ttl_hours=1)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test storing image
    metadata = {
        'test': True,
        'timestamp': time.time(),
        'bbox': [10, 10, 50, 50]
    }
    
    tags = ['test', 'random']
    key = storage.store_image(test_image, metadata, tags)
    print(f"Stored test image with key: {key}")
    
    # Test retrieving image
    retrieved_image = storage.retrieve_image(key, format="numpy")
    if retrieved_image is not None:
        print(f"Successfully retrieved image with shape: {retrieved_image.shape}")
    else:
        print("Failed to retrieve image")
    
    # Test metadata retrieval
    retrieved_metadata = storage.get_metadata(key)
    if retrieved_metadata:
        print(f"Retrieved metadata: {retrieved_metadata}")
    else:
        print("Failed to retrieve metadata")
    
    # Test tag search
    search_results = storage.search_by_tags(['test'], operator="OR")
    print(f"Search results: {len(search_results)} items found")
    
    # Test storage stats
    stats = storage.get_storage_stats()
    print(f"Storage stats: {stats}")
    
    # Cleanup
    storage.clear_all()
    print("Memory storage test completed\n")


def test_image_comparator():
    """Test image comparison functionality."""
    print("Testing ImageComparator...")
    
    try:
        comparator = ImageComparator()
        
        # Create two similar test images
        image1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        image2 = image1.copy() + np.random.randint(-20, 20, (64, 64, 3), dtype=np.uint8)
        image2 = np.clip(image2, 0, 255).astype(np.uint8)
        
        # Compare images
        result = comparator.compare_images(image1, image2, method='vgg16', normalize=True)
        print(f"Image comparison result: {result}")
        
        print("Image comparator test completed\n")
        
    except Exception as e:
        print(f"Image comparator test failed: {e}\n")


def test_redis_connection():
    """Test Redis connection (optional)."""
    print("Testing Redis connection...")
    
    try:
        redis_storage = RedisImageStorage(host="localhost", port=6379, ttl_hours=1)
        print("Redis connection successful")
        
        # Test basic operations
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        metadata = {'test': True, 'timestamp': time.time()}
        tags = ['test']
        
        key = redis_storage.store_image(test_image, metadata, tags)
        print(f"Stored test image in Redis with key: {key}")
        
        # Test retrieval
        retrieved_image = redis_storage.retrieve_image(key, format="numpy")
        if retrieved_image is not None:
            print(f"Successfully retrieved image from Redis with shape: {retrieved_image.shape}")
        else:
            print("Failed to retrieve image from Redis")
        
        # Cleanup
        redis_storage.delete_image(key)
        redis_storage.close()
        print("Redis test completed\n")
        
    except Exception as e:
        print(f"Redis test failed: {e}")
        print("Make sure Redis server is running on localhost:6379\n")


class TestVelocityTracking(unittest.TestCase):
    """Unit tests for velocity tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the required dependencies
        self.mock_depth_estimator = Mock()
        self.mock_object_detector = Mock()
        self.mock_image_segmenter = Mock()
        self.mock_image_comparator = Mock()
        
        # Create a real memory storage for testing
        self.memory_storage = MemoryImageStorage(max_memory_mb=100, ttl_hours=1)
        
        # Mock Redis storage
        self.mock_redis_storage = Mock()
        
        # Import and patch the VideoProcessor class
        with patch('rose.storage.redis_image_storage.RedisImageStorage') as mock_redis_class:
            mock_redis_class.return_value = self.mock_redis_storage
            
            # Import the VideoProcessor class
            from rose.exec.process_video_stream import VideoProcessor
            
            # Create instance with mocked dependencies
            self.processor = VideoProcessor(
                use_zoedepth=False,
                object_confidence=0.5,
                object_model='faster_rcnn',
                colormap='viridis',
                max_objects_for_segmentation=5,
                use_redis=False  # Disable Redis for testing
            )
            
            # Replace the dependencies with mocks
            self.processor.depth_estimator = self.mock_depth_estimator
            self.processor.object_detector = self.mock_object_detector
            self.processor.image_segmenter = self.mock_image_segmenter
            self.processor.image_comparator = self.mock_image_comparator
            self.processor.memory_storage = self.memory_storage
            self.processor.redis_storage = self.mock_redis_storage
            self.processor.frame_count = 0
    
    def tearDown(self):
        """Clean up after tests."""
        self.memory_storage.clear_all()
    
    def test_populate_initial_detections(self):
        """Test populating initial detections when storage is empty."""
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
        self.processor._populate_initial_detections(
            detections, depth_map, segmentation_masks, original_frame
        )
        
        # Verify storage was populated
        self.assertEqual(len(self.memory_storage), 2)
        
        # Check that objects were stored with correct tags
        person_objects = self.memory_storage.search_by_tags(['person'], operator="OR")
        car_objects = self.memory_storage.search_by_tags(['car'], operator="OR")
        
        self.assertEqual(len(person_objects), 1)
        self.assertEqual(len(car_objects), 1)
        
        # Verify metadata
        person_metadata = person_objects[0]['metadata']
        self.assertEqual(person_metadata['class_name'], 'person')
        self.assertEqual(person_metadata['confidence'], 0.9)
        self.assertEqual(person_metadata['depth'], 5.0)
        self.assertEqual(person_metadata['center_x'], 30.0)
        self.assertEqual(person_metadata['center_y'], 30.0)
    
    def test_calculate_object_depth(self):
        """Test depth calculation for objects."""
        # Create test data
        bbox = [10, 10, 50, 50]
        depth_map = np.ones((100, 100)) * 3.0  # 3 meters depth
        segmentation_masks = None
        
        # Test depth calculation
        depth = self.processor._calculate_object_depth(bbox, depth_map, segmentation_masks)
        
        # Should return the depth at the center point
        self.assertEqual(depth, 3.0)
        
        # Test with None depth map
        depth = self.processor._calculate_object_depth(bbox, None, segmentation_masks)
        self.assertEqual(depth, 0.0)
        
        # Test with out-of-bounds coordinates
        bbox_out_of_bounds = [95, 95, 105, 105]
        depth = self.processor._calculate_object_depth(bbox_out_of_bounds, depth_map, segmentation_masks)
        self.assertEqual(depth, 3.0)  # Should clamp to valid coordinates
    
    def test_find_best_image_match(self):
        """Test finding the best image match."""
        # Create test data
        current_object_region = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Mock the image comparator to return a high similarity score
        self.processor.image_comparator.compare_images.return_value = {
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
            result = self.processor._find_best_image_match(current_object_region, matching_objects)
            
            # Verify result
            self.assertIsNotNone(result)
            self.assertEqual(result['key'], 'test_key_1')
            
            # Verify image comparator was called
            self.processor.image_comparator.compare_images.assert_called_once()
    
    def test_calculate_3d_velocity(self):
        """Test 3D velocity calculation."""
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
        velocity = self.processor._calculate_3d_velocity(previous_match, current_bbox, current_depth)
        
        # Verify velocity is calculated (should be positive)
        self.assertGreater(velocity, 0.0)
        
        # Test with very small time difference
        previous_match['metadata']['timestamp'] = time.time() - 0.05  # 50ms ago
        velocity = self.processor._calculate_3d_velocity(previous_match, current_bbox, current_depth)
        self.assertEqual(velocity, 0.0)  # Should return 0 for very small time differences
    
    def test_update_metadata_with_match_id(self):
        """Test updating metadata with match ID."""
        # Store a test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        key = self.memory_storage.store_image(test_image, {'test': True}, ['test'])
        
        # Test updating metadata
        match_id = "test-match-123"
        success = self.processor._update_metadata_with_match_id(key, match_id)
        
        # Verify update was successful
        self.assertTrue(success)
        
        # Verify metadata was updated
        updated_metadata = self.memory_storage.get_metadata(key)
        self.assertEqual(updated_metadata['match_id'], match_id)
        self.assertIn('updated_at', updated_metadata)
    
    def test_store_new_detection(self):
        """Test storing new detections."""
        # Create test data
        object_region = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        detection = {
            'bbox': [10, 10, 50, 50],
            'class_name': 'bicycle',
            'confidence': 0.7
        }
        depth = 4.0
        
        # Call the method
        self.processor._store_new_detection(object_region, detection, depth)
        
        # Verify object was stored
        bicycle_objects = self.memory_storage.search_by_tags(['bicycle'], operator="OR")
        self.assertEqual(len(bicycle_objects), 1)
        
        # Verify metadata
        metadata = bicycle_objects[0]['metadata']
        self.assertEqual(metadata['class_name'], 'bicycle')
        self.assertEqual(metadata['confidence'], 0.7)
        self.assertEqual(metadata['depth'], 4.0)
    
    def test_process_detection_for_velocity(self):
        """Test processing a single detection for velocity tracking."""
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
        self.processor.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        # Patch the find_best_image_match method to return a match
        with patch.object(self.processor, '_find_best_image_match') as mock_find_match:
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
            self.processor._process_detection_for_velocity(
                current_detection, depth_map, segmentation_masks, original_frame
            )
            
            # Verify the method was called
            mock_find_match.assert_called_once()
    
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
        self.processor.identify_objects_with_velocity(
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
        self.processor.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        # Patch the find_best_image_match method to return a match
        with patch.object(self.processor, '_find_best_image_match') as mock_find_match:
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
            self.processor.identify_objects_with_velocity(
                detections, depth_map, segmentation_masks, original_frame
            )
            
            # Verify the method was called
            mock_find_match.assert_called_once()
    
    def test_identify_objects_with_velocity_no_detections(self):
        """Test velocity identification with no detections."""
        # Call the method with empty detections
        self.processor.identify_objects_with_velocity(
            [], None, None, None
        )
        
        # Should return early without processing
        self.assertEqual(len(self.memory_storage), 0)
    
    def test_identify_objects_with_velocity_redis_sync(self):
        """Test Redis synchronization during velocity identification."""
        # Create a new processor with Redis enabled for this test
        with patch('rose.storage.redis_image_storage.RedisImageStorage') as mock_redis_class:
            mock_redis_class.return_value = self.mock_redis_storage
            
            # Create a new processor instance with Redis enabled
            test_processor = VideoProcessor(
                use_zoedepth=False,
                object_confidence=0.5,
                object_model='faster_rcnn',
                colormap='viridis',
                max_objects_for_segmentation=5,
                use_redis=True  # Enable Redis for this test
            )
            
            # Replace the dependencies with mocks
            test_processor.depth_estimator = self.mock_depth_estimator
            test_processor.object_detector = self.mock_object_detector
            test_processor.image_segmenter = self.mock_image_segmenter
            test_processor.image_comparator = self.mock_image_comparator
            test_processor.memory_storage = self.memory_storage
            test_processor.redis_storage = self.mock_redis_storage
            test_processor.frame_count = 0
        
        # Mock the transfer and clear methods
        self.memory_storage.transfer_to_redis = Mock(return_value=2)
        self.memory_storage.clear_all = Mock(return_value=2)
        
        # Create test data
        depth_map = np.ones((100, 100)) * 3.0
        segmentation_masks = None
        original_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # First, populate storage with initial detections so Redis sync will be triggered
        initial_detections = [
            {
                'bbox': [5, 5, 45, 45],
                'class_name': 'person',
                'confidence': 0.9
            }
        ]
        
        # Call the method first time to populate storage
        test_processor.identify_objects_with_velocity(
            initial_detections, depth_map, segmentation_masks, original_frame
        )
        
        # Now call with new detections to trigger Redis sync
        new_detections = [
            {
                'bbox': [15, 15, 55, 55],
                'class_name': 'person',
                'confidence': 0.9
            }
        ]
        
        # Mock the image comparison to return a match so it processes the detection
        test_processor.image_comparator.compare_images.return_value = {
            'similarity_score': 0.8
        }
        
        # Patch the find_best_image_match method to return a match
        with patch.object(test_processor, '_find_best_image_match') as mock_find_match:
            mock_find_match.return_value = {
                'key': 'initial_key',
                'metadata': {
                    'center_x': 25.0,
                    'center_y': 25.0,
                    'depth': 3.0,
                    'timestamp': time.time() - 1.0
                }
            }
            
            # Call the method again - this should trigger Redis sync
            test_processor.identify_objects_with_velocity(
                new_detections, depth_map, segmentation_masks, original_frame
            )
        
        # Verify Redis sync was attempted
        self.memory_storage.transfer_to_redis.assert_called_once_with(self.mock_redis_storage)
        self.memory_storage.clear_all.assert_called_once()
    
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
        with patch.object(self.processor, '_populate_initial_detections', side_effect=Exception("Test error")):
            # Call the method - should handle the error gracefully
            self.processor.identify_objects_with_velocity(
                detections, None, None, None
            )
            
            # Should not crash and should log the error
            # The method should complete without raising an exception


def main():
    """Run all tests."""
    print("Starting velocity tracking functionality tests...\n")
    
    # Test memory storage
    test_memory_storage()
    
    # Test image comparator
    test_image_comparator()
    
    # Test Redis connection (optional)
    test_redis_connection()
    
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("All tests completed!")


if __name__ == "__main__":
    main()
