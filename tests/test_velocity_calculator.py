# Add the project root to the path to import from rose modules
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# flake8: noqa: E402

import pytest
import numpy as np
import time
import uuid
from unittest.mock import Mock, patch, MagicMock

from rose.processing.velocity_calculator import VelocityCalculator
from rose.storage.memory_image_storage import MemoryImageStorage
from rose.storage.redis_image_storage import RedisImageStorage
from rose.processing.image_comparator import ImageComparator


class TestVelocityCalculator:
    """Test cases for the VelocityCalculator class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock storage systems
        self.mock_memory_storage = Mock(spec=MemoryImageStorage)
        self.mock_redis_storage = Mock(spec=RedisImageStorage)
        self.mock_image_comparator = Mock(spec=ImageComparator)
        
        # Create test data
        self.sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.sample_depth_map = np.random.rand(480, 640).astype(np.float32)
        self.sample_segmentation_masks = np.random.rand(2, 480, 640).astype(np.float32)
        
        # Create sample detections
        self.sample_detections = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.85,
                'class_name': 'person',
                'class_id': 1
            },
            {
                'bbox': [300, 150, 400, 250],
                'confidence': 0.72,
                'class_name': 'car',
                'class_id': 3
            }
        ]

    @pytest.fixture
    def velocity_calculator_default(self):
        """Create a VelocityCalculator instance with default settings."""
        return VelocityCalculator(
            memory_storage=self.mock_memory_storage,
            redis_storage=None,
            image_comparator=self.mock_image_comparator
        )

    @pytest.fixture
    def velocity_calculator_with_redis(self):
        """Create a VelocityCalculator instance with Redis storage."""
        return VelocityCalculator(
            memory_storage=self.mock_memory_storage,
            redis_storage=self.mock_redis_storage,
            image_comparator=self.mock_image_comparator
        )

    def test_velocity_calculator_init_default(self):
        """Test VelocityCalculator initialization with default parameters."""
        calculator = VelocityCalculator(
            memory_storage=self.mock_memory_storage
        )
        
        assert calculator.memory_storage is self.mock_memory_storage
        assert calculator.redis_storage is None
        assert calculator.image_comparator is not None
        assert calculator.frame_count == 0

    def test_velocity_calculator_init_with_redis(self):
        """Test VelocityCalculator initialization with Redis storage."""
        calculator = VelocityCalculator(
            memory_storage=self.mock_memory_storage,
            redis_storage=self.mock_redis_storage,
            image_comparator=self.mock_image_comparator
        )
        
        assert calculator.memory_storage is self.mock_memory_storage
        assert calculator.redis_storage is self.mock_redis_storage
        assert calculator.image_comparator is self.mock_image_comparator
        assert calculator.frame_count == 0

    def test_velocity_calculator_init_with_custom_comparator(self):
        """Test VelocityCalculator initialization with custom image comparator."""
        custom_comparator = Mock(spec=ImageComparator)
        calculator = VelocityCalculator(
            memory_storage=self.mock_memory_storage,
            image_comparator=custom_comparator
        )
        
        assert calculator.image_comparator is custom_comparator

    def test_set_frame_count(self, velocity_calculator_default):
        """Test setting frame count."""
        velocity_calculator_default.set_frame_count(42)
        assert velocity_calculator_default.frame_count == 42

    def test_identify_objects_with_velocity_no_detections(self, velocity_calculator_default):
        """Test velocity identification with no detections."""
        velocity_calculator_default.identify_objects_with_velocity(
            [], self.sample_depth_map, self.sample_segmentation_masks, self.sample_frame
        )
        
        # Should handle gracefully and return early
        # No assertions needed as method returns None

    def test_identify_objects_with_velocity_empty_memory_storage(self, velocity_calculator_default):
        """Test velocity identification with empty memory storage."""
        # Mock empty memory storage
        self.mock_memory_storage.list_all_images.return_value = []
        
        # Mock the _populate_initial_detections method
        with patch.object(velocity_calculator_default, '_populate_initial_detections') as mock_populate:
            velocity_calculator_default.identify_objects_with_velocity(
                [{'bbox': [10, 10, 50, 50], 'class_name': 'person', 'confidence': 0.9}],
                np.ones((100, 100)) * 3.0,
                None,
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            
            # Verify _populate_initial_detections was called
            mock_populate.assert_called_once()
    
    def test_identify_objects_with_velocity_with_existing_data(self, velocity_calculator_default):
        """Test velocity identification with existing data in memory storage."""
        # Mock non-empty memory storage
        mock_existing_objects = [
            {'key': 'obj1', 'metadata': {'class_name': 'person'}},
            {'key': 'obj2', 'metadata': {'class_name': 'car'}}
        ]
        self.mock_memory_storage.list_all_images.return_value = mock_existing_objects
        
        # Mock the _process_detection_for_velocity method
        with patch.object(velocity_calculator_default, '_process_detection_for_velocity') as mock_process:
            velocity_calculator_default.identify_objects_with_velocity(
                [{'bbox': [10, 10, 50, 50], 'class_name': 'person', 'confidence': 0.9}],
                np.ones((100, 100)) * 3.0,
                None,
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            
            # Verify _process_detection_for_velocity was called
            mock_process.assert_called_once()
    
    def test_identify_objects_with_velocity_redis_sync(self, velocity_calculator_with_redis):
        """Test velocity identification with Redis synchronization."""
        # Mock non-empty memory storage
        mock_existing_objects = [
            {'key': 'obj1', 'metadata': {'class_name': 'person'}},
            {'key': 'obj2', 'metadata': {'class_name': 'car'}}
        ]
        self.mock_memory_storage.list_all_images.return_value = mock_existing_objects
        
        # Mock the _process_detection_for_velocity method
        with patch.object(velocity_calculator_with_redis, '_process_detection_for_velocity') as mock_process:
            velocity_calculator_with_redis.identify_objects_with_velocity(
                [{'bbox': [10, 10, 50, 50], 'class_name': 'person', 'confidence': 0.9}],
                np.ones((100, 100)) * 3.0,
                None,
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            
            # Verify _process_detection_for_velocity was called
            mock_process.assert_called_once()
    
    def test_identify_objects_with_velocity_redis_error(self, velocity_calculator_with_redis):
        """Test velocity identification with Redis error handling."""
        # Mock non-empty memory storage
        mock_existing_objects = [
            {'key': 'obj1', 'metadata': {'class_name': 'person'}},
            {'key': 'obj2', 'metadata': {'class_name': 'car'}}
        ]
        self.mock_memory_storage.list_all_images.return_value = mock_existing_objects
        
        # Mock the _process_detection_for_velocity method
        with patch.object(velocity_calculator_with_redis, '_process_detection_for_velocity') as mock_process:
            velocity_calculator_with_redis.identify_objects_with_velocity(
                [{'bbox': [10, 10, 50, 50], 'class_name': 'person', 'confidence': 0.9}],
                np.ones((100, 100)) * 3.0,
                None,
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            
            # Verify _process_detection_for_velocity was called
            mock_process.assert_called_once()

    def test_calculate_object_depth_with_depth_map(self, velocity_calculator_default):
        """Test object depth calculation with valid depth map."""
        bbox = [100, 100, 200, 200]
        depth = velocity_calculator_default._calculate_object_depth(
            bbox, self.sample_depth_map, self.sample_segmentation_masks
        )
        
        assert isinstance(depth, float)
        assert depth >= 0.0

    def test_calculate_object_depth_no_depth_map(self, velocity_calculator_default):
        """Test object depth calculation with no depth map."""
        bbox = [100, 100, 200, 200]
        depth = velocity_calculator_default._calculate_object_depth(
            bbox, None, self.sample_segmentation_masks
        )
        
        assert depth == 0.0

    def test_calculate_object_depth_edge_coordinates(self, velocity_calculator_default):
        """Test object depth calculation with edge coordinates."""
        # Test coordinates at image boundaries
        bbox = [0, 0, 639, 479]  # Full image dimensions
        depth = velocity_calculator_default._calculate_object_depth(
            bbox, self.sample_depth_map, self.sample_segmentation_masks
        )
        
        assert isinstance(depth, float)
        assert depth >= 0.0

    def test_find_best_image_match_success(self, velocity_calculator_default):
        """Test finding best image match successfully."""
        # Mock stored image
        mock_stored_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.mock_memory_storage.retrieve_image.return_value = mock_stored_image
        
        # Mock image comparison result
        self.mock_image_comparator.compare_images.return_value = {'similarity_score': 0.8}
        
        matching_objects = [
            {'key': 'test_key_1', 'metadata': {'class_name': 'person'}}
        ]
        
        best_match = velocity_calculator_default._find_best_image_match(
            self.sample_frame[100:200, 100:200], matching_objects
        )
        
        assert best_match is not None
        assert best_match['key'] == 'test_key_1'

    def test_find_best_image_match_no_matches(self, velocity_calculator_default):
        """Test finding best image match with no good matches."""
        # Mock stored image
        mock_stored_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.mock_memory_storage.retrieve_image.return_value = mock_stored_image
        
        # Mock low similarity score
        self.mock_image_comparator.compare_images.return_value = {'similarity_score': 0.3}
        
        matching_objects = [
            {'key': 'test_key_1', 'metadata': {'class_name': 'person'}}
        ]
        
        best_match = velocity_calculator_default._find_best_image_match(
            self.sample_frame[100:200, 100:200], matching_objects
        )
        
        assert best_match is None

    def test_find_best_image_match_empty_matching_objects(self, velocity_calculator_default):
        """Test finding best image match with empty matching objects."""
        best_match = velocity_calculator_default._find_best_image_match(
            self.sample_frame[100:200, 100:200], []
        )
        
        assert best_match is None

    def test_calculate_3d_velocity_success(self, velocity_calculator_default):
        """Test 3D velocity calculation successfully."""
        # Mock previous match with metadata
        previous_match = {
            'metadata': {
                'center_x': 150.0,
                'center_y': 150.0,
                'depth': 2.0,
                'timestamp': time.time() - 1.0  # 1 second ago
            }
        }
        
        current_bbox = [200, 200, 300, 300]
        current_depth = 2.5
        
        velocity = velocity_calculator_default._calculate_3d_velocity(
            previous_match, current_bbox, current_depth
        )
        
        assert isinstance(velocity, float)
        assert velocity >= 0.0

    def test_calculate_3d_velocity_small_time_diff(self, velocity_calculator_default):
        """Test 3D velocity calculation with very small time difference."""
        # Mock previous match with very recent timestamp
        previous_match = {
            'metadata': {
                'center_x': 150.0,
                'center_y': 150.0,
                'depth': 2.0,
                'timestamp': time.time() - 0.05  # 50ms ago (below threshold)
            }
        }
        
        current_bbox = [200, 200, 300, 300]
        current_depth = 2.5
        
        velocity = velocity_calculator_default._calculate_3d_velocity(
            previous_match, current_bbox, current_depth
        )
        
        assert velocity == 0.0

    def test_calculate_3d_velocity_missing_metadata(self, velocity_calculator_default):
        """Test 3D velocity calculation with missing metadata."""
        # Mock previous match with missing metadata
        previous_match = {
            'metadata': {}
        }
        
        current_bbox = [200, 200, 300, 300]
        current_depth = 2.5
        
        velocity = velocity_calculator_default._calculate_3d_velocity(
            previous_match, current_bbox, current_depth
        )
        
        assert velocity == 0.0

    def test_update_metadata_with_match_id_success(self, velocity_calculator_default):
        """Test updating metadata with match ID successfully."""
        # Mock successful metadata update
        self.mock_memory_storage.update_metadata.return_value = True
        
        success = velocity_calculator_default._update_metadata_with_match_id('test_key', 'match_123')
        
        assert success is True
        self.mock_memory_storage.update_metadata.assert_called_once_with('test_key', {'match_id': 'match_123'})

    def test_update_metadata_with_match_id_failure(self, velocity_calculator_default):
        """Test updating metadata with match ID failure."""
        # Mock failed metadata update
        self.mock_memory_storage.update_metadata.return_value = False
        
        success = velocity_calculator_default._update_metadata_with_match_id('test_key', 'match_123')
        
        assert success is False

    def test_store_new_detection_success(self, velocity_calculator_default):
        """Test storing new detection successfully."""
        detection = {
            'bbox': [100, 100, 200, 200],
            'class_name': 'person',
            'confidence': 0.85
        }
        
        object_region = self.sample_frame[100:200, 100:200]
        depth = 2.0
        
        velocity_calculator_default._store_new_detection(object_region, detection, depth)
        
        # Should store image in memory storage
        self.mock_memory_storage.store_image.assert_called_once()

    def test_store_new_detection_invalid_bbox(self, velocity_calculator_default):
        """Test storing new detection with invalid bounding box."""
        detection = {
            'bbox': [100, 100],  # Invalid: only 2 values
            'class_name': 'person',
            'confidence': 0.85
        }
        
        object_region = self.sample_frame[100:200, 100:200]
        depth = 2.0
        
        velocity_calculator_default._store_new_detection(object_region, detection, depth)
        
        # Should not store image with invalid bbox
        self.mock_memory_storage.store_image.assert_not_called()

    def test_get_velocity_stats_no_objects(self, velocity_calculator_default):
        """Test getting velocity stats with no tracked objects."""
        # Mock empty list_all_images results
        self.mock_memory_storage.list_all_images.return_value = []
        
        stats = velocity_calculator_default.get_velocity_stats()
        
        expected_stats = {'total_tracked_objects': 0, 'average_velocity': 0.0}
        assert stats == expected_stats
    
    def test_get_velocity_stats_with_objects(self, velocity_calculator_default):
        """Test getting velocity stats with tracked objects."""
        # Mock objects with velocity data
        mock_velocity_objects = [
            {'metadata': {'velocity_mps': 1.5}},
            {'metadata': {'velocity_mps': 2.5}},
            {'metadata': {'velocity_mps': 3.5}}
        ]
        self.mock_memory_storage.list_all_images.return_value = mock_velocity_objects
        
        stats = velocity_calculator_default.get_velocity_stats()
        
        assert stats['total_tracked_objects'] == 3
        assert stats['average_velocity'] == 2.5  # (1.5 + 2.5 + 3.5) / 3
        assert stats['min_velocity'] == 1.5
        assert stats['max_velocity'] == 3.5
        assert stats['velocity_samples'] == 3
    
    def test_get_velocity_stats_mixed_metadata(self, velocity_calculator_default):
        """Test getting velocity stats with mixed metadata."""
        # Mock objects with some missing velocity data
        mock_velocity_objects = [
            {'metadata': {'velocity_mps': 1.5}},
            {'metadata': {'other_field': 'value'}},  # No velocity
            {'metadata': {'velocity_mps': 2.5}}
        ]
        self.mock_memory_storage.list_all_images.return_value = mock_velocity_objects
        
        stats = velocity_calculator_default.get_velocity_stats()
        
        assert stats['total_tracked_objects'] == 2  # Only 2 have velocity
        assert stats['average_velocity'] == 2.0  # (1.5 + 2.5) / 2
        assert stats['min_velocity'] == 1.5
        assert stats['max_velocity'] == 2.5
        assert stats['velocity_samples'] == 2

    def test_get_velocity_stats_exception_handling(self, velocity_calculator_default):
        """Test getting velocity stats with exception handling."""
        # Mock exception in search
        self.mock_memory_storage.search_by_tags.side_effect = Exception("Storage error")
        
        stats = velocity_calculator_default.get_velocity_stats()
        
        assert stats == {}

    def test_process_detection_for_velocity_success(self, velocity_calculator_default):
        """Test processing detection for velocity tracking successfully."""
        # Mock search results
        mock_matching_objects = [
            {'key': 'test_key_1', 'metadata': {'class_name': 'person'}}
        ]
        self.mock_memory_storage.search_by_tags.return_value = mock_matching_objects
        
        # Mock image comparison result
        self.mock_image_comparator.compare_images.return_value = {'similarity_score': 0.8}
        
        # Mock stored image
        mock_stored_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.mock_memory_storage.retrieve_image.return_value = mock_stored_image
        
        # Mock successful metadata update
        self.mock_memory_storage.update_metadata.return_value = True
        
        detection = self.sample_detections[0]  # Use first detection
        
        velocity_calculator_default._process_detection_for_velocity(
            detection, self.sample_depth_map, self.sample_segmentation_masks, self.sample_frame
        )
        
        # Should process detection and store results
        self.mock_memory_storage.store_image.assert_called()

    def test_process_detection_for_velocity_invalid_bbox(self, velocity_calculator_default):
        """Test processing detection for velocity with invalid bounding box."""
        detection = {
            'bbox': [100, 100],  # Invalid: only 2 values
            'class_name': 'person',
            'confidence': 0.85
        }
        
        velocity_calculator_default._process_detection_for_velocity(
            detection, self.sample_depth_map, self.sample_segmentation_masks, self.sample_frame
        )
        
        # Should return early without processing
        self.mock_memory_storage.search_by_tags.assert_not_called()

    def test_process_detection_for_velocity_no_matches(self, velocity_calculator_default):
        """Test processing detection for velocity with no matching objects."""
        # Mock empty search results
        self.mock_memory_storage.search_by_tags.return_value = []
        
        detection = self.sample_detections[0]
        
        velocity_calculator_default._process_detection_for_velocity(
            detection, self.sample_depth_map, self.sample_segmentation_masks, self.sample_frame
        )
        
        # Should store as new detection
        self.mock_memory_storage.store_image.assert_called()

    def test_populate_initial_detections_success(self, velocity_calculator_default):
        """Test populating initial detections successfully."""
        velocity_calculator_default._populate_initial_detections(
            self.sample_detections, self.sample_depth_map, self.sample_segmentation_masks, self.sample_frame
        )
        
        # Should store initial detections
        self.mock_memory_storage.store_image.assert_called()

    def test_populate_initial_detections_invalid_bbox(self, velocity_calculator_default):
        """Test populating initial detections with invalid bounding box."""
        invalid_detections = [
            {
                'bbox': [100, 100],  # Invalid: only 2 values
                'class_name': 'person',
                'confidence': 0.85
            }
        ]
        
        velocity_calculator_default._populate_initial_detections(
            invalid_detections, self.sample_depth_map, self.sample_segmentation_masks, self.sample_frame
        )
        
        # Should not store invalid detections
        self.mock_memory_storage.store_image.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])

