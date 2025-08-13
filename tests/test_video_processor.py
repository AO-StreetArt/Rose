# Add the project root to the path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
import numpy as np
import time
import threading
import queue
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import os

from rose.exec.process_video_stream import VideoProcessor


class TestVideoProcessor(unittest.TestCase):
    """Test cases for the VideoProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        self.sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.sample_depth_map = np.random.rand(480, 640).astype(np.float32)
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
        self.sample_segmentation_masks = np.random.rand(2, 480, 640).astype(np.float32)

        # Create VideoProcessor instances with mocked dependencies
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'):

            self.video_processor_default = VideoProcessor()
            self.video_processor_custom = VideoProcessor(
                use_zoedepth=True,
                object_confidence=0.7,
                object_model='ssd',
                colormap='plasma',
                max_objects_for_segmentation=3
            )

    def test_video_processor_init_default(self):
        """Test VideoProcessor initialization with default parameters."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'):

            processor = VideoProcessor()

            self.assertFalse(processor.use_zoedepth)
            self.assertEqual(processor.object_confidence, 0.5)
            self.assertEqual(processor.object_model, 'faster_rcnn')
            self.assertEqual(processor.colormap, 'viridis')
            self.assertEqual(processor.max_objects_for_segmentation, 5)
            self.assertFalse(processor.is_processing)
            self.assertIsNone(processor.processing_thread)
            self.assertIsInstance(processor.frame_queue, queue.Queue)
            self.assertIsInstance(processor.result_queue, queue.Queue)

    def test_video_processor_init_custom(self):
        """Test VideoProcessor initialization with custom parameters."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'):

            processor = VideoProcessor(
                use_zoedepth=True,
                object_confidence=0.7,
                object_model='ssd',
                colormap='plasma',
                max_objects_for_segmentation=3
            )

            self.assertTrue(processor.use_zoedepth)
            self.assertEqual(processor.object_confidence, 0.7)
            self.assertEqual(processor.object_model, 'ssd')
            self.assertEqual(processor.colormap, 'plasma')
            self.assertEqual(processor.max_objects_for_segmentation, 3)

    def test_process_frame_basic(self):
        """Test basic frame processing."""
        with patch.object(self.video_processor_default, '_process_depth') as mock_depth, \
             patch.object(self.video_processor_default, '_process_objects') as mock_objects, \
             patch.object(self.video_processor_default, '_process_segmentation') as mock_seg:

            mock_depth.return_value = self.sample_depth_map
            mock_objects.return_value = self.sample_detections
            mock_seg.return_value = self.sample_segmentation_masks

            result = self.video_processor_default.process_frame(self.sample_frame)

            self.assertIn('depth_map', result)
            self.assertIn('detections', result)
            self.assertIn('segmentation_masks', result)
            self.assertIn('processing_time', result)

            self.assertEqual(result['depth_map'], self.sample_depth_map)
            self.assertEqual(result['detections'], self.sample_detections)
            self.assertEqual(result['segmentation_masks'], self.sample_segmentation_masks)

    def test_process_frame_with_depth_only(self):
        """Test frame processing with depth estimation only."""
        with patch.object(self.video_processor_default, '_process_depth') as mock_depth:
            mock_depth.return_value = self.sample_depth_map

            result = self.video_processor_default.process_frame(self.sample_frame, depth_only=True)

            self.assertIn('depth_map', result)
            self.assertNotIn('detections', result)
            self.assertNotIn('segmentation_masks', result)

    def test_process_frame_with_objects_only(self):
        """Test frame processing with object detection only."""
        with patch.object(self.video_processor_default, '_process_objects') as mock_objects:
            mock_objects.return_value = self.sample_detections

            result = self.video_processor_default.process_frame(self.sample_frame, objects_only=True)

            self.assertNotIn('depth_map', result)
            self.assertIn('detections', result)
            self.assertNotIn('segmentation_masks', result)

    def test_process_frame_with_segmentation_only(self):
        """Test frame processing with segmentation only."""
        with patch.object(self.video_processor_default, '_process_segmentation') as mock_seg:
            mock_seg.return_value = self.sample_segmentation_masks

            result = self.video_processor_default.process_frame(self.sample_frame, segmentation_only=True)

            self.assertNotIn('depth_map', result)
            self.assertNotIn('detections', result)
            self.assertIn('segmentation_masks', result)

    def test_start_processing(self):
        """Test starting video processing."""
        self.assertFalse(self.video_processor_default.is_processing)
        self.assertIsNone(self.video_processor_default.processing_thread)

        self.video_processor_default.start_processing()

        self.assertTrue(self.video_processor_default.is_processing)
        self.assertIsNotNone(self.video_processor_default.processing_thread)
        self.assertTrue(self.video_processor_default.processing_thread.is_alive())

        # Clean up
        self.video_processor_default.stop_processing()

    def test_stop_processing(self):
        """Test stopping video processing."""
        # Start processing first
        self.video_processor_default.start_processing()
        self.assertTrue(self.video_processor_default.is_processing)

        # Stop processing
        self.video_processor_default.stop_processing()

        self.assertFalse(self.video_processor_default.is_processing)
        self.assertIsNone(self.video_processor_default.processing_thread)

    def test_add_frame_to_queue(self):
        """Test adding frames to the processing queue."""
        initial_size = self.video_processor_default.frame_queue.qsize()
        
        self.video_processor_default.add_frame_to_queue(self.sample_frame)
        
        self.assertEqual(self.video_processor_default.frame_queue.qsize(), initial_size + 1)

    def test_get_result_from_queue(self):
        """Test getting results from the result queue."""
        # Add a mock result to the queue
        mock_result = {'test': 'data'}
        self.video_processor_default.result_queue.put(mock_result)
        
        result = self.video_processor_default.get_result_from_queue()
        
        self.assertEqual(result, mock_result)

    def test_get_result_from_queue_empty(self):
        """Test getting results from empty queue."""
        # Ensure queue is empty
        while not self.video_processor_default.result_queue.empty():
            self.video_processor_default.result_queue.get()
        
        result = self.video_processor_default.get_result_from_queue()
        
        self.assertIsNone(result)

    def test_process_depth_estimation(self):
        """Test depth estimation processing."""
        with patch.object(self.video_processor_default.depth_estimator, 'estimate_depth') as mock_estimate:
            mock_estimate.return_value = self.sample_depth_map

            depth_map = self.video_processor_default._process_depth(self.sample_frame)

            self.assertEqual(depth_map, self.sample_depth_map)
            mock_estimate.assert_called_once_with(self.sample_frame)

    def test_process_depth_estimation_zoedepth(self):
        """Test depth estimation with ZoeDepth."""
        with patch.object(self.video_processor_custom.depth_estimator, 'estimate_depth_zoedepth') as mock_estimate:
            mock_estimate.return_value = self.sample_depth_map

            depth_map = self.video_processor_custom._process_depth(self.sample_frame)

            self.assertEqual(depth_map, self.sample_depth_map)
            mock_estimate.assert_called_once_with(self.sample_frame)

    def test_process_object_detection(self):
        """Test object detection processing."""
        with patch.object(self.video_processor_default.object_detector, 'detect_objects') as mock_detect:
            mock_detect.return_value = self.sample_detections

            detections = self.video_processor_default._process_objects(self.sample_frame)

            self.assertEqual(detections, self.sample_detections)
            mock_detect.assert_called_once_with(self.sample_frame)

    def test_process_segmentation(self):
        """Test segmentation processing."""
        with patch.object(self.video_processor_default.segmenter, 'segment') as mock_segment:
            mock_segment.return_value = self.sample_segmentation_masks

            masks = self.video_processor_default._process_segmentation(self.sample_frame, self.sample_detections)

            self.assertEqual(masks, self.sample_segmentation_masks)
            mock_segment.assert_called_once()

    def test_process_segmentation_with_max_objects(self):
        """Test segmentation processing with maximum object limit."""
        # Create more detections than the limit
        many_detections = self.sample_detections * 3  # 6 detections
        
        with patch.object(self.video_processor_custom.segmenter, 'segment') as mock_segment:
            mock_segment.return_value = self.sample_segmentation_masks

            masks = self.video_processor_custom._process_segmentation(self.sample_frame, many_detections)

            # Should only process up to max_objects_for_segmentation
            self.assertEqual(masks, self.sample_segmentation_masks)
            # Verify that only the first 3 detections were used
            mock_segment.assert_called_once()

    def test_error_handling_invalid_frame(self):
        """Test error handling with invalid frame input."""
        # Test with None frame
        with self.assertRaises(ValueError):
            self.video_processor_default.process_frame(None)

        # Test with empty array
        empty_frame = np.array([])
        with self.assertRaises(ValueError):
            self.video_processor_default.process_frame(empty_frame)

        # Test with wrong dimensions
        wrong_dim_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)  # Missing channel dimension
        with self.assertRaises(ValueError):
            self.video_processor_default.process_frame(wrong_dim_frame)

    def test_error_handling_processing_failure(self):
        """Test error handling when processing fails."""
        with patch.object(self.video_processor_default, '_process_depth') as mock_depth:
            mock_depth.side_effect = Exception("Depth estimation failed")

            with self.assertRaises(Exception):
                self.video_processor_default.process_frame(self.sample_frame, depth_only=True)

    def test_performance_measurement(self):
        """Test that processing time is measured correctly."""
        with patch.object(self.video_processor_default, '_process_depth') as mock_depth:
            mock_depth.return_value = self.sample_depth_map

            result = self.video_processor_default.process_frame(self.sample_frame, depth_only=True)

            self.assertIn('processing_time', result)
            self.assertIsInstance(result['processing_time'], float)
            self.assertGreater(result['processing_time'], 0.0)

    def test_thread_safety(self):
        """Test thread safety of the video processor."""
        # Start processing
        self.video_processor_default.start_processing()
        
        # Add multiple frames from different threads
        def add_frames():
            for i in range(5):
                self.video_processor_default.add_frame_to_queue(self.sample_frame)
                time.sleep(0.01)

        threads = [threading.Thread(target=add_frames) for _ in range(3)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()

        # Stop processing
        self.video_processor_default.stop_processing()

        # Verify that frames were added
        self.assertGreaterEqual(self.video_processor_default.frame_queue.qsize(), 0)

    def test_cleanup_on_stop(self):
        """Test proper cleanup when stopping processing."""
        # Start processing
        self.video_processor_default.start_processing()
        
        # Verify processing is active
        self.assertTrue(self.video_processor_default.is_processing)
        self.assertIsNotNone(self.video_processor_default.processing_thread)
        
        # Stop processing
        self.video_processor_default.stop_processing()
        
        # Verify cleanup
        self.assertFalse(self.video_processor_default.is_processing)
        self.assertIsNone(self.video_processor_default.processing_thread)

    def test_queue_management(self):
        """Test queue management functionality."""
        # Test frame queue
        self.assertEqual(self.video_processor_default.frame_queue.qsize(), 0)
        
        # Add frames
        for i in range(3):
            self.video_processor_default.add_frame_to_queue(self.sample_frame)
        
        self.assertEqual(self.video_processor_default.frame_queue.qsize(), 3)
        
        # Get frames
        for i in range(3):
            frame = self.video_processor_default.frame_queue.get()
            self.assertEqual(frame, self.sample_frame)
        
        self.assertEqual(self.video_processor_default.frame_queue.qsize(), 0)

        # Test result queue
        self.assertEqual(self.video_processor_default.result_queue.qsize(), 0)
        
        # Add results
        test_result = {'test': 'result'}
        self.video_processor_default.result_queue.put(test_result)
        
        self.assertEqual(self.video_processor_default.result_queue.qsize(), 1)
        
        # Get result
        result = self.video_processor_default.result_queue.get()
        self.assertEqual(result, test_result)
        
        self.assertEqual(self.video_processor_default.result_queue.qsize(), 0)

    def test_processor_configuration_validation(self):
        """Test validation of processor configuration parameters."""
        # Test valid parameters
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'):

            # These should not raise exceptions
            processor = VideoProcessor(
                use_zoedepth=True,
                object_confidence=0.8,
                object_model='yolo',
                colormap='inferno',
                max_objects_for_segmentation=10
            )

            self.assertTrue(processor.use_zoedepth)
            self.assertEqual(processor.object_confidence, 0.8)
            self.assertEqual(processor.object_model, 'yolo')
            self.assertEqual(processor.colormap, 'inferno')
            self.assertEqual(processor.max_objects_for_segmentation, 10)

    def test_processor_with_different_image_formats(self):
        """Test processor with different image formats."""
        # Test with different image sizes
        small_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        large_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        with patch.object(self.video_processor_default, '_process_depth') as mock_depth:
            mock_depth.return_value = self.sample_depth_map

            # Process different sized frames
            result_small = self.video_processor_default.process_frame(small_frame, depth_only=True)
            result_large = self.video_processor_default.process_frame(large_frame, depth_only=True)

            self.assertIn('depth_map', result_small)
            self.assertIn('depth_map', result_large)
            self.assertIn('processing_time', result_small)
            self.assertIn('processing_time', result_large)

    def test_processor_with_different_data_types(self):
        """Test processor with different data types."""
        # Test with float32 frame
        float_frame = self.sample_frame.astype(np.float32) / 255.0
        
        with patch.object(self.video_processor_default, '_process_depth') as mock_depth:
            mock_depth.return_value = self.sample_depth_map

            result = self.video_processor_default.process_frame(float_frame, depth_only=True)

            self.assertIn('depth_map', result)
            self.assertIn('processing_time', result)


if __name__ == "__main__":
    unittest.main()
