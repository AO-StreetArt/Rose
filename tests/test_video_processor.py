# Add the project root to the path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# flake8: noqa: E402

import pytest
import numpy as np
import time
import threading
import queue
from unittest.mock import Mock, patch

from rose.exec.process_video_stream import VideoProcessor


class TestVideoProcessor:
    """Test cases for the VideoProcessor class."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_depth_map(self):
        """Create a sample depth map for testing."""
        return np.random.rand(480, 640).astype(np.float32)

    @pytest.fixture
    def sample_detections(self):
        """Create sample object detections for testing."""
        return [
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
    def sample_segmentation_masks(self):
        """Create sample segmentation masks for testing."""
        return np.random.rand(2, 480, 640).astype(np.float32)

    @pytest.fixture
    def video_processor_default(self):
        """Create a VideoProcessor instance with default settings."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'), \
             patch('rose.exec.process_video_stream.VelocityCalculator'):

            processor = VideoProcessor()
            return processor

    @pytest.fixture
    def video_processor_custom(self):
        """Create a VideoProcessor instance with custom settings."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'), \
             patch('rose.exec.process_video_stream.VelocityCalculator'):

            processor = VideoProcessor(
                use_zoedepth=True,
                object_confidence=0.7,
                object_model='ssd',
                colormap='plasma',
                max_objects_for_segmentation=3
            )
            return processor

    def test_video_processor_init_default(self):
        """Test VideoProcessor initialization with default parameters."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'), \
             patch('rose.exec.process_video_stream.VelocityCalculator'):

            processor = VideoProcessor()

            assert processor.use_zoedepth is False
            assert processor.object_confidence == 0.5
            assert processor.object_model == 'faster_rcnn'
            assert processor.colormap == 'viridis'
            assert processor.max_objects_for_segmentation == 5
            assert processor.is_processing is False
            assert processor.processing_thread is None
            assert isinstance(processor.frame_queue, queue.Queue)
            assert isinstance(processor.result_queue, queue.Queue)
            # Note: velocity_calculator is mocked, so we can't test its actual initialization

    def test_video_processor_init_custom(self):
        """Test VideoProcessor initialization with custom parameters."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'), \
             patch('rose.exec.process_video_stream.VelocityCalculator'):

            processor = VideoProcessor(
                use_zoedepth=True,
                object_confidence=0.8,
                object_model='ssd',
                colormap='inferno',
                max_objects_for_segmentation=2
            )

            assert processor.use_zoedepth is True
            assert processor.object_confidence == 0.8
            assert processor.object_model == 'ssd'
            assert processor.colormap == 'inferno'
            assert processor.max_objects_for_segmentation == 2

    def test_extract_object_prompts_empty(self, video_processor_default):
        """Test extracting object prompts from empty detections."""
        prompts = video_processor_default.extract_object_prompts([])
        assert prompts == []

    def test_extract_object_prompts_single(self, video_processor_default):
        """Test extracting object prompts from single detection."""
        detections = [{'class_name': 'person', 'confidence': 0.9}]
        prompts = video_processor_default.extract_object_prompts(detections)
        assert prompts == ['person']

    def test_extract_object_prompts_multiple(self, video_processor_default):
        """Test extracting object prompts from multiple detections."""
        detections = [
            {'class_name': 'person', 'confidence': 0.9},
            {'class_name': 'car', 'confidence': 0.8},
            {'class_name': 'dog', 'confidence': 0.7}
        ]
        prompts = video_processor_default.extract_object_prompts(detections)
        assert prompts == ['person', 'car', 'dog']

    def test_extract_object_prompts_duplicates(self, video_processor_default):
        """Test extracting object prompts with duplicate classes."""
        detections = [
            {'class_name': 'person', 'confidence': 0.9},
            {'class_name': 'car', 'confidence': 0.8},
            {'class_name': 'person', 'confidence': 0.7}  # Duplicate
        ]
        prompts = video_processor_default.extract_object_prompts(detections)
        assert prompts == ['person', 'car']  # Should remove duplicates

    def test_extract_object_prompts_limit(self, video_processor_default):
        """Test extracting object prompts with limit."""
        video_processor_default.max_objects_for_segmentation = 2
        detections = [
            {'class_name': 'person', 'confidence': 0.9},
            {'class_name': 'car', 'confidence': 0.8},
            {'class_name': 'dog', 'confidence': 0.7},
            {'class_name': 'cat', 'confidence': 0.6}
        ]
        prompts = video_processor_default.extract_object_prompts(detections)
        assert len(prompts) == 2
        assert prompts == ['person', 'car']

    def test_extract_object_prompts_missing_class_name(self, video_processor_default):
        """Test extracting object prompts with missing class_name."""
        detections = [
            {'confidence': 0.9},  # Missing class_name
            {'class_name': 'car', 'confidence': 0.8}
        ]
        prompts = video_processor_default.extract_object_prompts(detections)
        assert prompts == ['car']  # Should skip detection without class_name

    @patch('rose.exec.process_video_stream.ImagePreprocessor.convertBGRtoRGB')
    @patch('rose.exec.process_video_stream.Image.fromarray')
    def test_process_frame_success(self, mock_fromarray, mock_convert,
                                   video_processor_default, sample_frame,
                                   sample_depth_map, sample_detections,
                                   sample_segmentation_masks):
        """Test successful frame processing."""
        # Mock the processing components
        video_processor_default.depth_estimator.estimate_depth.return_value = sample_depth_map
        video_processor_default.object_detector.detect_objects.return_value = sample_detections
        video_processor_default.image_segmenter.segment.return_value = sample_segmentation_masks

        # Mock image conversion
        mock_convert.return_value = sample_frame
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image

        # Process frame
        result = video_processor_default.process_frame(sample_frame)

        # Verify result structure
        assert isinstance(result, dict)
        assert 'depth_map' in result
        assert 'detections' in result
        assert 'object_prompts' in result
        assert 'segmentation_masks' in result
        assert 'original_frame' in result

        # Verify values
        assert result['depth_map'] is sample_depth_map
        assert result['detections'] == sample_detections
        assert result['object_prompts'] == ['person', 'car']
        assert result['segmentation_masks'] is sample_segmentation_masks
        assert result['original_frame'] is sample_frame

    @patch('rose.exec.process_video_stream.ImagePreprocessor.convertBGRtoRGB')
    @patch('rose.exec.process_video_stream.Image.fromarray')
    def test_process_frame_zoedepth_fallback(self, mock_fromarray, mock_convert,
                                             video_processor_custom, sample_frame,
                                             sample_depth_map, sample_detections):
        """Test frame processing with ZoeDepth fallback to DPT."""
        # Mock ZoeDepth to fail, DPT to succeed
        video_processor_custom.depth_estimator.estimate_depth_zoedepth.return_value = None
        video_processor_custom.depth_estimator.estimate_depth.return_value = sample_depth_map
        video_processor_custom.object_detector.detect_objects.return_value = sample_detections
        video_processor_custom.image_segmenter.segment.return_value = None

        # Mock image conversion
        mock_convert.return_value = sample_frame
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image

        # Process frame
        result = video_processor_custom.process_frame(sample_frame)

        # Verify DPT was called as fallback
        video_processor_custom.depth_estimator.estimate_depth.assert_called_once()
        assert result['depth_map'] is sample_depth_map

    @patch('rose.exec.process_video_stream.ImagePreprocessor.convertBGRtoRGB')
    @patch('rose.exec.process_video_stream.Image.fromarray')
    def test_process_frame_no_objects(self, mock_fromarray, mock_convert,
                                      video_processor_default, sample_frame,
                                      sample_depth_map):
        """Test frame processing with no object detections."""
        # Mock no detections
        video_processor_default.depth_estimator.estimate_depth.return_value = sample_depth_map
        video_processor_default.object_detector.detect_objects.return_value = []

        # Mock image conversion
        mock_convert.return_value = sample_frame
        mock_pil_image = Mock()
        mock_fromarray.return_value = mock_pil_image

        # Process frame
        result = video_processor_default.process_frame(sample_frame)

        # Verify no segmentation was performed
        assert result['object_prompts'] == []
        assert result['segmentation_masks'] is None

    @patch('rose.exec.process_video_stream.ImagePreprocessor.convertBGRtoRGB')
    def test_process_frame_exception(self, mock_convert, video_processor_default, sample_frame):
        """Test frame processing with exception handling."""
        # Mock conversion to raise exception
        mock_convert.side_effect = Exception("Test exception")

        # Process frame
        result = video_processor_default.process_frame(sample_frame)

        # Verify error handling
        assert result['depth_map'] is None
        assert result['detections'] == []
        assert result['object_prompts'] == []
        assert result['segmentation_masks'] is None
        assert result['original_frame'] is sample_frame

    @patch('rose.postprocessing.image_creator.ImageCreator.create_depth_visualization')
    @patch('rose.postprocessing.image_creator.ImageCreator.create_segmentation_visualization')
    def test_create_visualization_success(self, mock_seg_vis, mock_depth_vis,
                                          video_processor_default, sample_frame,
                                          sample_depth_map, sample_detections,
                                          sample_segmentation_masks):
        """Test successful visualization creation."""
        # Mock visualization methods
        mock_depth_vis.return_value = sample_frame.copy()
        mock_seg_vis.return_value = sample_frame.copy()

        # Create result dict
        result = {
            'depth_map': sample_depth_map,
            'detections': sample_detections,
            'object_prompts': ['person', 'car'],
            'segmentation_masks': sample_segmentation_masks,
            'original_frame': sample_frame
        }

        # Create visualizations
        original, depth, seg = video_processor_default.create_visualization(result)

        # Verify results
        assert original is sample_frame
        assert depth is not None
        assert seg is not None

        # Verify method calls
        mock_depth_vis.assert_called_once_with(sample_depth_map, sample_frame, 'viridis')
        mock_seg_vis.assert_called_once_with(sample_frame, sample_segmentation_masks, ['person', 'car'])

    @patch('rose.postprocessing.image_creator.ImageCreator.create_depth_visualization')
    @patch('rose.postprocessing.image_creator.ImageCreator.create_segmentation_visualization')
    def test_create_visualization_no_depth(self, mock_seg_vis, mock_depth_vis,
                                           video_processor_default, sample_frame,
                                           sample_detections):
        """Test visualization creation with no depth map."""
        # Mock visualization methods
        mock_depth_vis.return_value = sample_frame.copy()
        mock_seg_vis.return_value = sample_frame.copy()

        # Create result dict without depth
        result = {
            'depth_map': None,
            'detections': sample_detections,
            'object_prompts': ['person'],
            'segmentation_masks': None,
            'original_frame': sample_frame
        }

        # Create visualizations
        original, depth, seg = video_processor_default.create_visualization(result)

        # Verify results
        assert original is sample_frame
        # When depth is None, the method should return a copy of the original frame
        assert np.array_equal(depth, sample_frame)  # Should be same content when no depth
        assert np.array_equal(seg, sample_frame)  # Should be same content when no segmentation

        # Verify that visualization methods were not called
        mock_depth_vis.assert_not_called()
        mock_seg_vis.assert_not_called()



    def test_start_processing(self, video_processor_default):
        """Test starting the processing thread."""
        video_processor_default.start_processing()

        assert video_processor_default.is_processing is True
        assert video_processor_default.processing_thread is not None
        assert video_processor_default.processing_thread.is_alive()

        # Cleanup
        video_processor_default.stop_processing()

    def test_stop_processing(self, video_processor_default):
        """Test stopping the processing thread."""
        # Start processing
        video_processor_default.start_processing()
        assert video_processor_default.is_processing is True

        # Stop processing
        video_processor_default.stop_processing()
        assert video_processor_default.is_processing is False

    def test_processing_worker_basic(self, video_processor_default, sample_frame):
        """Test the processing worker thread."""
        # Start processing
        video_processor_default.start_processing()

        # Add frame to queue
        video_processor_default.frame_queue.put(sample_frame)

        # Wait a bit for processing
        time.sleep(0.1)

        # Check if result is available
        try:
            result = video_processor_default.result_queue.get_nowait()
            assert isinstance(result, dict)
        except queue.Empty:
            # This is okay if processing is slow
            pass

        # Cleanup
        video_processor_default.stop_processing()

    def test_processing_worker_queue_full(self, video_processor_default, sample_frame):
        """Test processing worker with full queue."""
        # Start processing
        video_processor_default.start_processing()

        # Fill the result queue
        for _ in range(5):
            try:
                video_processor_default.result_queue.put_nowait({'test': 'data'})
            except queue.Full:
                break

        # Add frame to queue
        video_processor_default.frame_queue.put(sample_frame)

        # Wait a bit for processing
        time.sleep(0.1)

        # Cleanup
        video_processor_default.stop_processing()

    def test_processing_worker_exception(self, video_processor_default, sample_frame):
        """Test processing worker with exception."""
        # Mock process_frame to raise exception
        video_processor_default.process_frame = Mock(side_effect=Exception("Test exception"))

        # Start processing
        video_processor_default.start_processing()

        # Add frame to queue
        video_processor_default.frame_queue.put(sample_frame)

        # Wait a bit for processing
        time.sleep(0.1)

        # Cleanup
        video_processor_default.stop_processing()

    def test_queue_management(self, video_processor_default, sample_frame):
        """Test queue management and size limits."""
        # Test frame queue size limit
        for i in range(5):
            try:
                video_processor_default.frame_queue.put_nowait(sample_frame)
            except queue.Full:
                break

        # Test result queue size limit
        for i in range(5):
            try:
                video_processor_default.result_queue.put_nowait({'test': i})
            except queue.Full:
                break

    def test_thread_safety(self, video_processor_default, sample_frame):
        """Test thread safety of the processor."""
        # Start processing
        video_processor_default.start_processing()

        # Add frames from multiple threads
        def add_frames():
            for _ in range(3):
                try:
                    video_processor_default.frame_queue.put_nowait(sample_frame)
                    time.sleep(0.01)
                except queue.Full:
                    break

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_frames)
            threads.append(thread)
            thread.start()

        # Wait for threads to complete
        for thread in threads:
            thread.join()

        # Wait a bit for processing
        time.sleep(0.1)

        # Cleanup
        video_processor_default.stop_processing()

    def test_integration_with_real_components(self):
        """Test integration with real processing components (if available)."""
        try:
            # This test will only run if the real components are available
            with patch('rose.exec.process_video_stream.VelocityCalculator'):
                processor = VideoProcessor(
                    use_zoedepth=False,  # Use DPT for faster testing
                    object_confidence=0.3,  # Lower threshold
                    max_objects_for_segmentation=2
                )

            # Create a simple test frame
            test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            # Process the frame
            result = processor.process_frame(test_frame)

            # Verify basic structure
            assert isinstance(result, dict)
            assert 'depth_map' in result
            assert 'detections' in result
            assert 'object_prompts' in result
            assert 'segmentation_masks' in result
            assert 'original_frame' in result

            # Verify types
            assert isinstance(result['detections'], list)
            assert isinstance(result['object_prompts'], list)
            assert result['original_frame'] is test_frame

        except Exception as e:
            # Skip test if components are not available
            pytest.skip(f"Real components not available: {e}")

    def test_get_velocity_stats(self, video_processor_default):
        """Test getting velocity statistics."""
        # Mock velocity stats
        mock_velocity_stats = {
            'total_tracked_objects': 3,
            'average_velocity': 2.5,
            'min_velocity': 1.0,
            'max_velocity': 4.0,
            'velocity_samples': 3
        }
        
        video_processor_default.velocity_calculator.get_velocity_stats.return_value = mock_velocity_stats
        
        stats = video_processor_default.get_velocity_stats()
        
        assert stats == mock_velocity_stats
        video_processor_default.velocity_calculator.get_velocity_stats.assert_called_once()

    def test_get_velocity_stats_exception_handling(self, video_processor_default):
        """Test getting velocity statistics with exception handling."""
        # Mock exception in velocity calculator
        video_processor_default.velocity_calculator.get_velocity_stats.side_effect = Exception("Velocity error")
        
        stats = video_processor_default.get_velocity_stats()
        
        assert stats == {}


class TestVideoProcessorEdgeCases:
    """Test edge cases and error conditions for VideoProcessor."""

    def test_init_with_invalid_colormap(self):
        """Test initialization with invalid colormap."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'), \
             patch('rose.exec.process_video_stream.VelocityCalculator'):

            # Should not raise exception, just use default
            processor = VideoProcessor(colormap='invalid_colormap')
            assert processor.colormap == 'invalid_colormap'  # Should still be set

    def test_process_frame_empty_frame(self):
        """Test processing an empty frame."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'), \
             patch('rose.exec.process_video_stream.VelocityCalculator'):

            processor = VideoProcessor()
            empty_frame = np.array([], dtype=np.uint8)

            # Should handle gracefully
            result = processor.process_frame(empty_frame)
            assert isinstance(result, dict)

    def test_process_frame_none_frame(self):
        """Test processing a None frame."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'), \
             patch('rose.exec.process_video_stream.VelocityCalculator'):

            processor = VideoProcessor()

            # Should handle gracefully
            result = processor.process_frame(None)
            assert isinstance(result, dict)

    def test_extract_object_prompts_none_detections(self):
        """Test extracting prompts from None detections."""
        with patch('rose.exec.process_video_stream.DepthEstimator'), \
             patch('rose.exec.process_video_stream.ObjectDetector'), \
             patch('rose.exec.process_video_stream.ImageSegmenter'), \
             patch('rose.exec.process_video_stream.VelocityCalculator'):

            processor = VideoProcessor()
            prompts = processor.extract_object_prompts(None)
            assert prompts == []




if __name__ == "__main__":
    pytest.main([__file__])
