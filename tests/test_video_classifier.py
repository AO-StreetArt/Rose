import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import cv2

from rose.processing.video_classifier import VideoClassifier


class TestVideoClassifier(unittest.TestCase):
    """Test cases for the VideoClassifier class."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('rose.processing.video_classifier.AutoProcessor') as mock_processor, \
             patch('rose.processing.video_classifier.AutoModel') as mock_model:

            # Mock the processor and model
            mock_processor.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()

            self.video_classifier = VideoClassifier()
            
        # Create sample frames for testing
        self.sample_frames = []
        for i in range(8):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            self.sample_frames.append(frame)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any temporary files that might have been created
        pass

    def test_init(self):
        """Test VideoClassifier initialization."""
        self.assertEqual(self.video_classifier.model_name, "microsoft/xclip-base-patch32")
        self.assertIsNotNone(self.video_classifier.processor)
        self.assertIsNotNone(self.video_classifier.model)

    def test_extract_frames(self):
        """Test frame extraction from video file."""
        # Create a temporary video file for testing
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.close()

            try:
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, 30.0, (224, 224))

                # Write frames
                for frame in self.sample_frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()

                # Test frame extraction
                frames = self.video_classifier.extract_frames(temp_path, num_frames=4)

                self.assertEqual(len(frames), 4)
                self.assertTrue(all(isinstance(frame, np.ndarray) for frame in frames))
                self.assertTrue(all(frame.shape == (224, 224, 3) for frame in frames))
                self.assertTrue(all(frame.dtype == np.uint8 for frame in frames))

            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_extract_frames_invalid_path(self):
        """Test frame extraction with invalid video path."""
        with self.assertRaises(ValueError):
            self.video_classifier.extract_frames("nonexistent_video.mp4")

    def test_preprocess_frames(self):
        """Test frame preprocessing."""
        # Test that the method doesn't crash and returns something
        inputs = self.video_classifier.preprocess_frames(self.sample_frames)

        # Check that we got some result back
        self.assertIsNotNone(inputs)

    def test_classify_video(self):
        """Test video classification."""
        candidate_labels = ["playing sports", "cooking", "dancing", "reading"]

        with patch.object(self.video_classifier, 'extract_frames') as mock_extract, \
             patch.object(self.video_classifier, 'preprocess_frames') as mock_preprocess, \
             patch.object(self.video_classifier.model, '__call__') as mock_model_call:

            # Mock the frame extraction
            mock_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
            mock_extract.return_value = mock_frames

            # Mock the preprocessing
            mock_inputs = Mock()
            mock_preprocess.return_value = mock_inputs

            # Mock the model call
            mock_output = Mock()
            mock_output.logits_per_video = Mock()
            mock_output.logits_per_video.cpu.return_value.numpy.return_value = np.array([[0.1, 0.8, 0.05, 0.05]])
            mock_model_call.return_value = mock_output

            # Test classification
            result = self.video_classifier.classify_video("dummy_path.mp4", candidate_labels)

            # Verify the result structure
            self.assertIn('predictions', result)
            self.assertIn('scores', result)
            self.assertIn('top_prediction', result)

            # Verify predictions
            self.assertEqual(len(result['predictions']), len(candidate_labels))
            self.assertEqual(len(result['scores']), len(candidate_labels))

            # Verify top prediction
            self.assertIn(result['top_prediction'], candidate_labels)

    def test_classify_video_with_custom_frames(self):
        """Test video classification with custom frame extraction."""
        candidate_labels = ["action", "drama", "comedy"]

        with patch.object(self.video_classifier, 'preprocess_frames') as mock_preprocess, \
             patch.object(self.video_classifier.model, '__call__') as mock_model_call:

            # Mock the preprocessing
            mock_inputs = Mock()
            mock_preprocess.return_value = mock_inputs

            # Mock the model call
            mock_output = Mock()
            mock_output.logits_per_video = Mock()
            mock_output.logits_per_video.cpu.return_value.numpy.return_value = np.array([[0.7, 0.2, 0.1]])
            mock_model_call.return_value = mock_output

            # Test classification with custom frames
            result = self.video_classifier.classify_video_with_frames(self.sample_frames, candidate_labels)

            # Verify the result structure
            self.assertIn('predictions', result)
            self.assertIn('scores', result)
            self.assertIn('top_prediction', result)

            # Verify predictions
            self.assertEqual(len(result['predictions']), len(candidate_labels))
            self.assertEqual(len(result['scores']), len(candidate_labels))

            # Verify top prediction
            self.assertIn(result['top_prediction'], candidate_labels)

    def test_extract_frames_with_different_num_frames(self):
        """Test frame extraction with different numbers of frames."""
        # Create a temporary video file for testing
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.close()

            try:
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, 30.0, (224, 224))

                # Write frames
                for frame in self.sample_frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()

                # Test with different numbers of frames
                for num_frames in [1, 4, 8]:
                    frames = self.video_classifier.extract_frames(temp_path, num_frames=num_frames)
                    self.assertEqual(len(frames), num_frames)
                    self.assertTrue(all(isinstance(frame, np.ndarray) for frame in frames))

            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_preprocess_frames_with_different_frame_counts(self):
        """Test preprocessing with different numbers of frames."""
        # Test with 1 frame
        single_frame = [self.sample_frames[0]]
        inputs = self.video_classifier.preprocess_frames(single_frame)
        self.assertIsNotNone(inputs)

        # Test with 4 frames
        four_frames = self.sample_frames[:4]
        inputs = self.video_classifier.preprocess_frames(four_frames)
        self.assertIsNotNone(inputs)

        # Test with 8 frames
        inputs = self.video_classifier.preprocess_frames(self.sample_frames)
        self.assertIsNotNone(inputs)

    def test_classify_video_error_handling(self):
        """Test error handling in video classification."""
        # Test with empty candidate labels
        with self.assertRaises(ValueError):
            self.video_classifier.classify_video("dummy_path.mp4", [])

        # Test with None candidate labels
        with self.assertRaises(ValueError):
            self.video_classifier.classify_video("dummy_path.mp4", None)

    def test_classify_video_with_frames_error_handling(self):
        """Test error handling in video classification with frames."""
        # Test with empty frames
        with self.assertRaises(ValueError):
            self.video_classifier.classify_video_with_frames([], ["action", "drama"])

        # Test with None frames
        with self.assertRaises(ValueError):
            self.video_classifier.classify_video_with_frames(None, ["action", "drama"])

    def test_classify_video_performance(self):
        """Test performance of video classification."""
        import time
        
        candidate_labels = ["action", "drama", "comedy"]

        with patch.object(self.video_classifier, 'extract_frames') as mock_extract, \
             patch.object(self.video_classifier, 'preprocess_frames') as mock_preprocess, \
             patch.object(self.video_classifier.model, '__call__') as mock_model_call:

            # Mock the methods
            mock_extract.return_value = self.sample_frames
            mock_preprocess.return_value = Mock()
            mock_output = Mock()
            mock_output.logits_per_video = Mock()
            mock_output.logits_per_video.cpu.return_value.numpy.return_value = np.array([[0.7, 0.2, 0.1]])
            mock_model_call.return_value = mock_output

            # Measure classification time
            start_time = time.time()
            result = self.video_classifier.classify_video("dummy_path.mp4", candidate_labels)
            classification_time = time.time() - start_time

            # Should complete in reasonable time (less than 60 seconds)
            self.assertLess(classification_time, 60.0)
            self.assertIsNotNone(result)

    def test_video_classifier_with_different_model_names(self):
        """Test VideoClassifier with different model names."""
        # Test with default model name
        self.assertEqual(self.video_classifier.model_name, "microsoft/xclip-base-patch32")

        # Test with custom model name
        with patch('rose.processing.video_classifier.AutoProcessor') as mock_processor, \
             patch('rose.processing.video_classifier.AutoModel') as mock_model:

            mock_processor.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()

            custom_classifier = VideoClassifier(model_name="custom/model")
            self.assertEqual(custom_classifier.model_name, "custom/model")

    def test_video_classifier_device_handling(self):
        """Test VideoClassifier device handling."""
        # Test default device
        self.assertIsNotNone(self.video_classifier.device)

        # Test with custom device
        with patch('rose.processing.video_classifier.AutoProcessor') as mock_processor, \
             patch('rose.processing.video_classifier.AutoModel') as mock_model:

            mock_processor.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()

            custom_classifier = VideoClassifier(device="cpu")
            self.assertEqual(custom_classifier.device, "cpu")
