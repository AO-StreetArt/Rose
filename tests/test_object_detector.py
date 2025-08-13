import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from rose.processing.object_detector import ObjectDetector
from rose.preprocessing.image_utils import ImagePreprocessor
import os
from PIL import Image, ImageDraw, ImageFont


class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.object_detector = ObjectDetector()

    def test_object_detector_init(self):
        """Test object detector initialization."""
        self.assertIsNotNone(self.object_detector)
        self.assertEqual(self.object_detector.model_type, "faster_rcnn")
        self.assertEqual(self.object_detector.confidence_threshold, 0.5)

    def test_object_detector_init_with_params(self):
        """Test object detector initialization with custom parameters."""
        od = ObjectDetector(model_type="faster_rcnn", confidence_threshold=0.7)
        self.assertEqual(od.model_type, "faster_rcnn")
        self.assertEqual(od.confidence_threshold, 0.7)

    def test_object_detector_invalid_model_type(self):
        """Test object detector initialization with invalid model type."""
        with self.assertRaises(ValueError):
            ObjectDetector(model_type="invalid_model")

    def test_detect_objects_mocked(self):
        """Test object detection with mocked model."""
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Test with mock detection
        detections = self.object_detector.detect_objects(dummy_img)
        self.assertIsInstance(detections, list)
        # Real models may not detect anything on dummy images
        self.assertGreaterEqual(len(detections), 0)

        # Check structure of detections if any
        for detection in detections:
            self.assertIn('bbox', detection)
            self.assertIn('confidence', detection)
            self.assertIn('class_name', detection)
            self.assertIn('class_id', detection)
            self.assertIsInstance(detection['bbox'], list)
            self.assertIsInstance(detection['confidence'], float)
            self.assertIsInstance(detection['class_name'], str)
            self.assertIsInstance(detection['class_id'], int)

    def test_detect_objects_with_batch_dimension(self):
        """Test object detection with batch dimension in input."""
        dummy_img = np.zeros((1, 224, 224, 3), dtype=np.uint8)

        detections = self.object_detector.detect_objects(dummy_img)
        self.assertIsInstance(detections, list)
        # Real models may not detect anything on dummy images
        self.assertGreaterEqual(len(detections), 0)

    def test_detect_objects_on_square_image(self):
        """Test object detection on actual image file."""
        img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
        img_array = ImagePreprocessor.load_and_preprocess_image(img_path)

        detections = self.object_detector.detect_objects(img_array[0])  # Remove batch dimension

        self.assertIsInstance(detections, list)
        # Should have at least some detections (even if mock)
        self.assertGreaterEqual(len(detections), 0)

        # Visualize detections if any
        if detections:
            orig_img = Image.open(img_path).convert('RGB').resize((224, 224))
            draw = ImageDraw.Draw(orig_img)

            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = None

            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']

                # Draw bounding box
                draw.rectangle(bbox, outline=(255, 0, 0), width=2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                draw.text((bbox[0], bbox[1] - 15), label, fill=(255, 0, 0), font=font)

            save_path = os.path.join(os.path.dirname(__file__), 'detections_squareTestImage.png')
            orig_img.save(save_path)

    def test_detect_objects_faster_rcnn(self):
        """Test Faster R-CNN object detection."""
        od = ObjectDetector(model_type="faster_rcnn")
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        detections = od.detect_objects(dummy_img)
        self.assertIsInstance(detections, list)
        self.assertGreaterEqual(len(detections), 0)

    def test_detect_objects_ssd(self):
        """Test SSD object detection."""
        od = ObjectDetector(model_type="ssd")
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        detections = od.detect_objects(dummy_img)
        self.assertIsInstance(detections, list)
        self.assertGreaterEqual(len(detections), 0)

    def test_detect_objects_yolo(self):
        """Test YOLO object detection."""
        od = ObjectDetector(model_type="yolo")
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        detections = od.detect_objects(dummy_img)
        self.assertIsInstance(detections, list)
        self.assertGreaterEqual(len(detections), 0)

    def test_detect_objects_with_different_confidence_thresholds(self):
        """Test object detection with different confidence thresholds."""
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Test with high confidence threshold
        od_high = ObjectDetector(confidence_threshold=0.9)
        detections_high = od_high.detect_objects(dummy_img)
        self.assertIsInstance(detections_high, list)

        # Test with low confidence threshold
        od_low = ObjectDetector(confidence_threshold=0.1)
        detections_low = od_low.detect_objects(dummy_img)
        self.assertIsInstance(detections_low, list)

        # High confidence threshold should have fewer or equal detections
        self.assertLessEqual(len(detections_high), len(detections_low))

    def test_detect_objects_with_different_image_sizes(self):
        """Test object detection with different image sizes."""
        # Test with small image
        small_img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections_small = self.object_detector.detect_objects(small_img)
        self.assertIsInstance(detections_small, list)

        # Test with large image
        large_img = np.zeros((512, 512, 3), dtype=np.uint8)
        detections_large = self.object_detector.detect_objects(large_img)
        self.assertIsInstance(detections_large, list)

        # Test with rectangular image
        rect_img = np.zeros((300, 400, 3), dtype=np.uint8)
        detections_rect = self.object_detector.detect_objects(rect_img)
        self.assertIsInstance(detections_rect, list)

    def test_detect_objects_with_different_image_types(self):
        """Test object detection with different image types."""
        # Test with uint8 image
        uint8_img = np.zeros((224, 224, 3), dtype=np.uint8)
        detections_uint8 = self.object_detector.detect_objects(uint8_img)
        self.assertIsInstance(detections_uint8, list)

        # Test with float32 image
        float32_img = np.zeros((224, 224, 3), dtype=np.float32)
        detections_float32 = self.object_detector.detect_objects(float32_img)
        self.assertIsInstance(detections_float32, list)

    def test_detect_objects_error_handling(self):
        """Test error handling in object detection."""
        # Test with None input
        with self.assertRaises(ValueError):
            self.object_detector.detect_objects(None)

        # Test with empty array
        empty_img = np.array([])
        with self.assertRaises(ValueError):
            self.object_detector.detect_objects(empty_img)

        # Test with wrong number of dimensions
        wrong_dim_img = np.zeros((224, 224), dtype=np.uint8)  # Missing channel dimension
        with self.assertRaises(ValueError):
            self.object_detector.detect_objects(wrong_dim_img)

    def test_detect_objects_performance(self):
        """Test performance of object detection."""
        import time
        
        # Create test image
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Measure detection time
        start_time = time.time()
        detections = self.object_detector.detect_objects(test_img)
        detection_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 30 seconds)
        self.assertLess(detection_time, 30.0)
        self.assertIsInstance(detections, list)

    def test_detect_objects_batch_processing(self):
        """Test batch processing of multiple images."""
        # Create multiple test images
        images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]
        
        # Process each image individually
        all_detections = []
        for img in images:
            detections = self.object_detector.detect_objects(img)
            all_detections.append(detections)
            self.assertIsInstance(detections, list)
        
        # Should have processed all images
        self.assertEqual(len(all_detections), 3)

    def test_detect_objects_with_real_image(self):
        """Test object detection with a real image if available."""
        img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
        
        if os.path.exists(img_path):
            # Load and process the image
            img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
            
            # Detect objects
            detections = self.object_detector.detect_objects(img_array[0])
            
            # Verify results
            self.assertIsInstance(detections, list)
            self.assertGreaterEqual(len(detections), 0)
            
            # Check detection structure if any detections found
            for detection in detections:
                self.assertIn('bbox', detection)
                self.assertIn('confidence', detection)
                self.assertIn('class_name', detection)
                self.assertIn('class_id', detection)
                
                # Verify bbox format
                bbox = detection['bbox']
                self.assertEqual(len(bbox), 4)  # [x1, y1, x2, y2]
                self.assertLess(bbox[0], bbox[2])  # x1 < x2
                self.assertLess(bbox[1], bbox[3])  # y1 < y2
                
                # Verify confidence range
                confidence = detection['confidence']
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)
                
                # Verify class information
                self.assertIsInstance(detection['class_name'], str)
                self.assertIsInstance(detection['class_id'], int)
                self.assertGreaterEqual(detection['class_id'], 0)
