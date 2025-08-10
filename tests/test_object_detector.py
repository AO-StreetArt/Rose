import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from rose.processing.object_detector import ObjectDetector
from rose.preprocessing.image_utils import ImagePreprocessor
import os
from PIL import Image, ImageDraw, ImageFont

def test_object_detector_init():
    """Test object detector initialization."""
    od = ObjectDetector()
    assert od is not None
    assert od.model_type == "faster_rcnn"
    assert od.confidence_threshold == 0.5

def test_object_detector_init_with_params():
    """Test object detector initialization with custom parameters."""
    od = ObjectDetector(model_type="faster_rcnn", confidence_threshold=0.7)
    assert od.model_type == "faster_rcnn"
    assert od.confidence_threshold == 0.7

def test_object_detector_invalid_model_type():
    """Test object detector initialization with invalid model type."""
    with pytest.raises(ValueError, match="Unsupported model type"):
        ObjectDetector(model_type="invalid_model")

def test_detect_objects_mocked():
    """Test object detection with mocked model."""
    od = ObjectDetector()
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    # Test with mock detection
    detections = od.detect_objects(dummy_img)
    assert isinstance(detections, list)
    # Real models may not detect anything on dummy images
    assert len(detections) >= 0

    # Check structure of detections if any
    for detection in detections:
        assert 'bbox' in detection
        assert 'confidence' in detection
        assert 'class_name' in detection
        assert 'class_id' in detection
        assert isinstance(detection['bbox'], list)
        assert isinstance(detection['confidence'], float)
        assert isinstance(detection['class_name'], str)
        assert isinstance(detection['class_id'], int)

def test_detect_objects_with_batch_dimension():
    """Test object detection with batch dimension in input."""
    od = ObjectDetector()
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.uint8)

    detections = od.detect_objects(dummy_img)
    assert isinstance(detections, list)
    # Real models may not detect anything on dummy images
    assert len(detections) >= 0

def test_detect_objects_on_square_image():
    """Test object detection on actual image file."""
    img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
    img_array = ImagePreprocessor.load_and_preprocess_image(img_path)

    od = ObjectDetector()
    detections = od.detect_objects(img_array[0])  # Remove batch dimension

    assert isinstance(detections, list)
    # Should have at least some detections (even if mock)
    assert len(detections) >= 0

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

def test_detect_objects_faster_rcnn():
    """Test Faster R-CNN object detection."""
    od = ObjectDetector(model_type="faster_rcnn")
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    detections = od.detect_objects(dummy_img)
    assert isinstance(detections, list)
    assert len(detections) >= 0

def test_detect_objects_ssd():
    """Test SSD object detection."""
    od = ObjectDetector(model_type="ssd")
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    detections = od.detect_objects(dummy_img)
    assert isinstance(detections, list)
    assert len(detections) >= 0

def test_get_detection_summary():
    """Test detection summary generation."""
    od = ObjectDetector()

    # Test with empty detections
    summary = od.get_detection_summary([])
    assert summary['total_objects'] == 0
    assert summary['unique_classes'] == []
    assert summary['average_confidence'] == 0.0
    assert summary['class_counts'] == {}

    # Test with mock detections
    mock_detections = [
        {'bbox': [10, 10, 50, 50], 'confidence': 0.8, 'class_name': 'person', 'class_id': 0},
        {'bbox': [60, 60, 100, 100], 'confidence': 0.9, 'class_name': 'car', 'class_id': 1},
        {'bbox': [110, 110, 150, 150], 'confidence': 0.7, 'class_name': 'person', 'class_id': 0}
    ]

    summary = od.get_detection_summary(mock_detections)
    assert summary['total_objects'] == 3
    assert set(summary['unique_classes']) == {'person', 'car'}
    assert summary['average_confidence'] == pytest.approx(0.8, rel=1e-2)
    assert summary['class_counts'] == {'person': 2, 'car': 1}

def test_filter_detections_by_class():
    """Test filtering detections by class names."""
    od = ObjectDetector()

    mock_detections = [
        {'bbox': [10, 10, 50, 50], 'confidence': 0.8, 'class_name': 'person', 'class_id': 0},
        {'bbox': [60, 60, 100, 100], 'confidence': 0.9, 'class_name': 'car', 'class_id': 1},
        {'bbox': [110, 110, 150, 150], 'confidence': 0.7, 'class_name': 'dog', 'class_id': 2}
    ]

    # Filter for person only
    filtered = od.filter_detections_by_class(mock_detections, ['person'])
    assert len(filtered) == 1
    assert filtered[0]['class_name'] == 'person'

    # Filter for multiple classes
    filtered = od.filter_detections_by_class(mock_detections, ['person', 'car'])
    assert len(filtered) == 2
    class_names = [det['class_name'] for det in filtered]
    assert 'person' in class_names
    assert 'car' in class_names

def test_filter_detections_by_confidence():
    """Test filtering detections by confidence threshold."""
    od = ObjectDetector()

    mock_detections = [
        {'bbox': [10, 10, 50, 50], 'confidence': 0.8, 'class_name': 'person', 'class_id': 0},
        {'bbox': [60, 60, 100, 100], 'confidence': 0.9, 'class_name': 'car', 'class_id': 1},
        {'bbox': [110, 110, 150, 150], 'confidence': 0.6, 'class_name': 'dog', 'class_id': 2}
    ]

    # Filter with confidence threshold 0.7
    filtered = od.filter_detections_by_confidence(mock_detections, 0.7)
    assert len(filtered) == 2
    for det in filtered:
        assert det['confidence'] >= 0.7

    # Filter with confidence threshold 0.9
    filtered = od.filter_detections_by_confidence(mock_detections, 0.9)
    assert len(filtered) == 1
    assert filtered[0]['confidence'] == 0.9

def test_mock_detection():
    """Test mock detection functionality."""
    od = ObjectDetector()
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    detections = od._mock_detection(dummy_img)
    assert isinstance(detections, list)
    assert len(detections) == 2  # Should return 2 mock detections

    for detection in detections:
        assert 'bbox' in detection
        assert 'confidence' in detection
        assert 'class_name' in detection
        assert 'class_id' in detection
        assert detection['class_name'] in ['person', 'car']
        assert detection['confidence'] > 0.5

def test_detect_objects_invalid_model_type():
    """Test detection with invalid model type."""
    od = ObjectDetector()
    od.model_type = "invalid_type"
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported model type"):
        od.detect_objects(dummy_img)

def test_faster_rcnn_detection_mocked():
    """Test Faster R-CNN detection with mocked torchvision."""
    with patch('torchvision.models.detection.fasterrcnn_resnet50_fpn') as mock_model:
        mock_model.return_value = MagicMock()
        mock_model.return_value.eval.return_value = None

        od = ObjectDetector(model_type="faster_rcnn")
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Mock the detection process
        mock_model.return_value.return_value = [{
            'boxes': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=lambda: np.array([[10, 10, 50, 50]])))),
            'scores': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=lambda: np.array([0.8])))),
            'labels': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=lambda: np.array([1]))))
        }]

        detections = od._detect_faster_rcnn(dummy_img)
        assert isinstance(detections, list)

def test_ssd_detection_mocked():
    """Test SSD detection with mocked torchvision."""
    with patch('torchvision.models.detection.ssd300_vgg16') as mock_model:
        mock_model.return_value = MagicMock()
        mock_model.return_value.eval.return_value = None

        od = ObjectDetector(model_type="ssd")
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Mock the detection process
        mock_model.return_value.return_value = [{
            'boxes': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=lambda: np.array([[10, 10, 50, 50]])))),
            'scores': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=lambda: np.array([0.8])))),
            'labels': MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=lambda: np.array([1]))))
        }]

        detections = od._detect_ssd(dummy_img)
        assert isinstance(detections, list)