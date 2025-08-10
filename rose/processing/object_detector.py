import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import json
import os

class ObjectDetector:
    """
    Detects objects in images using various pre-trained models (Faster R-CNN, etc.).
    Supports multiple detection frameworks and provides unified interface.
    """

    def __init__(self, model_type: str = "faster_rcnn", confidence_threshold: float = 0.5):
        """
        Initialize the object detector.

        Args:
            model_type (str): Type of model to use ('faster_rcnn', 'ssd')
            confidence_threshold (float): Minimum confidence score for detections
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []

        # Initialize model based on type
        if model_type == "faster_rcnn":
            self._init_faster_rcnn()
        elif model_type == "ssd":
            self._init_ssd()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _init_faster_rcnn(self):
        """Initialize Faster R-CNN model."""
        try:
            import torchvision
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True
            )
            self.model.eval()
            # COCO class names for Faster R-CNN
            self.class_names = [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
        except ImportError:
            self.model = None
            self.class_names = ["person", "car", "dog", "cat", "chair", "table"]

    def _init_ssd(self):
        """Initialize SSD model."""
        try:
            import torchvision
            self.model = torchvision.models.detection.ssd300_vgg16(
                pretrained=True
            )
            self.model.eval()
            # Same class names as Faster R-CNN
            self.class_names = [
                '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
        except ImportError:
            self.model = None
            self.class_names = ["person", "car", "dog", "cat", "chair", "table"]

    def detect_objects(self, img_array: np.ndarray) -> List[Dict]:
        """
        Detect objects in the given image array.

        Args:
            img_array (np.ndarray): Input image array (shape: (H, W, 3) or (1, H, W, 3))

        Returns:
            List[Dict]: List of detected objects with bounding boxes, confidence scores, and class names
        """
        if img_array.ndim == 4:
            img_array = img_array[0]  # Remove batch dimension

        if self.model_type == "faster_rcnn":
            return self._detect_faster_rcnn(img_array)
        elif self.model_type == "ssd":
            return self._detect_ssd(img_array)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _detect_faster_rcnn(self, img_array: np.ndarray) -> List[Dict]:
        """Detect objects using Faster R-CNN model."""
        if self.model is None:
            return self._mock_detection(img_array)

        # Convert to PIL Image
        img = Image.fromarray(img_array.astype('uint8'))

        # Transform for torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(img_tensor)

        detections = []
        if len(predictions) > 0:
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score > self.confidence_threshold:
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(score),
                        'class_name': self.class_names[label] if label < len(self.class_names) else f"class_{label}",
                        'class_id': int(label)
                    })

        return detections

    def _detect_ssd(self, img_array: np.ndarray) -> List[Dict]:
        """Detect objects using SSD model."""
        if self.model is None:
            return self._mock_detection(img_array)

        # Convert to PIL Image
        img = Image.fromarray(img_array.astype('uint8'))

        # Transform for torchvision
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(img_tensor)

        detections = []
        if len(predictions) > 0:
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score > self.confidence_threshold:
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(score),
                        'class_name': self.class_names[label] if label < len(self.class_names) else f"class_{label}",
                        'class_id': int(label)
                    })

        return detections

    def _mock_detection(self, img_array: np.ndarray) -> List[Dict]:
        """Mock detection for testing purposes."""
        height, width = img_array.shape[:2]

        # Create some mock detections
        mock_detections = [
            {
                'bbox': [width//4, height//4, width//2, height//2],
                'confidence': 0.85,
                'class_name': 'person',
                'class_id': 0
            },
            {
                'bbox': [width//8, height//8, width//3, height//3],
                'confidence': 0.72,
                'class_name': 'car',
                'class_id': 1
            }
        ]

        return mock_detections

    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Get a summary of detections.

        Args:
            detections (List[Dict]): List of detections from detect_objects()

        Returns:
            Dict: Summary statistics
        """
        if not detections:
            return {
                'total_objects': 0,
                'unique_classes': [],
                'average_confidence': 0.0,
                'class_counts': {}
            }

        class_counts = {}
        total_confidence = 0.0

        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']

            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += confidence

        return {
            'total_objects': len(detections),
            'unique_classes': list(class_counts.keys()),
            'average_confidence': total_confidence / len(detections),
            'class_counts': class_counts
        }

    def filter_detections_by_class(self, detections: List[Dict], class_names: List[str]) -> List[Dict]:
        """
        Filter detections by class names.

        Args:
            detections (List[Dict]): List of detections
            class_names (List[str]): List of class names to keep

        Returns:
            List[Dict]: Filtered detections
        """
        return [det for det in detections if det['class_name'] in class_names]

    def filter_detections_by_confidence(self, detections: List[Dict], min_confidence: float) -> List[Dict]:
        """
        Filter detections by minimum confidence threshold.

        Args:
            detections (List[Dict]): List of detections
            min_confidence (float): Minimum confidence threshold

        Returns:
            List[Dict]: Filtered detections
        """
        return [det for det in detections if det['confidence'] >= min_confidence]