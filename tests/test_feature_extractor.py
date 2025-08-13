import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from rose.processing.feature_extractor import FeatureExtractor
from rose.preprocessing.image_utils import ImagePreprocessor
import os
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications.vgg16 import decode_predictions
import cv2


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.feature_extractor = FeatureExtractor()

    def test_feature_extractor_init(self):
        self.assertIsNotNone(self.feature_extractor)

    def test_extract_features_mocked(self):
        dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        with patch.object(self.feature_extractor.model, 'predict', return_value=['mocked_result']) as mock_predict:
            result = self.feature_extractor.extract_features(dummy_img)
            self.assertEqual(result, ['mocked_result'])
            mock_predict.assert_called_once()

    def test_extract_features_on_square_image(self):
        # Use the actual image file
        img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
        img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
        
        preds = self.feature_extractor.extract_features(img_array)
        # Assert that predictions are returned and not empty
        self.assertIsNotNone(preds)
        self.assertTrue(hasattr(preds, 'shape'))
        self.assertEqual(preds.shape[0], 1)
        self.assertTrue(np.any(preds > 0), "No features detected in the image.")
        
        # Overlay top-3 class labels on the image
        decoded = decode_predictions(preds, top=3)[0]
        labels = [f"{c[1]}: {c[2]:.2f}" for c in decoded]
        # Load original image for annotation
        orig_img = Image.open(img_path).convert('RGB').resize((224, 224))
        draw = ImageDraw.Draw(orig_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception as e:
            font = None
            print('Failed to load font')
            print(e)
        y = 5
        for label in labels:
            draw.text((5, y), label, fill=(255, 0, 0), font=font)
            y += 20
        save_path = os.path.join(os.path.dirname(__file__), 'features_squareTestImage.png')
        orig_img.save(save_path)

    def test_extract_features_vit_mocked(self):
        dummy_img = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        # Patch the vit_model and vit_processor to avoid downloading and running the real model
        with patch.object(self.feature_extractor, 'vit_processor') as mock_processor, \
             patch.object(self.feature_extractor, 'vit_model') as mock_model:
            mock_processor.return_value = {'pixel_values': np.zeros((1, 3, 224, 224))}
            mock_model.return_value = MagicMock()
            mock_model.__call__ = MagicMock(return_value={'logits': np.array([[0.1, 0.9]])})
            # Patch torch.no_grad with a context manager mock
            with patch('torch.no_grad', MagicMock(return_value=MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))):
                result = self.feature_extractor.extract_features_vit(dummy_img)
                self.assertIsNotNone(result)
                self.assertTrue('logits' in result or hasattr(result, 'logits'))

    def test_extract_features_dinov2_mocked(self):
        dummy_img = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        # Patch the dinov2_model and dinov2_processor to avoid downloading and running the real model
        with patch.object(self.feature_extractor, 'dinov2_processor') as mock_processor, \
             patch.object(self.feature_extractor, 'dinov2_model') as mock_model:
            mock_processor.return_value = {'pixel_values': np.zeros((1, 3, 224, 224))}
            mock_model.return_value = MagicMock()
            mock_model.__call__ = MagicMock(return_value={'logits': np.array([[0.2, 0.8]])})
            # Patch torch.no_grad with a context manager mock
            with patch('torch.no_grad', MagicMock(return_value=MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))):
                result = self.feature_extractor.extract_features_dinov2(dummy_img)
                self.assertIsNotNone(result)
                self.assertTrue('logits' in result or hasattr(result, 'logits'))

    def test_extract_features_vit_on_square_image(self):
        img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
        img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
        
        outputs = self.feature_extractor.extract_features_vit(img_array)
        # outputs can be a dict or a model output object with .logits
        logits = outputs['logits'] if isinstance(outputs, dict) else getattr(outputs, 'logits', None)
        self.assertIsNotNone(logits, "No logits returned by ViT model.")
        # logits shape: (1, num_classes)
        self.assertEqual(logits.shape[0], 1)
        self.assertTrue(np.any(logits.detach().cpu().numpy() != 0), "No features detected by ViT (all logits are zero).")

    def test_extract_features_dinov2_on_square_image(self):
        img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
        img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
        
        outputs = self.feature_extractor.extract_features_dinov2(img_array)
        # outputs can be a dict or a model output object with .logits
        logits = outputs['logits'] if isinstance(outputs, dict) else getattr(outputs, 'logits', None)
        self.assertIsNotNone(logits, "No logits returned by DINOv2 model.")
        # logits shape: (1, num_classes)
        self.assertEqual(logits.shape[0], 1)
        self.assertTrue(np.any(logits.detach().cpu().numpy() != 0), "No features detected by DINOv2 (all logits are zero).")

    def test_extract_features_with_different_input_sizes(self):
        """Test feature extraction with different input image sizes."""
        # Test with small image
        small_img = np.zeros((1, 100, 100, 3), dtype=np.float32)
        with patch.object(self.feature_extractor.model, 'predict', return_value=['mocked_result']) as mock_predict:
            result = self.feature_extractor.extract_features(small_img)
            self.assertEqual(result, ['mocked_result'])
            mock_predict.assert_called_once()

        # Test with large image
        large_img = np.zeros((1, 512, 512, 3), dtype=np.float32)
        with patch.object(self.feature_extractor.model, 'predict', return_value=['mocked_result']) as mock_predict:
            result = self.feature_extractor.extract_features(large_img)
            self.assertEqual(result, ['mocked_result'])
            mock_predict.assert_called_once()

    def test_extract_features_with_batch_processing(self):
        """Test feature extraction with batch processing."""
        # Test with batch of 2 images
        batch_img = np.zeros((2, 224, 224, 3), dtype=np.float32)
        with patch.object(self.feature_extractor.model, 'predict', return_value=['mocked_result1', 'mocked_result2']) as mock_predict:
            result = self.feature_extractor.extract_features(batch_img)
            self.assertEqual(result, ['mocked_result1', 'mocked_result2'])
            mock_predict.assert_called_once()

    def test_extract_features_error_handling(self):
        """Test error handling in feature extraction."""
        # Test with invalid input shape
        invalid_img = np.zeros((224, 224, 3), dtype=np.float32)  # Missing batch dimension
        
        with self.assertRaises(ValueError):
            self.feature_extractor.extract_features(invalid_img)

        # Test with None input
        with self.assertRaises(ValueError):
            self.feature_extractor.extract_features(None)
