import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from rose.processing.feature_extractor import FeatureExtractor
from rose.preprocessing.image_utils import ImagePreprocessor
import os
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications.vgg16 import decode_predictions
import cv2

def test_feature_extractor_init():
    fe = FeatureExtractor()
    assert fe is not None

def test_extract_features_mocked():
    fe = FeatureExtractor()
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    with patch.object(fe.model, 'predict', return_value=['mocked_result']) as mock_predict:
        result = fe.extract_features(dummy_img)
        assert result == ['mocked_result']
        mock_predict.assert_called_once()

def test_extract_features_on_square_image():
    # Use the actual image file
    img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
    img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
    fe = FeatureExtractor()
    preds = fe.extract_features(img_array)
    # Assert that predictions are returned and not empty
    assert preds is not None
    assert hasattr(preds, 'shape')
    assert preds.shape[0] == 1
    assert np.any(preds > 0), "No features detected in the image."
    # Overlay top-3 class labels on the image
    decoded = decode_predictions(preds, top=3)[0]
    labels = [f"{c[1]}: {c[2]:.2f}" for c in decoded]
    # Load original image for annotation
    orig_img = Image.open(img_path).convert('RGB').resize((224, 224))
    draw = ImageDraw.Draw(orig_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = None
    y = 5
    for label in labels:
        draw.text((5, y), label, fill=(255, 0, 0), font=font)
        y += 20
    save_path = os.path.join(os.path.dirname(__file__), 'features_squareTestImage.png')
    orig_img.save(save_path)

def test_extract_features_vit_mocked():
    fe = FeatureExtractor()
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    # Patch the vit_model and vit_processor to avoid downloading and running the real model
    with patch.object(fe, 'vit_processor') as mock_processor, \
         patch.object(fe, 'vit_model') as mock_model:
        mock_processor.return_value = {'pixel_values': np.zeros((1, 3, 224, 224))}
        mock_model.return_value = MagicMock()
        mock_model.__call__ = MagicMock(return_value={'logits': np.array([[0.1, 0.9]])})
        # Patch torch.no_grad with a context manager mock
        with patch('torch.no_grad', MagicMock(return_value=MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))):
            result = fe.extract_features_vit(dummy_img)
            assert result is not None
            assert 'logits' in result or hasattr(result, 'logits')

def test_extract_features_dinov2_mocked():
    fe = FeatureExtractor()
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.uint8)
    # Patch the dinov2_model and dinov2_processor to avoid downloading and running the real model
    with patch.object(fe, 'dinov2_processor') as mock_processor, \
         patch.object(fe, 'dinov2_model') as mock_model:
        mock_processor.return_value = {'pixel_values': np.zeros((1, 3, 224, 224))}
        mock_model.return_value = MagicMock()
        mock_model.__call__ = MagicMock(return_value={'logits': np.array([[0.2, 0.8]])})
        # Patch torch.no_grad with a context manager mock
        with patch('torch.no_grad', MagicMock(return_value=MagicMock(__enter__=lambda s: None, __exit__=lambda s, exc_type, exc_val, exc_tb: None))):
            result = fe.extract_features_dinov2(dummy_img)
            assert result is not None
            assert 'logits' in result or hasattr(result, 'logits')

def test_extract_features_vit_on_square_image():
    img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
    img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
    fe = FeatureExtractor()
    outputs = fe.extract_features_vit(img_array)
    # outputs can be a dict or a model output object with .logits
    logits = outputs['logits'] if isinstance(outputs, dict) else getattr(outputs, 'logits', None)
    assert logits is not None, "No logits returned by ViT model."
    # logits shape: (1, num_classes)
    assert logits.shape[0] == 1
    assert np.any(logits.detach().cpu().numpy() != 0), "No features detected by ViT (all logits are zero)."

def test_extract_features_dinov2_on_square_image():
    img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
    img_array = ImagePreprocessor.load_and_preprocess_image(img_path)
    fe = FeatureExtractor()
    outputs = fe.extract_features_dinov2(img_array)
    logits = outputs['logits'] if isinstance(outputs, dict) else getattr(outputs, 'logits', None)
    assert logits is not None, "No logits returned by DINOv2 model."
    assert logits.shape[0] == 1
    assert np.any(logits.detach().cpu().numpy() != 0), "No features detected by DINOv2 (all logits are zero)." 

def test_classify_image_vit_returns_label_and_score():
    extractor = FeatureExtractor()
    # Create a dummy image (random noise, shape: (1, 224, 224, 3))
    img_array = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)
    result = extractor.classify_image_vit(img_array)
    assert isinstance(result, dict), "Result should be a dictionary."
    assert 'label' in result, "Result should contain a 'label' key."
    assert 'score' in result, "Result should contain a 'score' key."
    assert isinstance(result['label'], str), "Label should be a string."
    assert isinstance(result['score'], float), "Score should be a float." 

def test_classify_image_vit_on_square_image():
    extractor = FeatureExtractor()
    img_path = os.path.join(os.path.dirname(__file__), 'squareTestImage.png')
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert image is not None, f"Failed to load image at {img_path}"
    # Resize to (224, 224) as required by ViT
    image_resized = cv2.resize(image, (224, 224))
    img_array = image_resized[np.newaxis, ...]
    result = extractor.classify_image_vit(img_array)
    assert isinstance(result, dict), "Result should be a dictionary."
    assert 'label' in result, "Result should contain a 'label' key."
    assert 'score' in result, "Result should contain a 'score' key."
    assert isinstance(result['label'], str), "Label should be a string."
    assert isinstance(result['score'], float), "Score should be a float."
    assert result['label'], "Label should not be empty." 