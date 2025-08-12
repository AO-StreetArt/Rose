import numpy as np
import tempfile
from PIL import Image
import os
import cv2
from rose.preprocessing.image_utils import ImagePreprocessor
import pytest

def test_image_preprocessor_init():
    ip = ImagePreprocessor()
    assert ip is not None

def test_load_and_preprocess_image():
    # Create a temporary grayscale image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = Image.new('L', (300, 300), color=128)
        img.save(tmp.name)
        tmp_path = tmp.name
    try:
        arr = ImagePreprocessor.load_and_preprocess_image(tmp_path)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 224, 224, 3)
    finally:
        os.remove(tmp_path)

def test_ensure_rgb_pil_image():
    # Grayscale numpy array
    gray_np = np.ones((100, 100), dtype=np.uint8) * 128
    rgb_img = ImagePreprocessor.ensure_rgb_pil_image(gray_np)
    assert isinstance(rgb_img, Image.Image)
    assert rgb_img.mode == 'RGB'

    # Grayscale PIL Image
    gray_pil = Image.new('L', (100, 100), color=128)
    rgb_img2 = ImagePreprocessor.ensure_rgb_pil_image(gray_pil)
    assert isinstance(rgb_img2, Image.Image)
    assert rgb_img2.mode == 'RGB'

    # Already RGB PIL Image
    rgb_pil = Image.new('RGB', (100, 100), color=128)
    rgb_img3 = ImagePreprocessor.ensure_rgb_pil_image(rgb_pil)
    assert isinstance(rgb_img3, Image.Image)
    assert rgb_img3.mode == 'RGB'

def test_ensure_bgr_image():
    """Test the ensure_bgr_image method with different input types."""
    # Test with grayscale image (2D array)
    gray_image = np.ones((100, 100), dtype=np.uint8) * 128
    bgr_image = ImagePreprocessor.ensure_bgr_image(gray_image)
    assert isinstance(bgr_image, np.ndarray)
    assert bgr_image.shape == (100, 100, 3)
    assert bgr_image.dtype == np.uint8

    # Test with already BGR image (3D array)
    bgr_input = np.ones((100, 100, 3), dtype=np.uint8) * 128
    bgr_output = ImagePreprocessor.ensure_bgr_image(bgr_input)
    assert isinstance(bgr_output, np.ndarray)
    assert bgr_output.shape == (100, 100, 3)
    assert bgr_output.dtype == np.uint8
    # Should be the same array (no conversion needed)
    assert np.array_equal(bgr_input, bgr_output)

def test_create_blob_for_hed():
    """Test the create_blob_for_hed method."""
    # Create a test BGR image
    bgr_image = np.ones((100, 150, 3), dtype=np.uint8) * 128

    # Create blob
    blob = ImagePreprocessor.create_blob_for_hed(bgr_image)

    # Verify blob properties
    assert isinstance(blob, np.ndarray)
    assert blob.shape == (1, 3, 100, 150)  # (batch, channels, height, width)
    assert blob.dtype == np.float32

    # Test with different image dimensions
    bgr_image_2 = np.ones((200, 300, 3), dtype=np.uint8) * 64
    blob_2 = ImagePreprocessor.create_blob_for_hed(bgr_image_2)
    assert blob_2.shape == (1, 3, 200, 300)

def test_create_blob_for_hed_with_mean_subtraction():
    """Test that the blob creation properly applies mean subtraction."""
    # Create a test BGR image with known values
    bgr_image = np.ones((50, 50, 3), dtype=np.uint8) * 128

    # Create blob
    blob = ImagePreprocessor.create_blob_for_hed(bgr_image)

    # The mean values used in the method are (104.00698793, 116.66876762, 122.67891434)
    # So the expected values should be approximately 128 - mean for each channel
    expected_b = 128 - 104.00698793
    expected_g = 128 - 116.66876762
    expected_r = 128 - 122.67891434

    # Check that the blob values are close to expected (allowing for floating point precision)
    assert abs(blob[0, 0, 0, 0] - expected_b) < 1e-5
    assert abs(blob[0, 1, 0, 0] - expected_g) < 1e-5
    assert abs(blob[0, 2, 0, 0] - expected_r) < 1e-5

def test_ensure_bgr_image_edge_cases():
    """Test edge cases for ensure_bgr_image method."""
    # Test with very small image
    small_gray = np.ones((1, 1), dtype=np.uint8) * 255
    small_bgr = ImagePreprocessor.ensure_bgr_image(small_gray)
    assert small_bgr.shape == (1, 1, 3)

    # Test with zero values
    zero_gray = np.zeros((10, 10), dtype=np.uint8)
    zero_bgr = ImagePreprocessor.ensure_bgr_image(zero_gray)
    assert zero_bgr.shape == (10, 10, 3)
    assert np.all(zero_bgr == 0)

def test_ensure_grayscale_image():
    """Test the ensure_grayscale_image method with different input types."""
    # Test with BGR image (3D array)
    bgr_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    gray_image = ImagePreprocessor.ensure_grayscale_image(bgr_image)
    assert isinstance(gray_image, np.ndarray)
    assert gray_image.shape == (100, 100)
    assert gray_image.dtype == np.uint8

    # Test with already grayscale image (2D array)
    gray_input = np.ones((100, 100), dtype=np.uint8) * 128
    gray_output = ImagePreprocessor.ensure_grayscale_image(gray_input)
    assert isinstance(gray_output, np.ndarray)
    assert gray_output.shape == (100, 100)
    assert gray_output.dtype == np.uint8
    # Should be the same array (no conversion needed)
    assert np.array_equal(gray_input, gray_output)

def test_ensure_grayscale_image_edge_cases():
    """Test edge cases for ensure_grayscale_image method."""
    # Test with very small BGR image
    small_bgr = np.ones((1, 1, 3), dtype=np.uint8) * 255
    small_gray = ImagePreprocessor.ensure_grayscale_image(small_bgr)
    assert small_gray.shape == (1, 1)

    # Test with zero values
    zero_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    zero_gray = ImagePreprocessor.ensure_grayscale_image(zero_bgr)
    assert zero_gray.shape == (10, 10)
    assert np.all(zero_gray == 0)

    # Test with different color values
    color_bgr = np.zeros((5, 5, 3), dtype=np.uint8)
    color_bgr[:, :, 0] = 100  # Blue channel
    color_bgr[:, :, 1] = 150  # Green channel
    color_bgr[:, :, 2] = 200  # Red channel
    color_gray = ImagePreprocessor.ensure_grayscale_image(color_bgr)
    assert color_gray.shape == (5, 5)
    # Grayscale conversion should produce different values than the original channels
    assert not np.array_equal(color_gray, color_bgr[:, :, 0])
    assert not np.array_equal(color_gray, color_bgr[:, :, 1])
    assert not np.array_equal(color_gray, color_bgr[:, :, 2])

def test_load_and_preprocess_for_feature_extraction():
    """Test the load_and_preprocess_for_feature_extraction method."""
    # Test with numpy array input
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    processed = ImagePreprocessor.load_and_preprocess_for_feature_extraction(test_image)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 224, 224, 3)  # (batch, height, width, channels)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0 and processed.max() <= 1.0  # Normalized to [0, 1]

    # Test with custom target size
    processed_custom = ImagePreprocessor.load_and_preprocess_for_feature_extraction(
        test_image, target_size=(64, 64)
    )
    assert processed_custom.shape == (1, 64, 64, 3)

def test_load_and_preprocess_for_feature_extraction_edge_cases():
    """Test edge cases for load_and_preprocess_for_feature_extraction method."""
    # Test with very small image
    small_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
    processed = ImagePreprocessor.load_and_preprocess_for_feature_extraction(small_image)
    assert processed.shape == (1, 224, 224, 3)

    # Test with zero values
    zero_image = np.zeros((50, 50, 3), dtype=np.uint8)
    processed_zero = ImagePreprocessor.load_and_preprocess_for_feature_extraction(zero_image)
    assert processed_zero.shape == (1, 224, 224, 3)
    assert np.all(processed_zero == 0.0)

    # Test with grayscale image (should be converted to RGB)
    gray_image = np.ones((50, 50), dtype=np.uint8) * 128
    processed_gray = ImagePreprocessor.load_and_preprocess_for_feature_extraction(gray_image)
    assert processed_gray.shape == (1, 224, 224, 3)

def test_load_and_preprocess_for_feature_extraction_file_path():
    """Test load_and_preprocess_for_feature_extraction with file path input."""
    # Create a temporary test image file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = Image.new('RGB', (100, 100), color=(128, 128, 128))
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        processed = ImagePreprocessor.load_and_preprocess_for_feature_extraction(tmp_path)
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (1, 224, 224, 3)
        assert processed.dtype == np.float32
    finally:
        os.remove(tmp_path)

def test_load_and_preprocess_for_feature_extraction_invalid_path():
    """Test that invalid file path raises appropriate error."""
    with pytest.raises(ValueError, match="Could not load image from path"):
        ImagePreprocessor.load_and_preprocess_for_feature_extraction("nonexistent_image.jpg")
