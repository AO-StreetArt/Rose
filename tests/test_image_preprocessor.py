import numpy as np
import tempfile
from PIL import Image
import os
from rose.preprocessing.image_utils import ImagePreprocessor

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