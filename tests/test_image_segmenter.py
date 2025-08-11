import os
import numpy as np
from PIL import Image
import pytest
from rose.processing.image_segmenter import ImageSegmenter

def test_image_segmenter_on_square_image():
    # Path to the test image
    img_path = os.path.join(os.path.dirname(__file__), "squareTestImage.png")
    assert os.path.exists(img_path), f"Test image not found: {img_path}"
    image = Image.open(img_path).convert("RGB")

    # Instantiate the segmenter
    segmenter = ImageSegmenter(device="cpu")
    prompts = ["square"]
    masks = segmenter.segment(image, prompts)

    # Check output shape and type
    assert isinstance(masks, np.ndarray), "Output should be a numpy array"
    assert masks.shape[0] == 1, f"Expected 1 mask, got {masks.shape[0]}"
    assert masks.shape[1] == image.height and masks.shape[2] == image.width, (
        f"Mask shape {masks.shape} does not match image size {(image.height, image.width)}"
    )
    assert (0.0 <= masks).all() and (masks <= 1.0).all(), "Mask values should be in [0, 1]"