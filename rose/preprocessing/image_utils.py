from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Handles image pre-processing tasks such as resizing, normalization, and file format conversion.
    """
    @staticmethod
    def load_and_preprocess_image(img_path, target_size=(224, 224)):
        """
        Loads an image file and converts it to a numpy array suitable for VGG16 input.
        Args:
            img_path (str): Path to the image file.
            target_size (tuple): Desired image size (default is (224, 224) for VGG16).
        Returns:
            np.ndarray: Preprocessed image array.
        """
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    @staticmethod
    def ensure_rgb_pil_image(img):
        """
        Ensures the input is a valid RGB PIL Image. Converts numpy arrays and non-RGB images as needed.
        Args:
            img (np.ndarray or PIL.Image.Image): Input image.
        Returns:
            PIL.Image.Image: RGB image.
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img 