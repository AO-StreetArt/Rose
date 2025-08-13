from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple


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

    @staticmethod
    def ensure_bgr_image(image: np.ndarray) -> np.ndarray:
        """
        Ensures the input image is in BGR format for OpenCV operations.
        Args:
            image (np.ndarray): Input image array.
        Returns:
            np.ndarray: BGR image array.
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def create_blob_for_hed(image: np.ndarray) -> np.ndarray:
        """
        Creates a blob from image for HED edge detection model input.
        Args:
            image (np.ndarray): Input image array in BGR format.
        Returns:
            np.ndarray: Blob suitable for HED model input.
        """
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(image.shape[1], image.shape[0]),
            mean=(104.00698793, 116.66876762, 122.67891434),
            swapRB=False,
            crop=False
        )
        return blob

    @staticmethod
    def ensure_grayscale_image(image: np.ndarray) -> np.ndarray:
        """
        Ensures the input image is in grayscale format for edge detection.
        Args:
            image (np.ndarray): Input image array (BGR, RGB, or grayscale).
        Returns:
            np.ndarray: Grayscale image array.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return gray

    @staticmethod
    def convertBGRtoRGB(image_input: np.ndarray):
        return cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_and_preprocess_for_feature_extraction(
            image_input: Union[np.ndarray, str],
            target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Load and preprocess an image for feature extraction.
        Args:
            image_input (Union[np.ndarray, str]): Image as numpy array or file path
            target_size (Tuple[int, int]): Target size for resizing (default: (224, 224))
        Returns:
            np.ndarray: Preprocessed image array with batch dimension
        """
        if isinstance(image_input, str):
            # Load image from file path
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            img = ImagePreprocessor.convertBGRtoRGB(img)
        else:
            # Assume it's already a numpy array
            img = image_input.copy()
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Resize to target size (standard size for many models)
        img = cv2.resize(img, target_size)
        # Normalize to [0, 1] range
        img = img.astype(np.float32) / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
