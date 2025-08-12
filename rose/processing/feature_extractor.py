from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
from typing import Any


class FeatureExtractor:
    """
    Extracts features and objects from images using pre-trained models (e.g., VGG16, ResNet, ViT, DINOv2).
    """
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=True)
        # Initialize ViT model and processor
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        # Initialize DINOv2 model and processor using Auto classes
        self.dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dinov2_model = AutoModelForImageClassification.from_pretrained('facebook/dinov2-base')

    def extract_features(self, img_array: np.ndarray) -> np.ndarray:
        """
        Accepts a numpy array (preprocessed image) and returns VGG16 predictions.
        Args:
            img_array (np.ndarray): Preprocessed image array (shape: (1, 224, 224, 3)).
        Returns:
            list: List of detected features (top predictions).
        """
        img_array = preprocess_input(img_array)
        preds = self.model.predict(img_array)
        return preds

    def extract_features_vit(self, img_array: np.ndarray) -> Any:
        """
        Accepts a numpy array (preprocessed image) and returns Google ViT predictions.
        Args:
            img_array (np.ndarray): Preprocessed image array (shape: (1, 224, 224, 3)).
        Returns:
            dict: ViT model outputs including logits and hidden states.
        """
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array[0].astype('uint8'))

        # Process image with ViT processor
        inputs = self.vit_processor(images=img, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            outputs = self.vit_model(**inputs)

        return outputs

    def extract_features_dinov2(self, img_array: np.ndarray) -> Any:
        """
        Accepts a numpy array (preprocessed image) and returns Facebook DINOv2 predictions.
        Args:
            img_array (np.ndarray): Preprocessed image array (shape: (1, 224, 224, 3)).
        Returns:
            dict: DINOv2 model outputs including logits and hidden states.
        """
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array[0].astype('uint8'))

        # Process image with DINOv2 processor
        inputs = self.dinov2_processor(images=img, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)

        return outputs

    def classify_image_vit(self, img_array: np.ndarray) -> dict:
        """
        Classifies an image using the ViT model and returns the top predicted class and its score.
        Args:
            img_array (np.ndarray): Preprocessed image array (shape: (1, 224, 224, 3)).
        Returns:
            dict: Dictionary with 'label' and 'score' for the top prediction.
        """
        img = Image.fromarray(img_array[0].astype('uint8'))
        inputs = self.vit_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            label = self.vit_model.config.id2label[top_idx.item()]
            score = top_prob.item()
        return {"label": label, "score": score}
