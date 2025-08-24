import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Union
import cv2

from .feature_extractor import FeatureExtractor
from .feature_detector import FeatureDetector
from ..preprocessing.image_utils import ImagePreprocessor


class ImageComparator:
    """
    Compares images using cosine similarity on extracted features.
    Supports multiple feature extraction methods (VGG16, ViT, DINOv2).
    """

    def __init__(self, feature_extractor: FeatureExtractor = None, feature_detector: FeatureDetector = None):
        """
        Initialize the ImageComparator with a feature extractor.

        Args:
            feature_extractor (FeatureExtractor, optional): Feature extractor instance.
                If None, creates a new one.
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.feature_detector = feature_detector or FeatureDetector(n_features=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def compare_images(self,
                       img1: Union[np.ndarray, str],
                       img2: Union[np.ndarray, str],
                       method: str = 'vgg16',
                       normalize: bool = True) -> Dict[str, Any]:
        """
        Compare two images using cosine similarity on extracted features.

        Args:
            img1 (Union[np.ndarray, str]): First image as numpy array or file path
            img2 (Union[np.ndarray, str]): Second image as numpy array or file path
            method (str): Feature extraction method ('vgg16', 'vit', 'dinov2')
            normalize (bool): Whether to normalize features before comparison

        Returns:
            Dict[str, Any]: Comparison results including similarity score and metadata
        """
        # Load and preprocess images
        img1_array = self._load_and_preprocess_image(img1)
        img2_array = self._load_and_preprocess_image(img2)

        # Extract features
        features1 = self._extract_features(img1_array, method)
        features2 = self._extract_features(img2_array, method)

        # Calculate similarity
        similarity_score = self._calculate_similarity(features1, features2, normalize)

        return {
            'similarity_score': similarity_score,
            'method': method,
            'normalized': normalize,
            'feature_shape_1': features1.shape,
            'feature_shape_2': features2.shape
        }

    def compare_multiple_images(self,
                                images: List[Union[np.ndarray, str]],
                                method: str = 'vgg16',
                                normalize: bool = True) -> Dict[str, Any]:
        """
        Compare multiple images pairwise using cosine similarity.

        Args:
            images (List[Union[np.ndarray, str]]): List of images as numpy arrays or file paths
            method (str): Feature extraction method ('vgg16', 'vit', 'dinov2')
            normalize (bool): Whether to normalize features before comparison

        Returns:
            Dict[str, Any]: Matrix of similarity scores and metadata
        """
        n_images = len(images)
        similarity_matrix = np.zeros((n_images, n_images))

        # Extract features for all images
        features_list = []
        for img in images:
            img_array = self._load_and_preprocess_image(img)
            features = self._extract_features(img_array, method)
            features_list.append(features)

        # Calculate pairwise similarities
        for i in range(n_images):
            for j in range(n_images):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Self-similarity
                else:
                    similarity_matrix[i, j] = self._calculate_similarity(
                        features_list[i], features_list[j], normalize
                    )

        return {
            'similarity_matrix': similarity_matrix,
            'method': method,
            'normalized': normalize,
            'num_images': n_images,
            'feature_shapes': [f.shape for f in features_list]
        }

    def find_most_similar(self,
                          query_image: Union[np.ndarray, str],
                          candidate_images: List[Union[np.ndarray, str]],
                          method: str = 'vgg16',
                          normalize: bool = True,
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar images to a query image.

        Args:
            query_image (Union[np.ndarray, str]): Query image as numpy array or file path
            candidate_images (List[Union[np.ndarray, str]]): List of candidate images
            method (str): Feature extraction method ('vgg16', 'vit', 'dinov2')
            normalize (bool): Whether to normalize features before comparison
            top_k (int): Number of top similar images to return

        Returns:
            List[Dict[str, Any]]: List of top-k similar images with scores and indices
        """
        # Extract query image features
        query_array = self._load_and_preprocess_image(query_image)
        query_features = self._extract_features(query_array, method)

        # Extract features for all candidate images
        similarities = []
        for i, candidate_img in enumerate(candidate_images):
            candidate_array = self._load_and_preprocess_image(candidate_img)
            candidate_features = self._extract_features(candidate_array, method)

            similarity = self._calculate_similarity(query_features, candidate_features, normalize)
            similarities.append({
                'index': i,
                'similarity_score': similarity,
                'image': candidate_img
            })

        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)

        return similarities[:top_k]

    def compare_images_orb(self,
                           img1: Union[np.ndarray, str],
                           img2: Union[np.ndarray, str],
                           method: str = 'vgg16',
                           normalize: bool = True) -> Dict[str, Any]:
        """
        Compare two images using ORB feature detection and Brute-Force matching.

        Args:
            img1 (Union[np.ndarray, str]): First image as numpy array or file path
            img2 (Union[np.ndarray, str]): Second image as numpy array or file path
            method (str): Feature extraction method ('vgg16', 'vit', 'dinov2') - kept for API consistency
            normalize (bool): Whether to normalize features before comparison - kept for API consistency

        Returns:
            Dict[str, Any]: Comparison results including match count, match ratio, and metadata
        """
        # Load and preprocess images for ORB (different from deep learning preprocessing)
        img1_array = self._load_and_preprocess_image_for_orb(img1)
        img2_array = self._load_and_preprocess_image_for_orb(img2)

        # Detect and compute features for both images
        keypoints1, descriptors1 = self.feature_detector.detect_and_compute(img1_array)
        keypoints2, descriptors2 = self.feature_detector.detect_and_compute(img2_array)

        # Check if descriptors were found
        if descriptors1 is None or descriptors2 is None:
            return {
                'similarity_score': 0.0,
                'method': 'orb',
                'normalized': normalize,
                'keypoints_1': len(keypoints1),
                'keypoints_2': len(keypoints2),
                'descriptors_1': None,
                'descriptors_2': None,
                'matches_count': 0,
                'match_ratio': 0.0,
                'error': 'No descriptors found in one or both images'
            }

        # Find matches
        matches = self.bf_matcher.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate match ratio (number of good matches / total possible matches)
        total_possible_matches = min(len(descriptors1), len(descriptors2))
        match_ratio = len(matches) / total_possible_matches if total_possible_matches > 0 else 0.0

        # Convert match ratio to similarity score (0-1 range)
        similarity_score = min(match_ratio, 1.0)

        return {
            'similarity_score': similarity_score,
            'method': 'orb',
            'normalized': normalize,
            'keypoints_1': len(keypoints1),
            'keypoints_2': len(keypoints2),
            'descriptors_1': descriptors1.shape if descriptors1 is not None else None,
            'descriptors_2': descriptors2.shape if descriptors2 is not None else None,
            'matches_count': len(matches),
            'match_ratio': match_ratio,
            'total_possible_matches': total_possible_matches
        }

    def _load_and_preprocess_image(self, image_input: Union[np.ndarray, str]) -> np.ndarray:
        """
        Load and preprocess an image for feature extraction.

        Args:
            image_input (Union[np.ndarray, str]): Image as numpy array or file path

        Returns:
            np.ndarray: Preprocessed image array
        """
        return ImagePreprocessor.load_and_preprocess_for_feature_extraction(image_input)

    def _load_and_preprocess_image_for_orb(self, image_input: Union[np.ndarray, str]) -> np.ndarray:
        """
        Load and preprocess an image specifically for ORB feature detection.
        ORB requires 3D uint8 arrays, not 4D float32 tensors.

        Args:
            image_input (Union[np.ndarray, str]): Image as numpy array or file path

        Returns:
            np.ndarray: Preprocessed image array suitable for ORB (3D, uint8)
        """
        if isinstance(image_input, str):
            # Load image from file path
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            # Convert BGR to RGB for consistency
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Handle numpy array input
            img = image_input.copy()
            
            # If it's a 4D tensor (from deep learning preprocessing), remove batch dimension
            if len(img.shape) == 4:
                img = img.squeeze(axis=0)
            
            # If it's float32 normalized [0,1], convert back to uint8 [0,255]
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            
            # Ensure it's 3D
            if len(img.shape) != 3:
                raise ValueError(f"Expected 3D image array, got shape: {img.shape}")
        
        # Resize to a reasonable size for ORB (not too small, not too large)
        target_size = (256, 256)  # Good balance for ORB
        img = cv2.resize(img, target_size)
        
        return img

    def _extract_features(self, img_array: np.ndarray, method: str) -> np.ndarray:
        """
        Extract features from image using specified method.

        Args:
            img_array (np.ndarray): Preprocessed image array
            method (str): Feature extraction method ('vgg16', 'vit', 'dinov2')

        Returns:
            np.ndarray: Extracted features
        """
        if method == 'vgg16':
            features = self.feature_extractor.extract_features(img_array)
            # Flatten features for comparison
            return features.flatten()

        elif method == 'vit':
            outputs = self.feature_extractor.extract_features_vit(img_array)
            # Use the last hidden state as features
            features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return features

        elif method == 'dinov2':
            outputs = self.feature_extractor.extract_features_dinov2(img_array)
            # Use the last hidden state as features
            features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return features

        else:
            raise ValueError(f"Unsupported method: {method}. Use 'vgg16', 'vit', or 'dinov2'")

    def _calculate_similarity(self,
                              features1: np.ndarray,
                              features2: np.ndarray,
                              normalize: bool = True) -> float:
        """
        Calculate cosine similarity between two feature vectors.

        Args:
            features1 (np.ndarray): First feature vector
            features2 (np.ndarray): Second feature vector
            normalize (bool): Whether to normalize features before comparison

        Returns:
            float: Cosine similarity score between 0 and 1
        """
        # Ensure features are 1D
        features1 = features1.flatten()
        features2 = features2.flatten()

        # Normalize if requested
        if normalize:
            features1 = features1 / (np.linalg.norm(features1) + 1e-8)
            features2 = features2 / (np.linalg.norm(features2) + 1e-8)

        # Calculate cosine similarity
        similarity = cosine_similarity(
            features1.reshape(1, -1),
            features2.reshape(1, -1)
        )[0, 0]

        return float(similarity)
