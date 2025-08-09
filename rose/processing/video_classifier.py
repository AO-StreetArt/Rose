import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
import cv2
from transformers import AutoProcessor, AutoModel
import logging

from ..preprocessing.image_utils import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoClassifier:
    """
    Video classification using Microsoft's X-CLIP model.
    X-CLIP is a cross-modal model that can understand both video and text,
    making it suitable for zero-shot video classification tasks.
    """
    
    def __init__(self, model_name: str = "microsoft/xclip-base-patch32"):
        """
        Initialize the video classifier with X-CLIP model.
        
        Args:
            model_name (str): Name of the X-CLIP model to use.
                             Default is "microsoft/xclip-base-patch32"
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            logger.info(f"Loading X-CLIP model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"X-CLIP model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load X-CLIP model: {e}")
            raise
    
    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to extract (default: 8)
            
        Returns:
            List[np.ndarray]: List of extracted frames as numpy arrays
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to extract
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = ImagePreprocessor.convertBGRtoRGB(frame)
                    frames.append(frame_rgb)
                else:
                    logger.warning(f"Failed to read frame at index {idx}")
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError("No frames were successfully extracted from the video")
            
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Preprocess frames for the X-CLIP model.
        
        Args:
            frames (List[np.ndarray]): List of video frames as numpy arrays
            
        Returns:
            Dict[str, Any]: Preprocessed video inputs
        """
        try:
            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Process with X-CLIP processor
            inputs = self.processor(
                videos=pil_frames,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            if isinstance(inputs, dict):
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preprocessing frames: {e}")
            raise
    
    def classify_video(self, video_path: str, candidate_labels: List[str], 
                      num_frames: int = 8) -> Dict[str, Any]:
        """
        Classify a video using X-CLIP with zero-shot learning.
        
        Args:
            video_path (str): Path to the video file
            candidate_labels (List[str]): List of possible class labels
            num_frames (int): Number of frames to extract (default: 8)
            
        Returns:
            Dict[str, Any]: Classification results with scores for each label
        """
        try:
            # Extract frames
            frames = self.extract_frames(video_path, num_frames)
            
            # Preprocess frames
            video_inputs = self.preprocess_frames(frames)
            
            # Process text labels
            text_inputs = self.processor(
                text=candidate_labels,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move text inputs to device
            if isinstance(text_inputs, dict):
                for key in text_inputs:
                    if isinstance(text_inputs[key], torch.Tensor):
                        text_inputs[key] = text_inputs[key].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**video_inputs, **text_inputs)
                
                # Calculate similarity scores
                logits_per_video = outputs.logits_per_video
                probs = torch.nn.functional.softmax(logits_per_video, dim=1)
            
            # Convert to numpy and create results
            scores = probs.cpu().numpy()[0]
            results = {
                "labels": candidate_labels,
                "scores": scores.tolist(),
                "top_label": candidate_labels[np.argmax(scores)],
                "top_score": float(np.max(scores)),
                "num_frames_processed": len(frames)
            }
            
            logger.info(f"Video classification completed. Top label: {results['top_label']} "
                       f"(score: {results['top_score']:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying video: {e}")
            raise
    
    def classify_video_with_frames(self, frames: List[np.ndarray], 
                                 candidate_labels: List[str]) -> Dict[str, Any]:
        """
        Classify video frames directly without loading from file.
        
        Args:
            frames (List[np.ndarray]): List of video frames as numpy arrays
            candidate_labels (List[str]): List of possible class labels
            
        Returns:
            Dict[str, Any]: Classification results with scores for each label
        """
        try:
            # Preprocess frames
            video_inputs = self.preprocess_frames(frames)
            
            # Process text labels
            text_inputs = self.processor(
                text=candidate_labels,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move text inputs to device
            if isinstance(text_inputs, dict):
                for key in text_inputs:
                    if isinstance(text_inputs[key], torch.Tensor):
                        text_inputs[key] = text_inputs[key].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**video_inputs, **text_inputs)
                
                # Calculate similarity scores
                logits_per_video = outputs.logits_per_video
                probs = torch.nn.functional.softmax(logits_per_video, dim=1)
            
            # Convert to numpy and create results
            scores = probs.cpu().numpy()[0]
            results = {
                "labels": candidate_labels,
                "scores": scores.tolist(),
                "top_label": candidate_labels[np.argmax(scores)],
                "top_score": float(np.max(scores)),
                "num_frames_processed": len(frames)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error classifying video frames: {e}")
            raise
    
    def get_video_embeddings(self, video_path: str, num_frames: int = 8) -> np.ndarray:
        """
        Extract video embeddings using X-CLIP.
        
        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to extract (default: 8)
            
        Returns:
            np.ndarray: Video embeddings
        """
        try:
            # Extract frames
            frames = self.extract_frames(video_path, num_frames)
            
            # Preprocess frames
            video_inputs = self.preprocess_frames(frames)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**video_inputs)
                video_embeddings = outputs.video_embeds.cpu().numpy()
            
            return video_embeddings
            
        except Exception as e:
            logger.error(f"Error extracting video embeddings: {e}")
            raise
    
    def get_text_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Extract text embeddings using X-CLIP.
        
        Args:
            text (Union[str, List[str]]): Text or list of texts to embed
            
        Returns:
            np.ndarray: Text embeddings
        """
        try:
            if isinstance(text, str):
                text = [text]
            
            # Process text
            text_inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            if isinstance(text_inputs, dict):
                for key in text_inputs:
                    if isinstance(text_inputs[key], torch.Tensor):
                        text_inputs[key] = text_inputs[key].to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**text_inputs)
                text_embeddings = outputs.text_embeds.cpu().numpy()
            
            return text_embeddings
            
        except Exception as e:
            logger.error(f"Error extracting text embeddings: {e}")
            raise 