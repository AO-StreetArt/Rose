# Video Classification with X-CLIP

This module provides video classification capabilities using Microsoft's X-CLIP (Cross-modal Language-Image Pre-training) model. X-CLIP is a powerful cross-modal model that can understand both video and text, making it suitable for zero-shot video classification tasks.

## Features

- **Zero-shot video classification**: Classify videos without training on specific classes
- **Frame extraction**: Extract frames from video files using OpenCV
- **Cross-modal understanding**: Understand both video content and text descriptions
- **Embedding extraction**: Extract video and text embeddings for further analysis
- **Flexible input**: Support for both video files and pre-extracted frames
- **GPU acceleration**: Automatic GPU detection and utilization

## Installation

The video classifier requires the following dependencies (already included in `requirements.txt`):

```bash
pip install torch transformers opencv-python pillow numpy
```

## Usage

### Basic Video Classification

```python
from rose.processing.video_classifier import VideoClassifier

# Initialize the classifier
classifier = VideoClassifier()

# Define candidate labels
candidate_labels = [
    "playing sports",
    "cooking food",
    "dancing",
    "reading a book"
]

# Classify a video
results = classifier.classify_video("path/to/video.mp4", candidate_labels)

print(f"Top prediction: {results['top_label']}")
print(f"Confidence: {results['top_score']:.4f}")
```

### Classification with Custom Frames

```python
import numpy as np

# Create sample frames
frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]

# Classify frames directly
results = classifier.classify_video_with_frames(frames, candidate_labels)
```

### Extracting Embeddings

```python
# Extract video embeddings
video_embeddings = classifier.get_video_embeddings("path/to/video.mp4")

# Extract text embeddings
text_embeddings = classifier.get_text_embeddings(["playing sports", "cooking"])
```

## API Reference

### VideoClassifier

#### `__init__(model_name: str = "microsoft/xclip-base-patch32")`

Initialize the video classifier with X-CLIP model.

**Parameters:**
- `model_name` (str): Name of the X-CLIP model to use

#### `extract_frames(video_path: str, num_frames: int = 8) -> List[np.ndarray]`

Extract frames from a video file.

**Parameters:**
- `video_path` (str): Path to the video file
- `num_frames` (int): Number of frames to extract

**Returns:**
- `List[np.ndarray]`: List of extracted frames as numpy arrays

#### `classify_video(video_path: str, candidate_labels: List[str], num_frames: int = 8) -> Dict[str, Any]`

Classify a video using X-CLIP with zero-shot learning.

**Parameters:**
- `video_path` (str): Path to the video file
- `candidate_labels` (List[str]): List of possible class labels
- `num_frames` (int): Number of frames to extract

**Returns:**
- `Dict[str, Any]`: Classification results with scores for each label

#### `classify_video_with_frames(frames: List[np.ndarray], candidate_labels: List[str]) -> Dict[str, Any]`

Classify video frames directly without loading from file.

**Parameters:**
- `frames` (List[np.ndarray]): List of video frames as numpy arrays
- `candidate_labels` (List[str]): List of possible class labels

**Returns:**
- `Dict[str, Any]`: Classification results with scores for each label

#### `get_video_embeddings(video_path: str, num_frames: int = 8) -> np.ndarray`

Extract video embeddings using X-CLIP.

**Parameters:**
- `video_path` (str): Path to the video file
- `num_frames` (int): Number of frames to extract

**Returns:**
- `np.ndarray`: Video embeddings

#### `get_text_embeddings(text: Union[str, List[str]]) -> np.ndarray`

Extract text embeddings using X-CLIP.

**Parameters:**
- `text` (Union[str, List[str]]): Text or list of texts to embed

**Returns:**
- `np.ndarray`: Text embeddings

## Example Output

```python
# Classification results
{
    'labels': ['playing sports', 'cooking food', 'dancing', 'reading a book'],
    'scores': [0.85, 0.08, 0.05, 0.02],
    'top_label': 'playing sports',
    'top_score': 0.85,
    'num_frames_processed': 8
}
```

## Supported Video Formats

The video classifier supports all video formats supported by OpenCV, including:
- MP4
- AVI
- MOV
- MKV
- And many others

## Performance Considerations

- **GPU Usage**: The model automatically uses GPU if available for faster inference
- **Frame Count**: More frames generally provide better accuracy but increase processing time
- **Memory**: Large videos may require significant memory for frame extraction
- **Model Size**: X-CLIP models are relatively large (~1GB), so initial loading may take time

## Error Handling

The module includes comprehensive error handling for:
- Invalid video file paths
- Corrupted video files
- Memory issues
- Model loading failures
- GPU/CPU compatibility issues

## Testing

Run the tests using pytest:

```bash
pytest tests/test_video_classifier.py -v
```

## Example Script

See `experiments/video_classification_example.py` for a complete example demonstrating all features of the video classifier.

## Model Information

X-CLIP (Cross-modal Language-Image Pre-training) is a state-of-the-art model for video-text understanding. It can:

- Perform zero-shot video classification
- Understand temporal relationships in videos
- Generate cross-modal embeddings
- Handle various video lengths and qualities

The model is pre-trained on large-scale video-text datasets and can be used without fine-tuning for many downstream tasks.