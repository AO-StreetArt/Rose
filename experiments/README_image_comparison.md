# Image Comparison with Cosine Similarity

This module provides image comparison functionality using cosine similarity on extracted features from various pre-trained models.

## Overview

The `ImageComparator` class in `rose/processing/image_comparator.py` allows you to compare images using cosine similarity on features extracted by different deep learning models:

- **VGG16**: Traditional CNN-based feature extraction
- **ViT (Vision Transformer)**: Transformer-based feature extraction  
- **DINOv2**: Self-supervised learning-based feature extraction

## Features

- **Multiple Feature Extraction Methods**: Support for VGG16, ViT, and DINOv2
- **Flexible Input**: Accept images as file paths or numpy arrays
- **Batch Processing**: Compare multiple images efficiently
- **Similarity Search**: Find most similar images in a collection
- **Normalization Options**: Control whether features are normalized before comparison
- **Comprehensive Results**: Detailed metadata about comparisons

## Installation

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

The module requires:
- `scikit-learn>=1.0.0` (for cosine similarity)
- `tensorflow>=2.0.0` (for VGG16)
- `torch>=1.9.0` (for ViT and DINOv2)
- `transformers>=4.21.0` (for ViT and DINOv2)
- `opencv-python>=4.5.0` (for image processing)
- `numpy>=1.19.0` (for numerical operations)

## Usage

### Basic Image Comparison

```python
from rose.processing.image_comparator import ImageComparator

# Initialize the comparator
comparator = ImageComparator()

# Compare two images
result = comparator.compare_images(
    "path/to/image1.jpg",
    "path/to/image2.jpg",
    method='vgg16'
)

print(f"Similarity score: {result['similarity_score']:.4f}")
print(f"Method used: {result['method']}")
```

### Multiple Image Comparison

```python
# Compare multiple images pairwise
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
result = comparator.compare_multiple_images(images, method='vgg16')

print(f"Similarity matrix shape: {result['similarity_matrix'].shape}")
print(f"Similarity matrix:\n{result['similarity_matrix']}")
```

### Similarity Search

```python
# Find most similar images to a query
query_image = "query.jpg"
candidate_images = ["candidate1.jpg", "candidate2.jpg", "candidate3.jpg"]

similar_images = comparator.find_most_similar(
    query_image,
    candidate_images,
    method='vgg16',
    top_k=3
)

for i, result in enumerate(similar_images):
    print(f"{i+1}. Image {result['index']}: {result['similarity_score']:.4f}")
```

### Using Different Feature Extraction Methods

```python
# Compare using ViT features
result_vit = comparator.compare_images(
    "image1.jpg", 
    "image2.jpg", 
    method='vit'
)

# Compare using DINOv2 features
result_dinov2 = comparator.compare_images(
    "image1.jpg", 
    "image2.jpg", 
    method='dinov2'
)
```

### Controlling Normalization

```python
# Compare without feature normalization
result = comparator.compare_images(
    "image1.jpg",
    "image2.jpg", 
    method='vgg16',
    normalize=False
)
```

## API Reference

### ImageComparator Class

#### Constructor
```python
ImageComparator(feature_extractor=None)
```
- `feature_extractor`: Optional FeatureExtractor instance. If None, creates a new one.

#### Methods

##### compare_images()
```python
compare_images(img1, img2, method='vgg16', normalize=True)
```
Compare two images using cosine similarity.

**Parameters:**
- `img1`: First image (file path or numpy array)
- `img2`: Second image (file path or numpy array)
- `method`: Feature extraction method ('vgg16', 'vit', 'dinov2')
- `normalize`: Whether to normalize features before comparison

**Returns:**
- Dictionary with similarity score and metadata

##### compare_multiple_images()
```python
compare_multiple_images(images, method='vgg16', normalize=True)
```
Compare multiple images pairwise.

**Parameters:**
- `images`: List of images (file paths or numpy arrays)
- `method`: Feature extraction method
- `normalize`: Whether to normalize features

**Returns:**
- Dictionary with similarity matrix and metadata

##### find_most_similar()
```python
find_most_similar(query_image, candidate_images, method='vgg16', normalize=True, top_k=5)
```
Find most similar images to a query image.

**Parameters:**
- `query_image`: Query image (file path or numpy array)
- `candidate_images`: List of candidate images
- `method`: Feature extraction method
- `normalize`: Whether to normalize features
- `top_k`: Number of top similar images to return

**Returns:**
- List of dictionaries with similarity scores and indices

## Example Script

Run the example script to see the functionality in action:

```bash
python3 experiments/image_comparison_example.py
```

This script demonstrates:
- Basic image comparison
- Multiple image comparison
- Similarity search
- Different feature extraction methods
- Normalization effects

## Testing

Run the test suite to verify functionality:

```bash
python3 -m pytest tests/test_image_comparator.py -v
```

## Performance Considerations

- **VGG16**: Fastest, good for general-purpose comparison
- **ViT**: Moderate speed, good for complex visual patterns
- **DINOv2**: Slower but excellent for fine-grained similarity

## Use Cases

- **Image Retrieval**: Find similar images in a database
- **Duplicate Detection**: Identify duplicate or near-duplicate images
- **Content-Based Search**: Search images by visual similarity
- **Quality Assessment**: Compare images for quality differences
- **Feature Analysis**: Analyze what makes images similar or different

## Notes

- Images are automatically resized to 224x224 pixels for feature extraction
- Features are normalized by default for better comparison results
- Cosine similarity scores range from -1 to 1, where 1 indicates identical features
- The module integrates with the existing `FeatureExtractor` class for consistency 