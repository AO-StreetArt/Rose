#!/usr/bin/env python3
"""
Example script demonstrating image comparison using cosine similarity.
This script shows how to use the ImageComparator class to compare images
using different feature extraction methods.
"""

import numpy as np
import cv2
import os
from pathlib import Path

# Add the parent directory to the path to import from rose
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rose.processing.image_comparator import ImageComparator


def create_test_images():
    """Create test images for demonstration."""
    test_dir = Path("experiments/test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create a red square
    red_square = np.zeros((200, 200, 3), dtype=np.uint8)
    red_square[:, :] = [255, 0, 0]
    cv2.imwrite(str(test_dir / "red_square.png"), red_square)
    
    # Create a blue square
    blue_square = np.zeros((200, 200, 3), dtype=np.uint8)
    blue_square[:, :] = [0, 0, 255]
    cv2.imwrite(str(test_dir / "blue_square.png"), blue_square)
    
    # Create a red circle
    red_circle = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(red_circle, (100, 100), 80, (255, 0, 0), -1)
    cv2.imwrite(str(test_dir / "red_circle.png"), red_circle)
    
    # Create a slightly different red square
    red_square_variant = np.zeros((200, 200, 3), dtype=np.uint8)
    red_square_variant[:, :] = [240, 10, 10]  # Slightly different red
    cv2.imwrite(str(test_dir / "red_square_variant.png"), red_square_variant)
    
    return test_dir


def demonstrate_basic_comparison():
    """Demonstrate basic image comparison between two images."""
    print("=== Basic Image Comparison ===")
    
    test_dir = create_test_images()
    
    # Initialize the comparator
    comparator = ImageComparator()
    
    # Compare red square with blue square (should be different)
    result = comparator.compare_images(
        str(test_dir / "red_square.png"),
        str(test_dir / "blue_square.png"),
        method='vgg16'
    )
    
    print(f"Red square vs Blue square similarity: {result['similarity_score']:.4f}")
    print(f"Method used: {result['method']}")
    print(f"Features normalized: {result['normalized']}")
    print()
    
    # Compare red square with red circle (should be more similar)
    result = comparator.compare_images(
        str(test_dir / "red_square.png"),
        str(test_dir / "red_circle.png"),
        method='vgg16'
    )
    
    print(f"Red square vs Red circle similarity: {result['similarity_score']:.4f}")
    print()
    
    # Compare red square with red square variant (should be very similar)
    result = comparator.compare_images(
        str(test_dir / "red_square.png"),
        str(test_dir / "red_square_variant.png"),
        method='vgg16'
    )
    
    print(f"Red square vs Red square variant similarity: {result['similarity_score']:.4f}")
    print()


def demonstrate_multiple_comparison():
    """Demonstrate comparing multiple images at once."""
    print("=== Multiple Image Comparison ===")
    
    test_dir = create_test_images()
    
    # Initialize the comparator
    comparator = ImageComparator()
    
    # List of images to compare
    images = [
        str(test_dir / "red_square.png"),
        str(test_dir / "blue_square.png"),
        str(test_dir / "red_circle.png"),
        str(test_dir / "red_square_variant.png")
    ]
    
    # Compare all images pairwise
    result = comparator.compare_multiple_images(images, method='vgg16')
    
    print(f"Similarity matrix shape: {result['similarity_matrix'].shape}")
    print(f"Number of images: {result['num_images']}")
    print("\nSimilarity Matrix:")
    print(result['similarity_matrix'])
    print()


def demonstrate_similarity_search():
    """Demonstrate finding most similar images."""
    print("=== Similarity Search ===")
    
    test_dir = create_test_images()
    
    # Initialize the comparator
    comparator = ImageComparator()
    
    # Query image
    query_image = str(test_dir / "red_square.png")
    
    # Candidate images
    candidate_images = [
        str(test_dir / "blue_square.png"),
        str(test_dir / "red_circle.png"),
        str(test_dir / "red_square_variant.png")
    ]
    
    # Find most similar images
    similar_images = comparator.find_most_similar(
        query_image,
        candidate_images,
        method='vgg16',
        top_k=2
    )
    
    print(f"Query image: {query_image}")
    print("\nTop 2 most similar images:")
    for i, result in enumerate(similar_images):
        print(f"{i+1}. Image {result['index']}: {result['similarity_score']:.4f}")
    print()


def demonstrate_different_methods():
    """Demonstrate comparison using different feature extraction methods."""
    print("=== Different Feature Extraction Methods ===")
    
    test_dir = create_test_images()
    
    # Initialize the comparator
    comparator = ImageComparator()
    
    # Test different methods
    methods = ['vgg16']  # Add 'vit' and 'dinov2' if you have the models
    
    for method in methods:
        print(f"\nUsing {method.upper()} method:")
        
        result = comparator.compare_images(
            str(test_dir / "red_square.png"),
            str(test_dir / "red_square_variant.png"),
            method=method
        )
        
        print(f"Similarity score: {result['similarity_score']:.4f}")
        print(f"Feature shapes: {result['feature_shape_1']} vs {result['feature_shape_2']}")


def demonstrate_normalization_effect():
    """Demonstrate the effect of feature normalization."""
    print("=== Normalization Effect ===")
    
    test_dir = create_test_images()
    
    # Initialize the comparator
    comparator = ImageComparator()
    
    # Compare with normalization
    result_normalized = comparator.compare_images(
        str(test_dir / "red_square.png"),
        str(test_dir / "red_square_variant.png"),
        method='vgg16',
        normalize=True
    )
    
    # Compare without normalization
    result_not_normalized = comparator.compare_images(
        str(test_dir / "red_square.png"),
        str(test_dir / "red_square_variant.png"),
        method='vgg16',
        normalize=False
    )
    
    print(f"With normalization: {result_normalized['similarity_score']:.4f}")
    print(f"Without normalization: {result_not_normalized['similarity_score']:.4f}")
    print()


def cleanup_test_images():
    """Clean up test images."""
    test_dir = Path("experiments/test_images")
    if test_dir.exists():
        for file in test_dir.glob("*.png"):
            file.unlink()
        test_dir.rmdir()


if __name__ == "__main__":
    try:
        print("Image Comparison Example")
        print("=" * 50)
        
        # Run demonstrations
        demonstrate_basic_comparison()
        demonstrate_multiple_comparison()
        demonstrate_similarity_search()
        demonstrate_different_methods()
        demonstrate_normalization_effect()
        
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cleanup_test_images() 