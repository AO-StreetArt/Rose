#!/usr/bin/env python3
"""
Example script demonstrating video classification using X-CLIP model.
This script shows how to use the VideoClassifier class for zero-shot video classification.
"""

# Add the project root to the path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rose.processing.video_classifier import VideoClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating video classification capabilities."""

    # Initialize the video classifier
    logger.info("Initializing VideoClassifier with X-CLIP model...")
    classifier = VideoClassifier()

    # Example 1: Define candidate labels for classification
    candidate_labels = [
        "playing sports",
        "cooking food",
        "dancing",
        "reading a book",
        "playing musical instrument",
        "walking",
        "running",
        "swimming"
    ]

    # Example 2: Classify a video file (if available)
    video_path = "path/to/your/video.mp4"  # Replace with actual video path

    if os.path.exists(video_path):
        logger.info(f"Classifying video: {video_path}")
        try:
            results = classifier.classify_video(video_path, candidate_labels, num_frames=8)

            print("\n=== Video Classification Results ===")
            print(f"Top prediction: {results['top_label']}")
            print(f"Confidence score: {results['top_score']:.4f}")
            print(f"Frames processed: {results['num_frames_processed']}")

            print("\nAll predictions:")
            for label, score in zip(results['labels'], results['scores']):
                print(f"  {label}: {score:.4f}")

        except Exception as e:
            logger.error(f"Error classifying video: {e}")
    else:
        logger.warning(f"Video file not found: {video_path}")
        logger.info("Skipping video classification example")

    # Example 3: Extract video embeddings
    if os.path.exists(video_path):
        logger.info("Extracting video embeddings...")
        try:
            video_embeddings = classifier.get_video_embeddings(video_path, num_frames=8)
            print(f"\nVideo embeddings shape: {video_embeddings.shape}")
            print(f"Video embeddings sample: {video_embeddings[0][:5]}...")  # Show first 5 values
        except Exception as e:
            logger.error(f"Error extracting video embeddings: {e}")

    # Example 4: Extract text embeddings
    logger.info("Extracting text embeddings...")
    try:
        text_embeddings = classifier.get_text_embeddings(candidate_labels)
        print(f"\nText embeddings shape: {text_embeddings.shape}")
        print(f"Number of text labels: {len(candidate_labels)}")

        # Show similarity between first two labels
        if text_embeddings.shape[0] >= 2:
            similarity = classifier.get_text_embeddings([candidate_labels[0], candidate_labels[1]])
            print(f"Similarity between '{candidate_labels[0]}' and '{candidate_labels[1]}': {similarity.shape}")

    except Exception as e:
        logger.error(f"Error extracting text embeddings: {e}")

    # Example 5: Demonstrate with custom frames (if no video file)
    if not os.path.exists(video_path):
        logger.info("Creating sample frames for demonstration...")
        import numpy as np

        # Create sample frames (simulating a video)
        sample_frames = []
        for i in range(8):
            # Create a simple pattern that changes over time
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 30) % 255  # Red channel varies
            frame[:, :, 1] = (i * 20) % 255  # Green channel varies
            frame[:, :, 2] = (i * 10) % 255  # Blue channel varies
            sample_frames.append(frame)

        logger.info("Classifying sample frames...")
        try:
            results = classifier.classify_video_with_frames(sample_frames, candidate_labels[:4])

            print("\n=== Sample Frames Classification Results ===")
            print(f"Top prediction: {results['top_label']}")
            print(f"Confidence score: {results['top_score']:.4f}")
            print(f"Frames processed: {results['num_frames_processed']}")

            print("\nAll predictions:")
            for label, score in zip(results['labels'], results['scores']):
                print(f"  {label}: {score:.4f}")

        except Exception as e:
            logger.error(f"Error classifying sample frames: {e}")

    logger.info("Video classification example completed!")


if __name__ == "__main__":
    main()