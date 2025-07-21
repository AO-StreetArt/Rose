# Rose: Spatial Intelligence Library

## What is Spatial Intelligence?
Spatial intelligence, or visuo-spatial ability, is the ability to generate, retain, retrieve, and transform well-structured visual images. It enables us to perceive, hold, manipulate, and problem-solve using visual information—such as assembling puzzles or finding our way in the world.

Rose is a Python library designed to explore and implement the foundations of spatial intelligence in machines. It answers questions like:
- Where am I?
- Where are the things around me?
- Are they moving? How fast and in what directions?
- What are the things around me?
- What might show up around me soon?
- What could exist around me?

## Core Concepts
Rose processes a series of images to:
- Detect and classify elements in each image (using pre-trained models like VGG16 and ResNet)
- Estimate the depth of elements (using models like Intel/dpt-large)
- Track movement and velocity of elements across frames
- Store detected features and their properties in a structured memory system (short-term and long-term)

## Project Structure
- `rose/` — Main library code
  - `processing/` — Feature extraction, depth estimation, tracking
  - `preprocessing/` — Image loading, conversion, and normalization utilities
  - `storage/` — Memory management for detected elements
- `tests/` — Unit tests and test images
- `experiments/` — Experimental scripts and data

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd spatial_intelligence
   ```
2. Install dependencies (recommended: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   pip install pillow tensorflow torch transformers
   ```

## Downloading HED Model Files

To use the Holistically-Nested Edge Detection (HED) features, you need to download the pre-trained model files:

1. Download the following files from the [Ashishkumar-hub/HED GitHub repository](https://github.com/Ashishkumar-hub/HED/tree/main):
   - `deploy.prototxt`
   - `hed_pretrained_bsds.caffemodel`
2. Place both files in a directory named `models/` at the root of this project (the folder is already in `.gitignore`).

Your directory structure should look like this:
```
spatial_intelligence/
  models/
    deploy.prototxt
    hed_pretrained_bsds.caffemodel
  ...
```

These files are required for running HED-based edge detection in the library.

## Running Tests
To run all unit tests:
```bash
PYTHONPATH=. pytest tests
```

This will also generate output images in the `tests/` folder for manual review.

## License
MIT License © 2024 Alex Barry 