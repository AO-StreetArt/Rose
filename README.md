# Rose: Spatial Intelligence Library

[![Unit Tests](https://github.com/AO-StreetArt/Rose/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/AO-StreetArt/Rose/actions/workflows/python-app.yml)

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
2. Install the CPU-only PyTorch stack (recommended: within a virtual environment):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
3. Export the PyTorch CMake path so `torchmcubes` can build against Torch:
   ```bash
   export Torch_DIR="$(python3 -c 'import torch, pathlib; print(pathlib.Path(torch.utils.cmake_prefix_path))')"
   ```
   On Windows PowerShell use:
   ```powershell
   $env:Torch_DIR = python -c "import torch, pathlib; print(pathlib.Path(torch.utils.cmake_prefix_path))"
   ```
4. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Advanced 3D Model Support: Installation Instructions for TRELLIS, VGGT, and Hunyuan3D 2.0

The library supports advanced image-to-3D conversion using three state-of-the-art models: Microsoft TRELLIS, Facebook VGGT, and Tencent Hunyuan3D 2.0. These models are **optional** and require additional dependencies and setup. You only need to install the requirements for the models you plan to use.

### System Requirements (for all models)
- **OS:** Linux recommended (Windows/macOS may work for some models)
- **Python:** 3.8+ (Hunyuan3D 2.0 requires Python 3.11)
- **GPU:** NVIDIA GPU with at least 12GB VRAM (16GB+ recommended for full pipelines)
- **CUDA:** Version matching your PyTorch install (see each model's instructions)

---

### 1. Microsoft TRELLIS
- **Repo:** https://github.com/microsoft/TRELLIS
- **Installation:**
  ```bash
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
  cd TRELLIS
  . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
  # This creates a conda environment and installs all dependencies
  ```
- **Notes:**
  - Requires conda and CUDA 11.8 or 12.2
  - See the TRELLIS repo for troubleshooting and advanced options

---

### 2. Facebook VGGT
- **Repo:** https://github.com/facebookresearch/vggt
- **Installation:**
  ```bash
  git clone https://github.com/facebookresearch/vggt.git
  cd vggt
  pip install -r requirements.txt
  pip install -r requirements_demo.txt
  ```
- **Notes:**
  - Requires PyTorch, Pillow, numpy, and other dependencies (handled by requirements.txt)
  - CUDA version must match your PyTorch install
  - See the VGGT repo for details and troubleshooting

---

### 3. Tencent Hunyuan3D 2.0
- **Repo:** https://github.com/Tencent/Hunyuan3D-2
- **Installation:**
  ```bash
  git clone https://github.com/Tencent/Hunyuan3D-2.git
  cd Hunyuan3D-2
  pip install -r requirements.txt
  pip install -e .
  # For texture generation, also run:
  cd hy3dgen/texgen/custom_rasterizer
  python3 setup.py install
  cd ../../..
  cd hy3dgen/texgen/differentiable_renderer
  python3 setup.py install
  ```
- **Notes:**
  - Requires Python 3.11 and CUDA 12.4
  - See the Hunyuan3D-2 repo for troubleshooting and advanced usage

---

**Tip:**
- Each model will download its own pre-trained weights from Hugging Face or the official repo on first use.
- For best results, use a dedicated virtual environment for each model.
- These models are large and require significant GPU memory and disk space.

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

## Texture Generation (SDXL + ControlNet)
- Models (Hugging Face):
  - SDXL Base: `stabilityai/stable-diffusion-xl-base-1.0`
  - ControlNet (default): `diffusers/controlnet-lineart-sdxl-1.0`
  - Alternatives: `diffusers/controlnet-canny-sdxl-1.0`, `diffusers/controlnet-depth-sdxl-1.0`
- Install extras:
  - `pip install diffusers transformers accelerate safetensors huggingface_hub scipy`
- Usage example:
  ```python
  from PIL import Image
  from rose.processing.texture_generator import TextureGenerator

  # Load a UV-layout/edge/AO image in UV space as control
  control_img = Image.open("uv_layout_overlay.png")

  tg = TextureGenerator(
      base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
      controlnet_id="diffusers/controlnet-lineart-sdxl-1.0",
  )

  result = tg.generate_textures(
      control_image=control_img,
      prompt="brushed stainless steel with subtle wear and fine grain",
      negative_prompt="blurry, low detail",
      seed=42,
      steps=25,
      guidance=5.0,
      conditioning_scale=1.0,
  )

  result["albedo"].save("albedo.png")
  result["normal"].save("normal.png")
  result["roughness"].save("roughness.png")
  ```
Notes:
- Accept the SDXL license on Hugging Face and log in (`huggingface-cli login`) if required.
- For AMD/ROCm, install PyTorch ROCm wheels; diffusers uses the GPU if available.

## Running Tests
To run all unit tests:
```bash
PYTHONPATH=. pytest tests
```

This will also generate output images in the `tests/` folder for manual review.

## License
MIT License © 2024 Alex Barry
