# CUDA-enabled image exposing the image-to-3D Flask server
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Common runtime flags and server defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IMAGE_TO_3D_SERVER_HOST=0.0.0.0 \
    IMAGE_TO_3D_SERVER_PORT=8080 \
    TEXTURE_SERVER_HOST=0.0.0.0 \
    TEXTURE_SERVER_PORT=8080 \
    ROSE_WSGI_APP=rose.exec.image_to_3d_server:app \
    CMAKE_PREFIX_PATH=/opt/torch/cmake:/opt/pybind11/cmake:${CMAKE_PREFIX_PATH} \
    Torch_DIR=/opt/torch/cmake/Torch \
    PIP_NO_BUILD_ISOLATION=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=8.0 \
    PYTORCH_NVCC_ARCH_LIST=8.0 \
    CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=80

# System dependencies and Python toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        cmake \
        git \
        libgl1 \
        libglib2.0-0 \
        wget \
        unzip \
        libsm6 \
        libxext6 \
        libxrender1 \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libopenexr-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libavutil-dev \
        libx264-dev \
        libssl-dev \
        libcurl4-openssl-dev \
        libyaml-dev \
        rsync \
        ffmpeg \
        libomp-dev \
        libopenblas-dev \
        gfortran \
        liblzma-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.11 from deadsnakes PPA
RUN apt-get update && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        python3.11-tk && \
    rm -rf /var/lib/apt/lists/*

# Ensure python / pip commands are available
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip

WORKDIR /app

# Install core GPU libraries then project dependencies
COPY requirements_gpu.txt ./requirements_gpu.txt
RUN python -m pip install --no-cache-dir "numpy<2"
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2+cu121 torchvision==0.17.2+cu121
RUN python -m pip install --no-cache-dir scikit-build-core ninja
RUN python -m pip install --no-cache-dir --no-build-isolation -r requirements_gpu.txt
RUN python - <<'PY'
import pathlib
import torch
import pybind11

torch_share_dir = pathlib.Path(torch.__file__).resolve().parent / "share" / "cmake"
torch_dest_dir = pathlib.Path("/opt/torch/cmake")
torch_dest_dir.mkdir(parents=True, exist_ok=True)

for name in ("Torch", "Caffe2"):
    source = torch_share_dir / name
    target = torch_dest_dir / name
    if not target.exists():
        target.symlink_to(source, target_is_directory=True)

pybind_src = pathlib.Path(pybind11.get_cmake_dir())
pybind_dest = pathlib.Path("/opt/pybind11/cmake")
pybind_dest.parent.mkdir(parents=True, exist_ok=True)
if not pybind_dest.exists():
    pybind_dest.symlink_to(pybind_src, target_is_directory=True)

print(f"Linked Torch CMake directory: {torch_share_dir}")
print(f"Linked pybind11 CMake directory: {pybind_src}")
PY
RUN python -m pip install --no-cache-dir --no-build-isolation -r requirements_gpu.txt

# Build native CUDA extensions shipped with the 3DTopia package
RUN python - <<'PY'
import importlib
import pathlib
import subprocess
import sys

dva_module = importlib.import_module("dva")
root = pathlib.Path(dva_module.__file__).resolve().parents[1]
extensions = [
    root / "mvp" / "extensions" / "mvpraymarch",
    root / "mvp" / "extensions" / "utils",
]
for ext_path in extensions:
    subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(ext_path))
PY

# Copy application source and entrypoint
COPY rose ./rose
COPY docker/run_rose_app.sh /usr/local/bin/run_rose_app.sh
RUN chmod +x /usr/local/bin/run_rose_app.sh

# NVIDIA container runtime integration
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8444

ENTRYPOINT ["/usr/local/bin/run_rose_app.sh"]
