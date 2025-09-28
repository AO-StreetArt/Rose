# CUDA-enabled image for the texture generation HTTPS server
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Ensure Python logs flush immediately and expose server defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TEXTURE_SERVER_HOST=0.0.0.0 \
    TEXTURE_SERVER_PORT=8443 \
    CMAKE_PREFIX_PATH=/opt/torch/cmake:${CMAKE_PREFIX_PATH} \
    Torch_DIR=/opt/torch/cmake/Torch \
    PIP_NO_BUILD_ISOLATION=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=8.0 \
    PYTORCH_NVCC_ARCH_LIST=8.0 \
    CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=80

# Install system dependencies and Python toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-distutils \
        build-essential \
        cmake \
        git \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Ensure python / pip commands are available
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip

WORKDIR /app

# Install core GPU libraries first and expose CMake metadata
COPY requirements_gpu.txt ./requirements_gpu.txt
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2+cu121 torchvision==0.17.2+cu121
RUN python - <<'PY'
import pathlib
import torch

share_dir = pathlib.Path(torch.__file__).resolve().parent / "share" / "cmake"
dest_dir = pathlib.Path("/opt/torch/cmake")
dest_dir.mkdir(parents=True, exist_ok=True)

for name in ("Torch", "Caffe2"):
    source = share_dir / name
    target = dest_dir / name
    if target.exists():
        continue
    target.symlink_to(source, target_is_directory=True)

print(f"Linked Torch CMake directory: {share_dir}")
PY
RUN python -m pip install --no-cache-dir -r requirements_gpu.txt

# Copy application source and reusable entrypoint
COPY rose ./rose
COPY docker/run_rose_app.sh /usr/local/bin/run_rose_app.sh
RUN chmod +x /usr/local/bin/run_rose_app.sh

# CUDA runtime containers expect NVIDIA runtime env vars
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8443

ENTRYPOINT ["/usr/local/bin/run_rose_app.sh"]
