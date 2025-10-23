# CUDA-enabled image exposing the TripoSR Flask server
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime defaults tailored for SageMaker-style deployments
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRIPOSR_SERVER_HOST=0.0.0.0 \
    TRIPOSR_SERVER_PORT=8080 \
    TRIPOSR_SERVER_APP=triposr_server:app \
    TRIPOSR_SERVER_MODEL_ID=stabilityai/TripoSR \
    TRIPOSR_SERVER_DEVICE=cuda \
    TRIPOSR_SERVER_DEFAULT_RESPONSE=mesh_base64 \
    TRIPOSR_SERVER_INCLUDE_RAW_RESULT=0 \
    TRIPOSR_SERVER_HF_CACHE_DIR=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST=8.0 \
    PYTORCH_NVCC_ARCH_LIST=8.0

# System dependencies and Python toolchain (Ubuntu 22.04 ships Python 3.10)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        python3-distutils \
        python3-pip \
        python-is-python3 \
        build-essential \
        cmake \
        git \
        libgl1 \
        libglib2.0-0 \
        wget \
        unzip \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir "scikit-build-core>=0.7.0" "pybind11>=2.10.0"

WORKDIR /app

# Install project dependencies
COPY requirements_triposr_server_gpu.txt ./requirements_triposr_server_gpu.txt
COPY resolve_torch_cmake.py ./resolve_torch_cmake.py
RUN set -eux; \
    python -m pip install --no-cache-dir --no-build-isolation \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.2.2+cu121 \
        torchvision==0.17.2+cu121 \
        torchaudio==2.2.2+cu121; \
    eval "$(python resolve_torch_cmake.py)"; \
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"; \
    export Torch_DIR="${Torch_DIR}"; \
    python -m pip install --no-cache-dir --no-build-isolation -r requirements_triposr_server_gpu.txt

RUN mkdir -p /app/.cache/huggingface /app/models/triposr
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

# Copy application source and entrypoint script
COPY image_to_3d.py triposr_server.py ./
COPY run_triposr_server.sh /usr/local/bin/run_triposr_server.sh
RUN chmod +x /usr/local/bin/run_triposr_server.sh

# Validate imports early to fail fast during build
RUN python - <<'PY'
from image_to_3d import ImageTo3DConverter
print("ImageTo3DConverter import OK:", ImageTo3DConverter)
PY

# NVIDIA container runtime integration
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080

ENV PYTHONPATH=/app

ENTRYPOINT ["/usr/local/bin/run_triposr_server.sh"]
