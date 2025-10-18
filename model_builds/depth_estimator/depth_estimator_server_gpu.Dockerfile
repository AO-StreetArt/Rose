# CUDA-enabled image exposing the DepthEstimator Flask server
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime defaults tailored for SageMaker-style deployments
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEPTH_SERVER_HOST=0.0.0.0 \
    DEPTH_SERVER_PORT=8080 \
    DEPTH_SERVER_APP=depth_estimator_server:app \
    DEPTH_SERVER_DEFAULT_ESTIMATOR=dpt \
    DEPTH_SERVER_LOCAL_FILES_ONLY=1 \
    DEPTH_SERVER_HF_CACHE_DIR=/app/.cache/huggingface \
    DEPTH_SERVER_DPT_MODEL_ID=/app/models/dpt-large \
    DEPTH_SERVER_ZOE_MODEL_ID=/app/models/zoedepth-nyu-kitti \
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
        git \
        libgl1 \
        libglib2.0-0 \
        wget \
        unzip \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install project dependencies
COPY requirements_depth_estimator_gpu.txt ./requirements_depth_estimator_gpu.txt
RUN python -m pip install --no-cache-dir --no-build-isolation -r requirements_depth_estimator_gpu.txt

RUN mkdir -p /app/models/dpt-large /app/models/zoedepth-nyu-kitti /app/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface

COPY download_models.py ./download_models.py
RUN python download_models.py

# Ensure required artifacts exist so runtime never falls back to network access.
RUN test -f /app/models/dpt-large/preprocessor_config.json \
    && test -f /app/models/dpt-large/config.json \
    && test -f /app/models/zoedepth-nyu-kitti/config.json

# Copy application source and entrypoint
COPY depth_estimator.py depth_estimator_server.py ./ 
COPY run_rose_app.sh /usr/local/bin/run_depth_estimator_server.sh
RUN chmod +x /usr/local/bin/run_depth_estimator_server.sh

# NVIDIA container runtime integration
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8080

ENV PYTHONPATH=/app

ENTRYPOINT ["/usr/local/bin/run_depth_estimator_server.sh"]
