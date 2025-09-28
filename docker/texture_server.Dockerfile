# Docker image for the texture generation HTTPS server
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TEXTURE_SERVER_HOST=0.0.0.0 \
    TEXTURE_SERVER_PORT=8443 \
    CMAKE_PREFIX_PATH=/usr/local/lib/python3.10/site-packages/torch/share/cmake:${CMAKE_PREFIX_PATH} \
    Torch_DIR=/usr/local/lib/python3.10/site-packages/torch/share/cmake/Torch \
    PIP_NO_BUILD_ISOLATION=1

# Install system deps required by Pillow, torch, and diffusers stack
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY rose ./rose
COPY docker/run_rose_app.sh /usr/local/bin/run_rose_app.sh
RUN chmod +x /usr/local/bin/run_rose_app.sh

# Expose HTTPS port
EXPOSE 8443

# Run the WSGI app under Gunicorn (TLS handled via env configuration)
ENTRYPOINT ["/usr/local/bin/run_rose_app.sh"]
