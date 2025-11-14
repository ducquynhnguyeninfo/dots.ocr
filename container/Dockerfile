# Use NVIDIA CUDA base image with Python 3.11 (matching host CUDA 12.9)
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    python3.11-venv \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python3
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and setup files first for better caching
COPY requirements.txt setup.py ./
COPY dots_ocr/ ./dots_ocr/

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p weights/DotsOCR output data

# Download model weights (optional - can be mounted as volume)
RUN python3 tools/download_model.py || echo "Model download failed - will download at runtime"

# Expose the API port
EXPOSE 3803

# Create a startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Check if GPU is available\n\
if ! nvidia-smi > /dev/null 2>&1; then\n\
    echo "Warning: NVIDIA GPU not detected. DotsOCR requires CUDA-capable GPU."\n\
    exit 1\n\
fi\n\
\n\
echo "Starting DotsOCR API Server..."\n\
echo "GPU Memory Status:"\n\
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits\n\
\n\
# Start the API server\n\
exec python3 demo/api_server.py --host 0.0.0.0 --port 3803 --workers 1\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3803/ || exit 1

# Default command
CMD ["/app/start.sh"]