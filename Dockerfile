# Edict Veritas — RunPod Serverless GPU worker
#
# Base image: official COLMAP Docker image built with CUDA 12.x + cuDNN.
# Provides: `colmap` binary (GPU-enabled), Ubuntu 22.04, Python 3.
# See tags: https://hub.docker.com/r/colmap/colmap/tags
FROM colmap/colmap:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install pip + any runtime libs the Python packages need
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python pipeline dependencies — pin versions to avoid surprise breakage.
# pycolmap jumped from 0.6.x to 3.10.x (COLMAP versioning) with a completely
# different API; pin to 0.6.1 which matches the code in process_zone.py.
# runpod 1.x is the stable serverless SDK; 2.x changed the startup API.
RUN pip3 install --no-cache-dir \
        "runpod==1.7.9" \
        "pycolmap==0.6.1" \
        "Pillow>=10,<11" \
        "numpy<2" \
        "scipy>=1.11" \
        "trimesh>=4" \
        "requests>=2.31"

WORKDIR /app

# Copy pipeline logic shared with the local runner
COPY process_zone.py .
# Copy the RunPod handler (entry point)
COPY runpod/handler.py .

CMD ["python3", "-u", "/app/handler.py"]
