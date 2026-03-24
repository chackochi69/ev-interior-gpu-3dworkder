# Edict Veritas — RunPod Serverless GPU worker
#
# Base image: official COLMAP Docker image built with CUDA 12.x + cuDNN.
# Provides: `colmap` binary (GPU-enabled), Ubuntu 22.04, Python 3.
# See tags: https://hub.docker.com/r/colmap/colmap/tags
FROM colmap/colmap:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        "runpod>=1.7" pycolmap Pillow numpy trimesh requests

WORKDIR /app
COPY process_zone.py .
COPY handler.py .

CMD ["python3", "-u", "/app/handler.py"]
