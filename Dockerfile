# Dockerfile for Microsoft TRELLIS3D
# -----------------
# Author: Samuel Tagliabracci
# GitHub: https://github.com/SamuelTagliabracci
# Date: Dec 8 2024
# License: MIT

# Tested on Ubuntu 22.04 Host with RTX 4090
#
# Description: Dockerfile for TRELLIS (https://github.com/JeffreyXiang/TRELLIS)
# This container provides a complete environment for running TRELLIS with CUDA support.
# Optimized for RTX 4090 and similar GPUs.

# *** IMPORTANT (SET YOUR CUDA ARCHITECTURE, MINE WAS 8.9+PTX) ***

# CUDA Architecture Guide:
# ----------------------
# 6.0, 6.1: Pascal (GTX 1080, 1070, 1060)
# 7.0: Volta (V100)
# 7.2: Turing (T4)
# 7.5: Turing (RTX 2080 Ti, 2080, 2070, 2060)
# 8.0: Ampere (A100)
# 8.6: Ampere Consumer (RTX 3090, 3080, 3070, 3060)
# 8.9: Ada Lovelace (RTX 4090, 4080, 4070)

# Usage Instructions:
# 1. Build: docker build -t samueltagliabracci/trellis3d-ubuntu22:latest .
# 2. Run: docker run -it --rm -p 7860:7860 --gpus all samueltagliabracci/trellis3d-ubuntu22:latest

# Note: It took 30 minutes to build the docker image on an i9-12900K, 64GB RAM, 24GB RTX 4090 (particularily the flash-attn, feel free to increase the MAX_JOBS - I put it there to prevent hanging)
# Note: It will take 10-20 minutes the first time you run to download all the model weights and dependencies

# Note: You can add share=True in /app/TRELLIS/app.py ie demo.launch(server_name="0.0.0.0", share=True) to enable external access to the web interface via gradio.live

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    cmake \
    libgl1-mesa-glx \
    xvfb \
    libxml2-dev \
    zlib1g-dev \
    ninja-build \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*
    
# Copy and install requirements
RUN git clone https://github.com/jasonxiaojx-3d/TRELLIS-live.git 

WORKDIR /TRELLIS-live

RUN . setup.sh
# # Copy your handler code
# COPY src/handler.py .

# Command to run when the container starts
# CMD ["python", "-u", "/handler.py"]