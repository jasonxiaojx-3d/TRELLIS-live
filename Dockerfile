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

FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

WORKDIR /

ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="8.0"

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get install -y git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils && \
    apt-get install -y wget cmake libgl1-mesa-glx xvfb libxml2-dev libjpeg-dev zlib1g-dev ninja-build nano vim && \
    rm -rf /var/lib/apt/lists/*

# Update alternatives to use Python 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py

# Copy and install requirements
RUN git clone --recurse-submodules https://github.com/jasonxiaojx-3d/TRELLIS-live.git 
WORKDIR /TRELLIS-live

RUN pip install --no-cache-dir runpod \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    sympy==1.13.1 \
    fsspec \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 \
    tensorboard pandas lpips \
    pillow-simd \
    xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu124 \
    flash-attn \
    kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html \
    spconv-cu124 \
    gradio==4.44.1 gradio_litmodel3d==0.0.1 && \
    pip uninstall -y pillow && \
    mkdir -p /tmp/extensions && \
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast && \
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast && \
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    cp -r extensions/vox2seq /tmp/extensions/vox2seq && \
    pip install /tmp/extensions/vox2seq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# COPY rp_handler.py /TRELLIS-live

# RUN chmod +x setup.sh && ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
# Copy your handler code
RUN chmod +x rp_handler.py

# Command to run when the container starts
CMD ["python3", "-u", "rp_handler.py"]

