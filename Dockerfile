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

# Set default CUDA architecture to 8.0 (A100)
ENV TORCH_CUDA_ARCH_LIST="8.0"

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Clone TRELLIS repository
RUN git clone --recurse-submodules https://github.com/jasonxiaojx-3d/TRELLIS-live.git
WORKDIR /app/TRELLIS

# Install Miniconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# Initialize conda for shell interaction
RUN conda init bash
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH


# Run setup script with required flags and demo dependencies
RUN chmod +x setup.sh && \
    ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast && \
    ./setup.sh --demo

# Set conda environment activation in PATH
ENV PATH=/opt/conda/envs/trellis/bin:$PATH

# Install PyTorch and basic dependencies
RUN conda run -n trellis pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    conda run -n trellis pip install gradio==4.44.1 && \
    conda run -n trellis pip install gradio-litmodel3d==0.0.1 && \
    conda run -n trellis pip install imageio imageio-ffmpeg && \
    conda run -n trellis pip install easydict && \
    conda run -n trellis pip install rembg && \
    conda run -n trellis pip install onnxruntime-gpu && \
    conda run -n trellis pip install plyfile && \
    conda run -n trellis pip install trimesh[easy] && \
    conda run -n trellis pip install pyvista && \
    conda run -n trellis pip install pymeshfix && \
    conda run -n trellis pip install python-igraph && \
    conda run -n trellis pip install safetensors && \
    conda run -n trellis pip install ninja && \
    conda run -n trellis pip install spconv-cu118

# Install Kaolin with FlexiCubes support
RUN cd /app && \
    conda run -n trellis pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
    #conda run -n trellis pip install kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
    

# Install nvdiffrast
RUN cd /app && \
    git clone https://github.com/NVlabs/nvdiffrast.git && \
    cd nvdiffrast && \
    conda run -n trellis pip install .

# Install CUDA development tools and required packages
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-8 \
    ninja-build \
    build-essential \
    libglm-dev \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CUDA_INCLUDE_PATH=${CUDA_HOME}/include
ENV CUDA_LIB_PATH=${CUDA_HOME}/lib64
ENV CPATH=${CUDA_HOME}/include

# Completely remove and reinstall Ninja
RUN apt-get update && \
    apt-get remove -y ninja-build && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda run -n trellis pip uninstall -y ninja && \
    conda run -n trellis pip install --no-cache-dir ninja==1.10.2.3

# Verify Ninja installation
RUN ninja --version

# Install GLM library first
RUN apt-get update && \
    apt-get install -y libglm-dev && \
    rm -rf /var/lib/apt/lists/*

# Install the correct version of diff-gaussian-rasterization from mip-splatting
RUN rm -rf /app/mip-splatting && \
    conda run -n trellis bash -c "\
    cd /app && \
    git clone https://github.com/autonomousvision/mip-splatting.git && \
    cd mip-splatting/submodules/diff-gaussian-rasterization && \
    pip install -e ."

# Install xformers with correct version
RUN conda run -n trellis pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118

# Install flash-attn with specific version
RUN conda run -n trellis pip uninstall -y flash-attn && \
    conda run -n trellis MAX_JOBS=8 pip install flash-attn==2.3.6 --no-build-isolation

# Install DISO with CUDA paths
RUN cd /app && \
    git clone https://github.com/SarahWeiii/diso.git && \
    cd diso && \
    CUDA_HOME=/usr/local/cuda \
    CUDA_INCLUDE_PATH=/usr/local/cuda/include \
    CUDA_LIB_PATH=/usr/local/cuda/lib64 \
    CPATH=/usr/local/cuda/include \
    conda run -n trellis pip install -v .

# Install utils3d from source
RUN conda run -n trellis pip install git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d

# Make flexicubes a proper Python package
#RUN touch /app/TRELLIS/trellis/representations/mesh/flexicubes/__init__.py

# Set environment variables
ENV PYTHONPATH=/app/TRELLIS
ENV DISPLAY=:99

# First, we need to modify app.py to use the correct launch parameters
RUN sed -i 's/demo.launch()/demo.launch(server_name="0.0.0.0")/' /app/TRELLIS/app.py

# Enable this if you want to share the web interface externally (or change /app/TRELLIS/app.py manually)
#RUN sed -i 's/demo.launch(server_name="0.0.0.0")/demo.launch(server_name="0.0.0.0", share=True)/' /app/TRELLIS/app.py

# Launch with network access
CMD ["/bin/bash", "-c", "\
    source /opt/conda/etc/profile.d/conda.sh && \
    conda activate trellis && \
    exec python3 -i app.py \
"]
