# Use NGC's prebuilt PyTorch container with CUDA 11.8 and Python 3
FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    wget unzip git cmake build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    python3-pyqt5 pkg-config libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy cleaned requirements file (excluding torch/torchvision/torchaudio)
COPY requirements.txt /tmp/requirements.txt

# Install project-specific dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your DEVO code
COPY . /workspace
WORKDIR /workspace

# Install your DEVO package
RUN pip install .

# Default shell
CMD ["/bin/bash"]
