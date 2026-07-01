# Use official NVIDIA CUDA runtime as parent image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install system dependencies, Python 3, and build utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as the default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Set working directory in container
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support from PyTorch index
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install remaining dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Install PennyLane-Lightning-GPU plugin for accelerated simulation
RUN pip3 install --no-cache-dir pennylane-lightning-gpu

# Copy all project source code into container
COPY . /app

# Expose server port
EXPOSE 8000

# Start FastAPI server using Uvicorn
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
