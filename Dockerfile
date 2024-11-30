# Use an official NVIDIA CUDA base image with PyTorch and Python
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables for Python
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Install system dependencies and tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501
