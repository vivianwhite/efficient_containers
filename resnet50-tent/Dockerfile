# Use the stable PyTorch base
#FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
#ENV TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6 9.0 10.0 12.0"
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

WORKDIR /app

# TTA experiments often require 'git' to pull specific TENT or Robustness 
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Blackwell fix: install PyTorch CUDA 12.8 wheels
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --break-system-packages

# Install specific TTA libraries
RUN pip3 install --no-cache-dir timm scipy pandas matplotlib robustbench codecarbon numpy \
    --break-system-packages
RUN pip3 install --no-cache-dir git+https://github.com/fra31/auto-attack.git --break-system-packages

COPY . .

ENTRYPOINT ["python3", "tta_experiment.py"]
