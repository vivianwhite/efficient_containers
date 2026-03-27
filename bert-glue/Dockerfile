# Use a lightweight PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies for tokenizers and git-lfs
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Install NLP-specific libraries
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    evaluate \
    scikit-learn \
    accelerate

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Environmental variable to point Hugging Face to our volume
ENV HF_HOME=/workspace/hf_cache

COPY train.py .

# Entry point
ENTRYPOINT ["python", "train.py"]
