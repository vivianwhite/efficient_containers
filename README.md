# efficient_containers
A container-based toolkit for reusable, efficient, and energy-aware machine learning experiments.
The repository consists of three reproducible, containerized ML experiments.

## Experiment 1: training ResNet18 on CIFAR10
### For Developers (Building from Source)
#### Setup
```
git clone git@github.com:vivianwhite/efficient_containers.git
cd efficient_containers
```
#### Use this if you are modifying the Dockerfile or environment for use on a GPU.
```
### Build the GPU  image
docker build -f Dockerfile.gpu -t vivwhite/ml-experiments:resnet18-cifar10-gpu .

### Run locally with volume mounting for instant code updates
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-gpu
```
#### Use this if you are modifying the Dockerfile or environment for use on a CPU.
```
### Build the CPU  image
docker build -f Dockerfile.cpu -t vivwhite/ml-experiments:resnet18-cifar10-cpu .

### Run locally with volume mounting for instant code updates
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-cpu
```
#### Use this if you are building a SIF file to run Apptainer on an HPC.
```
# 1. Set scratch paths to avoid 'No space left' errors during build
export APPTAINER_TMPDIR=$SCRATCH/app_tmp
export APPTAINER_CACHEDIR=$SCRATCH/app_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# 2. Build the SIF (Singularity Image File) from Docker Hub
apptainer build resnet_gpu.sif docker://vivwhite/ml-experiments:resnet18-cifar10-gpu
```

### For Users (Pulling the Pre-built Image)
#### Use this to run experiments directly on a local GPU.
```
# Pull the GPU image
docker pull vivwhite/ml-experiments:resnet18-cifar10-gpu
# Run the training experiment
# Note: Results (emissions.csv) will be saved to your current directory
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-gpu
```
#### Use this to run Apptainer experiments directly on an HPC GPU.
```
# Run with --nv to enable NVIDIA GPU pass-through
# Note: Results (emissions.csv) will be saved to your current directory
apptainer run --nv resnet_gpu.sif --batch-size 64 --epochs 1
```

#### Use this to run experiments directly on a local CPU.
```
# Pull the CPU image
docker pull vivwhite/ml-experiments:resnet18-cifar10-cpu

# Run the training experiment (No --gpus flag needed)
# Note: Results (emissions.csv) will be saved to your current directory
docker run --rm \
  -v $(pwd):/app -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-cpu
```


## Experiment 2: finetuning Bert on GLUE SST2
### For Developers (Building from Source)
```
### Build the GPU  image
docker build -f Dockerfile -t vivwhite/ml-experiments:bert-sst2-gpu .

### Run locally with volume mounting for instant code updates
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  -v hf_master_cache:/app/hf_cache \
  -e HF_HOME=/app/hf_cache \
  vivwhite/ml-experiments:bert-sst2-gpu
```

### For Users (Pulling the Pre-built Image)
#### Use this to run experiments directly on a local GPU.
```
# Pull the GPU image
docker pull vivwhite/ml-experiments:bert-sst2-gpu
# Run the training experiment
# Note: Results (emissions.csv) will be saved to your current directory
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  -v hf_master_cache: /app/hf_cache
  -e HF_HOME=/workspace/hf_cache
  vivwhite/ml-experiments:bert-sst2-gpu
```
