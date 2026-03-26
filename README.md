# efficient_containers
A container-based toolkit for reusable, efficient, and energy-aware machine learning experiments.
The repository consists of three reproducible, containerized ML experiments.

## Experiment 1: training ResNet18 on CIFAR10
### For Developers (Building from Source)
#### Use this if you are modifying the Dockerfile or environment.
```
git clone git@github.com:vivianwhite/efficient_containers.git
cd efficient_containers

### Build the GPU-optimized image
docker build -f Dockerfile.gpu -t vivwhite/ml-experiments:resnet18-cifar10-gpu .`

### Run locally with volume mounting for instant code updates
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-gpu \
  python3 train.py --batch-size 64 --epochs 1
```

### For Users (Pulling the Pre-built Image)
#### Use this to run experiments directly on the local GPU.
```
# Pull the latest image
docker pull vivwhite/ml-experiments:resnet18-cifar10-gpu
# Run the training experiment
# Note: Results (emissions.csv) will be saved to your current directory
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-gpu \
  python3 train.py --batch-size 64 --epochs 1
```
