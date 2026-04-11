# efficient_containers
A container-based toolkit for reusable, efficient, and energy-aware machine learning experiments.
The repository consists of three reproducible, containerized ML experiments.

#### Setup
```
git clone git@github.com:vivianwhite/efficient_containers.git
cd efficient_containers
```

## Experiment 1: training ResNet18 on CIFAR10
`cd resnet18-cifar10 `
### For Developers (Building from Source)
##### Modify the Dockerfile or environment for use on a GPU:
`docker build -f Dockerfile.gpu -t vivwhite/ml-experiments:resnet18-cifar10-gpu .`

##### Modify the Dockerfile or environment for use on a CPU:
`docker build -f Dockerfile.cpu -t vivwhite/ml-experiments:resnet18-cifar10-cpu .`

##### Build a SIF file to run Apptainer on an HPC:
```
# 1. Set scratch paths to avoid 'No space left' errors during build
export APPTAINER_TMPDIR=$SCRATCH/app_tmp
export APPTAINER_CACHEDIR=$SCRATCH/app_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# 2. Build the SIF (Singularity Image File) from Docker Hub
apptainer build resnet_gpu.sif docker://vivwhite/ml-experiments:resnet18-cifar10-gpu
```

### For Users (Pulling the Pre-built Image)
##### Pull the GPU image:
`docker pull vivwhite/ml-experiments:resnet18-cifar10-gpu`

##### Pull the CPU image:
`docker pull vivwhite/ml-experiments:resnet18-cifar10-cpu`

### Running the experiment
##### Run the experiment on a GPU:
```
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-gpu
```
##### Run the experiment on a HPC GPU with Apptainer:
```
apptainer run --nv resnet_gpu.sif --batch-size 64 --epochs 1
```

##### Run the experiment on a CPU:
```
docker run --rm \
  -v $(pwd):/app -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-cpu
```


## Experiment 2: finetuning Bert on GLUE SST2
`cd bert-glue`
### For Developers (Building from Source)
##### Modify the Dockerfile or environment for use on a GPU:
`docker build -f Dockerfile -t vivwhite/ml-experiments:bert-sst2-gpu .`
##### Modify the Dockerfile or environment for use on a CPU:
`docker build -f Dockerfile -t vivwhite/ml-experiments:bert-sst2-cpu .`

##### Build a SIF file to run Apptainer on an HPC:
```
# 1. Set scratch paths to avoid 'No space left' errors during build
export APPTAINER_TMPDIR=$SCRATCH/app_tmp
export APPTAINER_CACHEDIR=$SCRATCH/app_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
mkdir -p $SCRATCH/hf_master_cache

# 2. Build the SIF (Singularity Image File) from Docker Hub
apptainer build bert_gpu.sif docker://vivwhite/ml-experiments:bert-sst2-gpu
```

### For Users (Pulling the Pre-built Image)
##### Pull the GPU image:
`docker pull vivwhite/ml-experiments:bert-sst2-gpu`
##### Pull the CPU image:
`docker pull vivwhite/ml-experiments:bert-sst2-cpu`

### Running the experiment
##### Run the experiment on a GPU:
```
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  -v hf_master_cache:/app/hf_cache \
  -e HF_HOME=/app/hf_cache \
  vivwhite/ml-experiments:bert-sst2-gpu
```
##### Run the experiment on a HPC GPU with Apptainer:
```
apptainer run --nv  -bind $SCRATCH/hf_master_cache:/app/hf_cache bert_gpu.sif 
```
##### Run the experiment on a CPU:
```
docker pull vivwhite/ml-experiments:bert-sst2-cpu
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  -v hf_master_cache:/app/hf_cache \
  -e HF_HOME=/app/hf_cache \
  vivwhite/ml-experiments:bert-sst2-cpu
```
