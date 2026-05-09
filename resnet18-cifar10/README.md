
## Training ResNet18 on CIFAR10
#### Arguments
* `--batch_size`, default=128
* `--epochs`, default=10
* `--lr`, default=0.1
* `--num-workers`, default=4

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
apptainer run --nv resnet_gpu.sif
```
##### _Optional: log emissions to CodeCarbon API_
First log into CodeCarbon, then create a project, then create an experiment within the project and generate an API key.
```
apptainer run --nv --env CODECARBON_API_TOKEN=X --env CODECARBON_EXPERIMENT_ID=Y resnet_gpu.sif
```
##### Run the experiment on a CPU:
```
docker run --rm \
  -v $(pwd):/app -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-cpu
```

### Results
Emissions are saved to `emissions/emissions.csv`. 
Model is saved to `models/resnet18_cifar10.pt`.
Results are saved to `results/results.csv`.
