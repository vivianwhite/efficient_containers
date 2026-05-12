
## Training ResNet18 on CIFAR10
### Arguments
* `--batch_size`, default=128
* `--epochs`, default=10
* `--lr`, default=0.1
* `--num-workers`, default=4

### To build (for developers)
##### Modify the Dockerfile or environment for use on a GPU or CPU:
`docker build -f Dockerfile.gpu -t vivwhite/ml-experiments:resnet18-cifar10-gpu .` \
`docker build -f Dockerfile.cpu -t vivwhite/ml-experiments:resnet18-cifar10-cpu .`


### To run (for users)
##### Pull the pre-built GPU or CPU image:
`docker pull vivwhite/ml-experiments:resnet18-cifar10-gpu` \
`docker pull vivwhite/ml-experiments:resnet18-cifar10-cpu`

##### Run the experiment on a GPU:
```
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-gpu
```
##### Run the experiment on a CPU:
```
docker run --rm \
  -v $(pwd):/app -w /app \
  vivwhite/ml-experiments:resnet18-cifar10-cpu
```
 _Optional: log emissions to CodeCarbon API_
 
First log into CodeCarbon, then create a project, then create an experiment within the project and generate an API key. 

Add environment variables to the run command: `-e CODECARBON_API_TOKEN=X -e CODECARBON_EXPERIMENT_ID=Y `.

##### Run the experiment on a HPC GPU with Apptainer:
First, build a SIF file to run Apptainer on an HPC:
```
# 1. Set scratch paths to avoid 'No space left' errors during build
export APPTAINER_TMPDIR=$SCRATCH/app_tmp
export APPTAINER_CACHEDIR=$SCRATCH/app_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# 2. Build the SIF (Singularity Image File) from Docker Hub
apptainer build resnet_gpu.sif docker://vivwhite/ml-experiments:resnet18-cifar10-gpu
```
Then run:
```
apptainer run --nv resnet_gpu.sif
```
 _Optional: log emissions to CodeCarbon API_
```
apptainer run --nv --env CODECARBON_API_TOKEN=X --env CODECARBON_EXPERIMENT_ID=Y resnet_gpu.sif
```


### Results
Emissions are saved to `emissions/emissions.csv`. 
Model is saved to `models/resnet18_cifar10.pt`.
Results are saved to `results/results.csv`.
The model gets 77.36% after 10 epochs and 86.62% after 50 epochs.
<img width="640" height="480" alt="acc_energy_tradeoff" src="https://github.com/user-attachments/assets/8a79a4c5-3dec-4764-87ed-c6c080e0110d" />

