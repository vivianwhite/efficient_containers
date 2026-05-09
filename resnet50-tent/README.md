## Adapting ResNet50 with Tent

### For Developers (Building from Source)
##### Modify the Dockerfile or environment for use on a GPU:
`docker build -f Dockerfile -t vivwhite/ml-experiments:resnet50-tent .`

##### Build a SIF file to run Apptainer on an HPC:
```
# 1. Set scratch paths to avoid 'No space left' errors during build
export APPTAINER_TMPDIR=$SCRATCH/app_tmp
export APPTAINER_CACHEDIR=$SCRATCH/app_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# 2. Build the SIF (Singularity Image File) from Docker Hub
apptainer build resnet50_tent.sif docker://vivwhite/ml-experiments:resnet50-tent
```

### For Users (Pulling the Pre-built Image)
##### Pull the GPU image:
`docker pull vivwhite/ml-experiments:resnet50-tent`

### Running the experiment
##### Run the experiment on a local GPU _(Untested)_
```
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  -v /datasets/imagenet-c:/app/data/ImageNet-C \
  vivwhite/ml-experiments:resnet50-tent
```
##### Run the experiment on a HPC GPU with Apptainer:
```
apptainer run --nv --bind /datasets/imagenet-c:/app/data/ImageNet-C resnet50_tent.sif
```
##### _Optional: log emissions to CodeCarbon API_
First log into CodeCarbon, then create a project, then create an experiment within the project and generate an API key.
```
apptainer run --nv --bind /datasets/imagenet-c:/app/data/ImageNet-C --env CODECARBON_API_TOKEN=X --env
CODECARBON_EXPERIMENT_ID=Y resnet50_tent.sif 
```
### Results
Energy emissions are logged to `emissions/emissions.csv`.
Adaptation results are saved in `results/results.csv`.
Adapted model weights saved in `models/`.
<img width="640" height="480" alt="energy_consumed" src="https://github.com/user-attachments/assets/b1b12396-28cc-43e9-a37d-06645c2b4e7b" />

