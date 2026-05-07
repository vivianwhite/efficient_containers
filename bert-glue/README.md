## Finetuning Bert on GLUE SST2

### For Developers (Building from Source)
##### Modify the Dockerfile or environment for use on a GPU:
`docker build -f Dockerfile -t vivwhite/ml-experiments:bert-sst2-glue .`

##### Build a SIF file to run Apptainer on an HPC:
```
# 1. Set scratch paths to avoid 'No space left' errors during build
export APPTAINER_TMPDIR=$SCRATCH/app_tmp
export APPTAINER_CACHEDIR=$SCRATCH/app_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
mkdir -p $SCRATCH/hf_master_cache

# 2. Build the SIF (Singularity Image File) from Docker Hub
apptainer build bert_glue.sif docker://vivwhite/ml-experiments:bert-sst2-glue
```

### For Users (Pulling the Pre-built Image)
##### Pull the GPU image:
`docker pull vivwhite/ml-experiments:bert-sst2-glue`

### Running the experiment
##### Run the experiment on a GPU:
```
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  -v hf_master_cache:/app/hf_cache \
  -e HF_HOME=/app/hf_cache \
  vivwhite/ml-experiments:bert-sst2-glue
```
##### Run the experiment on a HPC GPU with Apptainer:
```
apptainer run --nv  -bind $SCRATCH/hf_master_cache:/app/hf_cache bert_glue.sif 
```
### Results
Energy emissions are logged to `emissions/emissions.csv`.
Finetuning results are saved in `results/results.csv`.
Finetuned model and tokenizer are saved in `models/`.
