## Finetuning Bert on GLUE SST2
### Arguments
* `--batch_size`, default=32
* `--epochs`, default=3
* `--lr`, default=2e-5
### To build (for developers)
#### Modify the Dockerfile or environment:
`docker build -f Dockerfile -t vivwhite/ml-experiments:bert-sst2-glue .`



### To run (for users)
#### Pull the pre-built image:
`docker pull vivwhite/ml-experiments:bert-sst2-glue`

#### Run the experiment on a GPU:
```
docker run --rm --gpus all \
  -v $(pwd):/app \
  -w /app \
  -v hf_master_cache:/app/hf_cache \
  -e HF_HOME=/app/hf_cache \
  vivwhite/ml-experiments:bert-sst2-glue
```
##### _Optional: log emissions to CodeCarbon API_
First log into CodeCarbon, then create a project, then create an experiment within the project and generate an API key.
Add `CODECARBON_API_TOKEN` and `CODECARBON_EXPERIMENT_ID` to a `.env` file.
```
docker run --rm --gpus all --env-file .env \
  -v $(pwd):/app \
  -w /app \
  -v hf_master_cache:/app/hf_cache \
  -e HF_HOME=/app/hf_cache \
  vivwhite/ml-experiments:bert-sst2-glue
```
#### Run the experiment on a HPC GPU with Apptainer:
First, build a SIF file to run Apptainer on an HPC:
```
# 1. Set scratch paths to avoid 'No space left' errors during build
export APPTAINER_TMPDIR=$SCRATCH/app_tmp
export APPTAINER_CACHEDIR=$SCRATCH/app_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
mkdir -p $SCRATCH/hf_master_cache

# 2. Build the SIF (Singularity Image File) from Docker Hub
apptainer build bert_glue.sif docker://vivwhite/ml-experiments:bert-sst2-glue
```
Then run:
```
apptainer run --nv  --bind $SCRATCH/hf_master_cache:/app/hf_cache bert_glue.sif 
```
_Optional: log emissions to CodeCarbon API_

`apptainer run --nv --env CODECARBON_API_TOKEN=X --env CODECARBON_EXPERIMENT_ID=Y --bind $SCRATCH/hf_master_cache:/app/hf_cache bert_glue bert_glue.sif`

### Results
Energy emissions are logged to `emissions/emissions.csv`.
Finetuning results are saved in `results/results.csv`.
Finetuned model and tokenizer are saved in `models/`.
