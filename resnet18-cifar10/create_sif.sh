#!/bin/bash
#SBATCH --output=build_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --account=aip-evanesce

## set scratch paths
export APPTAINER_TMPDIR=$SCRATCH/app_tmp
export APPTAINER_CACHEDIR=$SCRATCH/app_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# run the build
#apptainer build resnet_gpu.sif docker://vivwhite/ml-experiments:resnet18-cifar10-gpu
print("Build the thing")

exit
