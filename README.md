# Efficiency of Computational Experiments
A container-based toolkit for reusable, efficient, and energy-aware machine learning experiments.
The repository provides three reproducible, containerized ML experiments: training ResNet18 on CIFAR10, finetuning Bert on GLUE SST2, and adapting ResNet50 to ImageNet-C.

Docker is a system for consolidating everything needed for an environment into an “image”. Images are reproducible and can be pushed and pulled from the Docker hub. This enables consistency across machines and portability: images can be pulled from the Docker Hub and run on local workstations or HPC clusters (via Apptainer).

We integrate CodeCarbon to monitor the environmental impact of these experiments. CodeCarbon measures CPU, GPU, and RAM power consumption, total energy usage (kWh), and hardware utilization. CodeCarbon can store data locally or on the cloud:
* Local: energy metrics are saved automatically to `emissions/emissions.csv` within each experiment folder.
* Cloud (Optional): provide your `CODECARBON_API_TOKEN` and `CODECARBON_EXPERIMENT_ID` as environment variables to sync data to your web dashboard.
  * you must log into CodeCarbon, create a project, create an experiment within the project, and generate an API key.



### Structure
Each experiment folder contains its own `Dockerfile`, source code, and a `README.md` with instructions for running and developing the code.
* `resnet18-cifar10/` trains a ResNet18 from scratch on CIFAR-10 dataset
* `bert-glue/` fine-tunes BERT-Base on the GLUE (SST2) task.
* `resnet50-tent/` adapts a ResNet50 using Tent to ImageNet-C at test-time.
  
Each experiment folder additionally contains a `plot.py` script to visualize the energy emissions.

### Setup
```
git clone git@github.com:vivianwhite/efficient_containers.git
cd efficient_containers
```
Navigate to any experiment folder and follow the local `README.md`.
