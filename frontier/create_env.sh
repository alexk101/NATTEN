#!/bin/bash

# Load miniforge
module load miniforge3
conda init

# Create conda environment
mamba create -p $HOME/.miniforge/envs/natten_dev python=3.11 -y

# Activate environment
mamba activate $HOME/.miniforge/envs/natten_dev

# Install PyTorch and other dependencies
mamba install pytorch=2.0.1 torchvision pytorch-rocm=5.7 -c pytorch -c conda-forge -y
mamba install cmake ninja -y
mamba install fvcore -y

# Additional development dependencies
mamba install pytest -y

# Export the environment
conda env export > natten_env.yaml

echo "Environment created and exported to natten_env.yaml" 