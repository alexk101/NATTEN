#!/bin/bash

# Load miniforge
module load miniforge3

# Create conda environment
conda create -n natten_dev python=3.8 -y

# Activate environment
conda activate natten_dev

# Install PyTorch and other dependencies
conda install pytorch=2.0.1 torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install cmake ninja -y
conda install fvcore -y

# Additional development dependencies
conda install pytest -y

# Export the environment
conda env export > natten_env.yaml

echo "Environment created and exported to natten_env.yaml" 