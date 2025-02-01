#!/bin/bash

ROCM_VERSION=6.2.4

# Load required modules
module load PrgEnv-cray
module load rocm/$ROCM_VERSION
module load craype-accel-amd-gfx90a
module load cmake
module load ninja/1.11.1

# Set NATTEN environment variables
# AdaptiveCpp will be detected from submodule automatically
export NATTEN_WITH_ACPP=1
export NATTEN_ACPP_ARCH=gfx90a
export NATTEN_N_WORKERS=8

# Print confirmation
echo "NATTEN development environment loaded"
echo "NATTEN_ACPP_ARCH: $NATTEN_ACPP_ARCH"

# Load miniforge
module load miniforge3
conda init

# Check if environment exists
ENV_PATH="$HOME/.miniforge/envs/natten_dev"
if [ -d "$ENV_PATH" ]; then
    read -p "Environment natten_dev already exists. Do you want to delete and recreate it? [y/N] " response
    response=${response:-N}  # Default to N if empty
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        mamba env remove -p $ENV_PATH
    else
        echo "Keeping existing environment. Exiting..."
        mamba activate $ENV_PATH

        # Build and install NATTEN
        cd ..
        $ENV_PATH/bin/pip install -e .
        cd frontier

        echo "NATTEN package built and installed in development mode"

        exit 0
    fi
fi

# Create conda environment
mamba create -p $ENV_PATH python=3.11 -y

# Activate environment
mamba activate $ENV_PATH

# Install PyTorch and other dependencies
$ENV_PATH/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm$ROCM_VERSION/
mamba install -p $ENV_PATH fvcore -y

# Additional development dependencies
mamba install -p $ENV_PATH pytest -y

# Export the environment
conda env export -p $ENV_PATH > natten_env.yaml

echo "Environment created and exported to natten_env.yaml" 

export CC=craycc
export CXX=craycxx

# Build and install NATTEN
cd ..
$ENV_PATH/bin/pip install -e .
cd frontier

echo "NATTEN package built and installed in development mode"
