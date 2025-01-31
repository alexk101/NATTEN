#!/bin/bash

# Load miniforge
module load miniforge3

# Create environment from yaml if it doesn't exist
if ! conda env list | grep -q "natten_dev"; then
    conda env create -f natten_env.yaml
fi

# Activate environment
conda activate natten_dev

# Set NATTEN environment variables
export ACPP_HOME=/opt/rocm-5.7.0/lib/adaptivecpp
export NATTEN_WITH_ACPP=1
export NATTEN_ACPP_ARCH=gfx90a
export NATTEN_N_WORKERS=8

echo "NATTEN development environment loaded"
echo "ACPP_HOME: $ACPP_HOME"
echo "NATTEN_ACPP_ARCH: $NATTEN_ACPP_ARCH" 