#!/bin/bash

# Load required modules
module load PrgEnv-cray
module load rocm/5.7.0
module load pytorch/2.0.1
module load cmake

# Set NATTEN environment variables
export ACPP_HOME=/opt/rocm-5.7.0/lib/adaptivecpp
export NATTEN_WITH_ACPP=1
export NATTEN_ACPP_ARCH=gfx90a
export NATTEN_N_WORKERS=8

# Print confirmation
echo "NATTEN development environment loaded"
echo "ACPP_HOME: $ACPP_HOME"
echo "NATTEN_ACPP_ARCH: $NATTEN_ACPP_ARCH"
