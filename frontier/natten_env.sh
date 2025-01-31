#!/bin/bash

# Load required modules
module load PrgEnv-cray
module load rocm/6.3.1
module load cmake

# Set NATTEN environment variables
export NATTEN_WITH_ACPP=1
export NATTEN_ACPP_ARCH=gfx90a
export NATTEN_N_WORKERS=8

# Print confirmation
echo "NATTEN development environment loaded"
echo "NATTEN_ACPP_ARCH: $NATTEN_ACPP_ARCH"
