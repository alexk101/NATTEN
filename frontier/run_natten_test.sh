#!/bin/bash

# Load environment if not already loaded
if [[ -z "${NATTEN_WITH_ACPP}" ]]; then
    source $(dirname "$0")/load_env.sh
fi

# Configure CMake variables for AdaptiveCpp
export CMAKE_PREFIX_PATH="/opt/rocm:$CMAKE_PREFIX_PATH"
export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DWITH_ROCM_BACKEND=ON"

# Run all ACPP tests by default
if [[ $# -eq 0 ]]; then
    # First ensure NATTEN is built with proper CMake config
    cd ..
    pip install -e .
    
    # Run tests
    python -m pytest tests/test_*acpp.py -v
else
    # First ensure NATTEN is built with proper CMake config
    cd ..
    pip install -e .
    
    # Run specific tests if provided
    python -m pytest "$@" -v
fi