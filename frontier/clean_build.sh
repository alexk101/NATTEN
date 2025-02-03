#!/bin/bash

echo "Cleaning NATTEN build artifacts..."

# Get the project root directory (one level up from frontier/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Clean pip build artifacts
rm -rf "$PROJECT_ROOT/build"
rm -rf "$PROJECT_ROOT/dist"
rm -rf "$PROJECT_ROOT/*.egg-info"

# Clean CMake build directories
rm -rf "$PROJECT_ROOT/csrc/build"
rm -rf "$PROJECT_ROOT/third_party/AdaptiveCpp/build"

# Clean Python cache files
find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} +
find "$PROJECT_ROOT" -type f -name "*.pyc" -delete
find "$PROJECT_ROOT" -type f -name "*.pyo" -delete
find "$PROJECT_ROOT" -type f -name "*.pyd" -delete

# Clean any remaining CMake cache files
find "$PROJECT_ROOT" -type f -name "CMakeCache.txt" -delete
find "$PROJECT_ROOT" -type d -name "CMakeFiles" -exec rm -rf {} +

echo "Cleanup complete!" 