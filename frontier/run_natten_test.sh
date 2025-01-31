#!/bin/bash

# Load environment if not already loaded
if [[ -z "${NATTEN_WITH_ACCP}" ]]; then
    source $(dirname "$0")/load_env.sh
fi

# Run all ACPP tests by default
if [[ $# -eq 0 ]]; then
    python -m pytest tests/test_*acpp.py -v
else
    # Run specific tests if provided
    python -m pytest "$@" -v
fi