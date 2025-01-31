#!/bin/bash

# Load environment if not already loaded
if [[ -z "${NATTEN_WITH_ACPP}" ]]; then
    source $(dirname "$0")/natten_env.sh
fi

# Run all ACPP tests by default
if [[ $# -eq 0 ]]; then
    python -m unittest tests/test_*acpp.py
else
    # Run specific tests if provided
    python -m unittest "$@"
fi
