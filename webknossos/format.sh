#!/usr/bin/env bash
set -eEuo pipefail

if [ $# -eq 1 ] && [ "$1" = "check" ]; then
    black --check .
    isort --check-only .
else
    isort .
    black .
fi
