#!/usr/bin/env bash
set -eEuo pipefail

if [ $# -eq 1 ] && [ "$1" = "check" ]; then
    poetry run black --check .
    poetry run isort --check-only .
else
    poetry run isort .
    poetry run black .
fi
