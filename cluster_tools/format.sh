#!/usr/bin/env bash
set -eEuo pipefail

if [ $# -eq 1 ] && [ "$1" = "check" ]; then
    uv run ruff format --check .
else
    uv run ruff check --select I --fix . # format the imports 
    uv run ruff format .
fi
