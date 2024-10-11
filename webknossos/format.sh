#!/usr/bin/env bash
set -eEuo pipefail

if [ $# -eq 1 ] && [ "$1" = "check" ]; then
    uv run --frozen ruff format --check .
else
    uv run --frozen ruff check --select I --fix . # format the imports 
    uv run --frozen ruff format .
fi
