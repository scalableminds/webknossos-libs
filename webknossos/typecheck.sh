#!/usr/bin/env bash
set -eEuo pipefail

export MYPYPATH=./stubs 

echo "Typecheck webknossos..."
uv run --frozen python -m mypy -n 4 -p webknossos

echo "Typecheck tests..."
uv run --frozen python -m mypy -n 4 -p tests

echo "Typecheck examples..."
uv run --frozen python -m mypy -n 4 -p examples

echo "Typecheck script_collection..."
uv run --frozen python -m mypy -n 4 -p script_collection
