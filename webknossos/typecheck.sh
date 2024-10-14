#!/usr/bin/env bash
set -eEuo pipefail

export MYPYPATH=./stubs 

echo "Typecheck webknossos..."
uv run --frozen python -m mypy -p webknossos

echo "Typecheck tests..."
uv run --frozen python -m mypy -p tests

echo "Typecheck examples..."
uv run --frozen python -m mypy -p examples

echo "Typecheck script_collection..."
uv run --frozen python -m mypy -p script_collection
