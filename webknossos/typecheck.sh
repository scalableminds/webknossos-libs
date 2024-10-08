#!/usr/bin/env bash
set -eEuo pipefail

export MYPYPATH=./stubs 

echo "Typecheck webknossos..."
uv run python -m mypy -p webknossos

echo "Typecheck tests..."
uv run python -m mypy -p tests

echo "Typecheck examples..."
uv run python -m mypy -p examples

echo "Typecheck script_collection..."
uv run python -m mypy -p script_collection
