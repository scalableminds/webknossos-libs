#!/usr/bin/env bash
set -eEuo pipefail

echo "Typecheck webknossos..."
python -m mypy -p webknossos

echo "Typecheck tests..."
python -m mypy -p tests

echo "Typecheck examples..."
python -m mypy -p examples
