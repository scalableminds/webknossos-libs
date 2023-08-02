#!/usr/bin/env bash
set -eEuo pipefail

echo "Typecheck cluster_tools..."
poetry run python -m mypy -p cluster_tools

echo "Typecheck tests..."
poetry run python -m mypy -p tests
