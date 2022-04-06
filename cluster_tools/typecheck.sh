#!/usr/bin/env bash
set -eEuo pipefail

echo "typecheck.sh is not available for cluster_tools yet"

echo "Typecheck cluster_tools..."
poetry run python -m mypy -p cluster_tools

echo "Typecheck tests..."
poetry run python -m mypy -p tests
