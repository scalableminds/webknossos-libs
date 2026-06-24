#!/usr/bin/env bash
set -eEuo pipefail

echo "Typecheck cluster_tools..."
uv run python -m mypy -n 4 -p cluster_tools

echo "Typecheck tests..."
uv run python -m mypy -n 4 -p tests
