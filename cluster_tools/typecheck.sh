#!/usr/bin/env bash
set -eEuo pipefail

echo "Typecheck cluster_tools and tests..."
uv run pyrefly check --min-severity error
