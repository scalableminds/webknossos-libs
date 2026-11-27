#!/usr/bin/env bash
set -eEuo pipefail

echo "Typecheck webknossos, tests, examples and script_collection..."
uv run --frozen pyrefly check --min-severity error
