#!/usr/bin/env bash
set -eEuo pipefail

echo "Typecheck webknossos..."
python -m mypy -p webknossos --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages --no-implicit-optional

echo "Typecheck tests..."
python -m mypy -p tests --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages --no-implicit-optional

echo "Typecheck examples..."
python -m mypy -p examples --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages --no-implicit-optional

echo "Typecheck script-collection..."
python -m mypy -p script-collection --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages --no-implicit-optional
