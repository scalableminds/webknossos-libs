#!/bin/bash
set -eEuo pipefail

echo "Typecheck wkcuber module..."
poetry run python -m mypy -p wkcuber --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages --no-implicit-optional

echo "Typecheck tests..."
poetry run python -m mypy -p tests --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages --no-implicit-optional
