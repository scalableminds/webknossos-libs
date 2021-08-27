#!/bin/bash
set -eEuo pipefail
echo "Typecheck wklib module:"
python -m mypy -p wklib --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages

echo "Typecheck tests:"
python -m mypy -p tests --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages
