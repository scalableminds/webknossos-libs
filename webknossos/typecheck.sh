#!/usr/bin/env bash
set -eEuo pipefail

echo -n "Typecheck webknossos: "
python -m mypy -p webknossos --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages

echo -n "Typecheck tests:      "
python -m mypy -p tests --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages

echo -n "Typecheck examples:   "
python -m mypy -p examples --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages
