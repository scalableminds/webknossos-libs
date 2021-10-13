#!/usr/bin/env bash
set -eEuo pipefail

echo -n "Typecheck webknossos: "
python -m mypy -p webknossos

echo -n "Typecheck tests:      "
python -m mypy -p tests

echo -n "Typecheck examples:   "
python -m mypy -p examples
