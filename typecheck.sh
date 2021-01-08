#!/bin/bash
set -eEuo pipefail
python -m mypy wkcuber --disallow-untyped-defs --show-error-codes --strict-equality --namespace-packages
