#!/usr/bin/env bash
set -eEuo pipefail

if [ $# -eq 1 ] && [ "$1" = "check" ]; then
    ruff check .
else
    ruff check --fix .
    ruff format .
fi
