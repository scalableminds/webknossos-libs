#!/usr/bin/env bash
set -eEuo pipefail

if [ $# -eq 1 ] && [ "$1" = "check" ]; then
    poetry run ruff format --check .
else
    poetry run ruff check --fix .
    poetry run ruff format .
fi
