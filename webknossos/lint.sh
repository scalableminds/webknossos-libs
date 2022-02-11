#!/usr/bin/env bash
set -eEuo pipefail

# Pylint doesn't lint files in directories that don't have an __init__.py
# This will maybe be fixed by https://github.com/PyCQA/pylint/issues/352
# In the meantime, find all python files, except for the ./webknossos/client/_generated directory and lint them
# Inspired by https://stackoverflow.com/questions/4210042/how-to-exclude-a-directory-in-find-command
find . -type d \( -path ./webknossos/client/_generated -o -path ./tests/.webknossos-server -o -path ./stubs \) -prune -o -iname "*.py" -print | xargs poetry run python -m pylint -j2
