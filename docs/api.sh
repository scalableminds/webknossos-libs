#! /usr/bin/env bash
set -Eeo pipefail

PROJECT_DIR="$(dirname "$(dirname "$0")")"

cd "$PROJECT_DIR/."

if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
    pdoc wkcuber  -o docs/api
else
    pdoc wkcuber  -p 8095 -h 0.0.0.0
fi