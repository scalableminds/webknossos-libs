#! /usr/bin/env bash
set -Eeo pipefail

PROJECT_DIR="$(dirname "$(dirname "$0")")"

cd "$PROJECT_DIR/."

if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
    pdoc ./webknossos/webknossos ./wkcuber/wkcuber -o docs/api
else
    pdoc ./webknossos/webknossos ./wkcuber/wkcuber -p 8095 -h 0.0.0.0
fi