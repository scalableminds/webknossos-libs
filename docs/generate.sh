#! /usr/bin/env bash
set -Eeo pipefail

if [ ! -d "src/webknossos" ]; then
    echo
    echo ERROR!
    echo 'Either link or clone the webknossos repository to "webknossos-libs/docs/src/webknossos", e.g. with'
    echo 'git clone --depth 1 git@github.com:scalableminds/webknossos.git webknossos-libs/docs/src/webknossos'
    exit 1
fi
rm -rf src/api/webknossos
uv run --frozen generate_api_doc_pages.py 

if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
    uv run --with black --frozen mkdocs build
else
    uv run --with black --frozen mkdocs serve -a localhost:8197 --watch-theme
fi
    