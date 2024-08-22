#! /usr/bin/env bash
set -Eeo pipefail

poetry install

if [ ! -d "wk-repo" ]; then
    echo
    echo ERROR!
    echo 'Either link or clone the webknossos repository to "docs/wk-repo", e.g. with'
    echo 'git clone --depth 1 git@github.com:scalableminds/webknossos.git docs/wk-repo'
    exit 1
fi
rm -rf src/api
PYTHONPATH=$PYTHONPATH poetry run python generate_api_doc_pages.py

if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
    PYTHONPATH=$PYTHONPATH poetry run mkdocs build
else
    PYTHONPATH=$PYTHONPATH poetry run mkdocs serve -a localhost:8197 --watch-theme
fi
