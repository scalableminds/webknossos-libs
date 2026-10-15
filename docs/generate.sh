#! /usr/bin/env bash
set -Eeo pipefail

# src/webknossos: the docs of the webknossos repository, cloned or linked to wk-repo
if [ ! -e "src/webknossos" ] && [ -d "wk-repo/docs" ]; then
    ln -sfn ../wk-repo/docs src/webknossos
fi
if [ ! -d "src/webknossos" ]; then
    echo
    echo ERROR!
    echo 'Either link or clone the webknossos repository to "webknossos-libs/docs/wk-repo", e.g. with'
    echo 'git clone --depth 1 git@github.com:scalableminds/webknossos.git webknossos-libs/docs/wk-repo'
    exit 1
fi
rm -rf src/api/webknossos
uv run --frozen generate_api_doc_pages.py 

if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
    uv run --with black --frozen mkdocs build
else
    uv run --with black --frozen mkdocs serve -a localhost:8197 --watch-theme
fi
    