#! /usr/bin/env bash
set -Eeo pipefail

PROJECT_DIR="$(dirname "$(dirname "$0")")"


cd "$PROJECT_DIR/docs"
poetry install

if [ ! -d "src/wk-repo" ]; then
    echo
    echo ERROR!
    echo 'Either link or clone the webknossos repository to "docs/src/wk-repo", e.g. with'
    echo 'git clone git@github.com:scalableminds/webknossos.git docs/src/wk-repo'
    exit 1
fi

if [ $# -eq 1 ] && [ "$1" = "--api" ]; then
    poetry run pdoc ../webknossos/webknossos ../wkcuber/wkcuber -h 0.0.0.0 -p 8196
else
    rm -rf src/api
    poetry run pdoc ../webknossos/webknossos ../wkcuber/wkcuber -t pdoc_templates -o src/api
    # rename .html files to .md
    find src/api -iname "*.html" -exec sh -c 'mv "$0" "${0%.html}.md"' {} \;
    if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
        PYTHONPATH=$PYTHONPATH:. poetry run mkdocs build
    else
        PYTHONPATH=$PYTHONPATH:. poetry run mkdocs serve -a localhost:8197 --watch-theme
    fi
fi
