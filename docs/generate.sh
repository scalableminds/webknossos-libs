#! /usr/bin/env bash
set -Eeo pipefail

PROJECT_DIR="$(dirname "$(dirname "$0")")"


cd "$PROJECT_DIR/docs"
poetry install

if [ ! -d "wk-repo" ]; then
    echo
    echo ERROR!
    echo 'Either link or clone the webknossos repository to "docs/wk-repo", e.g. with'
    echo 'git clone --depth 1 git@github.com:scalableminds/webknossos.git docs/wk-repo'
    exit 1
fi

export PDOC_CLASS_MODULES="$(poetry run python get_keyword_mapping.py)"
if [ $# -eq 1 ] && [ "$1" = "--api" ]; then
    poetry run pdoc ../webknossos/webknossos ../wkcuber/wkcuber -t pdoc_templates/pure_pdoc -h 0.0.0.0 -p 8196
else
    rm -rf src/api
    poetry run pdoc ../webknossos/webknossos ../wkcuber/wkcuber -t pdoc_templates/with_mkdocs -o src/api
    # rename .html files to .md
    find src/api -iname "*.html" -exec sh -c 'mv "$0" "${0%.html}.md"' {} \;
    # assert that API docs are written
    webknossos_files="$(find src/api/webknossos -type f -name "*.md" | wc -l)"
    wkcuber_files="$(find src/api/wkcuber -type f -name "*.md" | wc -l)"
    if ! [ "$webknossos_files" -gt "50" ]; then
       echo "Error: There are too few ($webknossos_files, expected > 80) files in src/api/webknossos,"
       echo "probably there was an error with pdoc before!"
       exit 1
    fi
    if ! [ "$wkcuber_files" -gt "25" ]; then
       echo "There are too few ($wkcuber_files, expected > 25) files in src/api/wkcuber,"
       echo "probably there was an error with pdoc before!"
       exit 1
    fi
    if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
        PYTHONPATH=$PYTHONPATH:. poetry run mkdocs build
    else
        PYTHONPATH=$PYTHONPATH:. poetry run mkdocs serve -a localhost:8197 --watch-theme
    fi
fi
