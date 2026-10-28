#! /usr/bin/env bash
set -Eeo pipefail

# wk-repo: the webknossos repository, cloned or linked to wk-repo, or next to
# webknossos-libs in the monorepo (scalableminds/mono). Its docs are rendered
# as-is, its frontend sources are used for the frontend API reference.
if [ ! -e "wk-repo" ]; then
    if [ -e "src/webknossos" ]; then
        ln -sfn "$(cd -P src/webknossos/.. && pwd)" wk-repo
    elif [ -d "../../webknossos/docs" ]; then
        ln -sfn ../../webknossos wk-repo
    fi
fi
if [ ! -e "src/webknossos" ] && [ -d "wk-repo/docs" ]; then
    ln -sfn ../wk-repo/docs src/webknossos
fi
if [ ! -d "src/webknossos" ] || [ ! -d "wk-repo/frontend" ]; then
    echo
    echo ERROR!
    echo 'Either link or clone the webknossos repository to "webknossos-libs/docs/wk-repo", e.g. with'
    echo 'git clone --depth 1 git@github.com:scalableminds/webknossos.git webknossos-libs/docs/wk-repo'
    exit 1
fi
rm -rf src/api/webknossos
uv run --frozen generate_api_doc_pages.py 

# typedoc is called by mkdocstrings-typescript to document the frontend API
npm ci --no-audit --no-fund
export PATH="$PWD/node_modules/.bin:$PATH"

if [ $# -eq 1 ] && [ "$1" = "--persist" ]; then
    uv run --with black --frozen mkdocs build
else
    uv run --with black --frozen mkdocs serve -a localhost:8197 --watch-theme
fi
