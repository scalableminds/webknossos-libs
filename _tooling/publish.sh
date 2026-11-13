#!/usr/bin/env bash
set -eEuo pipefail
set +x

for PKG in {cluster_tools,webknossos}/pyproject.toml; do
    PKG="$(dirname "$PKG")"
    echo Publishing "$PKG"

    pushd "$PKG" > /dev/null

    cp pyproject.toml pyproject.toml.bak
    # set by the monorepo's dev releases (tools/dev_version.py); its release tags are webknossos-libs/vX.Y.Z
    PKG_VERSION="${PKG_VERSION:-$(uvx dunamai from git --pattern-prefix '(?:webknossos-libs/)?' --strict)}"

    echo "__version__ = '$PKG_VERSION'" > ./"$PKG"/version.py

    # Update version number in pyproject.toml
    sed -i 's/version = "0.0.0"/version = "'"${PKG_VERSION}"'"/g' pyproject.toml    

    # replace relative path dependencies (i.e. cluster-tools) with the current version:
    sed -i 's/"cluster-tools"/"cluster-tools=='"${PKG_VERSION}"'"/g' pyproject.toml    
    
    uv build
    if [ "${DRY_RUN:-}" = true ]; then
        echo "Dry run, not publishing:" dist/*
    else
        uv publish
    fi

    # Restore files
    mv pyproject.toml.bak pyproject.toml

    popd > /dev/null
done
