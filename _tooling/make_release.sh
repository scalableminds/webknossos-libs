#!/usr/bin/env bash
set -eEuo pipefail
set +x

if [[ $# -eq 0 ]] ; then
    echo "Please supply a 'version' as argument."
    exit 1
fi

PKG_VERSION="$1"

if ! python _tooling/check_version.py ${PKG_VERSION}; then
    echo "A higher version is already present."
    exit 1
fi

for PKG in {cluster_tools,webknossos}/pyproject.toml; do
    PKG="$(dirname "$PKG")"
    echo "Creating release for $PKG"

    pushd "$PKG" > /dev/null

    python ../_tooling/changelog_bump_version.py "$PKG_VERSION"

    popd > /dev/null
done
