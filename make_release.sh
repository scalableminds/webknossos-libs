#!/usr/bin/env bash
set -eEuo pipefail
set +x

if [[ $# -eq 0 ]] ; then
    echo "Please supply a 'version' as argument."
    exit 1
fi

PKG_VERSION="$1"

if ! python check_version.py ${PKG_VERSION}; then
    echo "A higher version is already present."
    exit 1
fi

for PKG in */pyproject.toml; do
    PKG="$(dirname "$PKG")"
    if [[ "$PKG" == "docs" ]]; then
        echo Skipping "$PKG"
        continue
    fi
    echo "Creating release for $PKG"

    pushd "$PKG" > /dev/null

    python ../make_changelog.py "$PKG_VERSION"

    popd > /dev/null
done
