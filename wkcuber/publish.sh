#!/usr/bin/env bash
set -eEuo pipefail
set +x

cp pyproject.toml pyproject.toml.bak
PKG_VERSION=$(dunamai from git)

sed -i "0,/version = \".*\"/{s/version = \".*\"/version = \"$PKG_VERSION\"/}" pyproject.toml
poetry publish --build -u $PYPI_USERNAME -p $PYPI_PASSWORD

# Restore files
mv pyproject.toml.bak pyproject.toml
