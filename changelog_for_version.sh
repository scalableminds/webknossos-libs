#!/usr/bin/env bash
set -eEuo pipefail
set +x

if [ $# -eq 0 ]; then
    VERSION_START="^## Unreleased"
else
    VERSION_START="^## \[$1\]"
fi
VERSION_END="^## \["

for PKG in */pyproject.toml; do
    PKG="$(dirname "$PKG")"
    if [ ! -f "$PKG/Changelog.md" ]; then
        continue
    fi

    CHANGES="$(awk "/$VERSION_START/{flag=1;next} /$VERSION_END/{flag=0} flag" "$PKG/Changelog.md" | tail -n +2)"

    WORDS_IN_CHANGES="$(echo "${CHANGES%x}" | grep --invert-match "###" | wc -w)"

    if [ "$WORDS_IN_CHANGES" != "0" ]; then
        echo "## $PKG"
        echo "${CHANGES%x}"
        echo
        echo
    fi
done
