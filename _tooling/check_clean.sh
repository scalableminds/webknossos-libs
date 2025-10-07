#!/usr/bin/env bash
set -e

echo "🔍 Checking for uncommitted changes..."

# Include both committed and untracked changes
if ! git diff --quiet HEAD || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  echo "❌ Git working tree is dirty:"
  git status --short
  exit 1
fi

echo "✅ Working tree clean."