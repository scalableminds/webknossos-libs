#!/usr/bin/env bash
set -eEuo pipefail

uv run --frozen ruff check --fix .