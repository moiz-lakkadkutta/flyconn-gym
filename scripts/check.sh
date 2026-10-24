#!/usr/bin/env bash
# Full local gate: lint, format, types, all tests (unit + physics, both offline). Run before every commit.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check
uv run ruff format --check
uv run pyright
uv run pytest -q
