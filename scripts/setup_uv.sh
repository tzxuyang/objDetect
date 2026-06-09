#!/usr/bin/env bash
set -euo pipefail

# Lightweight helper to install uv (user) and create/sync the project environment.
PYTHON=python3.10
if ! command -v $PYTHON >/dev/null 2>&1; then
  echo "Error: python3.10 not found. Install system Python 3.10 first." >&2
  exit 1
fi

# Install uv in user site if missing
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv into user site packages..."
  $PYTHON -m pip install --user --upgrade pip
  $PYTHON -m pip install --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create/activate uv environment
echo "Creating uv virtual environment and syncing dependencies..."
uv lock || true
uv sync || true

echo "Done. Use 'uv run python <script>' to run within the environment."