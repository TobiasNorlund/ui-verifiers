#!/bin/bash

set -euo pipefail

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Update ui_verifiers to bleeding edge
python -m pip install --user --upgrade --force-reinstall git+https://github.com/TobiasNorlund/ui-verifiers

# Start server
uvicorn ui_verifiers.server:app --host 0.0.0.0
