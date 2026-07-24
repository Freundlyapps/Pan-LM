#!/usr/bin/env bash
# Launcher — mirrors the Transcribe repo's convention of running via a .sh wrapper.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${PUNJABI_VENV:-$HOME/.venvs/punjabi-lm}"
cd "$HERE"
exec "$VENV/bin/python" app.py "$@"
