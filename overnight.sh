#!/usr/bin/env bash
# Unattended overnight transcription. Detaches from this terminal so it survives
# the session closing; only `touch data/STOP` or a reboot stops it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${PUNJABI_VENV:-$HOME/.venvs/punjabi-lm}"
cd "$HERE"
mkdir -p data
setsid nohup "$VENV/bin/python" overnight.py "$@" >> data/overnight.out 2>&1 &
echo "started pid $! — tail -f '$HERE/data/overnight.log'"
echo "stop with: touch '$HERE/data/STOP'"
