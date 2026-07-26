#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-$(pwd)}"

echo "[CHECK] pwd: $(pwd)"
echo "[CHECK] PATH: ${PATH}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[ERROR] Neither python nor python3 is available in the image" >&2
  exit 1
fi

echo "[CHECK] python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

mods = ["torch", "slime"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"[CHECK] import {name}: OK ({getattr(mod, '__file__', 'built-in')})")

print(f"[CHECK] executable: {sys.executable}")
PY
