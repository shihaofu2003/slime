#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_ROOT:-$(pwd)}"

echo "[CHECK] pwd: $(pwd)"

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

checks = [
    ("megatron.core", "Megatron-Core"),
    ("sglang", "SGLang"),
]

for module_name, display_name in checks:
    mod = importlib.import_module(module_name)
    print(f"[CHECK] import {display_name}: OK ({getattr(mod, '__file__', 'built-in')})")
    print(f"[CHECK] {display_name} module: {module_name}")

print(f"[CHECK] executable: {sys.executable}")
PY

if "${PYTHON_BIN}" -m sglang.launch_server --help >/tmp/sglang_help.txt 2>&1; then
  echo "[CHECK] sglang CLI: OK"
  sed -n '1,5p' /tmp/sglang_help.txt
else
  echo "[ERROR] sglang CLI failed" >&2
  sed -n '1,50p' /tmp/sglang_help.txt >&2 || true
  exit 1
fi
