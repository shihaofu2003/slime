#!/usr/bin/env bash
#
# tau2 dependency-discovery probe (run in the fresh slime cluster image).
#
# The cluster image ships neither tau2 nor its runtime deps. This probe:
#   1. prints the python/pip env,
#   2. runs setup/setup_tau2_bench.sh (the --no-deps editable re-point; installs
#      hatchling first if the image lacks the build backend -- setup_editable.sh
#      uses --no-build-isolation),
#   3. repeatedly invokes `tau2 --help` and pip-installs each missing runtime
#      module (logging every install), surfacing the MINIMAL set the image lacks,
#   4. installs gymnasium (the [gym] extra; not exercised by `tau2 run`),
#   5. runs the real `tau2 run --domain airline ...` smoke.
#
# Read the run log: the "DISCOVERED RUNTIME DEPS" block is the answer to
# "which packages does the image need to download".

set -uo pipefail   # no -e: we drive through install attempts deliberately

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PIP_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYBIN="$(command -v python || command -v python3 || echo python)"

log(){ echo "[tau2-probe] $*"; }

# import-name -> pip-name for tau2 deps where they differ
pipname_for(){
  case "$1" in
    yaml)              echo PyYAML ;;
    dotenv)            echo python-dotenv ;;
    docstring_parser)  echo docstring-parser ;;
    *)                 echo "$1" ;;
  esac
}

log "=== python / pip / tau2 ==="
command -v python python3 pip tau2 2>&1 || true
"$PYBIN" --version 2>&1 || true
"$PYBIN" -m pip --version 2>&1 || true

# 1. tau2 editable install (--no-deps). setup_editable.sh uses --no-build-isolation,
#    so the build backend (hatchling) AND its deps must be present in the image.
#    The image ships hatchling but not, e.g., `editables` (hatchling needs it for
#    editable wheels). Retry: on each failure, install the module it names, retry.
BUILD_DEPS=()
log "=== tau2 editable install via setup_tau2_bench.sh ==="
for attempt in $(seq 1 12); do
  OUT=$(bash "${SERVICE_AGENT_ROOT}/setup/setup_tau2_bench.sh" 2>&1); RC=$?
  if [ "$RC" -eq 0 ]; then log "setup OK (attempt ${attempt})"; break; fi
  # Extract a missing module from either error shape pip/hatchling emit:
  #   "No module named 'editables'"   or   "Cannot import 'hatchling.build'"
  ERRLINE=$(printf '%s\n' "$OUT" | grep -m1 -E "No module named|Cannot import")
  MOD=""
  if printf '%s\n' "$ERRLINE" | grep -q "No module named"; then
    MOD=$(printf '%s\n' "$ERRLINE" | sed -E "s/.*No module named '([^']+)'.*/\1/")
  elif [ -n "$ERRLINE" ]; then
    MOD=$(printf '%s\n' "$ERRLINE" | sed -E "s/.*Cannot import '([^']+)'.*/\1/")
    MOD="${MOD%%.*}"   # hatchling.build -> hatchling (pip dist name)
  fi
  if [ -z "$MOD" ]; then
    log "setup failed with non-fixable error (attempt ${attempt}); tail:"
    printf '%s\n' "$OUT" | tail -25
    break
  fi
  PKG="$(pipname_for "$MOD")"
  log "setup blocked on missing '${MOD}' -> pip install '${PKG}' (build-backend dep)"
  "$PYBIN" -m pip install "$PKG" -i "${PIP_INDEX}" && BUILD_DEPS+=("${MOD}=${PKG}")
done
log "=== BUILD-BACKEND DEPS INSTALLED (${#BUILD_DEPS[@]}) ==="
[ "${#BUILD_DEPS[@]}" -gt 0 ] && printf '  %s\n' "${BUILD_DEPS[@]}" || echo "  (none)"

command -v tau2 >/dev/null 2>&1 || { log "tau2 console script still missing -- editable build failed; aborting"; exit 1; }

# 2. discover missing runtime deps by repeatedly running `tau2 --help`.
log "=== runtime dep discovery (target: tau2 --help) ==="
DISCOVERED=()
for _ in $(seq 1 60); do
  if OUT=$(tau2 --help 2>&1); then
    log "tau2 --help OK after ${#DISCOVERED[@]} package(s) installed"
    break
  fi
  MOD=$(printf '%s\n' "$OUT" | grep "No module named" | head -1 | sed "s/.*'\\([^']*\\)'.*/\\1/")
  if [ -z "$MOD" ]; then
    log "non-ModuleNotFoundError -- discovery stopped. Last output:"
    printf '%s\n' "$OUT"
    break
  fi
  PKG="$(pipname_for "$MOD")"
  log "missing import '${MOD}' -> pip install '${PKG}'"
  if "$PYBIN" -m pip install "$PKG" -i "${PIP_INDEX}"; then
    DISCOVERED+=("${MOD}=${PKG}")
  else
    log "pip install '${PKG}' FAILED -- discovery stopped"
    break
  fi
done

log "=== DISCOVERED RUNTIME DEPS (${#DISCOVERED[@]}) ==="
if [ "${#DISCOVERED[@]}" -gt 0 ]; then printf '  %s\n' "${DISCOVERED[@]}"; else echo "  (none -- all already present)"; fi

# 3. [gym] extra: gymnasium (not imported by `tau2 run`; probe it explicitly)
log "=== [gym] extra: gymnasium ==="
if "$PYBIN" -c "import gymnasium" 2>/dev/null; then
  log "gymnasium already present"
else
  "$PYBIN" -m pip install gymnasium -i "${PIP_INDEX}" && log "installed gymnasium ([gym] extra)"
fi

# 4. the real smoke run (also makes LLM calls via the .env endpoint)
log "=== tau2 run (airline, 5 tasks) ==="
cd "${SERVICE_AGENT_ROOT}/tau2-bench"
exec tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 1 --num-tasks 5
