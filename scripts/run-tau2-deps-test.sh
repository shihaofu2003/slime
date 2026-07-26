#!/usr/bin/env bash
#
# Validate setup_tau2_bench_deps.sh is self-sufficient (no probe retry loops):
# install the deps, build tau2 editable (--no-deps), then run the airline smoke.
# Succeeds iff setup_tau2_bench_deps.sh put everything in place by itself.

set -uo pipefail

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
log(){ echo "[tau2-deps-test] $*"; }

# 1. deps (build backend + runtime + [gym]) -- must run BEFORE setup_tau2_bench.sh
log "=== setup_tau2_bench_deps.sh ==="
bash "${SERVICE_AGENT_ROOT}/setup/setup_tau2_bench_deps.sh"

# 2. tau2 editable build (--no-deps); works now that hatchling + editables are present
log "=== setup_tau2_bench.sh ==="
bash "${SERVICE_AGENT_ROOT}/setup/setup_tau2_bench.sh"

# 3. import sanity (also proves the runtime deps landed)
log "=== tau2 --help ==="
tau2 --help >/dev/null && log "tau2 --help OK"

# 4. real smoke run (makes LLM calls via the .env endpoint)
log "=== tau2 run (airline, 5 tasks) ==="
cd "${SERVICE_AGENT_ROOT}/tau2-bench"
exec tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 1 --num-tasks 5
