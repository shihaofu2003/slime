#!/usr/bin/env bash
# One-shot status snapshot for the two tau2-eval full jobs. Prints: job state,
# current domain, log size, real-error scan, and completion/result line.
# Usage: bash examples/tau2-bench/eval/legacy/monitor_eval.sh
# No `set -e`: greps legitimately return non-zero on no-match.
set -uo pipefail
JOBGRP="pt-t7u66jtq"
JOBINS="pt-nm2duwk2"
BASE="/mnt/afs/users/fush/projects/ServiceAgent/slime/output/experiments/tau2-eval/jobs"
GRPO_GLOB="${BASE}/fsh--examples-tau2-bench-run_full_qwen3-4b-tau2-grpo-v1-*/run_*.log"
INST_GLOB="${BASE}/fsh--examples-tau2-bench-run_full_qwen3-4b-instruct-2507-*/run_*.log"

snapshot() {  # $1 = label  $2 = jobid  $3 = log-glob
  local label="$1" jobid="$2" glob="$3"
  local log; log=$(ls -t $glob 2>/dev/null | head -1)
  local state; state=$(sco acp jobs describe "$jobid" --workspace-name project-one 2>/dev/null \
    | sed -n 's/.*"state": *"\([^"]*\)".*/\1/p' | head -1)
  printf '\n=== %s (%s) state=%s ===\n' "$label" "$jobid" "${state:-?}"
  if [ -z "$log" ]; then echo "  (no run log)"; return; fi
  printf '  log: %s (%s lines)\n' "$(basename "$log")" "$(wc -l < "$log")"
  echo "  domains started:"
  grep -E "Evaluating domain=.*tasks=" "$log" 2>/dev/null | sed -E 's/.*INFO //' | sed 's/^/    /'
  echo "  episodes started (Step 1.): $(grep -cE 'orchestrator:step:[0-9]+ - Step 1\.' "$log" 2>/dev/null || echo 0)"
  echo "  gemini user-sim calls (ok): $(grep -cE "model': 'gemini-2.5-flash'" "$log" 2>/dev/null || echo 0)"
  echo "  REAL errors (traceback/ratelimit/auth/badrequest): $(grep -cE 'Traceback|RateLimitError|AuthenticationError|BadRequestError|NotFoundException|ConnectionError' "$log" 2>/dev/null || echo 0)"
  echo "  completion:"
  grep -E "Wrote [0-9]+ results|Overall:|eval complete" "$log" 2>/dev/null | sed 's/^/    /' | tail -6
  [ -z "$(grep -E 'Overall:' "$log" 2>/dev/null)" ] || {
    echo "  per-domain summary:"
    grep -E "(airline|retail|telecom): \{'total'" "$log" 2>/dev/null | sed 's/^/    /'
  }
}

snapshot "grpo-v1"      "$JOBGRP" "$GRPO_GLOB"
snapshot "instruct-2507" "$JOBINS" "$INST_GLOB"
echo
