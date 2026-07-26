#!/bin/bash
#
# Generate tau-bench task-index JSONL files (tau1) for slime training.
#
# Runs examples/tau-bench/tau1_mock.py, which uses the tau_bench package's bundled
# tasks with mock/human providers -- no GPU and no external API needed; it just
# dumps {index, metadata} rows for each env/split (retail: train/test/dev,
# airline: test). The training run consumes retail_train_tasks.jsonl; eval
# consumes retail_dev_tasks.jsonl.
#
# tau_bench must be importable -- submit.sh runs setup/setup_tau_bench.sh first,
# re-pointing the editable install at our /mnt/afs checkout.

set -ex

SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-/mnt/afs/users/fush/projects/ServiceAgent}"
PROJECT_ROOT="${PROJECT_ROOT:-${SERVICE_AGENT_ROOT}/slime}"
DATA_DIR="${TAU_DATA_DIR:-${SERVICE_AGENT_ROOT}/datasets/tau-bench}"

cd "${PROJECT_ROOT}"

# tau_bench.envs imports litellm at module load, which the image lacks. Install
# tau_bench's runtime deps (skips if already importable) before any tau import.
bash "${SERVICE_AGENT_ROOT}/setup/setup_tau_bench_deps.sh"

# Confirm tau_bench resolves to our checkout (find_spec, no heavy import).
verify_resolves_under() {
    local pkg="$1" expected resolved
    expected="$(cd "${2}" && pwd)"
    resolved="$(IMPORT_NAME="${pkg}" python3 -c 'import importlib.util,os; s=importlib.util.find_spec(os.environ["IMPORT_NAME"]); exit(1) if s is None else print(os.path.realpath((s.submodule_search_locations or [os.path.dirname(s.origin)])[0]))' 2>/dev/null)" || resolved=""
    case "${resolved}" in
        "${expected}"|"${expected}"/*) echo "[VERIFY] OK: ${pkg} -> ${resolved}" ;;
        *) echo "[VERIFY] FAIL: ${pkg} -> '${resolved:-<unresolvable>}', expected under '${expected}'" >&2; return 1 ;;
    esac
}

verify_resolves_under tau_bench "${SERVICE_AGENT_ROOT}/tau-bench" \
    || { echo "[VERIFY] tau_bench is not our checkout; run setup/setup_tau_bench.sh first." >&2; exit 1; }

mkdir -p "${DATA_DIR}"

python examples/tau-bench/tau1_mock.py --local_dir "${DATA_DIR}"

echo "[INFO] Generated tau-bench mock data in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"
