#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"
ROLE="${2:-all}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "pilot" ]]; then
  echo "Usage: $0 <smoke|pilot> [teacher|student|all]" >&2
  exit 2
fi
if [[ "${ROLE}" != "teacher" && "${ROLE}" != "student" && "${ROLE}" != "all" ]]; then
  echo "Usage: $0 <smoke|pilot> [teacher|student|all]" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SERVICE_AGENT_ROOT="${SERVICE_AGENT_ROOT:-$(dirname "${PROJECT_ROOT}")}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tau2-opd-airline-retail-pilot}"
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/${EXPERIMENT_NAME}"
# The corrected recipe starts separately from the historical pilot/v2/v3 runs.
# Override the tag for each new ablation; reusing a tag resumes that run.
TAU2_OPD_RUN_TAG="${TAU2_OPD_RUN_TAG-current-bounded-lr2e6}"
RUN_TAG="${TAU2_OPD_RUN_TAG:+-${TAU2_OPD_RUN_TAG}}"
ARTIFACT_DIR="${EXPERIMENT_DIR}/${MODE}${RUN_TAG}"
# Teachers are a separate TP2 service per domain, shared by student runs.
TEACHER_ENDPOINT_FILE="${TEACHER_ENDPOINT_FILE:-${EXPERIMENT_DIR}/teacher_endpoints.env}"
TEACHER_STOP_FILE="${TEACHER_STOP_FILE:-${EXPERIMENT_DIR}/teacher.stop}"
TAU2_OPD_TEACHER_PERSISTENT="${TAU2_OPD_TEACHER_PERSISTENT:-1}"

mkdir -p "${ARTIFACT_DIR}/logs" "${ARTIFACT_DIR}/data" "${ARTIFACT_DIR}/checkpoints"

AIRLINE_EXPERT_MODEL="${AIRLINE_EXPERT_MODEL:-${EXPERIMENT_DIR}/../tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-airline-train/checkpoints/iter_0000029_hf}"
RETAIL_EXPERT_MODEL="${RETAIL_EXPERT_MODEL:-${EXPERIMENT_DIR}/../tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-retail-train/checkpoints/iter_0000009_hf}"
TELECOM_EXPERT_MODEL="${TELECOM_EXPERT_MODEL:-${EXPERIMENT_DIR}/../tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-telecom-train/checkpoints/iter_0000009_hf}"
BANKING_EXPERT_MODEL="${BANKING_EXPERT_MODEL:-${EXPERIMENT_DIR}/../tau2-domain-experts-sft4505-b128/arms/async/20260920_sft4505-banking-train/checkpoints/iter_0000009_hf}"
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${SERVICE_AGENT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${SFT_CKPT_ROOT}/iter_0004505_hf}"
REF_LOAD="${REF_LOAD:-${SFT_CKPT_ROOT}}"
REF_CKPT_STEP="${REF_CKPT_STEP:-4505}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/output/experiments/tau2-domain-experts-sft4505-b128/data}"
TAU2_OPD_DOMAINS="${TAU2_OPD_DOMAINS:-airline,retail}"
case "${TAU2_OPD_DOMAINS}" in
  airline,retail|airline,retail,telecom,banking) ;;
  *) echo "[ERROR] TAU2_OPD_DOMAINS must be airline,retail or airline,retail,telecom,banking" >&2; exit 2 ;;
esac
IFS=',' read -r -a OPD_DOMAINS <<<"${TAU2_OPD_DOMAINS}"
PREPARED_RL_DATA="${PREPARED_RL_DATA:-${ARTIFACT_DIR}/data/${TAU2_OPD_DOMAINS//,/_}_train.jsonl}"
TAU2_USER_ENDPOINT_FILE="${TAU2_USER_ENDPOINT_FILE:-${EXPERIMENT_DIR}/user_endpoint.env}"
TAU2_USER_API_BASE="${TAU2_USER_API_BASE:-}"
TAU2_USER_API_KEY="${TAU2_USER_API_KEY:-EMPTY}"

declare -A EXPERT_MODELS=(
  [airline]="${AIRLINE_EXPERT_MODEL}" [retail]="${RETAIL_EXPERT_MODEL}"
  [telecom]="${TELECOM_EXPERT_MODEL}" [banking]="${BANKING_EXPERT_MODEL}"
)
declare -A EXPERT_PORTS=(
  [airline]="${AIRLINE_PORT:-31001}" [retail]="${RETAIL_PORT:-31002}"
  [telecom]="${TELECOM_PORT:-31003}" [banking]="${BANKING_PORT:-31004}"
)
DATA_ARGS=()
REQUIRED_INPUTS=("${HF_CHECKPOINT}" "${REF_LOAD}")
for domain in "${OPD_DOMAINS[@]}"; do
  REQUIRED_INPUTS+=("${EXPERT_MODELS[${domain}]}" "${DATA_DIR}/${domain}_train.jsonl")
  DATA_ARGS+=("--${domain}" "${DATA_DIR}/${domain}_train.jsonl")
done
for required in "${REQUIRED_INPUTS[@]}"; do
  [[ -e "${required}" ]] || { echo "[ERROR] Missing OPD input: ${required}" >&2; exit 1; }
done
EXPERT_MEM_FRACTION="${EXPERT_MEM_FRACTION:-0.75}"
EXPERT_CONTEXT_LENGTH="${EXPERT_CONTEXT_LENGTH:-32768}"
EXPERT_MAX_TOTAL_TOKENS="${EXPERT_MAX_TOTAL_TOKENS:-32768}"
EXPERT_MAX_RUNNING_REQUESTS="${EXPERT_MAX_RUNNING_REQUESTS:-1}"
EXPERT_PIDS=()

# The teacher job uses four GPUs for two domains or eight GPUs for four domains.
# The student/Ray process has its own eight-GPU job and reads the AFS endpoint file.
if [[ "${ROLE}" == "teacher" ]]; then
  HOST="${HOST:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
  [[ -n "${HOST}" ]] || { echo "[ERROR] Cannot determine routable teacher host" >&2; exit 1; }
else
  HOST="${HOST:-127.0.0.1}"
fi

cleanup() {
  trap - EXIT TERM INT
  for pid in "${EXPERT_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${EXPERT_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT TERM INT

stop_teacher() {
  if [[ "${ROLE}" == "student" && "${TAU2_OPD_TEACHER_PERSISTENT}" != "1" ]]; then
    touch "${TEACHER_STOP_FILE}"
  fi
  cleanup
}
trap stop_teacher EXIT TERM INT

start_expert() {
  local label="$1" model="$2" port="$3" devices="$4" log_path="$5"
  CUDA_VISIBLE_DEVICES="${devices}" python3 -m sglang.launch_server \
    --model-path "${model}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --tp 2 \
    --dtype bfloat16 \
    --mem-fraction-static "${EXPERT_MEM_FRACTION}" \
    --context-length "${EXPERT_CONTEXT_LENGTH}" \
    --max-total-tokens "${EXPERT_MAX_TOTAL_TOKENS}" \
    --max-running-requests "${EXPERT_MAX_RUNNING_REQUESTS}" \
    >"${log_path}" 2>&1 &
  EXPERT_PIDS+=("$!")
  echo "[OPD] started ${label} expert pid=${EXPERT_PIDS[-1]} port=${port} devices=${devices}"
}

wait_for_expert() {
  local label="$1" pid="$2" port="$3" log_path="$4"
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] ${label} expert exited before readiness" >&2
      tail -n 100 "${log_path}" >&2 || true
      exit 1
    fi
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[OPD] ${label} expert ready at http://${HOST}:${port}"
      return
    fi
    sleep 5
  done
  echo "[ERROR] ${label} expert did not become ready" >&2
  tail -n 100 "${log_path}" >&2 || true
  exit 1
}

teacher_preflight() {
  local domain
  local endpoints=()
  for domain in "${OPD_DOMAINS[@]}"; do
    endpoints+=("${domain}" "${EXPERT_PORTS[${domain}]}")
  done
  python3 - "${endpoints[@]}" <<'PY'
import json
import sys
import urllib.request

endpoints = sys.argv[1:]
payload = json.dumps({
    "input_ids": [1, 2, 3],
    "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
    "return_logprob": True,
    "logprob_start_len": 0,
}).encode()
for label, port in zip(endpoints[::2], endpoints[1::2]):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.load(response)
    values = body.get("meta_info", {}).get("input_token_logprobs")
    if not isinstance(values, list) or len(values) != 3:
        raise SystemExit(f"{label} teacher preflight returned invalid input_token_logprobs: {values!r}")
    if not isinstance(values[0], (list, tuple)) or values[0][0] is not None:
        raise SystemExit(f"{label} teacher preflight missing first-token placeholder: {values!r}")
    print(f"OPD_TEACHER_PREFLIGHT label={label} tokens={len(values)}")
PY
}

start_experts() {
  local index domain devices
  for index in "${!OPD_DOMAINS[@]}"; do
    domain="${OPD_DOMAINS[${index}]}"
    devices="$((index * 2)),$((index * 2 + 1))"
    start_expert "${domain}" "${EXPERT_MODELS[${domain}]}" "${EXPERT_PORTS[${domain}]}" \
      "${devices}" "${ARTIFACT_DIR}/logs/${domain}_expert.log"
  done
  for index in "${!OPD_DOMAINS[@]}"; do
    domain="${OPD_DOMAINS[${index}]}"
    wait_for_expert "${domain}" "${EXPERT_PIDS[${index}]}" "${EXPERT_PORTS[${domain}]}" \
      "${ARTIFACT_DIR}/logs/${domain}_expert.log"
  done
  teacher_preflight
}

run_teacher() {
  rm -f "${TEACHER_ENDPOINT_FILE}" "${TEACHER_STOP_FILE}"
  start_experts
  local domain
  local tmp_file="${TEACHER_ENDPOINT_FILE}.tmp.$$"
  for domain in "${OPD_DOMAINS[@]}"; do
    printf 'TAU2_OPD_%s_URL=http://%s:%s/generate\n' "${domain^^}" "${HOST}" "${EXPERT_PORTS[${domain}]}"
  done >"${tmp_file}"
  cat >>"${tmp_file}" <<EOF
TAU2_OPD_TEACHER_HOST=${HOST}
TAU2_OPD_TEACHER_READY=1
EOF
  mv -f "${tmp_file}" "${TEACHER_ENDPOINT_FILE}"
  echo "[OPD] teacher endpoints published to ${TEACHER_ENDPOINT_FILE}"
  if [[ "${TAU2_OPD_TEACHER_PERSISTENT}" == "1" ]]; then
    echo "[OPD] teacher service is persistent; stop the teacher job explicitly when no longer needed"
    while :; do
      sleep 30
    done
  fi
  for _ in $(seq 1 2880); do
    [[ -e "${TEACHER_STOP_FILE}" ]] && { echo "[OPD] student stop marker found"; return; }
    sleep 10
  done
  echo "[ERROR] teacher wait timed out without student stop marker" >&2
  return 1
}

run_student_setup() {
  python3 "${PROJECT_ROOT}/examples/tau2-bench/opd/prepare_airline_retail_data.py" \
    "${DATA_ARGS[@]}" --output "${PREPARED_RL_DATA}"

  python3 -m pytest "${PROJECT_ROOT}/examples/tau2-bench/opd/test_tau2_opd.py" -q

  local domain key url ready=0
  for domain in "${OPD_DOMAINS[@]}"; do
    unset "TAU2_OPD_${domain^^}_URL"
  done
  for _ in $(seq 1 360); do
    if [[ -s "${TEACHER_ENDPOINT_FILE}" ]]; then
      # shellcheck disable=SC1090
      source "${TEACHER_ENDPOINT_FILE}"
      ready=1
      for domain in "${OPD_DOMAINS[@]}"; do
        key="TAU2_OPD_${domain^^}_URL"
        url="${!key:-}"
        if [[ -z "${url}" ]] || ! curl -fsS --max-time 10 "${url%/generate}/health" >/dev/null 2>&1; then
          ready=0
          break
        fi
        export "${key}"
      done
      [[ "${ready}" == 1 ]] && break
    fi
    sleep 5
  done
  [[ "${ready}" == 1 ]] || {
    echo "[ERROR] teacher endpoints are not ready for ${TAU2_OPD_DOMAINS}: ${TEACHER_ENDPOINT_FILE}" >&2
    exit 1
  }
  if [[ -z "${TAU2_USER_API_BASE}" ]]; then
    for _ in $(seq 1 360); do
      if [[ -s "${TAU2_USER_ENDPOINT_FILE}" ]]; then
        # shellcheck disable=SC1090
        source "${TAU2_USER_ENDPOINT_FILE}"
        break
      fi
      sleep 5
    done
  fi
  if [[ -z "${TAU2_USER_API_BASE}" ]]; then
    echo "[ERROR] User endpoint file not found; submit scripts/serve_tau2_user_pool.sh as a 4-GPU cluster job or set TAU2_USER_API_BASE: ${TAU2_USER_ENDPOINT_FILE}" >&2
    exit 1
  fi
  if ! curl -fsS --max-time 10 "${TAU2_USER_API_BASE}/models" >/dev/null; then
    echo "[ERROR] External User service is not reachable: ${TAU2_USER_API_BASE}" >&2
    exit 1
  fi
}

if [[ "${ROLE}" == "teacher" ]]; then
  run_teacher
  exit 0
fi

if [[ "${ROLE}" == "student" ]]; then
  run_student_setup
else
  # Local/manual use requires separate GPUs for teachers and the student.
  HOST="127.0.0.1"
  start_experts
  python3 "${PROJECT_ROOT}/examples/tau2-bench/opd/prepare_airline_retail_data.py" \
    "${DATA_ARGS[@]}" --output "${PREPARED_RL_DATA}"
  python3 -m pytest "${PROJECT_ROOT}/examples/tau2-bench/opd/test_tau2_opd.py" -q
  if ! curl -fsS --max-time 10 "${TAU2_USER_API_BASE}/models" >/dev/null; then
    echo "[ERROR] External User service is not reachable: ${TAU2_USER_API_BASE}" >&2
    exit 1
  fi
  for domain in "${OPD_DOMAINS[@]}"; do
    export "TAU2_OPD_${domain^^}_URL=http://${HOST}:${EXPERT_PORTS[${domain}]}/generate"
  done
fi

case "${MODE}" in
  smoke) NUM_ROLLOUT=2; SAVE_INTERVAL=1 ;;
  pilot) NUM_ROLLOUT="${TAU2_OPD_NUM_UPDATES:-60}"; SAVE_INTERVAL=10 ;;
esac

export PROJECT_ROOT SERVICE_AGENT_ROOT
export SOURCE_RL_DATA="${PREPARED_RL_DATA}"
export PREPARED_RL_DATA SKIP_PREPARE_RL_DATA=1
export TAU2_SRC="${TAU2_SRC:-${SERVICE_AGENT_ROOT}/tau2-bench/src}"
export TAU2_DATA_DIR="${SERVICE_AGENT_ROOT}/tau2-bench/data"
export SFT_CKPT_ROOT HF_CHECKPOINT REF_LOAD REF_CKPT_STEP
export SAVE_DIR="${ARTIFACT_DIR}/checkpoints" LOAD_DIR="${ARTIFACT_DIR}/checkpoints"
export CKPT_STEP="" START_ROLLOUT_ID=""
export DATA_SOURCE_PATH="random_tasks.ShuffledTaskDataSource"
export TAU2_RL_DOMAIN="" TAU2_RL_DOMAIN_QUOTA=""
export TAU2_OPD_PURE=1
export TAU2_OPD_TEACHER_TIMEOUT="${TAU2_OPD_TEACHER_TIMEOUT:-120}"
export USER_SGLANG=0 TAU2_USER_API_BASE TAU2_USER_API_KEY
export TAU2_USER_MODEL="Qwen3.6-27B-tau2-user-nonthinking"
export TAU2_USER_TEMPERATURE=0.0 TAU2_USER_TOP_P=1.0 TAU2_USER_MAX_TOKENS=512
export TAU2_USER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
export TRAIN_ENTRYPOINT=train_async.py TOTAL_GPUS=8 RAY_GPUS=8
export AGENT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export TAU2_DISJOINT_GPUS=1 ROLLOUT_NUM_GPUS_PER_ENGINE=2
export TAU2_RAW_TOKENS=1 TAU2_RL_RAISE_ERRORS=1
export TAU2_SAMPLING_MODE="${TAU2_SAMPLING_MODE:-async}"
export TAU2_POOL_CAPACITY="${TAU2_POOL_CAPACITY:-32}" TAU2_MAX_PENDING_GROUPS="${TAU2_MAX_PENDING_GROUPS:-64}"
export TAU2_MAX_BUFFERED_GROUPS="${TAU2_MAX_BUFFERED_GROUPS:-32}" TAU2_MAX_POLICY_LAG="${TAU2_MAX_POLICY_LAG:--1}"
export TAU2_ENVIRONMENT_WORKERS="${TAU2_ENVIRONMENT_WORKERS:-4}"
export TAU2_AGENT_CONCURRENCY=48 TAU2_STEP_CONCURRENCY=160 TAU2_AGENT_TIMEOUT=600
export ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=16 NUM_ROLLOUT SAVE_INTERVAL
export AGENT_MAX_TOKENS=1200 ROLLOUT_TEMPERATURE=1.0 ROLLOUT_TOP_P=1.0
export TAU2_MAX_STEPS=200 TAU2_MAX_ERRORS=10
export TAU2_RL_MAX_TRAIN_TOKENS=16384 TAU2_RL_MAX_ROLLOUT_RETRIES=2
export TAU2_AGENT_PROTOCOL_PROFILE=official-native LOSS_MASK_TYPE=qwen3_full
export TAU2_TURN_CREDIT_VERSION="" TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0 TAU2_DROP_UNIFORM_OUTCOME_GROUPS=0
export USE_REWARD_SHAPING=0 KL_LOSS_TYPE=k2 KL_LOSS_COEF=0 KL_COEF=0 ENTROPY_COEF=0
export EPS_CLIP=0.2 EPS_CLIP_HIGH=0.2 LR="${TAU2_OPD_LR:-2e-6}"
# OPD_KL_COEF scales the per-token reverse-KL advantage. Under Adam a global
# advantage scale cancels in m_hat/sqrt(v_hat), so this knob is nearly a no-op
# while OPD is the only loss term; it becomes meaningful only if a task-reward
# term is mixed in later. The real step-size lever is LR above.
export OPD_KL_COEF="${TAU2_OPD_KL_COEF:-1.0}"
# Current-student advantage; behavior probabilities remain the TIS denominator.
# Opt in only to reproduce the historical v3 surrogate.
export OPD_USE_BEHAVIOR_LOGPROBS="${TAU2_OPD_USE_BEHAVIOR_LOGPROBS:-0}"
export OPD_POST_UPDATE_LOG_INTERVAL="${OPD_POST_UPDATE_LOG_INTERVAL:-10}"
export TRAIN_SEED="${TRAIN_SEED:-1234}" ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export SGLANG_SERVER_CONCURRENCY=16
export SKIP_PREFLIGHT=0 USE_DYNAMIC_FILTER=0 TAU2_RL_CLEANUP=0 TAU2_SERIAL_CLEANUP=1
export USE_WANDB=0 WANDB_MODE=offline
export TAU2_RL_TRAJECTORY_DUMP_PATH="${ARTIFACT_DIR}/trajectories.jsonl"
mkdir -p "${ARTIFACT_DIR}/trajectories"

RUNNER_SNAPSHOT="${ARTIFACT_DIR}/run_runner_$(date +%Y%m%d_%H%M%S).sh"
cp "${PROJECT_ROOT}/examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh" "${RUNNER_SNAPSHOT}"

echo "[OPD] starting ${MODE} training with eight student GPUs (physical GPUs 0-7 in this job)"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" bash "${RUNNER_SNAPSHOT}"
