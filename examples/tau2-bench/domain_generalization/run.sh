#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the current worktree}"
STAGE="${1:?Usage: run.sh prepare|train|convert|eval [arm] [smoke|full] [seed]}"
ARM="${2:-mixed}"
MODE="${3:-full}"
export SEED="${4:-300}"
export EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-sft-domain-generalization"
BATCH="${BATCH:-20260912}"
ROOT="${EXPERIMENT_DIR}/${BATCH}"
CODE="${PROJECT_ROOT}/examples/tau2-bench/domain_generalization"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SAVE_DIR="${ROOT}/${ARM}/${MODE}/checkpoints"
export MODEL_PATH="${SAVE_DIR}/final_hf"
case "${STAGE}" in
  test)
    python3 "${PROJECT_ROOT}/tests/test_tau2_sft_domain_generalization.py"
    python3 "${PROJECT_ROOT}/tests/test_tau2_official_async_eval.py"
    ;;
  prepare)
    python3 "${CODE}/prepare.py" --output "${ROOT}/data" --tokenize
    ;;
  train)
    export SFT_DATA_PATH="${ROOT}/data/${ARM}.jsonl"
    export NUM_EPOCH=2
    if [[ "${MODE}" == smoke ]]; then
      export SFT_DATA_PATH="${ROOT}/data/smoke.jsonl" NUM_EPOCH=1
    fi
    export TRAIN_ENTRYPOINT=train.py TRAIN_SEED=1234 ROLLOUT_SEED=1234 START_ROLLOUT_ID=0
    export NUM_GPUS=8 GLOBAL_BATCH_SIZE=16 ROLLOUT_BATCH_SIZE=16
    export MAX_TOKENS_PER_GPU=16384 CONTEXT_PARALLEL_SIZE=1 LOSS_MASK_TYPE=qwen3_full TOOL_KEY=tools
    export LR=1e-5 MIN_LR=1e-6
    export SAVE_INTERVAL
    SAVE_INTERVAL="$(python3 -c 'import os; n=sum(1 for _ in open(os.environ["SFT_DATA_PATH"])); print(int(os.environ["NUM_EPOCH"])*(n//16))')"
    mkdir -p "${SAVE_DIR}"
    start="$(date +%s)"
    bash "${PROJECT_ROOT}/examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh"
    echo "training_elapsed_seconds=$(( $(date +%s) - start )) updates=${SAVE_INTERVAL}"
    ;;
  convert)
    iteration="$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
    printf -v iteration_dir 'iter_%07d' "${iteration}"
    export ITER_DIR="${SAVE_DIR}/${iteration_dir}" OUTPUT_DIR="${MODEL_PATH}"
    bash "${PROJECT_ROOT}/scripts/convert_tau2_sft_qwen3_4b_instruct_iter_to_hf.sh"
    ;;
  eval)
    export TAU2_DATA_DIR="${ROOT}/benchmark"
    export MODEL_NAME="domain-generalization-${ARM}-${MODE}"
    if [[ "${ARM}" == raw ]]; then
      export MODEL_PATH=/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B-Instruct-2507
    fi
    export EVAL_LABEL="${BATCH}/${ARM}/${MODE}"
    export OUTPUT_DIR="${ROOT}/${ARM}/${MODE}/eval/seed${SEED}/services"
    export SUMMARY_OUTPUT="${ROOT}/${ARM}/${MODE}/eval/seed${SEED}/summary.json"
    export AGENT_EVAL_MODE=official-native AGENT_IMPLEMENTATION=llm_agent AGENT_PROTOCOL_PROFILE=""
    export SIMULATION_TIMEOUT="" MAX_RETRIES=3
    bash "${PROJECT_ROOT}/examples/tau2-bench/eval/official/models/run_qwen3_4b_qwen36_user_async_four_domain.sh" "${MODE}"
    ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
