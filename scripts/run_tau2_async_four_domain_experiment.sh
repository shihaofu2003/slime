#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TAU2_SERIAL_CLEANUP=1
EXPERIMENT_DIR="${PROJECT_ROOT}/output/experiments/tau2-async-db-count-four-domain"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
VANILLA_RUN="${RUN_STAMP}-vanilla-grpo-train"
PROGRESS_RUN="${RUN_STAMP}-progress-db-count-v1-train"

for recipe in vanilla-grpo progress-db-count-v1; do
  run_name="${RUN_STAMP}-${recipe}-train"
  run_dir="${EXPERIMENT_DIR}/arms/async/${run_name}"
  mkdir -p "${run_dir}"
  echo "Four-domain training: ${run_name}"
  env -u LOAD_DIR -u CKPT_STEP TAU2_RUN_NAME="${run_name}" \
    bash "${PROJECT_ROOT}/scripts/run_tau2_async_four_domain.sh" "${recipe}" train \
    2>&1 | tee "${run_dir}/run.log"
  ray stop --force
done

for label in sft vanilla-grpo progress-db-count-v1; do
  iteration=555
  [[ "${label}" != sft ]] || iteration=4673
  eval_dir="${EXPERIMENT_DIR}/eval/${label}-iter${iteration}"
  mkdir -p "${eval_dir}"
  echo "Four-domain official evaluation: ${label} iter${iteration}"
  TAU2_RUN_NAME="${RUN_STAMP}-${label}-train" RUN_STAMP="${RUN_STAMP}" \
    bash "${PROJECT_ROOT}/scripts/eval_tau2_async_four_domain.sh" "${label}" \
    2>&1 | tee "${eval_dir}/run_${RUN_STAMP}.log"
done

python3 "${PROJECT_ROOT}/examples/tau2-bench/analysis/summarize_tau2_areal_async.py" \
  --four-domain --experiment-dir "${EXPERIMENT_DIR}" \
  --sft-eval "${EXPERIMENT_DIR}/eval/sft-iter4673/seed300_${RUN_STAMP}_summary.json" \
  --vanilla-grpo-eval "${EXPERIMENT_DIR}/eval/vanilla-grpo-iter555/seed300_${RUN_STAMP}_summary.json" \
  --progress-db-count-v1-eval "${EXPERIMENT_DIR}/eval/progress-db-count-v1-iter555/seed300_${RUN_STAMP}_summary.json" \
  --vanilla-grpo-logs "${EXPERIMENT_DIR}/arms/async/${VANILLA_RUN}/run.log" \
  --vanilla-grpo-run-dir "${EXPERIMENT_DIR}/arms/async/${VANILLA_RUN}" \
  --progress-db-count-v1-logs "${EXPERIMENT_DIR}/arms/async/${PROGRESS_RUN}/run.log" \
  --progress-db-count-v1-run-dir "${EXPERIMENT_DIR}/arms/async/${PROGRESS_RUN}"
python3 - "${EXPERIMENT_DIR}/README.md" "${RUN_STAMP}" <<'PYDOC'
import sys
from pathlib import Path
readme = Path(sys.argv[1])
text = readme.read_text().replace(
    "Implementation and smoke are verified; controlled training/evaluation results are pending.",
    "The serial training and official evaluation workflow completed; see the controlled comparison below.")
text += (f"\nFormal run `{sys.argv[2]}` completed both 556-update arms and three official evaluations. "
         "[Controlled comparison](reports/comparison.md), [efficiency](reports/efficiency.json), "
         "[reward curves](plots/reward_curves.png).\n")
readme.write_text(text)
PYDOC
echo "Four-domain training and evaluation completed: ${EXPERIMENT_DIR}/reports/comparison.md"
