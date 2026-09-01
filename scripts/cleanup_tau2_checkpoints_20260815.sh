#!/usr/bin/env bash
set -euo pipefail

# Remove completed tau2 distributed checkpoints whose HF conversions are kept.
# Dry-run is the default; pass --execute after reviewing the printed targets.

CHECKPOINT_ROOT="/mnt/afs/users/fush/projects/ServiceAgent/checkpoints"
MODE="${1:---dry-run}"

if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--execute" ]]; then
  echo "usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

targets=()

# These completed RL roots retain their validated iter99/iter199 HF models.
for name in \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_20260809 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_20260812 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_raw_user_maxsteps120_20260812 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v1_matched_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v2_l000_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v2_l010_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v2_raw_user_maxsteps120_airline_20260815 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v2_raw_user_maxsteps120_retail_20260815 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v2_raw_user_maxsteps120_mixed_20260815 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v2_raw_user_maxsteps120_telecom_20260815
do
  root="${CHECKPOINT_ROOT}/${name}"
  [[ -d "${root}" ]] || continue
  while IFS= read -r -d '' path; do
    targets+=("${path}")
  done < <(find "${root}" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
    -regex '.*/iter_[0-9]{7}' -print0)
  [[ ! -f "${root}/latest_checkpointed_iteration.txt" ]] || \
    targets+=("${root}/latest_checkpointed_iteration.txt")
done

# Every expert iteration has a validated copy in its separate *_expert_hf root.
for name in \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_airline_expert_20260806 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_retail_expert_20260806 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_telecom_expert_20260806
do
  path="${CHECKPOINT_ROOT}/${name}"
  [[ ! -d "${path}" ]] || targets+=("${path}")
done

# One-update smoke checkpoints are diagnostic artifacts; their logs remain.
for name in \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_v1_smoke_20260809 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v1_matched_smoke_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v1_matched_det2_smoke_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v1_matched_det3_smoke_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v2_l000_smoke_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v2_l000_det2_smoke_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v2_l000_det3_smoke_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v2_l010_smoke_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v2_l010_det2_smoke_20260814 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_single_call_credit_v2_l010_det3_smoke_20260814
do
  path="${CHECKPOINT_ROOT}/${name}"
  [[ ! -d "${path}" ]] || targets+=("${path}")
done

# Keep the final SFT iter3211 and final_hf used by current RL jobs.
sft_epoch_boundary="${CHECKPOINT_ROOT}/Qwen3-4B-Instruct-2507_tau2_agent_sft_single_call_v1_20260809/iter_0001605"
[[ ! -d "${sft_epoch_boundary}" ]] || targets+=("${sft_epoch_boundary}")

if (( ${#targets[@]} > 0 )); then
  mapfile -d '' -t targets < <(printf '%s\0' "${targets[@]}" | sort -zu)
fi

total_bytes=0
for path in "${targets[@]}"; do
  [[ "${path}" == "${CHECKPOINT_ROOT}/"* ]] || {
    echo "refusing out-of-root target: ${path}" >&2
    exit 1
  }
  [[ ! -L "${path}" ]] || {
    echo "refusing symlink target: ${path}" >&2
    exit 1
  }
  bytes="$(du -sb -- "${path}" | awk '{print $1}')"
  total_bytes=$((total_bytes + bytes))
  printf '%12s  %s\n' "$(numfmt --to=iec-i --suffix=B "${bytes}")" "${path}"
done

printf 'targets=%d reclaimable=%s mode=%s\n' \
  "${#targets[@]}" "$(numfmt --to=iec-i --suffix=B "${total_bytes}")" "${MODE}"

[[ "${MODE}" == "--execute" ]] || exit 0

rm -rf -- "${targets[@]}"
echo "cleanup complete"
