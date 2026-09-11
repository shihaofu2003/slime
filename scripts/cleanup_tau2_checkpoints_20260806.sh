#!/usr/bin/env bash
set -euo pipefail

# Audited one-off cleanup for the tau2 checkpoint inventory on 2026-08-06.
# Dry-run is the default. Pass --execute only after reviewing every target.

CHECKPOINT_ROOT="/mnt/afs/users/fush/projects/ServiceAgent/checkpoints"
MODE="${1:---dry-run}"

if [[ "$MODE" != "--dry-run" && "$MODE" != "--execute" ]]; then
  echo "usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

declare -a targets=()

add_entire_root() {
  local name="$1"
  local path="${CHECKPOINT_ROOT}/${name}"
  if [[ -d "$path" ]]; then
    targets+=("$path")
  fi
}

add_native_except() {
  local name="$1"
  shift
  local root="${CHECKPOINT_ROOT}/${name}"
  local child base keep

  [[ -d "$root" ]] || return 0
  while IFS= read -r -d '' child; do
    base="${child##*/}"
    keep=0
    for keep_name in "$@"; do
      if [[ "$base" == "$keep_name" ]]; then
        keep=1
        break
      fi
    done
    (( keep == 1 )) || targets+=("$child")
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
    -regex '.*/iter_[0-9]{7}' -print0)
}

add_stale_marker() {
  local name="$1"
  local path="${CHECKPOINT_ROOT}/${name}/latest_checkpointed_iteration.txt"
  if [[ -f "$path" ]]; then
    targets+=("$path")
  fi
}

# Entire roots: smoke/preflight outputs, empty roots, and superseded failed or
# incomplete RL attempts with no retained HF artifact.
for name in \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_turn_credit_smoke_r3_20260804 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_20260720_172315 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_20260720_172906 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_20260720_174147 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_20260720_185650 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_20260720_192759 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_20260721_061807 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_airline_20260727_112403 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_airline_20260728_062859 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_airline_20260728_132439 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_user_base_20260726_091200 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_user_base_20260726_091711 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_user_v1_20260726_090827 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_user_v1_20260727_044321 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_user_v2_20260726_090752 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_real8g_user_v2_20260727_044308 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_verify8g_verify8g_20260726_075529 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_verify8g_verify8g_20260726_083229 \
  Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_boundary_longest32_smoke_20260804 \
  Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_only_longest32_smoke_20260804 \
  Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_smoke_20260803 \
  Qwen3-4B-Instruct-2507_tau2_sft \
  Qwen3-4B-Instruct-2507_tau2_sft_smoke
do
  add_entire_root "$name"
done

# Retained training state: selected long100 iter99, selected boundary SFT final,
# and the final checkpoints of the current v1/v2 User models.
add_native_except Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804 iter_0000099
add_native_except Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_boundary_20260804 iter_0002413
add_native_except Qwen3-4B-Instruct-2507_tau2_user_sft_stop iter_0006899
add_native_except Qwen3-4B-Instruct-2507_tau2_user_sft_stop_v2 iter_0006311

# Completed/rejected historical lineages retain their converted HF artifacts,
# but no distributed optimizer checkpoint. Their stale latest markers go too.
for name in \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_20260804 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long200_20260805 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_local_first_dependency_safe_k2_fieldreward_lr2e6_20260803 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_stability_k2_fieldreward_lr2e6_20260729_065304 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_stability_k2_fieldreward_lr3e6_20260729_064901 \
  Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_stability_k2_fieldreward_lr5e6_20260729_064904 \
  Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_only_20260804 \
  Qwen3-4B-Instruct-2507_tau2_agent_sft_local_first_relaxed_20260803 \
  Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714 \
  Qwen3-4B-Instruct-2507_tau2_sft_areal_strict_no_thinking_epoch2_20260713 \
  Qwen3-4B-Instruct-2507_tau2_user_sft
do
  add_native_except "$name"
  add_stale_marker "$name"
done

# Do not touch base-model torch-dist roots, isolated HF curve/control roots, or
# any of the three 2026-08-06 domain-expert training/HF roots.

if (( ${#targets[@]} > 0 )); then
  mapfile -d '' -t targets < <(printf '%s\0' "${targets[@]}" | sort -zu)
fi

for path in "${targets[@]}"; do
  [[ "$path" == "${CHECKPOINT_ROOT}/"* ]] || {
    echo "refusing out-of-root target: $path" >&2
    exit 1
  }
  [[ ! -L "$path" ]] || {
    echo "refusing symlink target: $path" >&2
    exit 1
  }
  [[ -e "$path" ]] || {
    echo "target disappeared before execution: $path" >&2
    exit 1
  }
done

total_bytes=0
for path in "${targets[@]}"; do
  bytes="$(du -sb -- "$path" | awk '{print $1}')"
  total_bytes=$((total_bytes + bytes))
  printf '%12s  %s\n' "$(numfmt --to=iec-i --suffix=B "$bytes")" "$path"
done

printf 'targets=%d reclaimable=%s mode=%s\n' \
  "${#targets[@]}" "$(numfmt --to=iec-i --suffix=B "$total_bytes")" "$MODE"

if [[ "$MODE" == "--dry-run" ]]; then
  exit 0
fi

rm -rf -- "${targets[@]}"

for path in "${targets[@]}"; do
  [[ ! -e "$path" ]] || {
    echo "failed to remove: $path" >&2
    exit 1
  }
done

echo "cleanup complete"
