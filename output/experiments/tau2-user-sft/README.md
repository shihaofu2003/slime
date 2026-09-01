# tau2-user-sft

Purpose: prepare tau2-bench user-simulator SFT data and train
Qwen3-4B-Instruct-2507 as a non-thinking user model from the local torch_dist
checkpoint.

Tasks:

- `tau2-user-sft-data` — convert AReaL tau2 user turns and APIGen-MT-5k human
  turns into slime `messages` JSONL files under
  `../../../../datasets/tau2-bench-user-sft/`. Default mixed data:
  `mixed_user_sft_no_thinking.jsonl`.
- `tau2-user-sft-mixed-0714` — full 8-GPU user-model SFT on the mixed user
  dataset. Uses
  `/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B-Instruct-2507_torch_dist`
  and `datasets/tau2-bench-user-sft/mixed_user_sft_no_thinking.jsonl`.
  Submitted as `pt-an1xhor2`; run log:
  [jobs/fsh-tau2-user-sft-mixed-0714-0714-174346/run_20260714_174346.log](jobs/fsh-tau2-user-sft-mixed-0714-0714-174346/run_20260714_174346.log).
- `tau2-user-sft-data-stop-0715` — regenerate the mixed user SFT data with
  terminal `###STOP###` targets for successful complete AReaL dialogs:
  submitted as `pt-ppwn2ktw`; run log:
  [jobs/fsh-tau2-user-sft-data-stop-0715-0715-121924/run_20260715_121924.log](jobs/fsh-tau2-user-sft-data-stop-0715-0715-121924/run_20260715_121924.log).
- `tau2-user-sft-stop-0715` — retrain the user simulator on the regenerated
  mixed data, saving to
  `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop`:
  submitted as `pt-d0irg31d`; run log:
  [jobs/fsh-tau2-user-sft-stop-0715-0715-122152/run_20260715_122152.log](jobs/fsh-tau2-user-sft-stop-0715-0715-122152/run_20260715_122152.log).
- `pt-y7l06dfs` / `fsh-tau2-user-sft-stop-v2-0720-202433` — retrain the
  aligned v2 user simulator on 8 GPUs. The v2 data removes the custom user-tool
  preamble, uses domain-aligned tau2 guidelines, cleans scenario metadata to
  `reason_for_call`, and drops `content_and_tool` mixed targets. The job
  succeeded and saved final checkpoint `iter_0006311` under
  `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_user_sft_stop_v2`:
  [jobs/fsh-tau2-user-sft-stop-v2-0720-202433/run_20260720_202433.log](jobs/fsh-tau2-user-sft-stop-v2-0720-202433/run_20260720_202433.log).
