# tau2-hf-convert

Purpose: convert tau2 SFT Megatron torch_dist checkpoints to Hugging Face
directories for sglang evaluation.

## Tasks

- `pt-1i9dl3zk` / `fsh-convert-user-sft-stop-iter-0006899-0715-153727` —
  convert STOP-trained user SFT checkpoint `iter_0006899` to
  `iter_0006899_hf`:
  [jobs/fsh-convert-user-sft-stop-iter-0006899-0715-153727/run_20260715_153727.log](jobs/fsh-convert-user-sft-stop-iter-0006899-0715-153727/run_20260715_153727.log).
- `pt-w69uj1rq` / `fsh-convert-user-sft-iter-0006637-0714-211143` — convert
  user SFT checkpoint `iter_0006637` to `iter_0006637_hf`:
  [jobs/fsh-convert-user-sft-iter-0006637-0714-211143/run_20260714_211144.log](jobs/fsh-convert-user-sft-iter-0006637-0714-211143/run_20260714_211144.log).
- `pt-yc7wxtlc` / `fsh-convert-agent-sft-iter-0002413-0714-211143` — convert
  multitool agent SFT checkpoint `iter_0002413` to `iter_0002413_hf`:
  [jobs/fsh-convert-agent-sft-iter-0002413-0714-211143/run_20260714_211143.log](jobs/fsh-convert-agent-sft-iter-0002413-0714-211143/run_20260714_211143.log).
- `pt-qge44j0l` / `fsh-convert-user-sft-stop-v2-iter-0006311-0720-234102` —
  convert aligned v2 STOP-trained user SFT checkpoint `iter_0006311` to
  `iter_0006311_hf` (Qwen3 config, two safetensor shards):
  [jobs/fsh-convert-user-sft-stop-v2-iter-0006311-0720-234102/run_20260720_234102.log](jobs/fsh-convert-user-sft-stop-v2-iter-0006311-0720-234102/run_20260720_234102.log).
