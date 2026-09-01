# tau2-sft

Purpose: prepare tau2-bench SFT datasets and train Qwen3-4B-Instruct-2507 from
the local torch_dist checkpoint.

Tasks:

- `tau2-sft-prepare` — convert AReaL tau2 SFT and APIGen-MT-5k into slime
  `messages` JSONL files under `../../../../datasets/tau2-bench-sft/`.
- `tau2-sft-smoke-8g` — minimal 8-GPU SFT smoke on the same AReaL strict
  no-thinking slice. `pt-zue67sk0` failed before training because the copied
  smoke wrapper resolved the sibling run script relative to the job log
  directory. `pt-vokkdd47` succeeded; run log:
  [jobs/fsh--tau2-sft-smoke-8g-0713-210343/run_20260713_210343.log](jobs/fsh--tau2-sft-smoke-8g-0713-210343/run_20260713_210343.log).
- `tau2-sft-areal-epoch2-wandb` — full 8-GPU SFT on AReaL strict no-thinking
  data, 2 epochs, W&B enabled, saving to
  `/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_sft_areal_strict_no_thinking_epoch2_20260713`.
  W&B project: `serviceagent-tau2-sft`; group/run name:
  `areal-strict-no-thinking-epoch2-20260713`.
  First submitted as `pt-7b9bylv4`; it failed before training because W&B had
  no API key in the job environment. Run log:
  [jobs/fsh--tau2-sft-areal-epoch2-wandb-0713-212430/run_20260713_212430.log](jobs/fsh--tau2-sft-areal-epoch2-wandb-0713-212430/run_20260713_212430.log).
  Retried with W&B key configured as `pt-dkngjlmy`; run log:
  [jobs/fsh--tau2-sft-areal-epoch2-wandb-retry-0713-214029/run_20260713_214029.log](jobs/fsh--tau2-sft-areal-epoch2-wandb-retry-0713-214029/run_20260713_214029.log).
  `pt-dkngjlmy` failed before training because an empty `WANDB_MODE` was passed
  into the W&B runtime environment. Retried with empty `WANDB_MODE` omitted as
  `pt-kdotb9i9`; this run reached W&B initialization and is running with W&B run
  `tu1zy40e`. Run log:
  [jobs/fsh--tau2-sft-areal-epoch2-wandb-retry2-0713-215146/run_20260713_215146.log](jobs/fsh--tau2-sft-areal-epoch2-wandb-retry2-0713-215146/run_20260713_215146.log).
- `tau2-sft-convert-iter-0000881-hf` — convert SFT Megatron checkpoint
  `iter_0000881` to Hugging Face format under sibling directory
  `iter_0000881_hf`. `pt-bfhqs9mc` succeeded; run log:
  [jobs/fsh--tau2-sft-convert-iter-0000881-hf-0713-234858/run_20260713_234858.log](jobs/fsh--tau2-sft-convert-iter-0000881-hf-0713-234858/run_20260713_234858.log).
- `tau2-sft-loss-mask-debug` — read-only diagnostic for the tau2 strict
  no-thinking SFT file. It loads the real Qwen3-4B-Instruct-2507 tokenizer and
  prints decoded spans selected by `loss_mask == 1` for sampled
  airline/retail/telecom rows. Submitted as `pt-ukqcbuci`; run log:
  [jobs/fsh-tau2-sft-loss-mask-debug-0714-112841/run_20260714_112841.log](jobs/fsh-tau2-sft-loss-mask-debug-0714-112841/run_20260714_112841.log).
- `tau2-agent-sft-multitool-0714` — full 8-GPU agent SFT after aligning tau2
  multi-tool data conversion with the official orchestrator. Uses
  `/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B-Instruct-2507_torch_dist`
  and `datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking.jsonl`.
  Submitted as `pt-jokgkopu`; it failed during training with CUDA OOM after a
  long multi-tool sample produced a 12.4k-token sequence. Run log:
  [jobs/fsh-tau2-agent-sft-multitool-0714-0714-174345/run_20260714_174345.log](jobs/fsh-tau2-agent-sft-multitool-0714-0714-174345/run_20260714_174345.log).
- `tau2-agent-sft-multitool-max8192-0714` — retry of the agent SFT with an
  in-job token-length filter that writes
  `output/datasets/tau2-bench-sft/areal_tau2_sft_strict_no_thinking_max8192.jsonl`
  and trains with `MAX_TOKENS_PER_GPU=4096`. Submitted as `pt-1cf3ger2`; run
  log:
  [jobs/fsh-tau2-agent-sft-multitool-max8192-0714-0714-184308/run_20260714_184308.log](jobs/fsh-tau2-agent-sft-multitool-max8192-0714-0714-184308/run_20260714_184308.log).
