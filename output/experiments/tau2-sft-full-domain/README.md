# tau2-sft-full-domain

Purpose: train Qwen3-4B-Instruct-2507 with the complete mixed-domain Agent SFT
dataset formed from the official-native AReaL data and the Banking expert data.

Name: `tau2-sft-full-domain`.

## Data

- Input: `data/agent_banking_mixed_sft.jsonl` (37,392 rows).
- Merge order: AReaL 30,376 rows, then Banking 7,016 rows.
- No deduplication, resampling, or field rewriting.
- Source statistics: `data/agent_banking_mixed_sft.stats.json`.

## Fixed recipe

The recipe matches the prior three-domain official-native SFT: 8 GPUs, batch
size 16, two epochs, cosine learning rate `1e-5` to `1e-6` with 10% warmup,
training seed 1234, `qwen3_full` target-only loss, and a 16,384-token dynamic
batch ceiling. Checkpoints are saved every 400 updates.

## Tasks

- Job `16543` (`full-domain-sft-0909`) — full mixed-domain SFT from the raw
  Qwen3-4B-Instruct-2507 torch-dist checkpoint. [Run log](jobs/16543-full-domain-sft-0909-0909-140305915/run_*.log).

Checkpoint root:
`/mnt/afs/users/fush/projects/ServiceAgent/checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_full_domain_20260909`.
