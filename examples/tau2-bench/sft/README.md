# tau2-bench SFT

Supervised fine-tuning data preparation and run scripts for tau2-bench.

## Data format

slime SFT uses the normal rollout dataset loader with:

```bash
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
--prompt-data <jsonl-or-parquet>
--input-key messages
```

Each row must contain a `messages` field: a list of OpenAI-style chat messages.
`sft_rollout` tokenizes those messages and computes loss only on `assistant`
turns. This tau2 SFT path stores tool calls as text to match the official eval
agent protocol:

```json
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"<tool_call>{\"name\":\"get_user_details\",\"arguments\":{\"user_id\":\"...\"}}</tool_call>\n\n<tool_call>{\"name\":\"get_order_details\",\"arguments\":{\"order_id\":\"...\"}}</tool_call>"},
  {"role":"user","content":"Tool result:\n..."},
  {"role":"user","content":"Tool result:\n..."},
  {"role":"assistant","content":"Plain user-facing reply"}
]}
```

No `tool` role is emitted by the converter. Tool observations become `user`
messages so training matches `examples/tau2-bench/eval/official/sglang_agent.py`.
When one assistant turn has multiple tool calls, they are stored as consecutive
`<tool_call>` blocks in the same assistant message; the following tool results
are stored as consecutive `Tool result:` user messages in the same order.

## Data preparation

Default output:

```bash
/mnt/afs/users/fush/projects/ServiceAgent/datasets/tau2-bench-sft/
```

Run locally or through the normal job helper:

```bash
bash examples/tau2-bench/sft/prepare_data.sh

bash scripts/submit.sh --experiment tau2-sft --name tau2-sft-prepare \
  examples/tau2-bench/sft/prepare_data.sh
```

Outputs include AReaL strict/loose and no-thinking/with-thinking variants. The
default training file is:

```bash
areal_tau2_sft_strict_no_thinking.jsonl
```

`strict` drops samples whose history or target assistant turn mixes user-facing
text with a tool call in the same turn. Multi-tool assistant turns are kept,
matching tau2's official half-duplex orchestrator. `loose` keeps the full source
shape for diagnostics or later experiments.

APIGen-MT-5k is converted too, but it is not used by the default tau2 SFT run.

## User model data

User-model SFT uses a separate converter and output directory because the target
role is different from agent SFT. The simulated user is stored as `assistant`
so slime masks user responses for loss; service-agent text is stored as `user`.
Agent tool calls and agent tool results are not visible to the user model and
are skipped. User-side tool calls are kept as `<tool_call>` blocks, with their
tool observations converted to `Tool result:` inputs.

```bash
bash examples/tau2-bench/sft/prepare_user_data.sh
```

Default output:

```bash
/mnt/afs/users/fush/projects/ServiceAgent/datasets/tau2-bench-user-sft/
```

The converter writes:

```bash
areal_tau2_user_sft_no_thinking.jsonl
apigen_mt_5k_user_sft_no_thinking.jsonl
mixed_user_sft_no_thinking.jsonl
```

`mixed_user_sft_no_thinking.jsonl` is the default user-model training file and
uses both AReaL tau2 and APIGen-MT-5k user turns.

## Training

Smoke:

```bash
bash scripts/submit.sh --experiment tau2-sft --gpus 8 --name tau2-sft-smoke \
  examples/tau2-bench/sft/smoke_qwen3_4b_instruct_2507_sft.sh
```

The smoke script defaults to 8 examples and `global_batch_size=8`, matching the
8-GPU submission shape.

Full AReaL strict no-thinking SFT:

```bash
bash scripts/submit.sh --experiment tau2-sft --gpus 8 --name tau2-sft-areal \
  examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft.sh
```

User-model SFT smoke:

```bash
bash scripts/submit.sh --experiment tau2-user-sft --gpus 8 --name tau2-user-sft-smoke \
  examples/tau2-bench/sft/smoke_qwen3_4b_instruct_2507_user_sft.sh
```

Full user-model SFT:

```bash
bash scripts/submit.sh --experiment tau2-user-sft --gpus 8 --name tau2-user-sft \
  examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_user_sft.sh
```

Loss-mask diagnostic:

```bash
bash scripts/submit.sh --experiment tau2-sft --gpus 1 --name tau2-sft-loss-mask-debug \
  examples/tau2-bench/sft/debug_loss_mask.sh
```

This diagnostic is read-only. It loads the real Qwen3 tokenizer, runs the same
`qwen3` multi-turn mask generator used by SFT, and prints the decoded spans
selected by `loss_mask == 1`.

Important overrides:

- `SFT_DATA_PATH` - alternate JSONL, supports slime row slices like `file.jsonl@[0:4]`.
- `SAVE_DIR` - checkpoint output directory.
- `NUM_GPUS`, `ROLLOUT_BATCH_SIZE`, `GLOBAL_BATCH_SIZE`, `NUM_EPOCH`.
- `USE_WANDB=1` plus `WANDB_KEY` or `WANDB_API_KEY` for WandB logging.
