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

## Agent/User boundary v2

The boundary-v2 path is intentionally different from the legacy format above.
`build_agent_boundary_v2.py` uses the shared signed Agent contract, native
top-level `tools`, structured Assistant `tool_calls`, `role=tool` Agent results,
and the `qwen3_full` one-render loss mask. User/device calls and results are
hidden from the Agent view but retained as paired provenance; their following
natural-language User reports remain visible. Rows are never truncated and
must fit 16,384 tokens.

Prepare/review/build and run the two matched arms through the experiment
scripts:

```bash
bash examples/tau2-bench/sft/prepare_agent_boundary_v2.sh
bash examples/tau2-bench/sft/review_agent_boundary_v2.sh
bash examples/tau2-bench/sft/build_agent_boundary_v2.sh
bash examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft_agent_boundary_v2.sh contract-only
bash examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft_agent_boundary_v2.sh contract-boundary
```

The matched signed evaluation selected `contract-boundary` under
turn-aware-rl-v3; `contract-only` remained ineligible because its namespace
probes and attributed terminations were too high. The selected HF checkpoint is
`Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_boundary_20260804/final_hf`.
Its downstream fresh 100-update RL run reached a two-seed mean
pass@1/pass@4(any)/pass^4 of `29.50/56.50/10.50%`, versus
`23.63/43.50/8.50%` for this SFT, and was classified `win`. See the
[SFT experiment](../../../output/experiments/tau2-sft-agent-user-boundary-v2/README.md)
and [long100 experiment](../../../output/experiments/tau2-rl-agent-user-boundary-v2-long100/README.md).

## Strict single-call v1

The independent `tau2-agent-single-call-v1` path serializes every Assistant
batch into one native call per message. Later targets retain earlier calls and
their observed tool results; prefix batches are serialized the same way. The
converter filters each expanded target independently at 16,384 tokens and
keeps the other calls from the same original target.

```bash
bash examples/tau2-bench/sft/build_agent_single_call_v1.sh
bash examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft_agent_single_call_v1.sh smoke
bash examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft_agent_single_call_v1.sh full
bash scripts/convert_tau2_agent_sft_single_call_v1_final_to_hf.sh
```

This path starts from raw `Qwen3-4B-Instruct-2507` and uses its own data and
checkpoint roots. It does not modify the selected boundary-v2 artifact.

## Official-native expanded data

`build_agent_official_native_expanded.py` reads the raw AReaL
`datasets/AReaL-tau2-data/tau2_sft_train.jsonl` twice without loading the full
file into memory. It keeps only dialog-consistent `correct=1, reward=1` rows
from Airline, Retail, and Telecom. It does not rerun trajectories and does not
create Banking data.

The converter replaces the source prompt and tool list with the current Tau2
`LLMAgent.system_prompt` and Agent tool schemas. User/device calls and their
results are hidden; a following natural-language User report remains visible.
Assistant batches are expanded in source order into one call per message and
one target per row, with observed results inserted before later targets. Only
the final Assistant is supervised. Rows longer than 16,384 native Qwen tokens
are dropped rather than truncated.

```bash
bash examples/tau2-bench/sft/build_agent_official_native_expanded.sh
```

The wrapper builds and validates both outputs, and accepts
`AREAL_SFT_PATH`, `TOKENIZER_PATH`, `OUTPUT_PATH`, `SMOKE_OUTPUT_PATH`,
`MAX_TOTAL_TOKENS`, and `SMOKE_SIZE` overrides. Defaults are:

```text
output/experiments/tau2-sft-official-native-expanded/data/agent_official_native_expanded.jsonl
output/experiments/tau2-sft-official-native-expanded/data/agent_official_native_expanded_longest32.jsonl
```

Each row contains only `messages`, runtime-ordered `tools`, and minimal source
metadata. The converter does not inherit the strict-single-v1 prompt, fixed row
count, hashes, signatures, or contract metadata.

## Raw-all control

`build_raw_all_sft.sh` streams the unmodified AReaL JSONL and drops only rows
whose native Qwen rendering of `messages + answer` exceeds 16,384 tokens. It
retains both successful and failed labels and also writes the 32 longest kept
rows for the two-update smoke. `prompt_answer_sft_rollout` masks every
historical Assistant turn and supervises only the top-level `answer`; hidden
`thinking` and `reasoning` fields are not rendered.

```bash
bash examples/tau2-bench/sft/build_raw_all_sft.sh
bash examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft_raw_all_max16384.sh smoke
bash examples/tau2-bench/sft/run_qwen3_4b_instruct_2507_sft_raw_all_max16384.sh full
```

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
text with a tool call in the same turn. Multi-tool assistant turns are currently
kept because the converter and orchestrator accept a call list; the orchestrator
executes that list sequentially, while all three tau2 domain policies require at
most one tool call per assistant turn. `loose` keeps the full source shape for
diagnostics or later experiments.

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

## Data quality audit

The exact max-8192 training file can be reconstructed at source-dialog level, reviewed
with the local-first model judge, and mapped back to a strict filtered JSONL using the
[analysis pipeline](../analysis/README.md). The audit keeps source labels, deterministic
policy/schema facts, model judgments, and final selection decisions as separate evidence.

The official-native expanded file also has a separate turn-only quality filter. It
reviews the existing final supervised Assistant message in each JSONL row with two
independent Qwen3.8-27B `xhigh` passes. Only unanimous `drop` rows are removed;
`review` and disagreements remain in the training output. The output is copied as a
byte-identical subset of the input, with no trajectory reconstruction or target repair:

```bash
python3 examples/tau2-bench/sft/filter_official_native_turns.py prepare
bash scripts/submit.sh --experiment tau2-sft-turn-quality-qwen38-xhigh-v1 \
  --name turn-quality-pilot --gpus 1 \
  examples/tau2-bench/sft/run_qwen38_turn_quality_filter.sh -- --stage pilot
bash scripts/submit.sh --experiment tau2-sft-turn-quality-qwen38-xhigh-v1 \
  --name turn-quality-full-00-07 --gpus 8 \
  examples/tau2-bench/sft/run_qwen38_turn_quality_filter.sh -- \
  --stage full --shard-start 0 --shard-count 8
bash scripts/submit.sh --experiment tau2-sft-turn-quality-qwen38-xhigh-v1 \
  --name turn-quality-full-08-15 --gpus 8 \
  examples/tau2-bench/sft/run_qwen38_turn_quality_filter.sh -- \
  --stage full --shard-start 8 --shard-count 8
python3 examples/tau2-bench/sft/filter_official_native_turns.py finalize
python3 examples/tau2-bench/sft/filter_official_native_turns.py validate
```
