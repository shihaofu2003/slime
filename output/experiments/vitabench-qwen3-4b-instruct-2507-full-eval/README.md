# vitabench-qwen3-4b-instruct-2507-full-eval

Purpose: run the complete Chinese VitaBench evaluation for the raw
Qwen3-4B-Instruct-2507 checkpoint with local user-simulator and evaluator
models.

Name: `vitabench-qwen3-4b-instruct-2507-full-eval`.

## Protocol

- Agent: HF checkpoint
  `/mnt/afs/users/fush/projects/ServiceAgent/models/Qwen3-4B-Instruct-2507`;
  BF16, non-thinking, 262,144-token input limit, 8,192-token output limit, and
  SGLang `qwen25` tool-call parser. The Megatron `torch_dist` checkpoint is not
  used for inference.
- User and evaluator: local Qwen3.6-27B BF16 replicas with Thinking disabled.
- Scope: four Chinese 100-task suites, four trials per task, 1,600 total
  trajectories, seed 300, temperature 0, max 300 steps, and max 10 errors.
- Topology: eight single-GPU H100 services: two Agent, two User, and four
  Evaluator instances; four dynamic workers run 16 trajectories in aggregate.
- The runner first requires a strict 16-trajectory four-suite gate, then runs
  40 fixed 10-task shards and produces validated JSON/CSV summaries plus
  lossless response journals and SFT exports.
- No published four-suite VitaBench score is recorded for this exact Agent;
  the Qwen3.5-4B reference score of 22.0 is not used for comparison.

## Jobs

- `pt-fd8ry5ns` — `fsh-vitabench-qwen3-4b-instruct-2507-full-0803-170949`,
  8×N6lS-80GB, priority normal; submitted for the complete gate-plus-full run:
  [run log](jobs/fsh-vitabench-qwen3-4b-instruct-2507-full-0803-170949/run_20260803_170949.log).
