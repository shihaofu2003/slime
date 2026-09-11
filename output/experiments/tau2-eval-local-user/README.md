# tau2-eval-local-user

Purpose: verify tau2-bench official eval with local sglang services for both
the agent and user simulator, then run full Pass@4 evaluations.

## Results

Full test split, 100 tasks (airline 20 / retail 40 / telecom 40), 4 trials per
task. User simulator is the default local sglang-served
`Qwen3-4B-Instruct-2507`.

| agent model | tasks/sims | pass^1 | pass^4 | pass@1 | pass@4 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 100/400 | 38.25% | 19.00% | 40.00% | 54.00% |
| Qwen3-4B-Instruct-2507-SFT | 100/400 | 24.50% | 13.00% | 25.00% | 43.00% |
| Qwen3-4B-Instruct-2507 | 100/400 | 18.00% | 4.00% | 16.00% | 35.00% |
| Qwen3-4B-tau2-grpo-v1 | 100/400 | 6.25% | 0.00% | 8.00% | 17.00% |

Metric notes: `pass^k` is tau2 official `pass_hat_k`; `pass@1` is whether
trial 0 succeeds for each task; `pass@4` is whether any of the 4 trials succeeds.
`Qwen3-4B-Instruct-2507-SFT` is the
`Qwen3-4B-tau2-sft-areal-strict-iter0000881` checkpoint evaluated under the
older display name.

## Tasks

- `pt-oebf8eoe` / `fsh-tau2-full-agent-sft-multitool-default-user-0715-150827`
  — full Pass@4 eval for
  `Qwen3-4B-tau2-agent-sft-multitool-iter0002413` with the default local
  `Qwen3-4B-Instruct-2507` user simulator:
  [jobs/fsh-tau2-full-agent-sft-multitool-default-user-0715-150827/run_20260715_150827.log](jobs/fsh-tau2-full-agent-sft-multitool-default-user-0715-150827/run_20260715_150827.log).
- `pt-m18koamq` / `fsh-tau2-local-user-smoke-0714-202024` — two-GPU smoke using
  Qwen3-4B-Instruct-2507 for both agent and user simulator:
  [jobs/fsh-tau2-local-user-smoke-0714-202024/run_20260714_202024.log](jobs/fsh-tau2-local-user-smoke-0714-202024/run_20260714_202024.log).
- `pt-vm6lz3kd` / `fsh-tau2-full-qwen3-4b-instruct-2507-0714-203048` —
  full Pass@4 eval for Qwen3-4B-Instruct-2507 agent with Qwen3-4B-Instruct-2507
  local user simulator. Failed because an existing tau2 `results.json` triggered
  an interactive resume prompt; deleted after rerun submission:
  [jobs/fsh-tau2-full-qwen3-4b-instruct-2507-0714-203048/run_20260714_203048.log](jobs/fsh-tau2-full-qwen3-4b-instruct-2507-0714-203048/run_20260714_203048.log).
- `pt-iaa78xr0` / `fsh-tau2-full-qwen3-4b-tau2-grpo-v1-0714-203048` —
  full Pass@4 eval for Qwen3-4B-tau2-grpo-v1 agent with
  Qwen3-4B-Instruct-2507 local user simulator. Failed because an existing tau2
  `results.json` triggered an interactive resume prompt; deleted after rerun
  submission:
  [jobs/fsh-tau2-full-qwen3-4b-tau2-grpo-v1-0714-203048/run_20260714_203048.log](jobs/fsh-tau2-full-qwen3-4b-tau2-grpo-v1-0714-203048/run_20260714_203048.log).
- `pt-o3rza9sd` / `fsh-tau2-full-qwen3-5-4b-0714-203048` — full Pass@4 eval
  for Qwen3.5-4B agent with Qwen3-4B-Instruct-2507 local user simulator. Failed
  because an existing tau2 `results.json` triggered an interactive resume prompt;
  deleted after rerun submission:
  [jobs/fsh-tau2-full-qwen3-5-4b-0714-203048/run_20260714_203048.log](jobs/fsh-tau2-full-qwen3-5-4b-0714-203048/run_20260714_203048.log).
- `pt-8h8znxq8` / `fsh-tau2-full-qwen3-4b-instruct-2507-rerun-0714-205205` —
  rerun after adding timestamped `SAVE_PREFIX` to avoid tau2 resume prompts:
  [jobs/fsh-tau2-full-qwen3-4b-instruct-2507-rerun-0714-205205/run_20260714_205205.log](jobs/fsh-tau2-full-qwen3-4b-instruct-2507-rerun-0714-205205/run_20260714_205205.log).
- `pt-hp0jde5v` / `fsh-tau2-full-qwen3-4b-tau2-grpo-v1-rerun-0714-205205` —
  rerun after adding timestamped `SAVE_PREFIX` to avoid tau2 resume prompts:
  [jobs/fsh-tau2-full-qwen3-4b-tau2-grpo-v1-rerun-0714-205205/run_20260714_205205.log](jobs/fsh-tau2-full-qwen3-4b-tau2-grpo-v1-rerun-0714-205205/run_20260714_205205.log).
- `pt-exbjemy0` / `fsh-tau2-full-qwen3-5-4b-rerun-0714-205205` — rerun
  after adding timestamped `SAVE_PREFIX` to avoid tau2 resume prompts:
  [jobs/fsh-tau2-full-qwen3-5-4b-rerun-0714-205205/run_20260714_205205.log](jobs/fsh-tau2-full-qwen3-5-4b-rerun-0714-205205/run_20260714_205205.log).
