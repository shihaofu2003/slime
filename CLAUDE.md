## Project Overview

- **Goal:** RL post-training of LLM agents in agentic / tool-use environments. The
  framework is general; **tau-bench** (retail/airline) is the original test bed.
  **Next: full post-training stack on tau2-bench** — SFT, RFT, RL, and OPD
  (tau2 eval + a trained user simulator are already in place).
- **Framework:** slime 0.3.0, an RL post-training codebase on **Megatron-LM + Ray +
  sglang** (`requirements.txt`); entry points are `train.py` / `train_async.py`.
- **RL algorithms:** GRPO, PPO, REINFORCE, OPD, agentic-RL and more (currently testing
  with GRPO).
- **Models:** base is the Qwen3-4B family — Qwen3-4B / Qwen3-4B-Instruct-2507
  (non-thinking) / Qwen3.5-4B (thinking); other LLMs supported via
  `scripts/models/`. Trained checkpoints (HF, under `../checkpoints/`):
  - *Selected Agent SFT* — `Qwen3-4B-Instruct-2507_tau2_agent_sft_boundary_v2_contract_boundary_20260804/final_hf`
    (signed Agent-owned Contract + boundary anchors; turn-aware-rl-v3 init).
  - *Selected Agent RL* — `Qwen3-4B-Instruct-2507_tau2_agent_rl_boundary_v2_long100_20260804/iter_0000099_hf`
    (three-domain turn-credit-v1 GRPO, `2e-6`, K2 KL `0.01`, 100 updates;
    final two-seed decision `win`).
  - *Historical Agent SFT/RL* — `Qwen3-4B-Instruct-2507_tau2_agent_sft_multitool_max8192_20260714/iter_0002413_hf`
    and `Qwen3-4B-Instruct-2507_tau2_agent_rl_grpo_real8g_stability_k2_fieldreward_lr2e6_20260729_065304/iter_0000099_hf`
    (older parser-on/stability controls, not the current selected lineage).
  - *User simulator* — `Qwen3-4B-Instruct-2507_tau2_user_sft_stop/iter_0006899_hf`
    (v1, usable) and `..._tau2_user_sft_stop_v2/iter_0006311_hf` (v2, trained).
- **Current phase:** tau2-bench **GRPO RL** (`examples/tau2-bench/rl/`); SFT,
  user-SFT, and official eval are all in place. **Pipeline fixed end-to-end.**
  OOM root cause: a single over-cap trajectory forms its own oversized
  microbatch whose `[T,V]` logits OOM the log-prob forward (the old
  `_limit_training_context` also broke on-policy by dropping leading messages).
  Fix: full-conversation verbatim tokenization (on-policy), per-trajectory cap
  `TAU2_RL_MAX_TRAIN_TOKENS=16384` (== `max_tokens_per_gpu`), per-sample resample
  of over-long trajectories (`TAU2_RL_MAX_ROLLOUT_RETRIES=2`), and a
  `filters.drop_zero_std_or_unsampleable` group filter; a `test_rollout_logic.py`
  preflight gates every run. The first stability studies were **airline-only**
  (added `--domain` to `prepare_rl_data.py`, `TAU2_RL_DOMAIN` to the run script):
  AReaL's 1.7B recipe
  is airline-only, and the 3-domain mix is dominated by hard telecom failures.
  The first airline run at the AReaL-aligned `lr=1e-5` + `low_var_kl`
  **collapsed at step 49–71** (KL→0.42, reward→0, truncation→60%, backward NaN).
  The **stability sweep** (`output/experiments/tau2-rl-stability-k2-fieldreward-lr-sweep/`)
  replaced it with polynomial `k2` KL (`kl_loss_coef=0.01`, `entropy_coef=0`),
  field-level reward credit (tool-name / argument / DB / env / communication),
  explicit behavior penalties (malformed JSON, nonexistent tools, repetition,
  `max_steps`), and shaped-zero-std dynamic group replacement. All three LR arms
  (`2e-6` / `3e-6` / `5e-6`, jobs `pt-wcd6eguo` / `pt-g5oxa4ib` / `pt-i4h2m1rm`)
  completed 100 updates with no NaN. **`k2` KL fixes the numerical failure;
  behavioral stability still needs the small LR** — `2e-6` is stable, `5e-6`
  degenerates (repeated calls 0.14→2.56/traj).
  **Held-out official eval** (`tau2-rl-stability-k2-fieldreward-final-eval`,
  iter99, same v1 User, 100 tasks × 4 trials, seed 300, temp 0.6), RL vs SFT:

  | Agent | pass@1/pass^1 | pass@4(any) | pass^4 |
  |---|---:|---:|---:|
  | Instruct (raw, no SFT) | 23.50% | 45.00% | 6.00% |
  | multitool SFT | 23.25% | 46.00% | 8.00% |
  | RL 2e-6 iter99 | 25.50% | 49.00% | 7.00% |
  | RL 3e-6 iter99 | 23.50% | 47.00% | 5.00% |
  | RL 5e-6 iter99 | 18.50% | 39.00% | 3.00% |

  Raw Instruct (no SFT, same v1 User, from `tau2-eval-user-stop-parser`) is
  already pass@1 23.5% — SFT barely moves single-shot success (its gain is
  consistency, pass^4 6→8pp); `2e-6` lifts pass@1 +2pp over raw but its pass^4
  falls back to 7pp. `2e-6` is the best arm but still **does not establish
  RL > SFT**: pass@1 +2.25pp with 95% CI `[-2.84, +7.34]` (includes 0), pass^4
  −1pp. Metrics are monotonic
  in LR under the same User → the cause is **RL optimization instability /
  high-LR degeneration, not the user SFT model**. Known bottleneck: binary task
  reward is sparse (field credit 0.72–0.74 tool-name vs 0.26 DB success).
  **Trajectory analysis** (`output/experiments/tau2-traj-pattern/`) confirms
  failures are action/db mismatch (80–93%) — a capability gap, not process;
  SFT→RL raised multitool batch rate (0.3%→24.1%) but barely moved
  golden-action accuracy (46.7%→49.9%), since field credit rewards "call more
  tools" over "call the right one".

  The longer-training hypothesis is now resolved by the signed three-domain
  Contract + boundary line. Its first turn-aware Pilot A was healthy and improved
  namespace/capability at iter9, but the historical `ITER9_SAFETY_GATE.json`
  stopped on framework truncation, malformed, execution-error, and `max_steps`
  rates. Those are behavior/termination diagnostics, not evidence of OOM,
  non-finite training, corrupt resume state, quota drift, or turn-credit
  misalignment. The follow-up
  `tau2-rl-agent-user-boundary-v2-long100` therefore preserved that gate as
  `fail`, restarted from the selected SFT with a fresh optimizer/checkpoint root,
  and used health-only iter9/iter19 continuation gates while retaining LR
  `2e-6`, K2 `0.01`, K=8, reward/penalties, `max_steps=60`, and the v1 User.
  All health gates passed and iter99 completed. Strict same-protocol results:

  | Agent | two-seed pass@1 | pass@4(any) | pass^4 |
  |---|---:|---:|---:|
  | selected Contract + boundary SFT | 23.63% | 43.50% | 8.50% |
  | historical boundary RL iter9 | 25.25% | 50.00% | 9.00% |
  | **boundary RL long100 iter99** | **29.50%** | **56.50%** | **10.50%** |

  Iter99 improves over SFT by `+5.88/+13.00/+2.00pp` and iter9 by
  `+4.25/+6.50/+1.50pp`; every domain's mean pass@1 improves. Both seeds strictly
  improve pass@1 and pass@4(any) over both controls, so the predeclared final
  decision is **`win`**. Seed 300 has wider paired uncertainty and all overall
  pass^4 CIs include zero, so this is not a claim that every delta is independently
  significant. Full problem/solution evidence is in `output/doc/EXP_QA.md` and
  `output/experiments/tau2-rl-agent-user-boundary-v2-long100/README.md`. Stronger
  credit assignment remains a separate next option. tau1 (retail) GRPO is also
  verified end-to-end, `kl_coef=0`. Jobs use `scripts/submit.sh` on SCO ACP with
  `--priority normal` (HIGH is blocked by `tjNotAllowCreatePrioritJob`).

## Key Directories

Paths are relative to the repo root (`slime/`).

- **`../`** (`/mnt/afs/users/fush/projects/ServiceAgent`) — workspace root, mounted into
  every job container via `/mnt/afs`. Holds `slime/` (this repo), `tau-bench/` and
  `tau2-bench/` (env source, editable-installed), `models/`, `datasets/`, `checkpoints/`,
  and `setup/` (env setup scripts run before each job).
- **`scripts/`** — job launch & run configs: `submit.sh` (cluster submission),
  per-model training scripts `run-*.sh` (e.g. `run-qwen3-4B-serviceagent.sh`), model-arg
  configs `scripts/models/*.sh`, and `convert_qwen3_4b_to_torch_dist.sh`.
- **`examples/tau-bench/`** — tau-bench integration: `run_qwen3_4B.sh` (upstream training
  reference), `generate_with_tau.py` (custom rollout fn), `trainable_agents.py`,
  `openai_tool_adapter.py`, `sglang_tool_parser.py`, `tau1_mock.py` (mock data).
- **`output/`** — job artifacts. `experiments/<exp>/jobs/<job-name>/` holds a named
  experiment's runs (`run_*.log`, `submit_*.log`, copied script); `jobs/<job-name>/` holds
  ad-hoc runs (no `--experiment`); `doc/` holds `INDEX.md` (see Conventions).

## Commands

All `sco` commands need `--workspace-name project-one`.

**Submit a job**
```bash
bash scripts/submit.sh [--experiment <name>] [--gpus <n>] [--name <name>] [--spot] [--tail] [--dry-run] <script>
```
- `--experiment <name>` — experiment name; groups this run's logs under `output/experiments/<name>/jobs` (see Conventions).
- `--gpus <n>` — GPUs per job (default 1; 8 uses the `8.64c1024g` worker spec).
- `<script>` — run script to execute (e.g. `scripts/run-qwen3-4B-serviceagent.sh`).
- `--tail` tails the run log after submit; `--dry-run` prints the command without submitting.

```bash
bash scripts/submit.sh --experiment smoke --gpus 8 scripts/run-qwen3-4B-serviceagent.sh
```

**View jobs**
```bash
sco acp jobs list        --workspace-name project-one --page-size 500   # list all
sco acp jobs describe    <jobid> --workspace-name project-one            # details of one
sco acp jobs stream-logs <jobid> --workspace-name project-one -f         # live logs (-f follow)
```

**Delete a job**
```bash
sco acp jobs delete <jobid> --workspace-name project-one
```

## Conventions

**Experiment logging — do this on every submission.** `submit.sh` auto-writes each job's
logs (`run_*.log`, `submit_*.log`, copied script) to:
- `output/experiments/<experiment>/jobs/<job-name>/` when run with `--experiment <name>`, or
- `output/jobs/<job-name>/` for ad-hoc runs (no `--experiment`).

The logs are automatic, but the docs below are not — when you submit, also maintain:

- **`output/doc/INDEX.md`** — the master index: one entry per experiment (name + purpose
  + link to its README, e.g. `../experiments/<experiment>/README.md`).
- **`output/experiments/<experiment>/README.md`** — opens with the experiment **purpose**
  and **name**, then lists every task in the experiment with its run-log as a *relative*
  link: `jobs/<job-name>/run_*.log` (relative to this README).

**Doc style.** Keep every doc minimal and factual — no filler, no redundancy. No
markdown file may exceed **10 headings**.
