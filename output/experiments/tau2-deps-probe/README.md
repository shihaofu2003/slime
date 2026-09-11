# Experiment: tau2-deps-probe

**Name:** tau2-deps-probe
**Purpose:** Determine which packages the fresh slime cluster image needs to
download to run tau2-bench, by submitting a probe job that installs tau2 with
`--no-deps` (mirroring `setup_tau_bench.sh`) and pip-installs each module that
errors on import, then running a 5-task airline smoke. This defines the
surgical dep set for an eventual `setup_tau2_bench_deps.sh`.

## Result

The image (Python 3.12.3, pip 26.1.2) already ships most of tau2's declared
deps. Missing packages, by role:

- **Build backend** (needed because `setup_editable.sh` uses
  `--no-build-isolation`): `hatchling`, `editables`
- **Runtime:** `toml`, `deepdiff`, `litellm`
- **`[gym]` extra:** `gymnasium`
- **`[knowledge]` BM25:** `rank-bm25==0.2.2`

`litellm` is the same dep `setup_tau_bench_deps.sh` adds for tau1. The airline
smoke completed: 5/5 simulations, Average Reward 0.6000.

Consolidated into `setup/setup_tau2_bench_deps.sh` (idempotent, Tsinghua mirror),
which installs the selected dependencies and verifies `torch.utils.tensorboard` still imports --
clearing the tau1 protobuf concern (litellm does not break tensorboard here).

## Task

### 1. tau2 deps-probe + airline smoke

`scripts/run-tau2-deps-probe.sh` — prints env, runs `setup/setup_tau2_bench.sh`
inside a retry loop that installs each missing build-backend module, discovers
runtime deps by repeatedly running `tau2 --help` and pip-installing each
`ModuleNotFoundError`, installs `gymnasium` (`[gym]`), then runs
`tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 1 --num-tasks 5`.

- Latest run (SUCCEEDED, reward 0.60): [jobs/fsh--run-tau2-deps-probe-0711-144232/run_20260711_144232.log](jobs/fsh--run-tau2-deps-probe-0711-144232/run_20260711_144232.log)
- Earlier failed probes (probe script iterated -- parser/hatchling handling): [jobs/fsh--run-tau2-deps-probe-0711-143621/](jobs/fsh--run-tau2-deps-probe-0711-143621/), [jobs/fsh--run-tau2-deps-probe-0711-143936/](jobs/fsh--run-tau2-deps-probe-0711-143936/)

### 2. Validate setup_tau2_bench_deps.sh (self-sufficient)

`scripts/run-tau2-deps-test.sh` -- calls `setup/setup_tau2_bench_deps.sh` then
`setup/setup_tau2_bench.sh`, then `tau2 --help` and the airline smoke. Proves the
deps script alone (no probe retry loops) makes tau2 build and run.

- Latest run (SUCCEEDED, reward 0.60): [jobs/fsh--run-tau2-deps-test-0711-145008/run_20260711_145008.log](jobs/fsh--run-tau2-deps-test-0711-145008/run_20260711_145008.log)

### 3. banking_knowledge BM25 + grep smoke

Uses the existing `.env` API endpoint for one `banking_knowledge` task and one
trial with `--retrieval-config bm25_grep`; verifies `rank-bm25==0.2.2` before
the simulation.

- Job 11800 (SUCCEEDED, reward 1.0, DB match): [jobs/11800-tau2-bm25-grep-smoke-0901-151413243/run_0_20260901_151413243.log](jobs/11800-tau2-bm25-grep-smoke-0901-151413243/run_0_20260901_151413243.log)
