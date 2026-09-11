# tau2 Elastic Domain Concurrency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-domain initial trajectory concurrency and completion-triggered slot borrowing to official tau2 evaluation, then switch the eight-GPU production wrapper to two TP1 Agent replicas and three TP2 User replicas.

**Architecture:** tau2's existing batch runner receives an optional semaphore gate and a separate executor size, so all task/trial/seed, retry, checkpoint, and result behavior stays in the official runner. Slime owns the initial `2:2:5` allocation, cross-process semaphores, deterministic redistribution, summary diagnostics, and shell configuration.

**Tech Stack:** Python 3, `concurrent.futures`, `multiprocessing.Manager`, tau2 `run_domain`, Bash, `unittest`/`pytest`.

**Spec:** `docs/superpowers/specs/2026-09-01-tau2-elastic-domain-concurrency-design.md`

## Global Constraints

- Keep the official test workload at Airline/Retail/Telecom `20/40/40` tasks and four trials per task.
- Keep task/trial seed mapping, retry, checkpoint, metrics, and infrastructure-error failure semantics unchanged.
- Initial production concurrency is `airline:2,retail:2,telecom:5`; global active trajectory limit is 9.
- Borrow slots only after a domain Future returns successfully; do not preempt an in-flight trajectory.
- Production GPU topology is Agent TP1 on `0;1` and User TP2 on `2,3;4,5;6,7`.
- Preserve the old uniform `MAX_CONCURRENCY` path when new flags are absent.
- Do not commit unrelated dirty-worktree files; edits stay limited to files listed in the spec.

---

## Task 1: Add the optional tau2 trajectory gate

**Files:**
- Modify: `tests/test_tau2_official_async_eval.py`
- Modify: `../tau2-bench/src/tau2/runner/batch.py`

**Interfaces:**
- Produces: `run_tasks(..., executor_max_workers: int | None = None, slot_semaphore=None)`.
- Produces: `run_domain(..., executor_max_workers: int | None = None, slot_semaphore=None)`.
- Preserves: calls without these keyword arguments use `config.max_concurrency` and do not acquire a semaphore.

- [ ] **Step 1: Write the failing runner test**

Add a `unittest` test using three real mock-domain tasks, a `threading.Semaphore(2)`, and a fake `run_single_task` at the external-LLM boundary. Configure `max_concurrency=1`, pass `executor_max_workers=3`, and use a two-party barrier in the fake simulation function. Assert that the largest simultaneous fake simulations is exactly 2 and all three results return.

The production change caught by this test is either ignoring `executor_max_workers` or failing to acquire/release `slot_semaphore` around a complete trajectory.

- [ ] **Step 2: Verify RED**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/mnt/afs/users/fush/projects/ServiceAgent/tau2-bench/src \
  /mnt/afs/conda/envs/fsh-tau2/bin/python \
  tests/test_tau2_official_async_eval.py
```

Expected: FAIL because `run_tasks` does not accept `executor_max_workers` or `slot_semaphore`.

- [ ] **Step 3: Implement the minimal runner hook**

Extend `run_tasks` and `run_domain` with keyword-only optional arguments. Use:

```python
worker_count = executor_max_workers or config.max_concurrency
executor = ThreadPoolExecutor(max_workers=worker_count)
```

In `_run_tracked`, acquire before event-loop and monitor setup, and release in the outermost `finally`:

```python
if slot_semaphore is not None:
    slot_semaphore.acquire()
try:
    # existing tracked trajectory body
finally:
    if slot_semaphore is not None:
        slot_semaphore.release()
```

Pass both optional arguments unchanged from `run_domain` to `run_tasks`. Do not add them to the Pydantic run config or compatibility flat-argument API.

- [ ] **Step 4: Verify GREEN and compatibility**

Run the complete registered CPU test file again:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/mnt/afs/users/fush/projects/ServiceAgent/tau2-bench/src \
  /mnt/afs/conda/envs/fsh-tau2/bin/python \
  tests/test_tau2_official_async_eval.py
```

Expected: the new test and existing task-loading test pass.

## Task 2: Add per-domain configuration and deterministic redistribution

**Files:**
- Modify: `tests/test_tau2_official_async_eval.py`
- Modify: `examples/tau2-bench/eval/official/run_eval.py`

**Interfaces:**
- Produces: `_parse_domain_concurrency(value, domains, default_concurrency) -> dict[str, int]`.
- Produces: `_allocate_released_slots(slots, remaining_domains, initial_weights) -> dict[str, int]`.
- Parser flags: `--domain-concurrency`, `--global-concurrency`, `--borrow-completed-domain-slots`.

- [ ] **Step 1: Write failing parsing and allocation tests**

Add literal expectations for:

```python
_parse_domain_concurrency(
    "airline:2,retail:2,telecom:5",
    ["airline", "retail", "telecom"],
    3,
) == {"airline": 2, "retail": 2, "telecom": 5}

_allocate_released_slots(
    2,
    ["retail", "telecom"],
    {"airline": 2, "retail": 2, "telecom": 5},
) == {"retail": 1, "telecom": 1}
```

Also assert that no mapping yields the uniform default, a missing selected domain is rejected, and all parser flags are opt-in.

The production changes caught are accepting an incomplete allocation, changing backward-compatible defaults, or distributing Airline's two released slots as `0:2` instead of `1:1`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 tests/test_tau2_official_async_eval.py
```

Expected: FAIL because the parsing/allocation functions and flags do not exist.

- [ ] **Step 3: Implement minimal pure helpers and flags**

Parse comma-separated `domain:positive_integer` entries, allow entries for unselected standard domains, and require every selected domain. For redistribution, use integer `divmod(slots * weight, total_weight)` and assign leftover slots by descending remainder with the supplied remaining-domain order as the tie-breaker.

Do not introduce a scheduler class. Keep these two pure functions beside `_parse_csv`, where their behavior is directly visible.

- [ ] **Step 4: Verify GREEN**

Run the full existing CPU test file. Expected: all tests pass.

## Task 3: Coordinate elastic slots across domain processes

**Files:**
- Modify: `tests/tau2_async_eval_fixture.py`
- Modify: `tests/test_tau2_official_async_eval.py`
- Modify: `examples/tau2-bench/eval/official/run_eval.py`

**Interfaces:**
- `_run_domain_jobs(...)` additionally consumes the initial concurrency map, global concurrency, and borrow flag.
- `_run_domain_jobs(...)` returns ordered domain results plus a list of allocation events.
- `_evaluate_domain` passes `executor_max_workers` and `slot_semaphore` to tau2 only in elastic mode.

- [ ] **Step 1: Write failing orchestration tests**

Extend the process fixture so a controlled evaluator can return its received initial concurrency and optionally wait on its passed semaphore. Add tests that assert:

- static `2:2:5` reaches the three child payloads;
- result order remains the requested domain order even when completion order differs;
- successful Airline completion records `0:3:6`, then successful Retail completion records `0:0:9`;
- a child exception produces no borrowing event and propagates;
- every recorded active-domain allocation sums to 9.

The production changes caught are collecting Futures in submission order, releasing slots after failure, failing to wake blocked domain workers, or exceeding the global cap.

- [ ] **Step 2: Verify RED**

Run the async-eval CPU test and confirm failures refer to the absent elastic orchestration API.

- [ ] **Step 3: Implement cross-process coordination**

For elastic parallel execution, create one Manager semaphore per domain with its initial quota. Give each child a domain-specific payload containing its semaphore, base concurrency, and executor worker limit 9. Collect `future -> domain` with `concurrent.futures.as_completed()`.

After a Future returns successfully:

```python
released = current_concurrency[completed_domain]
current_concurrency[completed_domain] = 0
increments = _allocate_released_slots(released, remaining_domains, initial_weights)
```

Release each target semaphore once per increment, update the parent-side current map, and append a compact event with elapsed seconds, completed domain, released slots, and the new map. Reorder results to the original domain list before returning.

For static mode, set each child `TextRunConfig.max_concurrency` from the map and do not create a Manager or expanded executor.

- [ ] **Step 4: Wire summary fields and verify GREEN**

Add `initial_domain_concurrency`, `global_concurrency`, `borrow_completed_domain_slots`, and `concurrency_events` to the top-level summary. Preserve the existing `max_concurrency_per_domain` field for backward compatibility.

Run `python3 tests/test_tau2_official_async_eval.py`; expected: all tests pass.

## Task 4: Wire the 2-Agent/3-User production topology and validate

**Files:**
- Modify: `examples/tau2-bench/eval/official/run_eval.sh`
- Modify: `examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh`
- Modify: `examples/tau2-bench/eval/official/README.md`
- Modify: `output/experiments/tau2-eval-qwen36-user-async-timed/README.md`

**Interfaces:**
- Shell environment forwards the three new scheduler settings.
- Production topology is `AGENT_REPLICA_CUDA_GROUPS="0;1"` and `USER_REPLICA_CUDA_GROUPS="2,3;4,5;6,7"`.

- [ ] **Step 1: Write failing wrapper configuration test**

Extend `tests/test_tau2_official_async_eval.py` with a subprocess invocation of the production wrapper. Invoke the wrapper through `/bin/bash`, prepend a temporary directory containing a `bash` stub to `PATH`, and let that stub serialize the five exported settings to `ENV_CAPTURE_PATH` instead of starting `run_eval.sh`. Assert the captured values are exactly `0;1`, `2,3;4,5;6,7`, `airline:2,retail:2,telecom:5`, `9`, and `1`. This executes the wrapper and checks its exported behavior without asserting on source text.

The production changes caught are overlapping GPU groups, leaving the old four-Agent/two-User layout, or failing to forward elastic scheduler settings.

- [ ] **Step 2: Verify RED, then update shell wiring**

Run the targeted CPU test and confirm it reports the old topology. Update `run_eval.sh` to pass non-empty scheduler environment variables as CLI flags and update its topology log. Update the production wrapper to:

```bash
export AGENT_REPLICA_CUDA_GROUPS="0;1"
export USER_REPLICA_CUDA_GROUPS="2,3;4,5;6,7"
export DOMAIN_CONCURRENCY="airline:2,retail:2,telecom:5"
export GLOBAL_CONCURRENCY="9"
export BORROW_COMPLETED_DOMAIN_SLOTS="1"
```

Keep Agent TP1, User TP2, models, sampling, tasks, trials, and ports unchanged.

- [ ] **Step 3: Update factual documentation**

Document the new topology, initial allocation, borrowing transitions, summary fields, and the requirement for a matched timing smoke before treating the estimated speedup as measured.

- [ ] **Step 4: Run complete verification**

Run:

```bash
python3 tests/test_tau2_official_async_eval.py
bash examples/tau2-bench/eval/official/run_async_eval_preflight.sh
python3 -m py_compile examples/tau2-bench/eval/official/run_eval.py
bash -n examples/tau2-bench/eval/official/run_eval.sh
bash -n examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh
git diff --check -- \
  examples/tau2-bench/eval/official \
  tests/test_tau2_official_async_eval.py \
  tests/tau2_async_eval_fixture.py \
  docs/superpowers \
  output/experiments/tau2-eval-qwen36-user-async-timed/README.md
```

The registered CPU test already includes the tau2 runner gate from Task 1. Report its test count and note that a real GPU smoke remains a separate cluster submission unless explicitly submitted.
