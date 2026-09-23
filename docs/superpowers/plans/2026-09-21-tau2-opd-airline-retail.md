# Tau2 Airline/Retail Pure OPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a cluster-runnable pure OPD pilot for the Tau2 Airline/Retail mix, with one student sample per prompt and domain-routed SGLang teachers.

**Architecture:** The existing Tau2 rollout remains responsible for the complete on-policy conversation and binary task evaluation. A small wrapper preserves that task reward for diagnostics, clears the training reward, and lets a custom RM request token log-probs from the Airline or Retail expert selected from `sample.metadata.domain`; the framework’s existing OPD KL advantage path then supplies the only training signal. A launcher starts two local TP=2 experts, reserves the other eight GPUs for `train_async.py`, and runs independent smoke/pilot roots.

**Tech Stack:** Bash, Python, aiohttp, SGLang, Ray, slime async rollout/training, pytest.

**Spec:** `/mnt/afs/users/fush/softwares/gpu_cluster/codex/.codex/attachments/4c67fa3e-0e32-4ad7-84a3-0441030f12f8/goal-objective.md`

## Global Constraints

- Student initialization is SFT4505 with reference step 4505 and fresh smoke/pilot output roots.
- Airline and Retail are merged without domain quotas; `ShuffledTaskDataSource`, one shuffle, and rollout seed 42 preserve the natural row ratio.
- Pure OPD uses `n_samples_per_prompt=1`, zero task reward for training, OPD coefficient 1.0, K2 KL type, and all other reward/credit shaping disabled.
- Smoke and pilot are submitted through `scripts/submit.sh` to the compute cluster; no local GPU training is used.

## Review Focus

- Teacher routing must reject unknown domains and use the full on-policy token list.
- SGLang’s leading input-logprob placeholder must be removed, and any remaining token-length mismatch must raise.
- Failed or over-cap samples must not make teacher requests and must carry aligned zero vectors.
- A K=1 zero-reward group must survive the non-OPD dynamic-filter path because OPD, not GRPO reward variance, supplies the signal.
- Smoke/pilot logs must retain task-reward and domain-count diagnostics while training rewards remain zero.

### Task 1: OPD teacher contract

**Files:** Create `examples/tau2-bench/opd/tau2_opd.py`; create `examples/tau2-bench/opd/test_tau2_opd.py`; modify `examples/tau2-bench/rl/rollout.py`.

Implement the tested payload, domain routing, strict log-prob extraction, custom RM, post-processing, and `generate_opd` wrapper.

### Task 2: Mixed Airline/Retail data

**Files:** Create `examples/tau2-bench/opd/prepare_airline_retail_data.py`; extend the OPD tests with row/domain/db validation.

Merge the existing 1148 Airline and 563 Retail processed JSONL rows, validate each metadata record, and write one deterministic JSONL output.

### Task 3: Runner and cluster launcher

**Files:** Modify `examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh`; create `examples/tau2-bench/opd/run_tau2_opd_airline_retail.sh`.

Add the opt-in pure OPD argument branch and runtime environment, then provide smoke/pilot modes that start and health-check the two experts, preflight a real token-logprob request, reserve the student GPU set, and invoke the runner.

### Task 4: Experiment documentation and verification

**Files:** Modify `examples/tau2-bench/opd/README.md` and `output/doc/INDEX.md`; create the experiment README when submitting.

Run shell syntax checks, OPD tests, the existing Tau2 preflights, submit the two cluster jobs in order, inspect their terminal logs/checkpoints, and record exact job-log links and acceptance evidence.
