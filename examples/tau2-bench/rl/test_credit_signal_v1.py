"""CPU preflight for the opt-in credit-signal experiment; no GPU or Tau2 service."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import os
import runpy
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[3]
RL = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RL))

import continuous
import credit_signal_v1 as signal
import reward_postprocess as credit
from slime.rollout.agent_tokens import expand_training_segments, recorded_turn, training_segments
from slime.utils.types import Sample


@pytest.fixture(autouse=True)
def credit_environment(monkeypatch):
    for key, value in {
        "TAU2_TURN_CREDIT_VERSION": credit.PROGRESS_DB_COUNT_V1,
        "TAU2_DROP_UNIFORM_OUTCOME_GROUPS": "1",
        "TAU2_REPLACE_ZERO_SIGNAL_GROUPS": "0",
        "TAU2_PROGRESS_WEIGHT": "1", "TAU2_FORMAT_WEIGHT": "1", "TAU2_PROGRESS_GAMMA": "0.98",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("TAU2_PROGRESS_DIAGNOSTICS_PATH", raising=False)
    # Match the existing CPU tests: execute the real progress math without
    # importing optional environment/Ray/tokenizer dependencies.
    tree = ast.parse((RL / "progress.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"group_normalize", "db_count_group_advantages"}]
    namespace = {"math": math, "defaultdict": defaultdict}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(RL / "progress.py"), "exec"), namespace)
    monkeypatch.setitem(sys.modules, "progress", SimpleNamespace(**namespace))
    monkeypatch.setitem(sys.modules, "slime.rollout.data_source",
                        SimpleNamespace(RolloutDataSourceWithBuffer=object))
    monkeypatch.setattr(continuous, "AGENT_TURNS", continuous.AgentTurns())
    monkeypatch.setattr(continuous, "SAMPLING_STEPS", continuous.AgentTurns())


@pytest.fixture
def args():
    return SimpleNamespace(n_samples_per_prompt=8, advantage_estimator="grpo", reward_key=None,
                           rewards_normalization=True, grpo_std_normalization=True,
                           tau2_max_policy_lag=-1, tau2_sampling_mode="async", use_opd=False)


def make_group(reward=0.0, progress=True):
    samples = []
    for i in range(8):
        turns = [recorded_turn(prompt, {
            "completion_tokens": 2, "output_token_logprobs": [(-.7, output, None), (-.7, 99, None)],
        }, 0) for prompt, output in [([1, 2], 3), ([1, 2, 3, 99, 4], 5), ([1, 9, 3, 99], 6)]]
        segments = training_segments(turns, record_spans=True)
        details = [{**span, "simulation_message_index": 2 * span["turn_index"] + 1,
                    "segment_index": j, "errors": []}
                   for j, segment in enumerate(segments) for span in segment["assistant_spans"]]
        scores = [-10, -10 + i, -10 + i, -10 + 2 * i] if progress else [-10] * 4
        sample = Sample(index=i, group_index=0, rollout_id=i,
                        reward=float(i >= 4) if reward is None else reward,
                        metadata={"domain": "airline", "tau2_domain": "airline", "tau2_task_id": "task",
                                  "tau2_earliest_policy_version": 0,
                                  "tau2_turn_credit_version": credit.PROGRESS_DB_COUNT_V1,
                                  "tau2_turn_credits": details,
                                  "tau2_progress": {"source_indices": [1, 3, 5], "scores": scores,
                                                    "db_diff_counts": [0, 1, 2], "db_diff_side_counts": [{}] * 3,
                                                    "unavailable_reason": None}},
                        training_segments=segments,
                        train_metadata={"turn_credit_version": credit.PROGRESS_DB_COUNT_V1})
        for key in ("tokens", "response_length", "loss_mask", "rollout_log_probs"):
            setattr(sample, key, segments[0][key])
        samples.append(sample)
    return samples


@pytest.mark.parametrize("reward", [0.0, 1.0, None])
@pytest.mark.parametrize("progress", [False, True])
def test_filter_uses_actual_credit_instead_of_binary_uniformity(args, reward, progress):
    from filters import drop_zero_std_or_unsampleable

    samples = make_group(reward, progress)
    assert drop_zero_std_or_unsampleable(args, samples).keep == (reward is None)
    decision = signal.filter_group(args, samples)
    assert decision.keep == (progress or reward is None)
    assert decision.reason == (None if decision.keep else "credit_zero_signal")
    # Calling the new path did not change the legacy filter or process flags.
    assert drop_zero_std_or_unsampleable(args, samples).keep == (reward is None)


def test_identical_positive_progress_is_still_zero_signal(args):
    samples = make_group()
    for sample in samples:
        sample.metadata["tau2_progress"]["scores"] = [-10, -8, -8, -6]
    assert signal.filter_group(args, samples).reason == "credit_zero_signal"


@pytest.mark.parametrize("weight,keep", [(0, False), (1, True)])
def test_progress_weight_controls_the_signal_used_for_selection(args, monkeypatch, weight, keep):
    monkeypatch.setenv("TAU2_PROGRESS_WEIGHT", str(weight))
    assert signal.filter_group(args, make_group()).keep == keep


@pytest.mark.parametrize("weight,keep", [(0, False), (1, True)])
def test_local_format_penalty_can_train_an_all_failed_group(args, monkeypatch, weight, keep):
    monkeypatch.setenv("TAU2_FORMAT_WEIGHT", str(weight))
    samples = make_group(progress=False)
    samples[0].metadata["tau2_turn_credits"][0]["errors"] = [{"type": "malformed_json"}]
    assert signal.filter_group(args, samples).keep == keep


@pytest.mark.parametrize("reward,keep", [(0.0, False), (None, True)])
def test_unavailable_progress_retains_existing_outcome_fallback(args, reward, keep):
    samples = make_group(reward)
    samples[0].metadata["tau2_progress"]["unavailable_reason"] = "unsupported reward basis"
    assert signal.filter_group(args, samples).keep == keep


@pytest.mark.parametrize("too_long,reason", [(False, "removed_sample"), (True, "permanently_too_long")])
def test_unusable_group_is_rejected_before_credit_work(args, monkeypatch, too_long, reason):
    samples = make_group()
    samples[-1].remove_sample = True
    samples[-1].metadata["tau2_permanently_too_long"] = too_long
    prepare = Mock(side_effect=AssertionError("Unusable group must not be scored"))
    monkeypatch.setattr(credit, "attach_progress_group_advantages", prepare)
    assert signal.filter_group(args, samples).reason == reason
    prepare.assert_not_called()


def test_cached_credit_matches_legacy_formula_after_segment_expansion(args, monkeypatch, tmp_path):
    expected = make_group(reward=None)
    expected_raw, expected_rewards = credit.tau2_reward_post_process(args, expected)
    expected_parts, _, _ = expand_training_segments(expected, expected_raw, expected_rewards)
    diagnostics = tmp_path / "credit_candidates.jsonl"
    monkeypatch.setenv("TAU2_PROGRESS_DIAGNOSTICS_PATH", str(diagnostics))
    samples = make_group(reward=None)
    assert signal.filter_group(args, samples).keep
    assert len(diagnostics.read_text().splitlines()) == 8
    monkeypatch.setattr(credit, "attach_progress_group_advantages",
                        Mock(side_effect=AssertionError("Credit must be computed only once")))
    raw, rewards = signal.post_process_rewards(args, samples)
    parts, _, _ = expand_training_segments(samples, raw, rewards)
    assert raw == expected_raw and rewards == expected_rewards
    assert len(parts) == 16
    for actual, reference in zip(parts, expected_parts, strict=True):
        assert actual.train_metadata == reference.train_metadata
        values = actual.train_metadata["token_advantages"]
        assert all(value == 0 for value, mask in zip(values, actual.loss_mask, strict=True) if not mask)
        assert values[-1] == values[-2]  # Stop token retains its Assistant turn credit.
    assert len(diagnostics.read_text().splitlines()) == 8


def producer(args, samples):
    owner = object.__new__(signal.CreditSignalProducer)
    owner.args = args
    owner.condition = threading.Condition(threading.RLock())
    owner.groups = {0: {"remaining": 8, "start": 0.0}}
    owner.pending = {0: copy.deepcopy(samples)}
    owner.inflight = 8
    owner.ready, owner.taken = {}, []
    owner.replacements, owner.filter_counts = 0, {}
    owner.source = SimpleNamespace(metadata={})
    owner.version, owner.batch_size = 0, 1
    owner.closed, owner.error = False, None
    owner.refill = Mock()
    return owner


@pytest.mark.parametrize("progress,keep", [(True, True), (False, False)])
def test_collector_preserves_k8_ready_pool_accounting_and_prefilter_log(args, caplog, progress, keep):
    samples = make_group(progress=progress)
    owner = producer(args, samples)
    futures = [Future() for _ in samples]
    for i in reversed(range(8)):
        futures[i].set_result(samples[i])
    with caplog.at_level("INFO"):
        owner.collect_group(0, futures)
    assert owner.error is None and owner.inflight == 0
    assert owner.replacements == (0 if keep else 1)
    assert bool(owner.ready) == keep
    record = next(json.loads(r.message.split("tau2_credit_group ")[1])
                  for r in caplog.records if "tau2_credit_group " in r.message)
    assert record["outcome"] == "all_zero" and record["reward"] == 0
    assert record["kept"] == keep
    if keep:
        batch = owner.generate(args, 0, owner.source)
        assert [s.index for s in batch.samples[0]] == list(range(8))
        assert batch.metrics["rollout/zero_variance_group_rate"] == 1
        assert batch.metrics["rollout/trained_groups_airline"] == 1
        raw, normalized = signal.post_process_rewards(args, batch.samples[0])
        assert raw == normalized == [0.0] * 8
        assert any(s.metadata["tau2_turn_credit_group_has_signal"] for s in batch.samples[0])
    else:
        assert not owner.pending and not owner.groups
        assert owner.filter_counts == {"credit_zero_signal": 1}


def test_collector_propagates_rollout_errors_and_closes_gates(args):
    samples = make_group()
    owner = producer(args, samples)
    future = Future()
    error = RuntimeError("rollout failed")
    future.set_exception(error)
    owner.collect_group(0, [future])
    assert owner.error is error
    assert continuous.AGENT_TURNS.closed and continuous.SAMPLING_STEPS.closed


def test_wrong_credit_recipe_fails_before_starting_producer(args, monkeypatch):
    monkeypatch.setenv("TAU2_TURN_CREDIT_VERSION", "")
    with pytest.raises(ValueError, match="progress-db-count-v1"):
        signal.CreditSignalProducer(args, None)


def test_monitor_includes_rejected_groups_and_keeps_fixed_domain_weights(tmp_path):
    spec = importlib.util.spec_from_file_location("credit_signal_plot", ROOT / "scripts/plot_tau2_credit_signal_rewards.py")
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    records = [
        dict(group=0, domain="airline", trajectories=8, reward=0.0, outcome="all_zero", kept=True, reason=None),
        dict(group=1, domain="airline", trajectories=8, reward=1.0, outcome="all_one", kept=False, reason="credit_zero_signal"),
        dict(group=2, domain="banking", trajectories=8, reward=1.0, outcome="all_one", kept=True, reason=None),
    ]
    log = tmp_path / "run.log"
    log.write_text("".join("tau2_credit_group " + json.dumps(r) + "\n" for r in records)
                   + "tau2_pool_batch update=0 groups=[0, 2]\n"
                   + "tau2_trainer batch=0 start=1.0 end=2.0\n"
                   + "tau2_credit_group {\"group\":3")
    groups, batches, completed = monitor.read_log(log)
    windows, training = monitor.make_tables(groups, batches, completed, {"airline": .75, "banking": .25}, 400)
    assert len(groups) == 3  # Incomplete live line is ignored.
    assert windows[-1]["fixed_domain_reward"] == .625
    assert windows[-1]["reward"] == pytest.approx(2 / 3)
    assert windows[-1]["zero_signal_rate"] == pytest.approx(1 / 3)
    assert windows[-1]["kept_rate"] == pytest.approx(2 / 3)
    assert training[0]["reward"] == .5 and training[0]["trajectories"] == 16
    monitor.render(tmp_path, tmp_path / "plots", {"airline": .75, "banking": .25}, 400)
    assert (tmp_path / "plots/reward_diagnostics.png").stat().st_size > 0


def test_entrypoint_overrides_legacy_hooks_before_parsing(monkeypatch):
    import argparse

    parser = argparse.ArgumentParser()
    for option in ("rollout-producer-path", "dynamic-sampling-filter-path",
                   "custom-reward-post-process-path", "custom-advantage-function-path", "lr"):
        parser.add_argument("--" + option)
    captured = []
    monkeypatch.setitem(sys.modules, "slime.utils.arguments", SimpleNamespace(parse_args=parser.parse_args))
    monkeypatch.setitem(sys.modules, "train_async", SimpleNamespace(train=captured.append))
    monkeypatch.setattr(sys, "argv", ["train_tau2_credit_signal.py",
        "--rollout-producer-path", "continuous.Tau2Producer",
        "--dynamic-sampling-filter-path", "filters.drop_zero_std_or_unsampleable",
        "--custom-reward-post-process-path", "reward_postprocess.tau2_reward_post_process",
        "--custom-advantage-function-path", "reward_postprocess.turn_aware_grpo_advantage",
        "--lr", "2e-6"])
    runpy.run_path(str(ROOT / "train_tau2_credit_signal.py"), run_name="__main__")
    resolved = captured[0]
    assert resolved.rollout_producer_path == "credit_signal_v1.CreditSignalProducer"
    assert resolved.dynamic_sampling_filter_path == "credit_signal_v1.filter_group"
    assert resolved.custom_reward_post_process_path == "credit_signal_v1.post_process_rewards"
    assert resolved.custom_advantage_function_path == "reward_postprocess.turn_aware_grpo_advantage"
    assert resolved.lr == "2e-6"


def test_new_launcher_retains_recipe_and_uses_separate_paths():
    launcher = ROOT / "output/experiments/tau2-mix-rl-sft4505-credit-signal-v1-20260927/run_mix.sh"
    result = subprocess.run(["bash", str(launcher), "--dry-run"], check=True, capture_output=True, text=True,
                            env={**os.environ, "PROJECT_ROOT": str(ROOT), "NUM_ROLLOUT": "140"})
    config = json.loads(result.stdout)
    assert config["TRAIN_ENTRYPOINT"] == "train_tau2_credit_signal.py"
    assert config["TAU2_DROP_UNIFORM_OUTCOME_GROUPS"] == "0"
    for key, value in {"LR": "2e-6", "TRAIN_SEED": "1234", "ROLLOUT_SEED": "42",
                       "N_SAMPLES_PER_PROMPT": "8", "ROLLOUT_BATCH_SIZE": "16", "GLOBAL_BATCH_SIZE": "128",
                       "NUM_ROLLOUT": "140", "TAU2_POOL_CAPACITY": "32", "TAU2_MAX_PENDING_GROUPS": "64",
                       "TAU2_MAX_POLICY_LAG": "-1", "TAU2_ENVIRONMENT_WORKERS": "4",
                       "TAU2_PROGRESS_WEIGHT": "1.0", "TAU2_FORMAT_WEIGHT": "1.0", "TAU2_PROGRESS_GAMMA": "0.98",
                       "DATA_SOURCE_PATH": "random_tasks.RandomTaskDataSource"}.items():
        assert config[key] == value
    assert config["HF_CHECKPOINT"].endswith("20260920/iter_0004505_hf")
    assert "credit-signal-v1-20260927/data/" in config["PREPARED_RL_DATA"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
