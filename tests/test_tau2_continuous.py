"""CPU tests for recorded tokens and bounded, turn-interleaved Tau2 GRPO."""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.rollout.agent_tokens import BehaviorLogprobError, recorded_turn, training_segments, expand_training_segments

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tau2_continuous_test", ROOT / "examples/tau2-bench/rl/continuous.py")
continuous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuous)


def turn(prompt, output, version=0):
    return recorded_turn(prompt, {"output_token_logprobs": [(-0.7, t, None) for t in output], "completion_tokens": len(output), "finish_reason": {"type": "stop", "matched": 99}}, version)


def test_exact_prefix_merge_retains_stop_and_masks_framework_text():
    segments = training_segments([turn([1, 2], [3, 99]), turn([1, 2, 3, 99, 4, 5], [6, 99], 1)])
    assert segments == [{"tokens": [1, 2, 3, 99, 4, 5, 6, 99], "response_length": 6, "loss_mask": [1, 1, 0, 0, 1, 1], "rollout_log_probs": [-0.7, -0.7, 0, 0, -0.7, -0.7]}]


def test_json_reordering_keeps_full_original_context():
    # Raw arguments a,b become b,a in the next prompt's serialized history.
    segments = training_segments([turn([1, 2], [10, 11, 99]), turn([1, 2, 11, 10, 99, 4], [6, 99])])
    assert len(segments) == 2
    assert segments[0]["tokens"] == [1, 2, 10, 11, 99]
    assert segments[1]["tokens"] == [1, 2, 11, 10, 99, 4, 6, 99]
    assert sum(sum(s["loss_mask"]) for s in segments) == 5


@pytest.mark.parametrize("meta", [{}, {"completion_tokens": 2, "output_token_logprobs": [(-1, 1)]}, {"completion_tokens": 1, "output_token_logprobs": [(None, 1)]}, {"completion_tokens": 1, "output_token_logprobs": [(float("nan"), 1)]}])
def test_missing_behavior_probability_stops(meta):
    with pytest.raises(BehaviorLogprobError):
        recorded_turn([1], meta, 0)


def test_turn_boundary_waits_for_requests_and_blocks_admission():
    gate = continuous.AgentTurns(capacity=1)
    entered, release, updated = threading.Event(), threading.Event(), threading.Event()

    def first():
        with gate.request() as version:
            assert version == 0
            entered.set()
            assert release.wait(3)

    def update():
        gate.pause()
        updated.set()

    with ThreadPoolExecutor(3) as pool:
        first_future = pool.submit(first)
        assert entered.wait(3)
        update_future = pool.submit(update)
        assert not updated.wait(0.05)
        release.set()
        first_future.result(3)
        update_future.result(3)
        next_future = pool.submit(lambda: next_request(gate))
        assert not next_future.done()
        gate.resume(1)
        assert next_future.result(3) == 1


def next_request(gate):
    with gate.request() as version:
        return version


def test_slow_turn_logging_does_not_hold_sampling_gate(monkeypatch):
    gate = continuous.AgentTurns()
    logging_started, release_log = threading.Event(), threading.Event()

    def slow_log(*args):
        logging_started.set()
        assert release_log.wait(3)

    monkeypatch.setattr(continuous.logger, "debug", slow_log)
    with ThreadPoolExecutor(2) as pool:
        turn_future = pool.submit(next_request, gate)
        try:
            assert logging_started.wait(3)
            pool.submit(gate.pause).result(timeout=1)
            assert gate.active == 0
        finally:
            release_log.set()
        turn_future.result(3)


def extracted_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    for fn in functions:
        fn.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def test_k8_normalization_before_unequal_segments_and_zero_variance():
    import torch

    ns = extracted_functions(ROOT / "slime/ray/rollout.py", {"_post_process_rewards"}, {"torch": torch, "Sample": SimpleNamespace})
    args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True, n_samples_per_prompt=8, rollout_batch_size=2)
    owner = SimpleNamespace(args=args, custom_reward_post_process_func=None)
    values = [0, 0, 0, 0, 1, 1, 1, 1] + [0] * 8
    samples = [SimpleNamespace(index=i, rollout_id=i, training_segments=[{"tokens": [1, 2]}] * (i + 1), get_reward_value=lambda args, v=v: v) for i, v in enumerate(values)]
    raw, advantages = ns["_post_process_rewards"](owner, samples)
    expanded, _, expanded_advantages = expand_training_segments(samples, raw, advantages)
    for part, advantage in zip(expanded, expanded_advantages, strict=True):
        if part.rollout_id < 8:
            assert advantage == pytest.approx((values[part.rollout_id] - 0.5) / (torch.tensor(values[:8], dtype=torch.float).std().item() + 1e-6))
        else:
            assert advantage == 0


def test_tis_stop_gradient_and_segment_loss_equivalence():
    import torch
    from typing import Any

    ns = extracted_functions(ROOT / "slime/backends/megatron_utils/loss.py", {"vanilla_tis_function"}, {"torch": torch, "Any": Any})
    prox = torch.tensor([-1.0, -1.0, -1.0, -1.0], requires_grad=True)
    mu = torch.tensor([-3.0, -1.0, -2.0, -0.5], requires_grad=True)
    theta = prox.detach().clone().requires_grad_()
    pg_loss = -(theta - prox.detach()).exp()
    corrected, _, _ = ns["vanilla_tis_function"](SimpleNamespace(tis_clip=2.0, tis_clip_low=0.0), pg_loss=pg_loss, train_log_probs=[prox], rollout_log_probs=[mu], loss_masks=[torch.ones(4)])
    # Two segments of one trajectory retain the whole-trajectory denominator.
    merged = corrected.mean()
    separate = corrected[:1].sum() / 4 + corrected[1:].sum() / 4
    assert torch.equal(merged, separate)
    grad1 = torch.autograd.grad(merged, theta, retain_graph=True)[0]
    grad2 = torch.autograd.grad(separate, theta, retain_graph=True)[0]
    torch.testing.assert_close(grad1, grad2)
    merged.backward()
    assert prox.grad is None and mu.grad is None
    torch.testing.assert_close(theta.grad, -torch.tensor([2.0, 1.0, 2.0, torch.exp(torch.tensor(-0.5))]) / 4)


@pytest.fixture
def producer_fixture(monkeypatch):
    monkeypatch.setattr(continuous, "AGENT_TURNS", continuous.AgentTurns())
    monkeypatch.setattr(continuous, "SAMPLING_STEPS", continuous.AgentTurns(capacity=80))
    release = threading.Event()
    entered = threading.Event()

    def run(args, sample, params):
        with continuous.AGENT_TURNS.request() as first:
            pass
        # The first task stays unfinished while later tasks enter the ready pool.
        if sample.group_index == 0:
            entered.set()
            assert release.wait(10)
        with continuous.AGENT_TURNS.request() as last:
            sample.metadata["tau2_earliest_policy_version"] = first
            sample.metadata["tau2_latest_policy_version"] = last
        sample.reward = sample.index % 2
        return sample

    monkeypatch.setitem(sys.modules, "agent", SimpleNamespace(shared_client=lambda timeout: None, shared_tokenizer=lambda model: None))
    monkeypatch.setitem(sys.modules, "rollout", SimpleNamespace(_get_mask_generator=lambda args: None, _run_tau2_rollout_sync=run))
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", SimpleNamespace(GenerateState=lambda args: SimpleNamespace(sampling_params={})))
    monkeypatch.setitem(sys.modules, "slime.rollout.base_types", SimpleNamespace(RolloutFnTrainOutput=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, "filters", SimpleNamespace(drop_zero_std_or_unsampleable=lambda args, g: SimpleNamespace(keep=True)))

    class Source:
        def __init__(self):
            self.metadata = {}
            self.cursor = 0

        def get_samples(self, count):
            groups = []
            for _ in range(count):
                key = self.cursor
                self.cursor += 1
                domain = ["airline", "retail", "telecom"][key % 3]
                groups.append([SimpleNamespace(index=key * 8 + j, group_index=key,
                                               metadata={"domain": domain}) for j in range(8)])
            return groups

        def save(self, rollout_id):
            self.saved = copy.deepcopy((self.metadata, self.cursor))

    return Source, release, entered


def pool_args(**overrides):
    return SimpleNamespace(**(dict(hf_checkpoint="test", start_rollout_id=0, num_rollout=3,
                                  tau2_pool_capacity=10, tau2_sampling_mode="async",
                                  rollout_batch_size=5, n_samples_per_prompt=8) | overrides))


def test_environment_workers_preserve_total_concurrency_and_weight_barrier(producer_fixture, monkeypatch):
    from concurrent.futures import Future

    Source, _, _ = producer_fixture
    release = threading.Event()
    workers, options = [], {}
    monkeypatch.setattr(continuous.AGENT_TURNS, "capacity", 35)
    monkeypatch.setattr(continuous.SAMPLING_STEPS, "capacity", 81)

    class Method:
        def __init__(self, function):
            self.remote = function

    class Worker:
        def __init__(self, args, agent_capacity, step_capacity):
            self.gate = continuous.AgentTurns(agent_capacity)
            self.step_capacity = step_capacity
            self.pool = ThreadPoolExecutor(32)
            self.seen = []
            self.ready = Method(lambda: True)
            self.run = Method(lambda sample, params: self.pool.submit(self.sample, sample))
            self.pause = Method(self.gate.pause)
            self.resume = Method(self.gate.resume)
            self.close = Method(self.gate.close)

        def sample(self, sample):
            assert release.wait(5)
            with self.gate.request() as version:
                self.seen.append(sample.index)
                sample.metadata["tau2_earliest_policy_version"] = version
                sample.reward = sample.index % 2
            return sample

    class RemoteClass:
        def options(self, **kwargs):
            options.update(kwargs)
            return self

        def remote(self, *args):
            worker = Worker(*args)
            workers.append(worker)
            return worker

    def get(value):
        if isinstance(value, list):
            return [get(item) for item in value]
        return value.result(5) if isinstance(value, Future) else value

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(
        remote=lambda cls: RemoteClass(), get=get, kill=lambda worker: worker.pool.shutdown(wait=True)))
    args = pool_args(tau2_environment_workers=4, tau2_max_policy_lag=-1,
                     tau2_pool_capacity=2, rollout_batch_size=1, num_rollout=3)
    producer = continuous.Tau2Producer(args, Source())
    try:
        assert sum(worker.gate.capacity for worker in workers) == 35
        assert sum(worker.step_capacity for worker in workers) == 81
        assert options["num_cpus"] == 1
        assert options["max_concurrency"] > 16
        producer.pause()
        assert all(worker.gate.paused for worker in workers)
        producer.resume(2)
        release.set()
        output = producer.generate(args, 2, producer.source)
        assert all(sample.metadata["tau2_earliest_policy_version"] == 2
                   for group in output.samples for sample in group)
        assert all(worker.seen for worker in workers)
    finally:
        release.set()
        producer.close()


def test_producer_shutdown_releases_blocked_remote_dialogues(monkeypatch):
    monkeypatch.setattr(continuous, "AGENT_TURNS", continuous.AgentTurns())
    monkeypatch.setattr(continuous, "SAMPLING_STEPS", continuous.AgentTurns())
    entered, released = threading.Event(), threading.Event()
    worker = SimpleNamespace(close=SimpleNamespace(remote=lambda: None))
    producer = continuous.Tau2Producer.__new__(continuous.Tau2Producer)
    producer.condition = threading.Condition()
    producer.closed = False
    producer.environment_workers = [worker]
    producer.ray = SimpleNamespace(get=lambda value: value, kill=lambda worker: released.set())
    producer.workers = ThreadPoolExecutor(1)
    producer.collectors = ThreadPoolExecutor(1)

    def remote_dialogue():
        # Model a User request that cannot observe the closed Agent turn gate.
        entered.set()
        assert released.wait(5)

    pending = producer.workers.submit(remote_dialogue)
    producer.collectors.submit(pending.result)
    assert entered.wait(3)
    with ThreadPoolExecutor(1) as closer:
        closed = closer.submit(producer.close)
        try:
            closed.result(timeout=1)
        finally:
            # Also clean up when the old shutdown ordering fails this test.
            released.set()
            closed.result(timeout=5)


def test_environment_workers_reject_bounded_policy_lag(producer_fixture):
    Source, _, _ = producer_fixture
    with pytest.raises(ValueError, match="unlimited policy lag"):
        continuous.Tau2Producer(pool_args(tau2_environment_workers=4, tau2_max_policy_lag=1), Source())


def test_total_buffer_limit_includes_ready_and_training_groups(producer_fixture):
    Source, release, _ = producer_fixture
    release.set()
    args = pool_args(tau2_max_policy_lag=-1, tau2_max_buffered_groups=10, num_rollout=8)
    source = Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        with producer.condition:
            assert producer.condition.wait_for(lambda: len(producer.ready) == 10, timeout=3)
            producer.refill()
            assert source.cursor == 10
        batch = producer.generate(args, 0, source)
        assert batch.metrics["rollout/buffered_groups"] == 10
        assert batch.metrics["rollout/ready_groups"] == 5
        with producer.condition:
            producer.refill()
            assert len(producer.groups) + len(producer.taken) == 10
            assert source.cursor == 10
        producer.finish_training()
        with producer.condition:
            assert producer.condition.wait_for(lambda: len(producer.ready) == 10, timeout=3)
            assert source.cursor == 15
            assert len(producer.groups) + len(producer.taken) == 10
    finally:
        producer.close()


def test_total_buffer_limit_keeps_async_overtaking_and_replaces_filtered_groups(producer_fixture, monkeypatch):
    Source, release, entered = producer_fixture
    rejected = []

    def filter_group(args, group):
        if group[0].group_index == 1:
            rejected.append(1)
            return SimpleNamespace(keep=False, reason="unsampleable")
        return SimpleNamespace(keep=True)

    monkeypatch.setitem(sys.modules, "filters", SimpleNamespace(drop_zero_std_or_unsampleable=filter_group))
    args = pool_args(tau2_max_policy_lag=-1, tau2_max_buffered_groups=10, num_rollout=3)
    source = Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        assert entered.wait(3)
        consumed = []
        with ThreadPoolExecutor(1) as executor:
            for update in range(3):
                batch = executor.submit(producer.generate, args, update, source).result(3)
                consumed.extend(group[0].group_index for group in batch.samples)
                assert batch.metrics["rollout/buffered_groups"] <= 10
                assert batch.metrics["rollout/lag_wait_seconds"] == 0
                producer.finish_training()
                producer.pause()
                producer.resume(update + 1)
        assert rejected == [1]
        assert 0 not in consumed and 1 not in consumed
        assert len(set(consumed)) == 15
        assert producer.consumed == 15
    finally:
        release.set()
        producer.close()


def test_buffer_limit_smaller_than_batch_is_rejected(producer_fixture):
    Source, _, _ = producer_fixture
    with pytest.raises(ValueError, match="buffered group limit"):
        continuous.Tau2Producer(pool_args(tau2_max_buffered_groups=4), Source())


def test_resumed_pending_inputs_obey_smaller_buffer_limit(producer_fixture):
    Source, release, _ = producer_fixture
    release.set()
    source = Source()
    pending = source.get_samples(15)
    source.metadata["tau2_pending_groups"] = {group[0].group_index: group for group in pending}
    args = pool_args(start_rollout_id=1, num_rollout=3, tau2_max_policy_lag=-1, tau2_max_buffered_groups=5)
    producer = continuous.Tau2Producer(args, source)
    try:
        with producer.condition:
            assert producer.condition.wait_for(lambda: len(producer.ready) == 5, timeout=3)
            assert len(producer.groups) == 5
            assert source.cursor == 15
        first = producer.generate(args, 1, source)
        assert {g[0].group_index for g in first.samples} == set(range(5))
        producer.finish_training()
        producer.pause()
        producer.resume(2)
        second = producer.generate(args, 2, source)
        assert {g[0].group_index for g in second.samples} == set(range(5, 10))
        assert second.metrics["rollout/buffered_groups"] == 5
        assert source.cursor == 15
    finally:
        producer.close()


def test_opd_reports_consumed_domain_reward_and_token_weighted_lag(producer_fixture, monkeypatch):
    Source, _, _ = producer_fixture

    def run(args, sample, params):
        j = sample.index % 8
        sample.metadata.update(tau2_earliest_policy_version=0, tau2_latest_policy_version=4,
                               tau2_token_weighted_policy_version=4 * j / 7, tau2_response_tokens=j + 1,
                               tau2_opd_task_reward=float(sample.metadata["domain"] == "airline"))
        sample.reward = 0.0
        return sample

    monkeypatch.setenv("TAU2_OPD_PURE", "0")  # This fixture already provides scored samples.
    monkeypatch.setitem(sys.modules, "rollout", SimpleNamespace(_get_mask_generator=lambda _: None,
                                                               _run_tau2_rollout_sync=run))
    # Admit exactly one batch so both domains are present regardless of which
    # dialogues finish first. Extra ready groups would make membership random.
    args = pool_args(start_rollout_id=5, num_rollout=6, use_opd=True,
                     tau2_max_policy_lag=-1, tau2_max_buffered_groups=5)
    producer = continuous.Tau2Producer(args, Source())
    try:
        metrics = producer.generate(args, 5, producer.source).metrics
        for domain in ("airline", "retail"):
            prefix = f"rollout/opd/{domain}"
            assert metrics[f"{prefix}/raw_reward"] == float(domain == "airline")
            assert metrics[f"{prefix}/earliest_policy_lag"] == 5
            assert metrics[f"{prefix}/latest_policy_lag"] == 1
            assert metrics[f"{prefix}/token_weighted_policy_lag"] == pytest.approx(7 / 3)
    finally:
        producer.close()


def test_unbounded_lag_consumes_ready_groups_while_old_group_is_unfinished(producer_fixture):
    Source, release, entered = producer_fixture
    args, source = pool_args(tau2_max_policy_lag=-1), Source()
    producer = continuous.Tau2Producer(args, source)
    executor = ThreadPoolExecutor(1)
    try:
        assert entered.wait(3)
        consumed = []
        for update in (0, 1):
            batch = executor.submit(producer.generate, args, update, source).result(3)
            consumed.extend(g[0].group_index for g in batch.samples)
            assert len(batch.samples) == 5
            assert all(len(g) == 8 for g in batch.samples)
            assert 0 not in consumed
            assert batch.metrics["rollout/lag_wait_seconds"] == 0
            producer.finish_training()
            producer.pause()
            producer.resume(update + 1)
        assert 0 in producer.pending
        batch = executor.submit(producer.generate, args, 2, source).result(3)
        consumed.extend(g[0].group_index for g in batch.samples)
        assert batch.metrics["rollout/policy_lag_max"] == 2
        producer.finish_training()
        # The final update must not turn the unfinished old group into a
        # mandatory tail. Training consumes exactly its budget; spare work
        # stays pending until producer shutdown (or checkpoint resume).
        assert len(set(consumed)) == 15 and 0 not in consumed
        assert producer.consumed == args.num_rollout * args.rollout_batch_size
        cursor = source.cursor
        producer.resume(3)
        assert source.cursor == cursor
    finally:
        release.set()
        producer.close()
        executor.shutdown(wait=True)


@pytest.mark.parametrize("capacity", [10, 20, 40])
def test_unbounded_pool_refills_before_new_weights_arrive(producer_fixture, capacity):
    Source, release, entered = producer_fixture
    args, source = pool_args(tau2_max_policy_lag=-1, tau2_pool_capacity=capacity, num_rollout=20), Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        producer.generate(args, 0, source)
        producer.finish_training()
        with producer.condition:
            assert producer.awaiting_weights
            assert producer.condition.wait_for(lambda: source.cursor > capacity + 5, timeout=3)
            assert len(producer.groups) - len(producer.ready) <= capacity
            assert producer.inflight <= capacity * args.n_samples_per_prompt
    finally:
        release.set()
        producer.close()


def test_unbounded_pool_replaces_ready_group_before_training(producer_fixture, monkeypatch):
    Source, _, _ = producer_fixture
    release_one, release_rest = threading.Event(), threading.Event()

    def run(args, sample, params):
        assert (release_one if sample.group_index == 1 else release_rest).wait(5)
        sample.metadata["tau2_earliest_policy_version"] = 0
        sample.reward = sample.index % 2
        return sample

    monkeypatch.setattr(sys.modules["rollout"], "_run_tau2_rollout_sync", run)
    args, source = pool_args(tau2_max_policy_lag=-1), Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        assert source.cursor == 10
        release_one.set()
        with producer.condition:
            assert producer.condition.wait_for(lambda: 1 in producer.ready, timeout=3)
            assert source.cursor == 11
            assert len(producer.ready[1]) == 8
            assert len(producer.groups) - len(producer.ready) == 10
            assert producer.consumed == 0 and not producer.training
    finally:
        release_rest.set()
        producer.close()


def test_ready_groups_overtake_slow_task_but_cannot_make_it_stale(producer_fixture):
    Source, release, entered = producer_fixture
    args, source = pool_args(), Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        assert entered.wait(3)
        batch = producer.generate(args, 0, source)
        assert len(batch.samples) == 5 and all(len(g) == 8 for g in batch.samples)
        assert 0 not in [g[0].group_index for g in batch.samples]
        assert source.cursor == 10  # Taken groups still occupy capacity.
        producer.finish_training()
        producer.save(0)
        assert len(source.saved[0]["tau2_pending_groups"]) == 5
        assert 0 in source.saved[0]["tau2_pending_groups"]
        producer.pause()
        producer.resume(1)
        with producer.condition:
            assert producer.condition.wait_for(lambda: len(producer.ready) >= 5, timeout=3)
        # Plenty of new ready groups, but old unfinished task 0 cannot be skipped
        # again: another optimizer update would give its first turn lag two.
        with ThreadPoolExecutor(1) as executor:
            next_batch = executor.submit(producer.generate, args, 1, source)
            assert not next_batch.done()
            release.set()
            batch1 = next_batch.result(5)
        assert 0 in [g[0].group_index for g in batch1.samples]
        assert batch1.metrics["rollout/policy_lag_max"] == 1
        producer.finish_training()
        producer.pause()
        producer.resume(2)
        producer.generate(args, 2, source)
        producer.finish_training()
        producer.resume(3)
        assert source.cursor == 15
        assert not producer.pending and not producer.groups and not producer.taken
    finally:
        release.set()
        producer.close()

    # Resume regenerates only saved pending inputs at the recovered policy.
    continuous.AGENT_TURNS = continuous.AgentTurns()
    continuous.SAMPLING_STEPS = continuous.AgentTurns(capacity=80)
    restored = Source()
    restored.metadata, restored.cursor = source.saved
    saved_ids = set(restored.metadata["tau2_pending_groups"])
    args = pool_args(start_rollout_id=1, num_rollout=2)
    producer = continuous.Tau2Producer(args, restored)
    try:
        batch = producer.generate(args, 1, restored)
        assert {g[0].group_index for g in batch.samples} == saved_ids
        assert restored.cursor == 10
        assert batch.metrics["rollout/policy_lag_max"] == 0
    finally:
        producer.close()


def test_earliest_version_cannot_be_hidden_by_latest_turn(producer_fixture):
    Source, release, _ = producer_fixture
    release.set()
    args, source = pool_args(), Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        with producer.condition:
            assert producer.condition.wait_for(lambda: len(producer.ready) == 10, timeout=3)
            for group in producer.ready.values():
                for sample in group:
                    sample.metadata["tau2_latest_policy_version"] = 2
        with pytest.raises(RuntimeError, match="earliest-turn"):
            producer.generate(args, 2, source)
    finally:
        producer.close()


def test_rejected_group_refills_immediately_without_domain_constraint(producer_fixture, monkeypatch):
    Source, release, _ = producer_fixture
    release.set()
    rejected = []

    def filter_group(args, group):
        reject = group[0].group_index == 0
        if reject:
            rejected.append(group[0].metadata["domain"])
        return SimpleNamespace(keep=not reject)

    monkeypatch.setattr(sys.modules["filters"], "drop_zero_std_or_unsampleable", filter_group)
    args, source = pool_args(num_rollout=2), Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        with producer.condition:
            assert producer.condition.wait_for(lambda: len(producer.ready) == 10, timeout=3)
        # Group 10 (retail) replaces rejected group 0 (airline), before training.
        assert source.cursor == 11 and 10 in producer.ready
        assert rejected == ["airline"]
        for update in range(2):
            result = producer.generate(args, update, source)
            assert len(result.samples) == 5
            producer.finish_training()
            producer.pause()
            producer.resume(update + 1)
        assert not producer.pending
    finally:
        producer.close()


def test_replacement_can_start_during_training_with_bounded_old_groups(producer_fixture, monkeypatch):
    Source, release, entered = producer_fixture
    monkeypatch.setattr(sys.modules["filters"], "drop_zero_std_or_unsampleable",
                        lambda args, group: SimpleNamespace(keep=group[0].group_index != 0))
    args, source = pool_args(), Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        assert entered.wait(3)
        producer.generate(args, 0, source)
        release.set()  # Old group 0 fails while the five taken groups train.
        with producer.condition:
            assert producer.condition.wait_for(lambda: 10 in producer.ready, timeout=3)
            assert producer.ready[10][0].metadata["tau2_earliest_policy_version"] == 0
            assert len(producer.groups) + len(producer.taken) == 10
            assert source.cursor == 11
        producer.finish_training()
        producer.pause()
        producer.resume(1)
        batch = producer.generate(args, 1, source)
        assert 10 in [group[0].group_index for group in batch.samples]
        assert batch.metrics["rollout/policy_lag_max"] == 1
    finally:
        release.set()
        producer.close()


def test_producer_preserves_probability_failure_during_shutdown(producer_fixture, monkeypatch):
    Source, release, _ = producer_fixture
    release.set()
    def fail(args, sample, params):
        if sample.index == 7:
            raise BehaviorLogprobError("missing behavior probabilities")
        with continuous.AGENT_TURNS.condition:
            assert continuous.AGENT_TURNS.condition.wait_for(lambda: continuous.AGENT_TURNS.closed, timeout=3)
        with continuous.AGENT_TURNS.request():
            return sample

    monkeypatch.setattr(sys.modules["rollout"], "_run_tau2_rollout_sync", fail)
    args, source = pool_args(), Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        producer.workers.shutdown(wait=True)
        producer.collectors.shutdown(wait=True)
        with pytest.raises(BehaviorLogprobError, match="missing behavior"):
            producer.generate(args, 0, source)
    finally:
        producer.close()


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_sampling_stage_barrier_matches_training_mode(producer_fixture, mode):
    Source, release, _ = producer_fixture
    release.set()
    args, source = pool_args(tau2_sampling_mode=mode), Source()
    producer = continuous.Tau2Producer(args, source)
    try:
        producer.generate(args, 0, source)
        stepped = threading.Event()
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(continuous.sampling_step(stepped.set))
            if mode == "sync":
                assert not stepped.wait(0.05)
            else:
                assert stepped.wait(3)
            producer.finish_training()
            assert stepped.wait(3)  # Sampling resumes before checkpoint/weight synchronization.
            producer.pause()
            producer.resume(1)
            future.result(3)
            assert stepped.is_set()
    finally:
        producer.close()


def test_causal_model_gradient_matches_merged_and_individual_turns():
    import torch
    import torch.nn.functional as F

    turns = [turn([1, 2], [3, 7]), turn([1, 2, 3, 7, 4, 5], [6, 7])]
    weight = torch.randn(8, 8, generator=torch.Generator().manual_seed(42), requires_grad=True)

    def loss(segments):
        total = 0
        denominator = sum(sum(s["loss_mask"]) for s in segments)
        for segment in segments:
            ids = torch.tensor(segment["tokens"])
            # A causal toy model whose next-token distribution depends on the entire prefix.
            logits = weight[ids].cumsum(0)[:-1]
            nll = F.cross_entropy(logits, ids[1:], reduction="none")
            mask = torch.tensor(segment["loss_mask"], dtype=torch.float)
            total = total + (nll[-len(mask) :] * mask).sum() / denominator
        return total

    merged = loss(training_segments(turns))
    separate = loss([s for t in turns for s in training_segments([t])])
    torch.testing.assert_close(merged, separate)
    torch.testing.assert_close(torch.autograd.grad(merged, weight)[0], torch.autograd.grad(separate, weight)[0])


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("run_name", [None, "custom-resume"])
@pytest.mark.parametrize("profile,updates,save_interval", [("smoke", 3, 1), ("train20", 20, 10), ("train100", 100, 10), ("train200", 200, 10)])
@pytest.mark.parametrize("custom_data", [False, True])
def test_copied_launcher_honors_project_root(tmp_path, mode, run_name, profile, updates, save_interval, custom_data):
    import json
    import os
    import subprocess

    root = tmp_path / "worktree"
    launcher = root / "examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/bash\npython3 - <<'INNER'\nimport os,json\nprint(json.dumps(dict(os.environ)))\nINNER\n")
    snapshot = tmp_path / "snapshot.sh"
    snapshot.write_text((ROOT / "examples/tau2-bench/rl/run_tau2_areal_async.sh").read_text())
    env = dict(os.environ, PROJECT_ROOT=str(root), TAU2_RAW_TOKENS="1")
    env.pop("TAU2_RUN_NAME", None)
    # The launcher preflight may run inside an experiment arm that sets
    # these overrides.  This test checks the launcher's default behavior,
    # so keep outer arm configuration from leaking into the subprocess.
    env.pop("TAU2_POOL_CAPACITY", None)
    env.pop("TAU2_MAX_POLICY_LAG", None)
    env.pop("SMOKE_UPDATES", None)
    env.pop("NUM_ROLLOUT", None)
    env.pop("PREPARED_RL_DATA", None)
    env.pop("USER_SGLANG", None)
    env.pop("TAU2_USER_API_BASE", None)
    env.pop("TAU2_USER_API_KEY", None)
    env.pop("DATA_SOURCE_PATH", None)
    expected_source = "random_tasks.RandomTaskDataSource"
    expected_data = str(tmp_path / "datasets/tau2/rl/areal_tasks_slime.jsonl")
    if custom_data:
        expected_data = str(tmp_path / "airline_tasks.jsonl")
        env["PREPARED_RL_DATA"] = expected_data
        expected_source = "random_tasks.ShuffledTaskDataSource"
        env["DATA_SOURCE_PATH"] = expected_source
    if run_name is not None:
        env["TAU2_RUN_NAME"] = run_name
    result = subprocess.run(["bash", str(snapshot), mode, profile], env=env, capture_output=True, text=True, check=True)
    settings = json.loads(result.stdout)
    assert settings["PROJECT_ROOT"] == str(root)
    assert settings["TAU2_SAMPLING_MODE"] == mode
    assert settings["TAU2_POOL_CAPACITY"] == "10"
    assert settings["DATA_SOURCE_PATH"] == expected_source
    assert settings["TAU2_RL_DOMAIN_QUOTA"] == ""
    assert f"/{run_name or f'ready-{profile}'}/" in settings["SAVE_DIR"]
    assert settings["NUM_ROLLOUT"] == str(updates)
    assert settings["SAVE_INTERVAL"] == str(save_interval)
    if mode == "async":
        assert settings["USER_SGLANG"] == "0"
        assert settings["RAY_GPUS"] == "8"
        assert settings["AGENT_CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
        assert settings["USER_CUDA_VISIBLE_DEVICES"] == ""
        assert settings["TAU2_USER_API_BASE"].endswith(":30000/v1")
    else:
        assert settings["USER_SGLANG"] == "1"
        assert settings["RAY_GPUS"] == "6"
        assert settings["USER_CUDA_VISIBLE_DEVICES"] == "6,7"
    assert settings["ROLLOUT_NUM_GPUS_PER_ENGINE"] == "2"
    assert settings["SKIP_PREPARE_RL_DATA"] == "1"
    assert settings["PREPARED_RL_DATA"] == expected_data


def test_agent_posts_token_ids_and_records_original_sampling_output(monkeypatch):
    import os
    import time
    from typing import Any

    requests = []

    def post(url, json):
        requests.append(json)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {"text": "  answer  ", "meta_info": {"completion_tokens": 2, "output_token_logprobs": [(-0.2, 42), (-0.3, 99)], "finish_reason": {"type": "stop", "matched": 99}}})

    monkeypatch.setenv("TAU2_RAW_TOKENS", "1")
    gate = continuous.AgentTurns()
    gate.resume(7)
    monkeypatch.setitem(sys.modules, "continuous", SimpleNamespace(AGENT_TURNS=gate))
    ns = extracted_functions(ROOT / "examples/tau2-bench/rl/agent.py", {"_generate_raw"}, {"os": os, "time": time, "Any": Any, "shared_client": lambda timeout: SimpleNamespace(post=post), "recorded_turn": recorded_turn})
    agent = SimpleNamespace(timeout=600, sglang_url="http://test/generate", _sampling_params=lambda: {"temperature": 1.0})
    text, metadata = ns["_generate_raw"](agent, [1, 2, 3])
    assert requests == [{"input_ids": [1, 2, 3], "return_logprob": True, "logprob_start_len": -1, "sampling_params": {"temperature": 1.0}}]
    assert metadata["behavior_turn"]["output_token_ids"] == [42, 99]
    assert metadata["behavior_turn"]["log_probs"] == [-0.2, -0.3]
    assert metadata["behavior_turn"]["policy_version"] == 7


@pytest.mark.parametrize("prompt_length", [7, 8, 9])
def test_agent_resamples_over_cap_prompts_without_sending_http(monkeypatch, prompt_length):
    import os
    import time
    from typing import Any

    class RolloutContextOverflow(ValueError):
        pass

    requests = []

    def post(url, json):
        requests.append(json)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {
            "text": "answer", "meta_info": {"completion_tokens": 1, "output_token_logprobs": [(-0.2, 42)]}})

    monkeypatch.setenv("TAU2_RAW_TOKENS", "1")
    gate = continuous.AgentTurns()
    gate.resume(0)
    monkeypatch.setitem(sys.modules, "continuous", SimpleNamespace(AGENT_TURNS=gate))
    ns = extracted_functions(ROOT / "examples/tau2-bench/rl/agent.py", {"_generate_raw"}, {
        "os": os, "time": time, "Any": Any, "recorded_turn": recorded_turn,
        "RolloutContextOverflow": RolloutContextOverflow,
        "shared_client": lambda timeout: SimpleNamespace(post=post),
    })
    agent = SimpleNamespace(max_train_tokens=8, timeout=600, sglang_url="http://test/generate", _sampling_params=dict)
    if prompt_length >= 8:
        with pytest.raises(RolloutContextOverflow, match="training cap"):
            ns["_generate_raw"](agent, [1] * prompt_length)
        assert requests == []
    else:
        _, metadata = ns["_generate_raw"](agent, [1] * prompt_length)
        assert len(requests) == 1
        assert metadata["behavior_turn"]["prompt_token_ids"] == [1] * prompt_length
    assert gate.active == 0


@pytest.mark.parametrize("initial_error", ["read_error", "read_timeout"])
@pytest.mark.parametrize("outcome", ["success", "read_error", "read_timeout", "missing_probability"])
@pytest.mark.parametrize("agent_timeout", [60, 600])
def test_agent_read_retry_keeps_turn_version_and_propagates_failures(monkeypatch, initial_error, outcome, agent_timeout):
    import httpx
    import json
    import logging
    import os
    import time
    from typing import Any

    gate = continuous.AgentTurns()
    gate.resume(7)
    monkeypatch.setenv("TAU2_RAW_TOKENS", "1")
    monkeypatch.setitem(sys.modules, "continuous", SimpleNamespace(AGENT_TURNS=gate))
    calls = []
    shared_closed = []

    def failed_post(url, json):
        calls.append(("shared", json))
        raise (httpx.ReadTimeout("timed out") if initial_error == "read_timeout"
               else httpx.ReadError("[Errno 9] Bad file descriptor"))

    def retry_post(request):
        assert gate.active == 1 and gate.version == 7
        assert request.extensions["timeout"]["read"] == agent_timeout
        calls.append(("fresh", json.loads(request.content)))
        if outcome == "read_error":
            raise httpx.ReadError("connection still unavailable")
        if outcome == "read_timeout":
            raise httpx.ReadTimeout("retry also timed out")
        meta = {} if outcome == "missing_probability" else {
            "completion_tokens": 2, "output_token_logprobs": [(-0.2, 42), (-0.3, 99)],
            "finish_reason": {"type": "stop", "matched": 99},
        }
        return httpx.Response(200, json={"text": "answer", "meta_info": meta})

    client_class = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_class(
        transport=httpx.MockTransport(retry_post), **kwargs))
    ns = extracted_functions(ROOT / "examples/tau2-bench/rl/agent.py", {"_generate_raw"}, {
        "os": os, "time": time, "Any": Any, "httpx": httpx, "logging": logging, "__name__": __name__,
        "shared_client": lambda timeout: SimpleNamespace(post=failed_post, close=lambda: shared_closed.append(True)),
        "recorded_turn": recorded_turn,
    })
    agent = SimpleNamespace(timeout=agent_timeout, sglang_url="http://test/generate", _sampling_params=lambda: {"temperature": 1.0})
    if outcome == "success":
        _, metadata = ns["_generate_raw"](agent, [1, 2, 3])
        assert metadata["behavior_turn"] == recorded_turn([1, 2, 3], {
            "completion_tokens": 2, "output_token_logprobs": [(-0.2, 42), (-0.3, 99)],
            "finish_reason": {"type": "stop", "matched": 99},
        }, 7)
    else:
        error_type = {"read_error": httpx.ReadError, "read_timeout": httpx.ReadTimeout,
                      "missing_probability": BehaviorLogprobError}[outcome]
        with pytest.raises(error_type):
            ns["_generate_raw"](agent, [1, 2, 3])
    assert [kind for kind, _ in calls] == ["shared", "fresh"]
    assert calls[0][1] == calls[1][1]
    assert gate.active == 0 and shared_closed == []


def test_manager_conversion_preserves_k8_and_whole_trajectory_denominators():
    import torch
    from types import MethodType
    from slime.utils.types import Sample

    ns = extracted_functions(ROOT / "slime/ray/rollout.py", {"_post_process_rewards", "_convert_samples_to_train_data"}, {"torch": torch, "Sample": Sample})
    args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True, n_samples_per_prompt=8, rollout_batch_size=1, reward_key=None, rollout_top_p=1.0)
    owner = SimpleNamespace(args=args, custom_reward_post_process_func=None, custom_convert_samples_to_train_data_func=None)
    owner._post_process_rewards = MethodType(ns["_post_process_rewards"], owner)
    samples = []
    for index in range(8):
        segments = training_segments([turn([1], [2, 3])] * (index + 1))
        sample = Sample(index=index, reward=float(index >= 4), status=Sample.Status.COMPLETED, training_segments=segments)
        for key, value in segments[0].items():
            setattr(sample, key, value)
        assert sample.effective_response_length == 2 * (index + 1)
        samples.append(sample)
    data = ns["_convert_samples_to_train_data"](owner, samples)
    assert len(data["tokens"]) == 36
    assert data["trajectory_rewards"] == [0] * 4 + [1] * 4
    assert set(data["rollout_ids"]) == set(range(8))
    for index, advantage, mask_sum in zip(data["rollout_ids"], data["rewards"], data["rollout_mask_sums"], strict=True):
        assert mask_sum == 2 * (index + 1)
        assert advantage == pytest.approx((float(index >= 4) - 0.5) / (torch.tensor([0.0] * 4 + [1.0] * 4).std().item() + 1e-6))


def test_real_tokenizer_returns_json_token_array(monkeypatch):
    import json
    import os
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]")), unk_token="[UNK]", chat_template="{{ messages[0]['content'] }}")

    class UserMessage:
        is_audio = False

    ns = extracted_functions(ROOT / "examples/tau2-bench/rl/agent.py", {"_generate_next_message"}, {"os": os, "ValidAgentInputMessage": object, "AssistantMessage": object, "UserMessage": UserMessage, "MultiToolMessage": type("MultiToolMessage", (), {})})
    monkeypatch.setenv("TAU2_RAW_TOKENS", "1")
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        json.dumps({"input_ids": prompt})
        return "hello", {}

    agent = SimpleNamespace(tokenizer=tokenizer, contract=None, native_tools=None, _chat_messages=lambda state: [{"role": "user", "content": "hello"}], _generate_raw=generate, _assistant_from_text=lambda text, meta: text)
    ns["_generate_next_message"](agent, UserMessage(), SimpleNamespace(messages=[]))
    assert prompts == [[0]]


@pytest.mark.parametrize("overflow_attempts", [0, 1, 3])
def test_overlong_retry_rebuilds_user_and_mutable_environment(monkeypatch, overflow_attempts):
    import os
    import time
    import logging
    from typing import Any
    from slime.utils.types import Sample

    environments, users = [], []

    class RolloutContextOverflow(ValueError):
        pass

    def make_environment():
        env = SimpleNamespace(counter=0, get_user_tools=lambda: [], get_tools=lambda: [], get_policy=lambda: "policy")
        environments.append(env)
        return env

    def build_user(kind, environment, task, **kwargs):
        def generate(message, state):
            environment.counter += 1
            return message, state

        user = SimpleNamespace(generate_next_message=generate, environment=environment)
        users.append(user)
        return user

    class Orchestrator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            assert self.kwargs["environment"] is self.kwargs["user"].environment
            self.kwargs["user"].generate_next_message("hello", {})
            if len(users) <= overflow_attempts:
                raise RolloutContextOverflow("maximum context length")
            return SimpleNamespace(messages=[], termination_reason="user_stop")

        def _finalize(self):
            assert self.termination_reason == "agent_error"
            return SimpleNamespace(messages=[], termination_reason="agent_error")

    ns = extracted_functions(
        ROOT / "examples/tau2-bench/rl/rollout.py",
        {"_run_tau2_rollout_sync"},
        {
            "os": os,
            "time": time,
            "logging": logging,
            "Any": Any,
            "Sample": Sample,
            "task_from_metadata": lambda metadata: SimpleNamespace(id="task"),
            "domain_from_metadata": lambda *a: "retail",
            "db_path_from_metadata": lambda metadata: "db",
            "make_environment_constructor": lambda **kwargs: make_environment,
            "agent_protocol_profile": lambda: "official-native",
            "_sampling_params_for_agent": dict,
            "TrainableSGLangAgent": lambda **kwargs: SimpleNamespace(system_prompt="system", contract=None, native_tools=[], protocol_signature=None, llm_args={}),
            "_agent_api_base": lambda args: "http://test",
            "_get_mask_generator": lambda args: None,
            "max_train_tokens": lambda: 8,
            "_sampling_params_for_attempt": lambda params, **kwargs: params,
            "build_user": build_user,
            "_with_openai_prefix": str,
            "_user_llm_args": dict,
            "Orchestrator": Orchestrator,
            "RolloutContextOverflow": RolloutContextOverflow,
            "TerminationReason": SimpleNamespace(AGENT_ERROR="agent_error"),
            "configured_turn_credit_version": lambda: "",
            "PROGRESS_DB_COUNT_V1": "progress-db-count-v1",
            # An interrupted attempt can retain only short earlier turns. The
            # overflow still requires retry/removal, not partial-episode credit.
            "_conversation_token_len": lambda *args, **kwargs: 9 if not overflow_attempts and len(users) == 1 else 5,
            "evaluate_simulation_with_constructor": lambda **kwargs: SimpleNamespace(reward=1.0),
            "reward_info_to_dict": lambda info: {},
            "_field_reward_payload": lambda **kwargs: {},
            "_fill_sample_from_simulation": lambda **kwargs: kwargs["sample"],
            "_dump_trajectory": lambda *args: None,
            "AssistantMessage": type("AssistantMessage", (), {}),
            "UserMessage": type("UserMessage", (), {}),
        },
    )
    monkeypatch.setenv("TAU2_RAW_TOKENS", "1")
    monkeypatch.setenv("TAU2_USE_REWARD_SHAPING", "0")
    monkeypatch.setenv("TAU2_RL_MAX_ROLLOUT_RETRIES", "2")
    sample = Sample(index=0, reward=1.0, metadata={})
    result = ns["_run_tau2_rollout_sync"](SimpleNamespace(hf_checkpoint="test", n_samples_per_prompt=8, rollout_seed=42), sample, {})
    expected_attempts = 3 if overflow_attempts == 3 else 2
    assert len(users) == expected_attempts and users[0] is not users[1]
    assert [env.counter for env in environments] == [1] * expected_attempts
    assert result.metadata["tau2_rollout_attempts"] == expected_attempts
    assert result.metadata["tau2_permanently_too_long"] == (overflow_attempts == 3)


@pytest.mark.parametrize("conversation_length,cap,interrupted,removed", [(7, 8, False, False), (9, 8, False, True), (7, 7, False, True), (7, 8, True, True)])
def test_raw_segments_cap_uses_max_context_not_sum_and_ignores_tool_text(monkeypatch, conversation_length, cap, interrupted, removed):
    import os
    from typing import Any
    from slime.utils.types import Sample
    from dataclasses import dataclass

    # Match Tau2's field ownership: tool messages have no raw_data field.
    @dataclass
    class AssistantMessage:
        role: str
        content: str
        raw_data: dict | None = None

    @dataclass
    class UserMessage:
        role: str
        content: str

    @dataclass
    class ToolMessage:
        id: str
        role: str
        content: str

    turns = [turn([1] * 6, [2]), turn([3] * 6, [4, 99], version=2)]
    simulation = SimpleNamespace(
        messages=[
            AssistantMessage(role="assistant", content="framework greeting"),
            UserMessage(role="user", content="request"),
            AssistantMessage(role="assistant", content="first", raw_data={"tau2_rl_agent": True, "meta_info": {"behavior_turn": turns[0]}}),
            ToolMessage(id="tool", role="tool", content="not a model generation"),
            AssistantMessage(role="assistant", content="second", raw_data={"tau2_rl_agent": True, "meta_info": {"behavior_turn": turns[1]}}),
        ]
    )

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [0]}

        def decode(self, tokens):
            return str(tokens)

    generator = SimpleNamespace(tokenizer=Tokenizer(), get_loss_mask=lambda *args, **kwargs: ([0] * conversation_length, []))
    ns = extracted_functions(
        ROOT / "examples/tau2-bench/rl/rollout.py",
        {"_behavior_turns", "_conversation_token_len", "_fill_raw_sample"},
        {
            "os": os,
            "Any": Any,
            "AssistantMessage": AssistantMessage,
            "Sample": Sample,
            "BehaviorLogprobError": BehaviorLogprobError,
            "training_segments": training_segments,
            "configured_turn_credit_version": lambda: "",
            "PROGRESS_DB_COUNT_V1": "progress-db-count-v1",            "MultiTurnLossMaskGenerator": object,
            "SimulationRun": object,
            "AgentContract": object,
            "_training_messages": lambda *args: ([], {}),
            "max_train_tokens": lambda: cap,
            "_sample_status": lambda simulation: Sample.Status.COMPLETED,
        },
    )
    monkeypatch.setenv("TAU2_RAW_TOKENS", "1")
    sample = ns["_fill_raw_sample"](generator, Sample(index=0, metadata={}), simulation, 1.0, "system", None, None, {}, permanently_too_long=interrupted)
    assert sample.remove_sample is removed
    assert sample.metadata["tau2_original_total_tokens"] == max(8, conversation_length)
    assert sample.metadata["tau2_token_weighted_policy_version"] == pytest.approx(4 / 3)
    if removed:
        assert sample.training_segments is None and sample.loss_mask == [0]
    else:
        assert len(sample.training_segments) == 2
        assert sample.training_segments[-1]["tokens"] == [3] * 6 + [4, 99]
        assert sample.effective_response_length == 3


@pytest.fixture
def random_source(monkeypatch, tmp_path):
    from slime.utils.types import Sample
    spec = importlib.util.spec_from_file_location("tau2_random_tasks_test", ROOT / "examples/tau2-bench/rl/random_tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = [Sample(prompt=str(i), metadata={"domain": d, "task": {"id": str(i)}}) for i, d in enumerate(["airline", "airline", "retail", "telecom"])]

    def initialize(self, args):
        self.args = args
        self.dataset = SimpleNamespace(origin_samples=prompts, shuffle=lambda epoch: None)
        self.metadata = {}
        self.sample_index = self.sample_group_index = self.sample_offset = self.epoch_id = 0

    monkeypatch.setattr(module.RolloutDataSource, "__init__", initialize)
    args = SimpleNamespace(rollout_seed=42, n_samples_per_prompt=8, rollout_global_dataset=True, rollout_shuffle=False, save=str(tmp_path), load=str(tmp_path))
    return lambda class_name="RandomTaskDataSource": getattr(module, class_name)(args)


def test_shuffled_tasks_cover_deck_and_retry_only_after_two_policy_updates(random_source):
    source = random_source("ShuffledTaskDataSource")
    first = source.get_samples(4)
    assert len({g[0].prompt for g in first}) == 4
    failed = first[0]
    source.set_policy_version(0)
    source.record_outcome(failed, "official_outcome_all_zero", 0)
    source.record_outcome(first[1], "official_outcome_all_one", 0)
    for version in (0, 1):
        source.set_policy_version(version)
        assert all(g[0].prompt != failed[0].prompt for g in source.get_samples(12))
    source.set_policy_version(2)
    retry, fresh = source.get_samples(2)
    assert retry[0].prompt == failed[0].prompt
    assert len(retry) == 8 and all(s.index > failed[-1].index and s.reward is None for s in retry)
    assert source.metadata["tau2_task_retry_draws"] == 1
    assert fresh[0].prompt != failed[0].prompt


def test_shuffled_pending_retries_and_deck_resume(random_source):
    source = random_source("ShuffledTaskDataSource")
    failed = source.get_samples(1)[0]
    source.record_outcome(failed, "official_outcome_all_zero", 3)
    source.set_policy_version(4)
    source.save(4)
    expected_early = source.get_samples(2)
    source.set_policy_version(5)
    expected_late = source.get_samples(5)
    restored = random_source("ShuffledTaskDataSource")
    restored.load(4)
    actual_early = restored.get_samples(2)
    restored.set_policy_version(5)
    actual_late = restored.get_samples(5)

    def describe(groups):
        return [[(sample.index, sample.group_index, sample.prompt) for sample in group] for group in groups]

    assert describe(actual_early + actual_late) == describe(expected_early + expected_late)


def test_partial_group_completion_replenishes_dialogues_with_bounded_memory(producer_fixture, monkeypatch):
    Source, _, _ = producer_fixture
    release, replacement = threading.Event(), threading.Event()

    def run(args, sample, params):
        if sample.group_index >= 2:
            replacement.set()
        if sample.group_index >= 2 or sample.index % 8 == 7:
            assert release.wait(5)
        sample.metadata["tau2_earliest_policy_version"] = 0
        sample.reward = sample.index % 2
        return sample

    monkeypatch.setattr(sys.modules["rollout"], "_run_tau2_rollout_sync", run)
    producer = continuous.Tau2Producer(pool_args(tau2_pool_capacity=2, tau2_max_pending_groups=4,
                                                rollout_batch_size=1, num_rollout=8,
                                                tau2_max_policy_lag=-1), Source())
    try:
        assert replacement.wait(3), "Free trajectory slots should admit a new group before the slow tails finish"
        with producer.condition:
            assert 0 not in producer.ready and 1 not in producer.ready
            assert producer.inflight <= 16
            assert len(producer.groups) - len(producer.ready) <= 4
    finally:
        release.set()
        producer.close()


def test_uniform_task_sampling_with_replacement_keeps_all_wrong_tasks(random_source):
    import random
    source = random_source()
    groups = source.get_samples(20)
    expected_rng = random.Random(42)
    assert [g[0].prompt for g in groups] == [str(expected_rng.randrange(4)) for _ in groups]
    assert len({g[0].prompt for g in groups}) < len(groups)
    assert all(len(g) == 8 and len({s.prompt for s in g}) == 1 for g in groups)
    groups[0][0].metadata["task"]["id"] = "mutated"
    assert all(s.metadata["task"]["id"] != "mutated" for s in groups[0][1:])
    for group in groups:
        for sample in group:
            sample.reward = 0.0
    # Outcomes do not remove tasks or change the uniform draw distribution.
    more = source.get_samples(20)
    assert [g[0].prompt for g in more] == [str(expected_rng.randrange(4)) for _ in more]
    assert {g[0].prompt for g in groups} <= {g[0].prompt for g in more}
    assert sum(source.metadata["tau2_task_draw_counts"].values()) == 40


def test_random_task_rng_and_pending_inputs_resume(random_source):
    source = random_source()
    pending = source.get_samples(3)
    source.metadata["tau2_pending_groups"] = {g[0].group_index: g for g in pending}
    source.save(4)
    expected = source.get_samples(10)
    restored = random_source()
    restored.load(4)
    assert list(restored.metadata["tau2_pending_groups"]) == [0, 1, 2]
    actual = restored.get_samples(10)
    assert [[(s.index, s.group_index, s.prompt) for s in g] for g in actual] == [[(s.index, s.group_index, s.prompt) for s in g] for g in expected]


@pytest.mark.parametrize("start_at,fail_stage", [(None, None), (None, "async-train"),
                                               (None, "sync-eval300"), ("sync-eval301", None)])
def test_serial_pipeline_order_failure_and_environment(tmp_path, start_at, fail_stage):
    import os
    import subprocess

    trace = tmp_path / "trace"
    scripts = {
        "examples/tau2-bench/rl/run_tau2_areal_async.sh":
            'stage="$1-train"\n[[ "$2" == train100 && "$TAU2_RL_CLEANUP" == 0 ]]\n',
        "scripts/convert_tau2_areal_async_to_hf.sh": 'stage="$1-convert"\n',
        "examples/tau2-bench/eval/official/models/run_tau2_areal_async.sh":
            'stage="$1-eval$2"\n[[ "$TAU2_ITERATION" == 99 && "$TAU2_EVAL_SUFFIX" == -test-run ]]\n',
        "bin/python3": 'stage=report\n',
    }
    for relative, body in scripts.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/bash\nset -eu\n' + body +
                        '[[ -z "${SUMMARY_OUTPUT:-}" && -z "${LOAD_DIR:-}" ]]\n'
                        'echo "$stage" >> "$TRACE"\n'
                        'echo "stage-output:$stage"\n'
                        'export SUMMARY_OUTPUT=must-not-leak\n'
                        'if [[ "$stage" == "${FAIL_STAGE:-}" ]]; then exit 7; fi\n')
        path.chmod(0o755)
    ray = tmp_path / "bin/ray"
    ray.write_text('#!/bin/bash\necho ray-cleanup >> "$TRACE"\n')
    ray.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=str(tmp_path), TRACE=str(trace),
               FAIL_STAGE=fail_stage or "", PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
               SUMMARY_OUTPUT="inherited", LOAD_DIR="wrong-checkpoint")
    command = ["bash", str(ROOT / "scripts/run_tau2_serial_train100_eval.sh"), "--run-name", "test-run"]
    if start_at:
        command += ["--start-at", start_at]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    order = ["sync-train", "ray-cleanup", "async-train", "ray-cleanup", "sync-convert",
             "sync-eval300", "sync-eval301", "async-convert", "async-eval300", "async-eval301", "report"]
    if start_at:
        order = order[order.index(start_at):]
    if fail_stage:
        end = order.index(fail_stage) + 1 + int(fail_stage.endswith("-train"))
        order = order[:end]
    assert result.returncode == (7 if fail_stage else 0), result.stderr
    assert trace.read_text().splitlines() == order
    logs = list((tmp_path / "output/experiments/tau2-areal-async-rl/serial/test-run").glob("*_*.log"))
    assert len(logs) == len([s for s in order if s != "ray-cleanup"])
    for log in logs:
        assert log.read_text().count("stage-output:") == 1



@pytest.mark.parametrize("reward", [0.0, 1.0, None])
@pytest.mark.parametrize("aux_weight", [0.0, 1.0])
def test_db_count_raw_segments_keep_uniform_groups(monkeypatch, reward, aux_weight):
    monkeypatch.setenv("TAU2_DROP_UNIFORM_OUTCOME_GROUPS", "0")
    sys.path.insert(0, str(ROOT / "examples/tau2-bench/rl"))
    import reward_postprocess as rp
    from slime.utils.types import Sample
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput
    from collections import defaultdict
    import math
    import os
    import torch
    # Keep the CPU CI test independent of the optional Tau2 environment package.
    functions = extracted_functions(ROOT / "examples/tau2-bench/rl/progress.py",
        {"group_normalize", "db_count_group_advantages"}, {"math": math, "defaultdict": defaultdict})
    monkeypatch.setitem(sys.modules, "progress", SimpleNamespace(db_count_group_advantages=functions["db_count_group_advantages"]))
    filters = extracted_functions(ROOT / "examples/tau2-bench/rl/filters.py",
        {"_sample_domain", "_group_domain", "drop_zero_std_or_unsampleable"}, {
            "os": os, "torch": torch, "Sample": Sample, "DynamicFilterOutput": DynamicFilterOutput,
            "TURN_CREDIT_V1": rp.TURN_CREDIT_V1, "TURN_CREDIT_V2": rp.TURN_CREDIT_V2,
            "PROGRESS_DB_COUNT_V1": rp.PROGRESS_DB_COUNT_V1, "_REPLACE_ZERO_SIGNAL_ENV": "TAU2_REPLACE_ZERO_SIGNAL_GROUPS",
            "_ZERO_STD_TOL": 1e-6, "TurnCreditAlignmentError": rp.TurnCreditAlignmentError,
            "validate_token_penalties": rp.validate_token_penalties, "compute_rollout_score": rp.compute_rollout_score,
            "turn_credit_v2_group_has_signal": rp.turn_credit_v2_group_has_signal,
        })
    drop_zero_std_or_unsampleable = filters["drop_zero_std_or_unsampleable"]
    monkeypatch.setenv("TAU2_TURN_CREDIT_VERSION", rp.PROGRESS_DB_COUNT_V1)
    monkeypatch.setenv("TAU2_REPLACE_ZERO_SIGNAL_GROUPS", "0")
    monkeypatch.setenv("TAU2_PROGRESS_WEIGHT", str(aux_weight))
    monkeypatch.setenv("TAU2_FORMAT_WEIGHT", str(aux_weight))
    args = SimpleNamespace(n_samples_per_prompt=8, advantage_estimator="grpo",
                           rewards_normalization=True, grpo_std_normalization=True, reward_key=None)
    samples = []
    for i in range(8):
        # First two turns merge, third reserializes history and becomes a new segment.
        segments = training_segments([turn([1, 2], [3, 99]),
                                      turn([1, 2, 3, 99, 4], [5, 99]),
                                      turn([1, 9, 3, 99], [6, 99])], record_spans=True)
        details = [{**span, "simulation_message_index": 2 * span["turn_index"] + 1,
                    "segment_index": j, "errors": []}
                   for j, segment in enumerate(segments) for span in segment["assistant_spans"]]
        sample = Sample(index=i, group_index=0, reward=float(i >= 4) if reward is None else reward, rollout_id=i,
                        metadata={"domain": "banking", "tau2_domain": "banking", "tau2_task_id": "same",
                                  "tau2_turn_credit_version": rp.PROGRESS_DB_COUNT_V1,
                                  "tau2_turn_credits": details,
                                  "tau2_progress": {"source_indices": [1, 3, 5], "scores": [0, i, 0, 2 * i],
                                                    "db_diff_counts": [0, 1, 0], "db_diff_side_counts": [{}] * 3,
                                                    "unavailable_reason": None}},
                        training_segments=segments, train_metadata={"turn_credit_version": rp.PROGRESS_DB_COUNT_V1})
        for key in ("tokens", "response_length", "loss_mask", "rollout_log_probs"):
            setattr(sample, key, segments[0][key])
        samples.append(sample)
    assert drop_zero_std_or_unsampleable(args, samples).keep
    monkeypatch.setenv("TAU2_DROP_UNIFORM_OUTCOME_GROUPS", "1")
    decision = drop_zero_std_or_unsampleable(args, samples)
    assert decision.keep == (reward is None)
    if reward is not None:
        assert decision.reason == ("official_outcome_all_zero" if reward == 0 else "official_outcome_all_one")
    vanilla_samples = [Sample(index=i, reward=s.reward, metadata={"domain": "retail"})
                       for i, s in enumerate(samples)]
    assert drop_zero_std_or_unsampleable(args, vanilla_samples).keep == (reward is None)
    monkeypatch.setenv("TAU2_DROP_UNIFORM_OUTCOME_GROUPS", "0")
    raw, advantages = rp.tau2_reward_post_process(args, samples)
    expanded, _, _ = expand_training_segments(samples, raw, advantages)
    assert len(expanded) == 16
    for part in expanded:
        values = part.train_metadata["token_advantages"]
        assert len(values) == part.response_length
        assert all(value == 0 for value, mask in zip(values, part.loss_mask, strict=True) if not mask)
        # The stop token receives the same turn advantage as the output preceding it.
        assert values[-1] == values[-2]
        if aux_weight == 0:
            expected = [advantages[part.rollout_id] if mask else 0.0 for mask in part.loss_mask]
            assert values == expected
    assert expanded[0].train_metadata is not expanded[1].train_metadata
    assert all(s.metadata["tau2_turn_credit_group_has_signal"] == bool(aux_weight or reward is None) for s in samples)
    if reward is not None:
        assert all(value == 0 for value in advantages)




def test_four_domain_single_seed_report_checks_788_trials(tmp_path, monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(ROOT / "examples/tau2-bench/analysis"))
    from summarize_raw_processed_vanilla_grpo import load_evaluation, aggregate_evaluations, compare_models
    counts = {"airline": 20, "retail": 40, "telecom": 40, "banking_knowledge": 97}
    passing = {"pass_at_1": 1.0, "pass_at_4_any": 1.0, "pass_power_4": 1.0}
    summary = {"task_split_name": "test", "num_trials": 4, "seed": 300, "max_steps": 200,
               "agent_eval_mode": "official-native", "domains": {},
               "overall": {**passing, "tasks": 197, "simulations": 788, "diagnostics": {"infrastructure_errors": 0}}}
    for domain, n in counts.items():
        simulations = [{"task_id": str(i), "trial": trial, "reward_info": {"reward": 1}, "termination_reason": "user_stop"}
                       for i in range(n) for trial in range(4)]
        result = tmp_path / f"{domain}.json"
        result.write_text(json.dumps({"simulations": simulations}))
        summary["domains"][domain] = {"results_file": str(result),
            "pass_metrics": {**passing, "tasks": n, "simulations": n * 4},
            "metrics": {"infra_error_count": 0, "db_match_count": n * 4}}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    rows = {300: load_evaluation(path, expected_seed=300, results_root=tmp_path, label="test", expected_tasks=counts)}
    metrics = aggregate_evaluations(rows, seeds=(300,), domains=counts)
    assert set(metrics["by_domain"]) == set(counts)
    comparison = compare_models(key="same", candidate=rows, reference=rows, candidate_metrics=metrics,
        reference_metrics=metrics, samples=100, bootstrap_seed=1234, seeds=(300,), domains=counts)
    assert len(comparison["paired_bootstrap"]) == 15
    assert all(row["ci95"] == [0.0, 0.0] for row in comparison["paired_bootstrap"])



@pytest.mark.parametrize("recipe", ["vanilla-grpo", "progress-db-count-v1"])
@pytest.mark.parametrize("mode,updates", [("smoke", "5"), ("train", "556")])
def test_copied_four_domain_launcher_fixed_recipe(tmp_path, recipe, mode, updates):
    import json
    import os
    import subprocess
    project = tmp_path / "repo"
    entry = project / "examples/tau2-bench/rl/run_tau2_areal_async.sh"
    entry.parent.mkdir(parents=True)
    keys = ["PROJECT_ROOT", "TAU2_RL_RECIPE", "TAU2_TURN_CREDIT_VERSION", "NUM_ROLLOUT", "REF_CKPT_STEP",
            "TAU2_MAX_POLICY_LAG", "TAU2_POOL_CAPACITY", "TAU2_REPLACE_ZERO_SIGNAL_GROUPS", "TAU2_RAW_TOKENS", "HF_CHECKPOINT"]
    script = "import os,json; print(json.dumps({k:os.environ.get(k) for k in " + repr(keys) + "}))"
    import shlex
    entry.write_text("#!/usr/bin/env bash\npython3 -c " + shlex.quote(script) + "\n")
    data = project / "output/experiments/tau2-async-db-count-four-domain/data/train.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text("")
    copied = tmp_path / "job_snapshot.sh"
    copied.write_text((ROOT / "scripts/run_tau2_async_four_domain.sh").read_text())
    result = subprocess.run(["bash", str(copied), recipe, mode], env={**os.environ, "PROJECT_ROOT": str(project)},
                            text=True, capture_output=True, check=True)
    values = json.loads(result.stdout)
    assert values["PROJECT_ROOT"] == str(project)
    assert values["TAU2_RL_RECIPE"] == recipe
    assert values["TAU2_TURN_CREDIT_VERSION"] == (recipe if recipe == "progress-db-count-v1" else "")
    assert values["NUM_ROLLOUT"] == updates
    assert values["REF_CKPT_STEP"] == "4673"
    assert values["TAU2_MAX_POLICY_LAG"] == "-1"
    assert values["TAU2_POOL_CAPACITY"] == "20"
    assert values["TAU2_REPLACE_ZERO_SIGNAL_GROUPS"] == "0"
    assert values["TAU2_RAW_TOKENS"] == "1"
    assert values["HF_CHECKPOINT"].endswith("iter_0004673_hf")




def test_smoke_inspector_uses_logged_unlimited_lag(tmp_path, monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(ROOT / "examples/tau2-bench/analysis"))
    from inspect_tau2_ready_smoke import inspect
    log = tmp_path / "run.log"
    log.write_text("tau2_max_policy_lag ............. -1\ntau2_pool_batch update=2 groups=[0, 1, 2, 3, 4]\n"
                   "tau2_trainer batch=2 start=1.0 end=2.0\n")
    trace = tmp_path / "trajectories.jsonl"
    with trace.open("w") as stream:
        for i in range(40):
            row = {"sample_index": i, "group_index": i // 8, "task_id": str(i // 8), "remove_sample": False, "reward": 1,
                   "metadata": {"domain": "retail", "tau2_training_segments": 1, "tau2_train_total_tokens": 2,
                                "tau2_rollout_attempts": 1, "tau2_earliest_policy_version": 0},
                   "simulation": {"messages": [{"raw_data": {"meta_info": {"behavior_turn": {
                       "prompt_token_ids": [1], "output_token_ids": [99], "log_probs": [-0.1], "policy_version": 0,
                       "finish_reason": {"matched": 99}}}}}]}}
            stream.write(json.dumps(row) + "\n")
    assert inspect([log], trace)["violations"] == []
    assert any("lag" in v for v in inspect([log], trace, max_policy_lag=1)["violations"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_four_domain_monitor_waits_for_completed_save_and_deduplicates(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("four_domain_monitor", ROOT / "scripts/monitor_tau2_four_domain.py")
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    experiment = tmp_path / "output/experiments/tau2-async-db-count-four-domain"
    run = experiment / "arms/async/stamp-vanilla-grpo-train"
    run.mkdir(parents=True)
    log = run / "run.log"
    log.write_text("tau2_trainer batch=49 start=1.0 end=2.0\ntau2_checkpoint batch=49 start=2.0\n")
    monkeypatch.setattr(monitor, "job_status", lambda job: {"state": "RUNNING"})
    submitted = []

    def submit(root, experiment, stamp, recipe, iteration, name):
        submitted.append((recipe, iteration))
        (experiment / "jobs" / f"123-{name}-0915-020000000").mkdir(parents=True)
        return 123

    monkeypatch.setattr(monitor, "submit_evaluation", submit)
    monitor.inspect(tmp_path, 1, "stamp")
    assert submitted == []
    with log.open("a") as stream:
        stream.write("tau2_checkpoint batch=49 start=2.0 end=3.0\n"
                     "tau2_checkpoint batch=59 start=3.0 end=4.0\n")
    result = monitor.inspect(tmp_path, 1, "stamp")
    assert submitted == [("vanilla-grpo", 49)]
    assert result["arms"]["vanilla-grpo"]["updates"] == 1
    monkeypatch.setattr(monitor, "job_status", lambda job: {"state": "FAILED" if job == 123 else "RUNNING"})
    monitor.inspect(tmp_path, 1, "stamp")
    assert submitted == [("vanilla-grpo", 49)]
    assert monitor.ITERATIONS == tuple(range(49, 550, 50))
    assert set(monitor.RECIPES) == {"vanilla-grpo", "progress-db-count-v1"}

    db_run = experiment / "arms/async/stamp-progress-db-count-v1-train"
    db_run.mkdir(parents=True)
    (db_run / "run.log").write_text("tau2_trainer batch=49 start=1.0 end=2.0\n"
                                    "tau2_checkpoint batch=49 start=2.0 end=3.0\n")
    result = monitor.inspect(tmp_path, 1, "stamp", recipes=["progress-db-count-v1"])
    assert set(result["arms"]) == {"progress-db-count-v1"}
    assert set(result["evaluations"]) == {"progress-db-count-v1-iter49"}
    assert submitted == [("vanilla-grpo", 49), ("progress-db-count-v1", 49)]


def test_four_domain_monitor_submits_fixed_eval_and_documents_job(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("four_domain_monitor", ROOT / "scripts/monitor_tau2_four_domain.py")
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    experiment = tmp_path / "output/experiments/tau2-async-db-count-four-domain"
    experiment.mkdir(parents=True)
    (experiment / "README.md").write_text("# Experiment\n\n## Reporting\n")
    index = tmp_path / "output/doc/INDEX.md"
    index.parent.mkdir(parents=True)
    index.write_text("[Experiment](../experiments/tau2-async-db-count-four-domain/README.md)\n")
    command = []

    def execute(argv, **kwargs):
        command.extend(argv)
        return SimpleNamespace(stdout="[OK] id=321 queued")

    monkeypatch.setattr(monitor.subprocess, "run", execute)
    monkeypatch.setattr(monitor, "job_status", lambda job: {
        "run_log": str(experiment / "jobs/321-eval/run_*_20260915_020000000.log")})
    job = monitor.submit_evaluation(tmp_path, experiment, "stamp", "progress-db-count-v1", 99, "eval-name")
    assert job == 321
    assert command[-3:] == ["--", "progress-db-count-v1", "99"]
    assert command[command.index("--gpus") + 1] == "8"
    assert command[command.index("--cpus") + 1] == "64"
    assert command[command.index("--memory") + 1] == "1024"
    assert "TAU2_RUN_NAME=stamp-progress-db-count-v1-train" in command
    assert "jobs/321-eval/run_0_20260915_020000000.log" in (experiment / "README.md").read_text()
    assert "321 (progress-db-count-v1 iter99)" in index.read_text()


@pytest.mark.parametrize("status,body,overflow", [
    (400, "Requested token count exceeds the model's maximum context length of 262144 tokens.", True),
    (400, "The input (263229 tokens) is longer than the model's context length (262144 tokens).", True),
    (400, "Invalid sampling parameter", False),
    (500, "maximum context length", False),
])
def test_agent_context_overflow_is_distinct_from_other_http_errors(monkeypatch, status, body, overflow):
    import httpx
    import os
    import time
    from typing import Any

    class RolloutContextOverflow(ValueError):
        pass

    gate = continuous.AgentTurns()
    gate.resume(13)
    monkeypatch.setenv("TAU2_RAW_TOKENS", "1")
    monkeypatch.setitem(sys.modules, "continuous", SimpleNamespace(AGENT_TURNS=gate))
    response = httpx.Response(status, text=body, request=httpx.Request("POST", "http://test/generate"))
    ns = extracted_functions(ROOT / "examples/tau2-bench/rl/agent.py", {"_generate_raw"}, {
        "os": os, "time": time, "Any": Any, "httpx": httpx,
        "RolloutContextOverflow": RolloutContextOverflow,
        "shared_client": lambda timeout: SimpleNamespace(post=lambda *a, **k: response),
    })
    agent = SimpleNamespace(timeout=600, sglang_url="http://test/generate", _sampling_params=lambda: {"max_new_tokens": 1200})
    with pytest.raises(RolloutContextOverflow if overflow else httpx.HTTPStatusError):
        ns["_generate_raw"](agent, [1, 2, 3])
    assert gate.active == 0
