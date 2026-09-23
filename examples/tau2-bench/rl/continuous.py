"""Bounded Tau2 sampling and Agent-turn weight-update boundary."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class AgentTurns:
    def __init__(self, capacity=32):
        self.condition = threading.Condition()
        self.capacity = capacity
        self.active = 0
        self.paused = False
        self.closed = False
        self.version = 0

    @contextmanager
    def request(self):
        queued = time.time()
        with self.condition:
            self.condition.wait_for(lambda: self.closed or (not self.paused and self.active < self.capacity))
            if self.closed:
                raise RuntimeError("Tau2 producer closed")
            self.active += 1
            version = self.version
        started = time.time()
        try:
            if self is AGENT_TURNS and getattr(DIALOGUE, "record_version", None):
                DIALOGUE.record_version(version)
            yield version
        finally:
            event = "tau2_agent_turn" if self is AGENT_TURNS else "tau2_sampling_step"
            ended = time.time()
            timings = getattr(DIALOGUE, "timings", None)
            if timings is not None:
                name = "agent" if self is AGENT_TURNS else "step"
                timings[name + "_wait_seconds"] = timings.get(name + "_wait_seconds", 0.0) + started - queued
                timings[name + "_active_seconds"] = timings.get(name + "_active_seconds", 0.0) + ended - started
            with self.condition:
                self.active -= 1
                self.condition.notify_all()
            logger.debug("%s version=%s queued=%.6f start=%.6f end=%.6f", event, version, queued, started, ended)

    def pause(self):
        started = time.time()
        with self.condition:
            self.paused = True
            active = self.active
            self.condition.wait_for(lambda: self.active == 0)
        logger.info("tau2_pause_gate start=%.6f end=%.6f active_at_entry=%s", started, time.time(), active)

    def resume(self, version):
        with self.condition:
            self.version = version
            self.paused = False
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


AGENT_TURNS = AgentTurns(capacity=int(os.environ.get("TAU2_AGENT_CONCURRENCY", "32")))
SAMPLING_STEPS = AgentTurns(capacity=int(os.environ.get("TAU2_STEP_CONCURRENCY", "80")))
DIALOGUE = threading.local()


def _apply_opd_teacher(args, sample):
    """Fetch the teacher response when the producer bypasses custom RM hooks."""

    if os.environ.get("TAU2_OPD_PURE", "0") != "1":
        return sample
    from rollout import mark_opd_sample
    from tau2_opd import reward_func_for_training_segments

    mark_opd_sample(sample)
    teacher_responses = asyncio.run(reward_func_for_training_segments(args, sample))
    sample.metadata["tau2_opd_segment_responses"] = teacher_responses
    if len(teacher_responses) == 1:
        sample.metadata["tau2_opd_teacher_response"] = teacher_responses[0]
    # Keep producer metrics and filters numeric; post_process_rewards consumes
    # the response from metadata after the producer returns the group.
    sample.reward = 0.0
    return sample


def sampling_step(function):
    """Pause complete orchestrator steps at the synchronous training barrier."""
    def run(*args, **kwargs):
        with SAMPLING_STEPS.request():
            return function(*args, **kwargs)
    return run


class Tau2Producer:
    """Bounded public pool of complete K-trajectory groups, without batch binding."""

    def __init__(self, args, data_source):
        from rollout import _get_mask_generator
        from agent import shared_client, shared_tokenizer
        from slime.rollout.sglang_rollout import GenerateState

        self.args = args
        self.source = data_source
        self.condition = threading.Condition(threading.RLock())
        self.capacity = args.tau2_pool_capacity
        self.max_groups = getattr(args, "tau2_max_pending_groups", None) or self.capacity
        if self.max_groups < self.capacity:
            raise ValueError("Tau2 max pending groups must be at least pool capacity")
        self.inflight = 0
        self.batch_size = args.rollout_batch_size
        if self.capacity < self.batch_size:
            raise ValueError("Tau2 pool must hold at least one training batch")
        self.max_buffered_groups = getattr(args, "tau2_max_buffered_groups", -1)
        if self.max_buffered_groups != -1 and self.max_buffered_groups < self.batch_size:
            raise ValueError("Tau2 buffered group limit must hold at least one training batch, or be -1")
        self.environment_workers = []
        worker_count = getattr(args, "tau2_environment_workers", 1)
        if worker_count > 1:
            if args.tau2_sampling_mode != "async" or getattr(args, "tau2_max_policy_lag", 1) >= 0:
                raise ValueError("Tau2 environment workers require async sampling with unlimited policy lag")
            if worker_count > min(AGENT_TURNS.capacity, SAMPLING_STEPS.capacity):
                raise ValueError("Environment worker count exceeds the global request capacity")
            import ray
            self.ray = ray
            worker_class = ray.remote(Tau2EnvironmentWorker).options(
                num_cpus=1,
                # Control RPCs must run even when every admitted trajectory is
                # waiting at a paused Agent gate. The global slot bound limits
                # the number of run() calls across all workers.
                max_concurrency=self.capacity * args.n_samples_per_prompt + 2,
            )
            for index in range(worker_count):
                agent_capacity = AGENT_TURNS.capacity // worker_count + (index < AGENT_TURNS.capacity % worker_count)
                step_capacity = SAMPLING_STEPS.capacity // worker_count + (index < SAMPLING_STEPS.capacity % worker_count)
                self.environment_workers.append(worker_class.remote(args, agent_capacity, step_capacity))
            ray.get([worker.ready.remote() for worker in self.environment_workers])
        self.worker_inflight = [0] * len(self.environment_workers)
        self.workers = ThreadPoolExecutor(max_workers=self.capacity * args.n_samples_per_prompt,
                                          thread_name_prefix="tau2-dialogue")
        self.collectors = ThreadPoolExecutor(max_workers=self.max_groups, thread_name_prefix="tau2-group")
        self.pending = self.source.metadata.setdefault("tau2_pending_groups", {})
        self.groups = {}
        self.ready = {}  # Dict insertion order is group completion order.
        self.taken = []
        self.version = args.start_rollout_id
        self.consumed = self.version * self.batch_size
        self.training = False
        self.awaiting_weights = False
        self.closed = False
        self.error = None
        self.replacements = 0
        self.filter_counts = {}
        self.sampling_params = dict(GenerateState(args).sampling_params)
        shared_tokenizer(args.hf_checkpoint)
        shared_client(float(os.environ.get("TAU2_AGENT_TIMEOUT", 600.0)))
        _get_mask_generator(args)
        AGENT_TURNS.resume(self.version)
        SAMPLING_STEPS.resume(self.version)
        if hasattr(self.source, "set_policy_version"):
            self.source.set_policy_version(self.version)
        with self.condition:
            self.refill()

    def start_group(self, key):
        self.groups[key] = {"earliest": None, "start": time.time(), "remaining": self.args.n_samples_per_prompt}
        self.inflight += self.args.n_samples_per_prompt
        inputs = copy.deepcopy(self.pending[key])
        futures = [self.workers.submit(self.run_dialogue, key, sample) for sample in inputs]
        self.collectors.submit(self.collect_group, key, futures)

    def refill(self):
        # Dialogue concurrency and total prefetch are separate budgets. Ready
        # groups and groups currently training still occupy the prefetch budget.
        if self.closed or self.error:
            return
        remaining = self.args.num_rollout * self.batch_size - self.consumed
        limit = min(self.capacity, remaining)
        unlimited = getattr(self.args, "tau2_max_policy_lag", 1) < 0
        if not unlimited and (self.training or self.awaiting_weights):
            limit = min(limit, self.batch_size + len(self.taken))
        # Pending groups can still fail filtering. Near the last update, keep
        # supplying slots until enough *complete* groups cover the budget;
        # otherwise the last batch is forced to wait for every slow tail.
        while (len(self.ready) if unlimited else len(self.groups)) + len(self.taken) < remaining:
            if self.max_buffered_groups >= 0 and len(self.groups) + len(self.taken) >= self.max_buffered_groups:
                break
            if unlimited:
                if (len(self.groups) - len(self.ready) >= self.max_groups
                        or self.inflight + self.args.n_samples_per_prompt > self.capacity * self.args.n_samples_per_prompt):
                    break
            elif len(self.groups) + len(self.taken) >= limit:
                break
            # Resume saved task inputs under the same admission limits; do not
            # launch the entire old pending queue at once when its cap changed.
            key = next((key for key in self.pending if key not in self.groups and key not in self.taken), None)
            if key is None:
                drawn = self.source.get_samples(1)
                if not drawn:
                    break
                group = drawn[0]
                key = group[0].group_index
                self.pending[key] = group
            self.start_group(key)
        occupied = (len(self.ready) if unlimited else len(self.groups)) + len(self.taken)
        buffered = len(self.groups) + len(self.taken)
        if occupied >= remaining:
            reason = "budget"
        elif self.max_buffered_groups >= 0 and buffered >= self.max_buffered_groups:
            reason = "buffer"
        else:
            reason = "capacity" if unlimited or occupied >= self.capacity else "lag"
        logger.info("tau2_pool_admission time=%.6f reason=%s active=%s ready=%s training=%s consumed=%s inflight=%s",
                    time.time(), reason, len(self.groups) - len(self.ready), len(self.ready), len(self.taken), self.consumed, self.inflight)

    def run_dialogue(self, key, sample):
        if self.environment_workers:
            with self.condition:
                index = min(range(len(self.worker_inflight)), key=self.worker_inflight.__getitem__)
                self.worker_inflight[index] += 1
            try:
                return self.ray.get(self.environment_workers[index].run.remote(sample, self.sampling_params))
            finally:
                with self.condition:
                    self.worker_inflight[index] -= 1
        from rollout import _run_tau2_rollout_sync

        def record_version(version):
            with self.condition:
                entry = self.groups[key]
                entry["earliest"] = version if entry["earliest"] is None else min(entry["earliest"], version)
                self.condition.notify_all()

        DIALOGUE.record_version = record_version
        try:
            return _apply_opd_teacher(
                self.args,
                _run_tau2_rollout_sync(self.args, sample, dict(self.sampling_params)),
            )
        finally:
            DIALOGUE.record_version = None
            DIALOGUE.timings = None

    def collect_group(self, key, futures):
        from filters import drop_zero_std_or_unsampleable

        try:
            group = [None] * len(futures)
            positions = {future: index for index, future in enumerate(futures)}
            for future in as_completed(futures):
                group[positions[future]] = future.result()
                with self.condition:
                    self.inflight -= 1
                    self.groups[key]["remaining"] -= 1
                    # Complete groups are filtered below before replenishing.
                    if self.groups[key]["remaining"]:
                        self.refill()
            decision = drop_zero_std_or_unsampleable(self.args, group)
            keep = decision.keep
            with self.condition:
                entry = self.groups[key]
                ended = time.time()
                if keep:
                    # Finished retries may have replaced an earlier failed attempt.
                    entry["earliest"] = min(s.metadata["tau2_earliest_policy_version"] for s in group)
                    entry["duration"] = ended - entry["start"]
                    self.ready[key] = group
                else:
                    del self.groups[key]
                    del self.pending[key]
                    self.replacements += 1
                    reason = getattr(decision, "reason", None) or "filtered"
                    self.filter_counts[reason] = self.filter_counts.get(reason, 0) + 1
                    if hasattr(self.source, "record_outcome"):
                        self.source.record_outcome(group, reason, self.version)
                self.refill()
                logger.info("tau2_producer group=%s start=%.6f end=%.6f accepted=%s", key, entry["start"], ended, keep)
                self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                if not self.closed and self.error is None:
                    self.error = error
                self.condition.notify_all()
            AGENT_TURNS.close()
            SAMPLING_STEPS.close()

    def generate(self, args, rollout_id, data_source, evaluation=False):
        from slime.rollout.base_types import RolloutFnTrainOutput

        if evaluation:
            raise ValueError("Tau2 training producer requires separate official evaluation")
        started = time.time()
        waits = {"ready": 0.0, "lag": 0.0}
        with self.condition:
            while True:
                if self.error:
                    raise self.error
                if self.closed:
                    raise RuntimeError("Tau2 producer closed")
                lag_limit = getattr(self.args, "tau2_max_policy_lag", 1)
                urgent = []
                if lag_limit >= 0:
                    urgent = [key for key, entry in self.groups.items()
                              if entry["earliest"] is not None and entry["earliest"] < rollout_id]
                    stale = [key for key in urgent
                              if self.groups[key]["earliest"] < rollout_id - lag_limit]
                    if stale:
                        raise RuntimeError(
                            f"Tau2 earliest-turn policy lag exceeds {lag_limit} updates: {stale}"
                        )
                    if len(urgent) > self.batch_size:
                        raise RuntimeError("Too many old-policy groups for one Tau2 update")
                reason = "lag" if any(key not in self.ready for key in urgent) else "ready"
                if reason == "ready" and len(self.ready) >= self.batch_size:
                    if lag_limit >= 0:
                        keys = urgent + [key for key in self.ready if key not in urgent][:self.batch_size - len(urgent)]
                    else:
                        # Consume the oldest ready complete groups. With an
                        # unlimited lag gate there is no unfinished-group barrier.
                        keys = [key for key in self.ready][:self.batch_size]
                    break
                wait_start = time.time()
                self.condition.wait()
                waits[reason] += time.time() - wait_start
            groups = [self.ready.pop(key) for key in keys]
            duration = max(self.groups[key]["duration"] for key in keys)
            for key in keys:
                del self.groups[key]
            self.taken = keys
            self.training = True
            ready_count = len(self.ready)
            buffered_count = len(self.groups) + len(self.taken)
        barrier_start = time.time()
        if args.tau2_sampling_mode == "sync":
            SAMPLING_STEPS.pause()
        versions = [s.metadata["tau2_earliest_policy_version"] for g in groups for s in g]
        lags = [rollout_id - v for v in versions]
        lag_limit = getattr(self.args, "tau2_max_policy_lag", 1)
        if min(lags) < 0 or (lag_limit >= 0 and max(lags) > lag_limit):
            raise RuntimeError(f"Tau2 earliest-turn policy lag outside configured limit {lag_limit}: {lags}")
        logger.info("tau2_pool_batch update=%s groups=%s ready_wait=%.6f lag_wait=%.6f",
                    rollout_id, keys, waits["ready"], waits["lag"])
        counts = {domain: sum(g[0].metadata["domain"] == domain for g in groups)
                  for domain in ("airline", "retail", "telecom", "banking", "banking_knowledge")}
        timings = [s.metadata.get("tau2_timing", {}) for g in groups for s in g]
        timing_keys = {key for timing in timings for key in timing}
        opd_metrics = {}
        if getattr(args, "use_opd", False):
            for domain, count in counts.items():
                if not count:
                    continue
                samples = [s for g in groups for s in g if s.metadata["domain"] == domain]
                tokens = sum(s.metadata["tau2_response_tokens"] for s in samples)
                prefix = f"rollout/opd/{domain}"
                opd_metrics.update({
                    f"{prefix}/raw_reward": sum(s.metadata["tau2_opd_task_reward"] for s in samples) / len(samples),
                    f"{prefix}/earliest_policy_lag": sum(rollout_id - s.metadata["tau2_earliest_policy_version"] for s in samples) / len(samples),
                    f"{prefix}/latest_policy_lag": sum(rollout_id - s.metadata["tau2_latest_policy_version"] for s in samples) / len(samples),
                    f"{prefix}/token_weighted_policy_lag": sum(
                        (rollout_id - s.metadata["tau2_token_weighted_policy_version"]) * s.metadata["tau2_response_tokens"]
                        for s in samples
                    ) / max(1, tokens),
                })
        return RolloutFnTrainOutput(samples=groups, metrics={
            **opd_metrics,
            "rollout/producer_wait_seconds": time.time() - started,
            "rollout/producer_duration_seconds": duration,
            "rollout/ready_wait_seconds": waits["ready"],
            "rollout/lag_wait_seconds": waits["lag"],
            "rollout/sync_barrier_seconds": time.time() - barrier_start,
            "rollout/policy_lag_max": max(lags),
            "rollout/policy_lag_mean": sum(lags) / len(lags),
            "rollout/zero_variance_group_rate": sum(len({s.reward for s in g}) == 1 for g in groups) / len(groups),
            "rollout/inflight_trajectories": self.inflight,
            "rollout/ready_groups": ready_count,
            "rollout/buffered_groups": buffered_count,
            "rollout/filtered_groups_total": self.replacements,
            **{f"rollout/filtered_groups/{reason}": count for reason, count in self.filter_counts.copy().items()},
            **{f"rollout/timing/{key}_mean": sum(t.get(key, 0) for t in timings) / len(timings) for key in timing_keys},
            **{f"rollout/trained_groups_{d}": n for d, n in counts.items()},
            **{f"rollout/drawn_groups_{d}": n for d, n in self.source.metadata.get("tau2_task_draw_counts", {}).copy().items()},
        })

    def finish_training(self):
        with self.condition:
            counts = self.source.metadata.setdefault("tau2_task_trained_counts", {})
            for key in self.taken:
                domain = self.pending[key][0].metadata["domain"]
                counts[domain] = counts.get(domain, 0) + 1
                del self.pending[key]
            self.consumed += len(self.taken)
            self.taken = []
            self.training = False
            self.awaiting_weights = True
            self.refill()
            self.condition.notify_all()
        if self.args.tau2_sampling_mode == "sync":
            SAMPLING_STEPS.resume(self.version)

    def pause(self):
        logger.info("tau2_pause_producer_enter time=%.6f", time.time())
        if self.environment_workers:
            self.ray.get([worker.pause.remote() for worker in self.environment_workers])
        else:
            AGENT_TURNS.pause()

    def resume(self, version):
        with self.condition:
            self.version = version
            if hasattr(self.source, "set_policy_version"):
                self.source.set_policy_version(version)
            self.awaiting_weights = False
            if self.environment_workers:
                self.ray.get([worker.resume.remote(version) for worker in self.environment_workers])
            else:
                AGENT_TURNS.resume(version)
            SAMPLING_STEPS.resume(version)
            self.refill()
            self.condition.notify_all()

    def save(self, rollout_id):
        with self.condition:
            self.source.save(rollout_id)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        AGENT_TURNS.close()
        SAMPLING_STEPS.close()
        if self.environment_workers:
            self.ray.get([worker.close.remote() for worker in self.environment_workers])
            # Training and checkpoint saving have finished. Pending remote
            # dialogues are disposable; terminate them before joining threads
            # blocked in ray.get, including User HTTP calls outside Agent gates.
            for worker in self.environment_workers:
                self.ray.kill(worker)
        # Let queued calls observe the closed gates; cancelling queued futures
        # leaves as_completed collectors waiting for executor notification.
        self.workers.shutdown(wait=True)
        self.collectors.shutdown(wait=True)


class Tau2EnvironmentWorker:
    """CPU-only Ray process for independent environments and reward computation."""

    def __init__(self, args, agent_capacity, step_capacity):
        from rollout import _get_mask_generator
        from agent import shared_client, shared_tokenizer

        self.args = args
        AGENT_TURNS.capacity = agent_capacity
        SAMPLING_STEPS.capacity = step_capacity
        AGENT_TURNS.resume(args.start_rollout_id)
        SAMPLING_STEPS.resume(args.start_rollout_id)
        shared_tokenizer(args.hf_checkpoint)
        shared_client(float(os.environ.get("TAU2_AGENT_TIMEOUT", 600.0)))
        _get_mask_generator(args)

    def ready(self):
        return os.getpid()

    def run(self, sample, sampling_params):
        from rollout import _run_tau2_rollout_sync
        try:
            return _apply_opd_teacher(
                self.args,
                _run_tau2_rollout_sync(self.args, sample, dict(sampling_params)),
            )
        finally:
            DIALOGUE.timings = None

    def pause(self):
        AGENT_TURNS.pause()

    def resume(self, version):
        AGENT_TURNS.resume(version)

    def close(self):
        AGENT_TURNS.close()
        SAMPLING_STEPS.close()
