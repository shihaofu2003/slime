"""Dynamic sampling filter for tau2-bench GRPO rollouts.

Primary role: drop a whole GRPO group when at least one of its trajectories is
``permanently_too_long`` — it could not be sampled under the token cap even
after per-sample retries (see ``rollout._run_tau2_rollout_sync``). Its
``[T, V]`` logits would OOM the log-prob forward, and the prompt evidently
produces pathological lengths, so the whole group is dropped and a different
prompt is drawn. This is a *memory-safety* backstop and is always on.

Optional secondary role: replace groups with no usable advantage signal. V1
checks shaped trajectory-score variance plus local penalties. V2 checks the
actual combination of official outcome advantage and fixed-budget turn
reallocation. ``TAU2_REPLACE_ZERO_SIGNAL_GROUPS=0`` keeps those groups with
zero policy advantage so recipe comparisons retain the sampled prompt mix.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from collections import Counter, deque

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample
from reward_postprocess import (
    TURN_CREDIT_V1,
    TURN_CREDIT_V2,
    TurnCreditAlignmentError,
    compute_rollout_score,
    turn_credit_v2_group_has_signal,
    validate_token_penalties,
)

_ZERO_STD_TOL = 1e-6
_DOMAINS = ("telecom", "airline", "retail")
_DOMAIN_QUOTA_ENV = "TAU2_RL_DOMAIN_QUOTA"
_REPLACE_ZERO_SIGNAL_ENV = "TAU2_REPLACE_ZERO_SIGNAL_GROUPS"
_STRICT_SINGLE_PROFILE = "strict-single-v1"
_OFFICIAL_NATIVE_PROFILE = "official-native"
_STATE_KEY = "tau2_domain_quota_v1"


def parse_domain_quota(raw: str) -> dict[str, int]:
    """Parse an exact per-rollout quota such as ``telecom:3,airline:2,retail:1``."""
    quota: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid {_DOMAIN_QUOTA_ENV} item {part!r}; expected domain:count")
        domain, count_raw = (piece.strip() for piece in part.split(":", 1))
        if domain not in _DOMAINS:
            raise ValueError(f"Invalid {_DOMAIN_QUOTA_ENV} domain {domain!r}; expected one of {_DOMAINS}")
        if domain in quota:
            raise ValueError(f"Duplicate {_DOMAIN_QUOTA_ENV} domain {domain!r}")
        try:
            count = int(count_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid {_DOMAIN_QUOTA_ENV} count {count_raw!r} for {domain}") from exc
        if count < 0:
            raise ValueError(f"{_DOMAIN_QUOTA_ENV} count for {domain} must be non-negative")
        quota[domain] = count
    if not quota or sum(quota.values()) == 0:
        raise ValueError(f"{_DOMAIN_QUOTA_ENV} must contain at least one prompt")
    return {domain: quota.get(domain, 0) for domain in _DOMAINS}


def _sample_domain(sample: Sample) -> str | None:
    metadata = sample.metadata or {}
    domain = metadata.get("domain") or metadata.get("tau2_domain_quota_domain")
    return str(domain) if domain is not None else None


def _group_domain(samples: list[Sample]) -> str | None:
    domains = {_sample_domain(sample) for sample in samples}
    if len(domains) != 1:
        raise ValueError(f"A tau2 GRPO group must contain exactly one domain, got {sorted(map(str, domains))}")
    return next(iter(domains))


class DomainQuotaDataSource(RolloutDataSourceWithBuffer):
    """Draw exact domain quotas and issue only same-domain filter replacements.

    The generic rollout loop asks for an oversampling batch whenever a group is
    rejected. This data source deliberately returns the exact initial quota on
    the first request and only queued same-domain replacements thereafter.
    Domain cursors are checkpointed independently so resuming cannot change the
    per-domain sampling stream.
    """

    def __init__(self, args):
        super().__init__(args)
        raw_quota = os.environ.get(_DOMAIN_QUOTA_ENV)
        if not raw_quota:
            raise ValueError(f"{self.__class__.__name__} requires {_DOMAIN_QUOTA_ENV}")
        self.quota = parse_domain_quota(raw_quota)
        if sum(self.quota.values()) != args.rollout_batch_size:
            raise ValueError(
                f"{_DOMAIN_QUOTA_ENV} sums to {sum(self.quota.values())}, "
                f"but --rollout-batch-size is {args.rollout_batch_size}"
            )
        if self.dataset is None:
            raise ValueError(f"{self.__class__.__name__} requires --rollout-global-dataset and --prompt-data")

        self._domain_samples: dict[str, list[Sample]] = {domain: [] for domain in _DOMAINS}
        for sample in self.dataset.origin_samples:
            domain = _sample_domain(sample)
            if domain not in self._domain_samples:
                raise ValueError(f"Prompt sample has unsupported or missing domain: {domain!r}")
            self._domain_samples[domain].append(sample)
        for domain, required in self.quota.items():
            if required and not self._domain_samples[domain]:
                raise ValueError(f"Quota requires {domain}, but the prompt dataset has no {domain} rows")

        self._domain_offsets = {domain: 0 for domain in _DOMAINS}
        self._domain_epochs = {domain: 0 for domain in _DOMAINS}
        self._domain_orders: dict[str, list[int]] = {}
        self._dataset_fingerprint = None
        if os.environ.get("TAU2_AGENT_PROTOCOL_PROFILE") not in {
            _OFFICIAL_NATIVE_PROFILE,
            _STRICT_SINGLE_PROFILE,
        }:
            self._dataset_fingerprint = self._fingerprint_dataset()
        for domain in _DOMAINS:
            self._refresh_domain_order(domain)

        self._active_rollout_id: int | None = None
        self._initial_domains: deque[str] = deque()
        self._replacement_domains: deque[str] = deque()
        self._accepted: Counter[str] = Counter()
        self._rejected: Counter[str] = Counter()
        self._replacement_draws: Counter[str] = Counter()

    def _fingerprint_dataset(self) -> str:
        rows = []
        for domain in _DOMAINS:
            for sample in self._domain_samples[domain]:
                metadata = sample.metadata or {}
                task = metadata.get("task") or {}
                rows.append((domain, task.get("id"), metadata.get("source_row"), sample.prompt))
        payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _refresh_domain_order(self, domain: str) -> None:
        order = list(range(len(self._domain_samples[domain])))
        if self.args.rollout_shuffle:
            domain_salt = _DOMAINS.index(domain) * 1_000_003
            random.Random(self.args.rollout_seed + domain_salt + self._domain_epochs[domain]).shuffle(order)
        self._domain_orders[domain] = order

    def _next_prompt(self, domain: str) -> Sample:
        samples = self._domain_samples[domain]
        if not samples:
            raise RuntimeError(f"No prompt rows available for quota domain {domain}")
        if self._domain_offsets[domain] >= len(samples):
            self._domain_epochs[domain] += 1
            self._domain_offsets[domain] = 0
            self._refresh_domain_order(domain)
        order_index = self._domain_offsets[domain]
        self._domain_offsets[domain] += 1
        return samples[self._domain_orders[domain][order_index]]

    def _interleaved_quota(self) -> list[str]:
        remaining = dict(self.quota)
        order: list[str] = []
        while sum(remaining.values()):
            for domain in _DOMAINS:
                if remaining[domain] > 0:
                    order.append(domain)
                    remaining[domain] -= 1
        return order

    def begin_rollout(self, rollout_id: int) -> None:
        if self._active_rollout_id is not None:
            raise RuntimeError(f"Domain quota rollout {self._active_rollout_id} has not been finished")
        self._active_rollout_id = rollout_id
        self._initial_domains = deque(self._interleaved_quota())
        self._replacement_domains.clear()
        self._accepted.clear()
        self._rejected.clear()
        self._replacement_draws.clear()

    def _pop_buffer_for_domain(self, domain: str) -> list[Sample] | None:
        for index, group in enumerate(self.buffer):
            flat_group = group[0] if group and isinstance(group[0], list) else group
            if flat_group and _group_domain(flat_group) == domain:
                return self.buffer.pop(index)
        return None

    def _make_group(self, prompt_sample: Sample, *, domain: str, replacement: bool) -> list[Sample]:
        group: list[Sample] = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.group_index = self.sample_group_index
            sample.index = self.sample_index
            sample.metadata = dict(sample.metadata or {})
            sample.metadata.update(
                {
                    "tau2_domain_quota_domain": domain,
                    "tau2_domain_quota_replacement": replacement,
                    "tau2_domain_quota_rollout_id": self._active_rollout_id,
                }
            )
            self.sample_index += 1
            group.append(sample)
        self.sample_group_index += 1
        return group

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if self._active_rollout_id is None:
            raise RuntimeError("begin_rollout() must be called before drawing domain-quota samples")
        if num_samples <= 0:
            return []

        if self._replacement_domains:
            domains = [
                self._replacement_domains.popleft()
                for _ in range(min(num_samples, len(self._replacement_domains)))
            ]
            replacement = True
        elif self._initial_domains:
            domains = [self._initial_domains.popleft() for _ in range(min(num_samples, len(self._initial_domains)))]
            replacement = False
        else:
            raise RuntimeError("Domain quota is exhausted and no same-domain replacement was requested")

        groups: list[list[Sample]] = []
        for domain in domains:
            buffered = self._pop_buffer_for_domain(domain)
            if buffered is not None:
                group = buffered
                for sample in group:
                    sample.metadata = dict(sample.metadata or {})
                    sample.metadata.update(
                        {
                            "tau2_domain_quota_domain": domain,
                            "tau2_domain_quota_replacement": replacement,
                            "tau2_domain_quota_rollout_id": self._active_rollout_id,
                        }
                    )
            else:
                group = self._make_group(self._next_prompt(domain), domain=domain, replacement=replacement)
            groups.append(group)
            if replacement:
                self._replacement_draws[domain] += 1
        return groups

    def record_filter_result(self, domain: str | None, *, keep: bool) -> None:
        if self._active_rollout_id is None:
            raise RuntimeError("Received a filter result outside an active domain-quota rollout")
        if domain not in self.quota or self.quota[domain] <= 0:
            raise ValueError(f"Filter returned an unexpected quota domain: {domain!r}")
        if keep:
            self._accepted[domain] += 1
            if self._accepted[domain] > self.quota[domain]:
                raise RuntimeError(f"Accepted more {domain} groups than quota {self.quota[domain]}")
        else:
            self._rejected[domain] += 1
            self._replacement_domains.append(domain)

    def finish_rollout(self, rollout_id: int) -> dict[str, float]:
        if self._active_rollout_id != rollout_id:
            raise RuntimeError(f"Finishing rollout {rollout_id}, but active rollout is {self._active_rollout_id}")
        if self._initial_domains or self._replacement_domains:
            raise RuntimeError(
                "Domain quota finished with undrawn prompts: "
                f"initial={list(self._initial_domains)}, replacements={list(self._replacement_domains)}"
            )
        observed = {domain: self._accepted[domain] for domain in _DOMAINS}
        if observed != self.quota:
            raise RuntimeError(f"Domain quota mismatch: accepted={observed}, expected={self.quota}")
        metrics: dict[str, float] = {}
        for domain in _DOMAINS:
            prefix = f"rollout/domain_quota/{domain}"
            metrics[f"{prefix}/accepted"] = float(self._accepted[domain])
            metrics[f"{prefix}/rejected"] = float(self._rejected[domain])
            metrics[f"{prefix}/replacement_draws"] = float(self._replacement_draws[domain])
        self._active_rollout_id = None
        return metrics

    def checkpoint_state(self) -> dict[str, object]:
        if self._active_rollout_id is not None:
            raise RuntimeError("Refusing to checkpoint an unfinished domain-quota rollout")
        state: dict[str, object] = {
            "domain_offsets": dict(self._domain_offsets),
            "domain_epochs": dict(self._domain_epochs),
        }
        if self._dataset_fingerprint is not None:
            state["dataset_fingerprint"] = self._dataset_fingerprint
        return state

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        if (
            self._dataset_fingerprint is not None
            and state.get("dataset_fingerprint") != self._dataset_fingerprint
        ):
            raise ValueError("Cannot resume domain-quota sampling with a different prompt dataset")
        offsets = state.get("domain_offsets") or {}
        epochs = state.get("domain_epochs") or {}
        if not isinstance(offsets, dict) or not isinstance(epochs, dict):
            raise ValueError("Invalid domain-quota checkpoint state")
        for domain in _DOMAINS:
            self._domain_offsets[domain] = int(offsets.get(domain, 0))
            self._domain_epochs[domain] = int(epochs.get(domain, 0))
            if not 0 <= self._domain_offsets[domain] <= len(self._domain_samples[domain]):
                raise ValueError(f"Invalid resumed offset for {domain}: {self._domain_offsets[domain]}")
            self._refresh_domain_order(domain)

    def _sync_checkpoint_metadata(self) -> None:
        self.metadata[_STATE_KEY] = self.checkpoint_state()

    def save(self, rollout_id):
        self._sync_checkpoint_metadata()
        super().save(rollout_id)

    def load(self, rollout_id=None):
        super().load(rollout_id)
        state = self.metadata.get(_STATE_KEY)
        if not state:
            return
        self.restore_checkpoint_state(state)


def drop_zero_std_or_unsampleable(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Reject unsampleable groups and optionally replace signal-free groups.

    Rejection makes slime draw and generate a replacement group. A rejected
    group never enters training. Over-cap and invalid groups are always
    rejected; zero-signal replacement is controlled separately.
    """
    domain = _group_domain(samples)
    reason = None
    has_local_signal = False
    group_turn_credit_version = None
    replace_zero_signal = os.environ.get(_REPLACE_ZERO_SIGNAL_ENV, "1") == "1"
    for sample in samples:
        metadata = sample.metadata or {}
        if metadata.get("tau2_permanently_too_long"):
            reason = "permanently_too_long"
            break
        if sample.remove_sample:
            reason = str(metadata.get("tau2_turn_credit_invalid") or "removed_sample")
            break

        train_metadata = sample.train_metadata
        turn_credit_version = metadata.get("tau2_turn_credit_version")
        if turn_credit_version:
            group_turn_credit_version = turn_credit_version
        turn_credit_expected = turn_credit_version in {TURN_CREDIT_V1, TURN_CREDIT_V2}
        if train_metadata is None:
            if turn_credit_expected:
                reason = "missing_turn_credit_train_metadata"
                break
            continue
        if turn_credit_version == TURN_CREDIT_V2:
            if (
                not isinstance(train_metadata, dict)
                or train_metadata.get("turn_credit_version") != TURN_CREDIT_V2
                or not isinstance(metadata.get("tau2_turn_credits"), list)
            ):
                reason = "missing_turn_credit_v2_metadata"
                break
            continue
        if not isinstance(train_metadata, dict) or "token_penalties" not in train_metadata:
            if turn_credit_expected:
                reason = "missing_token_penalties"
                break
            continue
        try:
            penalties = validate_token_penalties(
                train_metadata["token_penalties"],
                response_length=sample.response_length,
                loss_mask=sample.loss_mask,
            )
        except TurnCreditAlignmentError as exc:
            reason = f"invalid_token_penalties:{exc}"
            break
        has_local_signal = has_local_signal or any(penalties)

    if reason is None and group_turn_credit_version == TURN_CREDIT_V2:
        has_group_signal = turn_credit_v2_group_has_signal(args, samples)
        for sample in samples:
            sample.metadata["tau2_turn_credit_group_has_signal"] = has_group_signal
        if not has_group_signal and replace_zero_signal:
            reason = "turn_credit_v2_zero_signal"
    elif reason is None and replace_zero_signal:
        global_scores = [compute_rollout_score(args, sample)[0] for sample in samples]
        if (
            len(global_scores) > 1
            and torch.tensor(global_scores, dtype=torch.float64).std() <= _ZERO_STD_TOL
            and not has_local_signal
            and replace_zero_signal
        ):
            reason = f"global_zero_std_no_local_signal_{round(float(global_scores[0]), 3)}"

    keep = reason is None
    data_source = kwargs.get("data_source")
    if data_source is not None and hasattr(data_source, "record_filter_result"):
        data_source.record_filter_result(domain, keep=keep)
    return DynamicFilterOutput(keep=keep, reason=reason)
