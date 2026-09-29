"""Opt-in progress-db-count filtering by the actual training advantages.

The legacy producer and reward postprocessor remain unchanged. This producer
prepares credit when a complete group arrives; its postprocessor consumes that
credit without computing or dumping it a second time.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import as_completed

import continuous
import reward_postprocess as credit
from slime.rollout.filter_hub.base_types import DynamicFilterOutput

logger = logging.getLogger(__name__)


def filter_group(args, samples, **kwargs):
    from filters import drop_zero_std_or_unsampleable

    # Reuse the existing usability checks. Only its binary-outcome rejection
    # is replaced; removed or over-cap trajectories must still reject the group.
    decision = drop_zero_std_or_unsampleable(args, samples)
    if not decision.keep and decision.reason not in {
        "official_outcome_all_zero", "official_outcome_all_one",
    }:
        return decision

    credit.attach_progress_group_advantages(args, samples)
    keep = samples[0].metadata["tau2_turn_credit_group_has_signal"]
    return DynamicFilterOutput(keep=keep, reason=None if keep else "credit_zero_signal")


def post_process_rewards(args, samples):
    """Keep outcome logging/normalization while reusing prepared token credit."""
    raw = [float(sample.get_reward_value(args)) for sample in samples]
    normalized = credit._group_normalize(
        raw, valid_mask=[1.0] * len(raw),
        n_samples_per_prompt=args.n_samples_per_prompt, apply_std=True,
    )
    return raw, normalized


class CreditSignalProducer(continuous.Tau2Producer):
    def __init__(self, args, data_source):
        if credit.configured_turn_credit_version() != credit.PROGRESS_DB_COUNT_V1:
            raise ValueError("CreditSignalProducer requires progress-db-count-v1")
        super().__init__(args, data_source)

    def collect_group(self, key, futures):
        # The collector is the legacy producer's only hard-coded filter site.
        # Keep its admission, ready order, exception propagation and accounting.
        try:
            group = [None] * len(futures)
            positions = {future: index for index, future in enumerate(futures)}
            for future in as_completed(futures):
                group[positions[future]] = future.result()
                with self.condition:
                    self.inflight -= 1
                    self.groups[key]["remaining"] -= 1
                    if self.groups[key]["remaining"]:
                        self.refill()
            decision = filter_group(self.args, group)
            rewards = [float(sample.get_reward_value(self.args)) for sample in group]
            outcome = ("all_zero" if not any(rewards) else
                       "all_one" if all(value == 1.0 for value in rewards) else "mixed")
            with self.condition:
                entry = self.groups[key]
                ended = time.time()
                if decision.keep:
                    entry["earliest"] = min(s.metadata["tau2_earliest_policy_version"] for s in group)
                    entry["duration"] = ended - entry["start"]
                    self.ready[key] = group
                else:
                    del self.groups[key]
                    del self.pending[key]
                    self.replacements += 1
                    reason = decision.reason
                    self.filter_counts[reason] = self.filter_counts.get(reason, 0) + 1
                    if hasattr(self.source, "record_outcome"):
                        self.source.record_outcome(group, reason, self.version)
                # Compact prefilter metrics include every complete group,
                # including unusable groups and groups never consumed by training.
                logger.info("tau2_credit_group %s", json.dumps({
                    "group": key, "domain": group[0].metadata["domain"],
                    "trajectories": len(group), "reward": sum(rewards) / len(rewards),
                    "outcome": outcome, "kept": bool(decision.keep), "reason": decision.reason,
                }))
                self.refill()
                logger.info("tau2_producer group=%s start=%.6f end=%.6f accepted=%s",
                            key, entry["start"], ended, decision.keep)
                self.condition.notify_all()
        except BaseException as error:
            with self.condition:
                if not self.closed and self.error is None:
                    self.error = error
                self.condition.notify_all()
            continuous.AGENT_TURNS.close()
            continuous.SAMPLING_STEPS.close()
