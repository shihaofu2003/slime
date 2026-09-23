"""Uniform task draws with replacement and checkpointed sampling state."""

import copy
import random

from slime.rollout.data_source import RolloutDataSource


class RandomTaskDataSource(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.rng = random.Random(args.rollout_seed)
        if self.dataset is None or not self.dataset.origin_samples:
            raise ValueError("Random Tau2 sampling requires a nonempty task dataset")

    def next_prompt(self):
        return self.rng.choice(self.dataset.origin_samples)

    def get_samples(self, num_samples):
        groups = []
        counts = self.metadata.setdefault("tau2_task_draw_counts", {})
        for _ in range(num_samples):
            prompt = self.next_prompt()
            if prompt is None:
                break
            domain = prompt.metadata["domain"]
            counts[domain] = counts.get(domain, 0) + 1
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt)
                sample.index = self.sample_index
                sample.group_index = self.sample_group_index
                sample.rollout_id = sample.index
                self.sample_index += 1
                group.append(sample)
            groups.append(group)
            self.sample_group_index += 1
            self.sample_offset += 1
        self.metadata["tau2_task_rng_state"] = self.rng.getstate()
        return groups

    def load(self, rollout_id=None):
        super().load(rollout_id)
        if "tau2_task_rng_state" in self.metadata:
            self.rng.setstate(self.metadata["tau2_task_rng_state"])


class ShuffledTaskDataSource(RandomTaskDataSource):
    """Visit a shuffled task deck; interleave fresh K8 retries after two updates."""

    def __init__(self, args):
        super().__init__(args)
        self.task_indices = {(s.metadata["domain"], s.metadata["task"]["id"]): i
                             for i, s in enumerate(self.dataset.origin_samples)}

    def set_policy_version(self, version):
        self.metadata["tau2_retry_policy_version"] = version

    def record_outcome(self, group, reason, version):
        if reason != "official_outcome_all_zero":
            return
        sample = group[0]
        index = self.task_indices[(sample.metadata["domain"], sample.metadata["task"]["id"])]
        self.metadata.setdefault("tau2_deferred_tasks", {})[index] = version + 2
        self.metadata["tau2_shuffled_tasks"] = [i for i in self.metadata.get("tau2_shuffled_tasks", []) if i != index]

    def next_prompt(self):
        deferred = self.metadata.setdefault("tau2_deferred_tasks", {})
        version = self.metadata.get("tau2_retry_policy_version", 0)
        due = next((i for i, ready in deferred.items() if ready <= version), None)
        if due is not None and self.metadata.get("tau2_retry_next", True):
            del deferred[due]
            self.metadata["tau2_retry_next"] = False
            self.metadata["tau2_task_retry_draws"] = self.metadata.get("tau2_task_retry_draws", 0) + 1
            return self.dataset.origin_samples[due]
        order = self.metadata.setdefault("tau2_shuffled_tasks", [])
        if not order:
            order.extend(i for i in range(len(self.dataset.origin_samples)) if i not in deferred)
            self.rng.shuffle(order)
        # A retry may have been queued while this deck was being traversed.
        while order:
            index = order.pop()
            if index not in deferred:
                self.metadata["tau2_retry_next"] = True
                return self.dataset.origin_samples[index]
        if due is not None:
            self.metadata["tau2_retry_next"] = True
            return self.next_prompt()
        return None
