"""Controlled v4 entry: consistent tasks, optionally adding goal-progress credit."""

import os
import sys

import ray

from slime.utils.arguments import parse_args
from train_async import train


if __name__ == "__main__":
    arm = os.environ.get("TAU2_V4_ARM", "goal_progress")
    hooks = {"task_consistent": "goal_progress_v4.setup_data_worker", "goal_progress": "goal_progress_v4.setup_worker"}
    sys.argv.extend([
        "--rollout-producer-path", "credit_signal_v1.CreditSignalProducer",
        "--dynamic-sampling-filter-path", "credit_signal_v1.filter_group",
        "--custom-reward-post-process-path", "credit_signal_v1.post_process_rewards",
    ])
    args = parse_args()
    ray.init(address="auto", runtime_env={"worker_process_setup_hook": hooks[arm]})
    train(args)
