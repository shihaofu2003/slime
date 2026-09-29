"""Separate RL entry for reward-v2 plus the existing credit-signal group filter."""

import sys

import ray

from slime.utils.arguments import parse_args
from train_async import train


if __name__ == "__main__":
    sys.argv.extend([
        "--rollout-producer-path", "credit_signal_v1.CreditSignalProducer",
        "--dynamic-sampling-filter-path", "credit_signal_v1.filter_group",
        "--custom-reward-post-process-path", "credit_signal_v1.post_process_rewards",
    ])
    args = parse_args()
    # Child actors inherit this setup hook, including the environment workers
    # created inside the rollout manager. Existing jobs have a separate runtime.
    ray.init(address="auto", runtime_env={"worker_process_setup_hook": "reward_v2.setup_worker"})
    train(args)
