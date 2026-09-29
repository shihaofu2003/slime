"""Separate entry point for credit-signal filtering; train_async.py is unchanged."""

import sys

from slime.utils.arguments import parse_args
from train_async import train


if __name__ == "__main__":
    # Append before parsing so the resolved configuration logs the new hooks,
    # even when the unchanged shell launcher supplied its legacy defaults.
    sys.argv.extend([
        "--rollout-producer-path", "credit_signal_v1.CreditSignalProducer",
        "--dynamic-sampling-filter-path", "credit_signal_v1.filter_group",
        "--custom-reward-post-process-path", "credit_signal_v1.post_process_rewards",
    ])
    train(parse_args())
