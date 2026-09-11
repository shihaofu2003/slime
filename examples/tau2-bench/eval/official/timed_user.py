"""Timing-preserving wrapper around tau2's official user simulator."""

from __future__ import annotations

import time

from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

from model_request import with_request_timing


USER_NAME = "slime_timed_user_simulator"


class TimedUserSimulator(UserSimulator):
    def _generate_next_message(self, message, state):
        started_at = time.perf_counter()
        response = super()._generate_next_message(message, state)
        response.raw_data = with_request_timing(
            response.raw_data,
            participant="user",
            elapsed_seconds=time.perf_counter() - started_at,
        )
        return response


def register_timed_user_simulator() -> None:
    if USER_NAME not in registry.get_users():
        registry.register_user(TimedUserSimulator, USER_NAME)
