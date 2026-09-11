"""Make blocking ``flock`` semantics reliable on the shared AFS mount.

This module is loaded only by the dedicated VitaBench A/B retry wrapper.  It
patches scheduler subprocesses, and leaves Vita, SGLang, and unrelated Python
processes untouched.
"""

import errno
import fcntl
import os
import sys
import time
from pathlib import Path


RETRY_INTERVAL_SECONDS = 0.05
RETRY_TIMEOUT_SECONDS = 120.0
RETRYABLE_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN})


def flock_with_retry(
    flock_function,
    file_descriptor,
    operation,
    *,
    timeout_seconds=RETRY_TIMEOUT_SECONDS,
    sleep_function=time.sleep,
    monotonic_function=time.monotonic,
):
    """Retry AFS ``EAGAIN`` while preserving blocking flock semantics."""

    if not operation & (fcntl.LOCK_EX | fcntl.LOCK_SH):
        return flock_function(file_descriptor, operation)

    deadline = monotonic_function() + timeout_seconds
    while True:
        try:
            return flock_function(file_descriptor, operation)
        except OSError as exc:
            if exc.errno not in RETRYABLE_ERRNOS or monotonic_function() >= deadline:
                raise
            sleep_function(RETRY_INTERVAL_SECONDS)


def _install_scheduler_patch():
    original_flock = fcntl.flock
    if getattr(original_flock, "_vitabench_afs_retry", False):
        return

    def retrying_flock(file_descriptor, operation):
        return flock_with_retry(original_flock, file_descriptor, operation)

    retrying_flock._vitabench_afs_retry = True
    fcntl.flock = retrying_flock


if os.environ.get("VITA_AFS_FLOCK_RETRY") == "1" and Path(sys.argv[0]).name == "eval_scheduler.py":
    _install_scheduler_patch()
