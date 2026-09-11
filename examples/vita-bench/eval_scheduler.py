#!/usr/bin/env python3
"""Validate the local VitaBench topology and coordinate dynamic shard claims."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


QUEUE_SCHEMA = "vitabench-dynamic-shard-queue/v1"
LOCAL_WORKER_COUNT = 4
LOCAL_AGENT_INSTANCE_COUNT = 2
LOCAL_USER_INSTANCE_COUNT = 2
LOCAL_EVALUATOR_INSTANCE_COUNT = 4
FLOCK_RETRY_INTERVAL_SECONDS = 0.05
FLOCK_RETRY_TIMEOUT_SECONDS = 120.0
FLOCK_RETRYABLE_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN})
QUEUE_LOCK_POLICY = {
    "primitive": "fcntl.flock",
    "retry_errno_names": ["EACCES", "EAGAIN"],
    "retry_interval_seconds": FLOCK_RETRY_INTERVAL_SECONDS,
    "retry_timeout_seconds": FLOCK_RETRY_TIMEOUT_SECONDS,
}


class SchedulerError(ValueError):
    """The topology or queue state is unsafe to use."""


@dataclass(frozen=True)
class PlanEntry:
    kind: str
    suite: str
    domain: str
    task_set: str
    shard_index: int
    result_relative: str
    task_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.suite}:{self.shard_index}"

    def to_tsv(self) -> str:
        return "\t".join(
            (
                self.kind,
                self.suite,
                self.domain,
                self.task_set,
                str(self.shard_index),
                self.result_relative,
                " ".join(self.task_ids),
            )
        )

    @classmethod
    def from_dict(cls, value: object) -> PlanEntry:
        if not isinstance(value, dict):
            raise SchedulerError("queue entry must be an object")
        try:
            task_ids = value["task_ids"]
            if not isinstance(task_ids, list) or not all(isinstance(task_id, str) and task_id for task_id in task_ids):
                raise SchedulerError("queue entry task_ids must be nonempty strings")
            return cls(
                kind=str(value["kind"]),
                suite=str(value["suite"]),
                domain=str(value["domain"]),
                task_set=str(value["task_set"]),
                shard_index=int(value["shard_index"]),
                result_relative=str(value["result_relative"]),
                task_ids=tuple(task_ids),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SchedulerError("queue entry is malformed") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_count(value: int, expected: int, name: str) -> None:
    if value != expected:
        raise SchedulerError(f"{name} must be {expected}; got {value}")


def flock_with_retry(
    flock_function: Callable[[int, int], object],
    file_descriptor: int,
    operation: int,
    *,
    timeout_seconds: float = FLOCK_RETRY_TIMEOUT_SECONDS,
    sleep_function: Callable[[float], object] = time.sleep,
    monotonic_function: Callable[[], float] = time.monotonic,
) -> object:
    """Preserve blocking flock semantics when AFS returns transient EAGAIN."""

    if not operation & (fcntl.LOCK_EX | fcntl.LOCK_SH):
        return flock_function(file_descriptor, operation)

    deadline = monotonic_function() + timeout_seconds
    while True:
        try:
            return flock_function(file_descriptor, operation)
        except OSError as exc:
            if exc.errno not in FLOCK_RETRYABLE_ERRNOS or monotonic_function() >= deadline:
                raise
            sleep_function(FLOCK_RETRY_INTERVAL_SECONDS)


def build_local_topology(
    gpu_selectors: Sequence[str],
    *,
    worker_count: int,
    agent_instance_count: int,
    user_instance_count: int,
    evaluator_instance_count: int,
    agent_port_base: int,
    role_port_base: int,
) -> dict[str, object]:
    """Return the fixed 2/2/4 topology after enforcing all local invariants."""

    _required_count(worker_count, LOCAL_WORKER_COUNT, "VITA_WORKER_COUNT")
    _required_count(
        agent_instance_count,
        LOCAL_AGENT_INSTANCE_COUNT,
        "VITA_AGENT_INSTANCE_COUNT",
    )
    _required_count(
        user_instance_count,
        LOCAL_USER_INSTANCE_COUNT,
        "VITA_USER_INSTANCE_COUNT",
    )
    _required_count(
        evaluator_instance_count,
        LOCAL_EVALUATOR_INSTANCE_COUNT,
        "VITA_EVALUATOR_INSTANCE_COUNT",
    )
    instance_count = agent_instance_count + user_instance_count + evaluator_instance_count
    if instance_count != 8:
        raise SchedulerError(f"local instance counts must sum to 8; got {instance_count}")

    selectors = [selector.strip() for selector in gpu_selectors]
    if len(selectors) != 8:
        raise SchedulerError(f"local evaluation requires exactly 8 GPU selectors; got {len(selectors)}")
    if any(not selector for selector in selectors):
        raise SchedulerError("GPU selectors must be nonempty")
    if len(set(selectors)) != len(selectors):
        raise SchedulerError("the 8 local GPU selectors must be unique")

    agent_instances = [
        {
            "instance_id": index,
            "gpu_slot": index,
            "gpu_selector": selectors[index],
            "port": agent_port_base + index,
        }
        for index in range(agent_instance_count)
    ]
    user_instances = [
        {
            "instance_id": index,
            "gpu_slot": agent_instance_count + index,
            "gpu_selector": selectors[agent_instance_count + index],
            "port": role_port_base + index,
        }
        for index in range(user_instance_count)
    ]
    evaluator_instances = [
        {
            "instance_id": index,
            "gpu_slot": agent_instance_count + user_instance_count + index,
            "gpu_selector": selectors[agent_instance_count + user_instance_count + index],
            "port": role_port_base + user_instance_count + index,
        }
        for index in range(evaluator_instance_count)
    ]

    agent_routes = (0, 1, 0, 1)
    user_routes = (0, 1, 0, 1)
    evaluator_routes = (0, 1, 2, 3)
    worker_endpoints = []
    for worker_id in range(worker_count):
        agent = agent_instances[agent_routes[worker_id]]
        user = user_instances[user_routes[worker_id]]
        evaluator = evaluator_instances[evaluator_routes[worker_id]]
        worker_endpoints.append(
            {
                "worker_id": worker_id,
                "agent_instance": agent["instance_id"],
                "agent_gpu_slot": agent["gpu_slot"],
                "agent_port": agent["port"],
                "user_instance": user["instance_id"],
                "user_gpu_slot": user["gpu_slot"],
                "user_port": user["port"],
                "evaluator_instance": evaluator["instance_id"],
                "evaluator_gpu_slot": evaluator["gpu_slot"],
                "evaluator_port": evaluator["port"],
            }
        )

    service_ports = {instance["port"] for instances in (agent_instances, user_instances, evaluator_instances) for instance in instances}
    if len(service_ports) != 8:
        raise SchedulerError("all 8 local SGLang instances must use unique ports")
    return {
        "scheduler": "dynamic_shard_queue",
        "queue_lock": dict(QUEUE_LOCK_POLICY),
        "worker_count": worker_count,
        "instance_count": instance_count,
        "agent_instances": agent_instances,
        "user_instances": user_instances,
        "evaluator_instances": evaluator_instances,
        "worker_endpoints": worker_endpoints,
    }


def read_shard_plan(path: Path) -> list[PlanEntry]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SchedulerError(f"cannot read run plan {path}: {exc}") from exc

    entries: list[PlanEntry] = []
    keys: set[str] = set()
    result_paths: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 7:
            raise SchedulerError(f"run plan line {line_number} must contain exactly 7 TSV fields")
        kind, suite, domain, task_set, shard_text, result_relative, ids_text = fields
        if kind != "shard":
            continue
        try:
            shard_index = int(shard_text)
        except ValueError as exc:
            raise SchedulerError(f"run plan line {line_number} has an invalid shard index") from exc
        task_ids = tuple(ids_text.split())
        if not suite or not domain or not task_set or not result_relative or not task_ids:
            raise SchedulerError(f"run plan line {line_number} is incomplete")
        entry = PlanEntry(
            kind=kind,
            suite=suite,
            domain=domain,
            task_set=task_set,
            shard_index=shard_index,
            result_relative=result_relative,
            task_ids=task_ids,
        )
        if entry.key in keys:
            raise SchedulerError(f"duplicate shard key in run plan: {entry.key}")
        if result_relative in result_paths:
            raise SchedulerError(f"duplicate shard result path in run plan: {result_relative}")
        keys.add(entry.key)
        result_paths.add(result_relative)
        entries.append(entry)
    if not entries:
        raise SchedulerError("run plan contains no shard entries")
    return entries


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _run_plan_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize_queue(
    run_plan: Path,
    state_path: Path,
    lock_path: Path,
    *,
    worker_count: int,
    completion_checker: Callable[[PlanEntry], str],
    shard_index: int | None = None,
) -> dict[str, object]:
    """Rebuild queue state, omitting shards already proven complete."""

    _required_count(worker_count, LOCAL_WORKER_COUNT, "VITA_WORKER_COUNT")
    entries = read_shard_plan(run_plan)
    if shard_index is not None:
        entries = [entry for entry in entries if entry.shard_index == shard_index]
        if not entries:
            raise SchedulerError(f"run plan contains no shard entries with index {shard_index}")
    available: list[dict[str, object]] = []
    skipped_complete: list[str] = []
    corrupt: list[str] = []
    for entry in entries:
        status = completion_checker(entry)
        if status == "complete":
            skipped_complete.append(entry.key)
        elif status in {"partial", "corrupt"}:
            available.append(asdict(entry))
            if status == "corrupt":
                corrupt.append(entry.key)
        else:
            raise SchedulerError(f"completion checker returned invalid status {status!r} for {entry.key}")

    state = {
        "schema": QUEUE_SCHEMA,
        "initialized_at": _utc_now(),
        "run_plan_sha256": _run_plan_sha256(run_plan),
        "worker_count": worker_count,
        "selected_shard_index": shard_index,
        "available": available,
        "claims": [],
        "skipped_complete": skipped_complete,
        "corrupt_pending": corrupt,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        flock_with_retry(fcntl.flock, lock_stream.fileno(), fcntl.LOCK_EX)
        _atomic_write_json(state_path, state)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return state


def _load_queue_state(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchedulerError(f"queue state is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchedulerError(f"cannot read queue state {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != QUEUE_SCHEMA:
        raise SchedulerError("queue state has an unsupported schema")
    if not isinstance(value.get("available"), list) or not isinstance(value.get("claims"), list):
        raise SchedulerError("queue state omits available entries or claims")
    return value


def claim_next_shard(state_path: Path, lock_path: Path, *, worker_id: int) -> PlanEntry | None:
    """Atomically claim one pending shard for a worker."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        flock_with_retry(fcntl.flock, lock_stream.fileno(), fcntl.LOCK_EX)
        state = _load_queue_state(state_path)
        worker_count = state.get("worker_count")
        if not isinstance(worker_count, int):
            raise SchedulerError("queue state has an invalid worker count")
        if not 0 <= worker_id < worker_count:
            raise SchedulerError(f"worker id {worker_id} is outside queue worker range 0..{worker_count - 1}")
        available = state["available"]
        assert isinstance(available, list)
        if not available:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            return None
        entry = PlanEntry.from_dict(available.pop(0))
        claims = state["claims"]
        assert isinstance(claims, list)
        if any(isinstance(claim, dict) and claim.get("key") == entry.key for claim in claims):
            raise SchedulerError(f"queue attempted to claim {entry.key} twice")
        claims.append(
            {
                "key": entry.key,
                "worker_id": worker_id,
                "claimed_at": _utc_now(),
                "pid": os.getpid(),
            }
        )
        _atomic_write_json(state_path, state)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        return entry


def _summarizer_checker(args: argparse.Namespace, entry: PlanEntry) -> str:
    result_file = args.artifact_dir / entry.result_relative
    command = [
        args.python,
        str(args.summarizer),
        "check-shard",
        "--file",
        str(result_file),
        "--task-ids",
        *entry.task_ids,
        "--num-trials",
        str(args.num_trials),
        "--base-seed",
        str(args.base_seed),
        "--expected-domain",
        entry.domain,
        "--expected-max-steps",
        str(args.expected_max_steps),
        "--expected-max-errors",
        str(args.expected_max_errors),
        "--expected-agent-model",
        args.expected_agent_model,
        "--expected-user-model",
        args.expected_user_model,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        return "complete"
    if completed.returncode == 1:
        return "partial"
    if completed.returncode == 2:
        return "corrupt"
    detail = (completed.stderr or completed.stdout).strip()
    raise SchedulerError(f"shard checker failed for {entry.key} with status {completed.returncode}: {detail}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    topology_parser = subparsers.add_parser("validate-topology")
    topology_parser.add_argument("--gpu-selector", action="append", required=True)
    topology_parser.add_argument("--worker-count", type=int, required=True)
    topology_parser.add_argument("--agent-instance-count", type=int, required=True)
    topology_parser.add_argument("--user-instance-count", type=int, required=True)
    topology_parser.add_argument("--evaluator-instance-count", type=int, required=True)
    topology_parser.add_argument("--agent-port-base", type=int, required=True)
    topology_parser.add_argument("--role-port-base", type=int, required=True)

    init_parser = subparsers.add_parser("init-queue")
    init_parser.add_argument("--run-plan", type=Path, required=True)
    init_parser.add_argument("--state", type=Path, required=True)
    init_parser.add_argument("--lock", type=Path, required=True)
    init_parser.add_argument("--artifact-dir", type=Path, required=True)
    init_parser.add_argument("--summarizer", type=Path, required=True)
    init_parser.add_argument("--python", required=True)
    init_parser.add_argument("--worker-count", type=int, required=True)
    init_parser.add_argument("--num-trials", type=int, required=True)
    init_parser.add_argument("--base-seed", type=int, required=True)
    init_parser.add_argument("--expected-max-steps", type=int, required=True)
    init_parser.add_argument("--expected-max-errors", type=int, required=True)
    init_parser.add_argument("--expected-agent-model", required=True)
    init_parser.add_argument("--expected-user-model", required=True)
    init_parser.add_argument("--shard-index", type=int)

    claim_parser = subparsers.add_parser("claim")
    claim_parser.add_argument("--state", type=Path, required=True)
    claim_parser.add_argument("--lock", type=Path, required=True)
    claim_parser.add_argument("--worker-id", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate-topology":
            topology = build_local_topology(
                args.gpu_selector,
                worker_count=args.worker_count,
                agent_instance_count=args.agent_instance_count,
                user_instance_count=args.user_instance_count,
                evaluator_instance_count=args.evaluator_instance_count,
                agent_port_base=args.agent_port_base,
                role_port_base=args.role_port_base,
            )
            print(json.dumps(topology, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "init-queue":
            state = initialize_queue(
                args.run_plan,
                args.state,
                args.lock,
                worker_count=args.worker_count,
                completion_checker=lambda entry: _summarizer_checker(args, entry),
                shard_index=args.shard_index,
            )
            print(
                json.dumps(
                    {
                        "pending": len(state["available"]),
                        "skipped_complete": len(state["skipped_complete"]),
                        "corrupt_pending": len(state["corrupt_pending"]),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        if args.command == "claim":
            entry = claim_next_shard(args.state, args.lock, worker_id=args.worker_id)
            if entry is None:
                return 3
            print(entry.to_tsv())
            return 0
    except (OSError, SchedulerError) as exc:
        print(f"eval scheduler error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
