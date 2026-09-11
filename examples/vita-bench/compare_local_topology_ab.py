#!/usr/bin/env python3
"""Validate and compare legacy 4/2/2 and dynamic 2/2/4 VitaBench shards."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUITE_NAMES = ("delivery", "instore", "ota", "cross_domain")
RESULT_TIME_FORMAT = "%Y%m%d_%H%M%S"
SERVER_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SERVER_METRIC_RE = re.compile(r"^\[(?P<time>[^]]+)] .*#running-req: (?P<running>\d+), .*#queue-req: (?P<queue>\d+)")
FATAL_SERVER_RE = re.compile(
    r"out of memory|cuda error|address already in use|traceback \(most recent call last\)",
    re.IGNORECASE,
)
NATIVE_QUEUE_LOCK_POLICY = {
    "primitive": "fcntl.flock",
    "retry_errno_names": ["EACCES", "EAGAIN"],
    "retry_interval_seconds": 0.05,
    "retry_timeout_seconds": 120.0,
}


class ABValidationError(ValueError):
    """The compared runs are incomplete, invalid, or not protocol-compatible."""


def _load_summarizer(example_dir: Path):
    path = example_dir / "summarize_full_eval.py"
    spec = importlib.util.spec_from_file_location("vita_ab_summarizer", path)
    if spec is None or spec.loader is None:
        raise ABValidationError(f"cannot load summarizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ABValidationError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ABValidationError(f"{path} must contain an object")
    return value


def _selected_specs(manifest: dict[str, Any], artifact_dir: Path, shard_index: int) -> dict[str, dict[str, Any]]:
    suites = manifest.get("suites")
    if not isinstance(suites, list):
        raise ABValidationError("manifest suites must be a list")
    selected: dict[str, dict[str, Any]] = {}
    for suite_name in SUITE_NAMES:
        suite = next(
            (value for value in suites if isinstance(value, dict) and value.get("name") == suite_name),
            None,
        )
        if suite is None:
            raise ABValidationError(f"manifest omits suite {suite_name}")
        shards = suite.get("shards")
        if not isinstance(shards, list):
            raise ABValidationError(f"manifest suite {suite_name} omits shards")
        shard = next(
            (value for value in shards if isinstance(value, dict) and value.get("index") == shard_index),
            None,
        )
        if shard is None:
            raise ABValidationError(f"manifest suite {suite_name} omits shard {shard_index}")
        result_file = artifact_dir / str(shard.get("result_file", ""))
        selected[suite_name] = {
            "suite": suite_name,
            "domain": suite.get("domain"),
            "task_ids": shard.get("task_ids"),
            "result_file": result_file,
        }
    return selected


def _validate_protocol_match(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    protocol_keys = (
        "language",
        "num_trials",
        "temperature",
        "seed",
        "max_steps",
        "max_errors",
        "evaluation_type",
        "enable_think",
        "max_concurrency",
        "shard_size",
    )
    for key in protocol_keys:
        if baseline.get("protocol", {}).get(key) != candidate.get("protocol", {}).get(key):
            raise ABValidationError(f"manifest protocol differs for {key}")
    stable_fields = (
        ("agent", "requested"),
        ("agent", "model_files_sha256"),
        ("user", "requested"),
        ("user", "actual"),
        ("user", "thinking_mode"),
        ("evaluator", "requested"),
        ("evaluator", "actual"),
        ("evaluator", "thinking_mode"),
        ("local_role_serving", "model_files_sha256"),
        ("local_role_serving", "context_length"),
    )
    for section, key in stable_fields:
        if baseline.get(section, {}).get(key) != candidate.get(section, {}).get(key):
            raise ABValidationError(f"manifest model configuration differs for {section}.{key}")
    execution = candidate.get("execution") or {}
    if candidate.get("schema_version") != 6 or execution.get("scheduler") != "dynamic_shard_queue" or execution.get("run_mode") != "ab" or execution.get("aggregate_max_concurrency") != 16:
        raise ABValidationError("candidate is not a schema-6 dynamic A/B run")


def _validate_scheduler_locking(
    candidate_dir: Path,
    candidate_manifest: dict[str, Any],
    example_dir: Path,
) -> dict[str, Any]:
    execution = candidate_manifest.get("execution") or {}
    native_policy = execution.get("queue_lock")
    if native_policy is not None:
        if native_policy != NATIVE_QUEUE_LOCK_POLICY:
            raise ABValidationError("candidate native queue lock policy mismatch")
        software = candidate_manifest.get("software") or {}
        scheduler_sha256 = software.get("eval_scheduler_sha256")
        local_scheduler_sha256 = hashlib.sha256(
            (example_dir / "eval_scheduler.py").read_bytes()
        ).hexdigest()
        if scheduler_sha256 != local_scheduler_sha256:
            raise ABValidationError(
                "candidate native queue lock scheduler hash is not manifest-locked"
            )
        return {
            "mode": "native_scheduler",
            "policy": native_policy,
            "scheduler_sha256": scheduler_sha256,
        }

    record = _load_json(candidate_dir / "scheduler_runtime_adapter.json")
    expected = {
        "schema": "vitabench-afs-flock-retry/v1",
        "enabled": True,
        "scope": "eval_scheduler.py subprocesses only",
        "retry_interval_seconds": 0.05,
        "retry_timeout_seconds": 120.0,
        "retry_errno_names": ["EACCES", "EAGAIN"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ABValidationError(f"candidate AFS lock adapter mismatch for {key}")
    software = candidate_manifest.get("software") or {}
    if record.get("base_runner_sha256") != software.get("runner_sha256"):
        raise ABValidationError("candidate AFS adapter runner hash is not manifest-locked")
    if record.get("base_scheduler_sha256") != software.get("eval_scheduler_sha256"):
        raise ABValidationError("candidate AFS adapter scheduler hash is not manifest-locked")
    local_files = {
        "adapter_sha256": example_dir / "afs_flock_retry" / "sitecustomize.py",
        "adapter_wrapper_sha256": example_dir / "run_qwen3_5_4b_full_eval_local_roles_afs_lock_retry.sh",
    }
    for field, path in local_files.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if record.get(field) != digest:
            raise ABValidationError(f"candidate AFS adapter hash differs for {path.name}")
    return {
        "mode": "runtime_adapter",
        "policy": NATIVE_QUEUE_LOCK_POLICY,
        "record": record,
    }


def _validate_results(
    summarizer: Any,
    manifest: dict[str, Any],
    specs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], datetime, datetime]:
    protocol = manifest["protocol"]
    simulations: list[dict[str, Any]] = []
    starts: list[datetime] = []
    persisted_at: list[datetime] = []
    observed_pairs: set[tuple[str, int]] = set()
    for suite_name in SUITE_NAMES:
        spec = specs[suite_name]
        result = _load_json(spec["result_file"])
        validation = summarizer.validate_shard_data(
            result,
            spec["task_ids"],
            protocol["num_trials"],
            protocol["seed"],
            expected_domain=spec["domain"],
            expected_max_steps=protocol["max_steps"],
            expected_max_errors=protocol["max_errors"],
            expected_agent_model=manifest["agent"]["requested"],
            expected_user_model=manifest["user"]["requested"],
        )
        if not validation["complete"] or validation["seed_validation"] != "validated":
            raise ABValidationError(f"{suite_name} shard is incomplete")
        for simulation in result.get("simulations", []):
            pair = (simulation.get("task_id"), simulation.get("trial"))
            if pair in observed_pairs:
                raise ABValidationError(f"duplicate task/trial pair: {pair}")
            observed_pairs.add(pair)
            try:
                start = datetime.strptime(simulation["start_time"], RESULT_TIME_FORMAT).replace(tzinfo=timezone.utc)
                duration = float(simulation["duration"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ABValidationError(f"simulation {simulation.get('id')} has invalid timing metadata") from exc
            if not math.isfinite(duration) or duration < 0:
                raise ABValidationError("simulation duration must be finite and nonnegative")
            starts.append(start)
            simulations.append(simulation)
        persisted_at.append(datetime.fromtimestamp(spec["result_file"].stat().st_mtime, tz=timezone.utc))
    if len(simulations) != 160 or len(observed_pairs) != 160:
        raise ABValidationError(f"A/B shard set must contain 160 unique trajectories; got {len(simulations)}")
    return simulations, min(starts), max(persisted_at)


def _queue_claims(
    candidate_dir: Path,
    candidate_manifest: dict[str, Any],
    shard_index: int,
) -> dict[str, int]:
    state = _load_json(candidate_dir / ".dynamic_shard_queue.json")
    if state.get("schema") != "vitabench-dynamic-shard-queue/v1" or state.get("selected_shard_index") != shard_index or state.get("available") != []:
        raise ABValidationError("candidate dynamic queue state is incomplete")
    claims = state.get("claims")
    if not isinstance(claims, list) or len(claims) != 4:
        raise ABValidationError("candidate queue must contain exactly four claims")
    claim_workers: dict[str, int] = {}
    for claim in claims:
        if not isinstance(claim, dict):
            raise ABValidationError("candidate queue claim is malformed")
        key = claim.get("key")
        worker_id = claim.get("worker_id")
        if not isinstance(key, str) or not isinstance(worker_id, int):
            raise ABValidationError("candidate queue claim fields are malformed")
        if key in claim_workers:
            raise ABValidationError(f"candidate queue claimed {key} more than once")
        claim_workers[key] = worker_id
    expected_keys = {f"{suite}:{shard_index}" for suite in SUITE_NAMES}
    if set(claim_workers) != expected_keys or set(claim_workers.values()) != set(range(4)):
        raise ABValidationError("candidate queue did not distribute four suite shards once each")

    endpoints = {worker["worker_id"]: worker for worker in candidate_manifest["execution"]["worker_endpoints"]}
    evaluator_model = candidate_manifest["evaluator"]["requested"]
    for key, worker_id in claim_workers.items():
        suite, index_text = key.split(":", 1)
        context = f"shard:{suite}:{index_text}:worker-{worker_id}"
        expected_endpoint = f"http://127.0.0.1:{endpoints[worker_id]['evaluator_port']}/v1/chat/completions"
        valid_journals = 0
        for path in (candidate_dir / "api_journal" / suite).rglob("*.json"):
            try:
                journal = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if (journal.get("context") or {}).get("runner_context") == context and journal.get("requested_model") == evaluator_model and journal.get("response_model") == evaluator_model and journal.get("endpoint") == expected_endpoint:
                valid_journals += 1
        if valid_journals == 0:
            raise ABValidationError(f"candidate evaluator worker {worker_id} has no journal")
    return claim_workers


def _server_queue_metrics(artifact_dir: Path, start: datetime, end: datetime) -> dict[str, dict[str, int]]:
    labels = (
        "agent-0",
        "agent-1",
        "user-0",
        "user-1",
        "evaluator-0",
        "evaluator-1",
        "evaluator-2",
        "evaluator-3",
    )
    metrics: dict[str, dict[str, int]] = {}
    for label in labels:
        samples = 0
        max_running = 0
        max_queue = 0
        queued_samples = 0
        consecutive_queue = 0
        max_consecutive_queue = 0
        for path in sorted(artifact_dir.glob(f"sglang_{label}_*.log")):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if FATAL_SERVER_RE.search(line):
                    raise ABValidationError(f"fatal server log entry in {path}: {line}")
                match = SERVER_METRIC_RE.search(line)
                if match is None:
                    continue
                try:
                    timestamp = datetime.strptime(match.group("time"), SERVER_TIME_FORMAT).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if not start <= timestamp <= end:
                    continue
                running = int(match.group("running"))
                queued = int(match.group("queue"))
                samples += 1
                max_running = max(max_running, running)
                max_queue = max(max_queue, queued)
                if queued:
                    queued_samples += 1
                    consecutive_queue += 1
                    max_consecutive_queue = max(max_consecutive_queue, consecutive_queue)
                else:
                    consecutive_queue = 0
        if samples == 0:
            raise ABValidationError(f"no in-window SGLang metrics found for {label}")
        metrics[label] = {
            "samples": samples,
            "max_running_requests": max_running,
            "max_queued_requests": max_queue,
            "queued_samples": queued_samples,
            "max_consecutive_queued_samples": max_consecutive_queue,
        }
        if label.startswith("evaluator-") and max_queue != 0:
            raise ABValidationError(f"{label} queued evaluator requests during A/B")
        if not label.startswith("evaluator-") and max_consecutive_queue > 10:
            raise ABValidationError(f"{label} had a sustained request queue during A/B")
    return metrics


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-artifact-dir", type=Path, required=True)
    parser.add_argument("--candidate-artifact-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-speedup", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        example_dir = Path(__file__).resolve().parent
        summarizer = _load_summarizer(example_dir)
        if summarizer._repair_json is None:
            raise ABValidationError("json-repair is required; run this check in the VitaBench environment")
        baseline_dir = args.baseline_artifact_dir.resolve()
        candidate_dir = args.candidate_artifact_dir.resolve()
        baseline_manifest = _load_json(baseline_dir / "manifest.json")
        candidate_manifest = _load_json(candidate_dir / "manifest.json")
        _validate_protocol_match(baseline_manifest, candidate_manifest)
        scheduler_locking = _validate_scheduler_locking(
            candidate_dir,
            candidate_manifest,
            example_dir,
        )
        if candidate_manifest["execution"].get("selected_shard_index") != args.shard_index:
            raise ABValidationError("candidate manifest selected a different shard index")
        baseline_specs = _selected_specs(baseline_manifest, baseline_dir, args.shard_index)
        candidate_specs = _selected_specs(candidate_manifest, candidate_dir, args.shard_index)
        for suite in SUITE_NAMES:
            if baseline_specs[suite]["task_ids"] != candidate_specs[suite]["task_ids"]:
                raise ABValidationError(f"task IDs differ for suite {suite}")
        _, baseline_start, baseline_end = _validate_results(summarizer, baseline_manifest, baseline_specs)
        _, candidate_start, candidate_end = _validate_results(summarizer, candidate_manifest, candidate_specs)
        baseline_seconds = (baseline_end - baseline_start).total_seconds()
        candidate_seconds = (candidate_end - candidate_start).total_seconds()
        speedup = baseline_seconds / candidate_seconds
        wall_reduction = 1 - candidate_seconds / baseline_seconds
        claim_workers = _queue_claims(candidate_dir, candidate_manifest, args.shard_index)
        queue_metrics = _server_queue_metrics(candidate_dir, candidate_start, candidate_end)
        passed = speedup >= args.target_speedup
        report = {
            "schema": "vitabench-local-topology-ab/v1",
            "status": "pass" if passed else "fail",
            "shard_index": args.shard_index,
            "suite_count": 4,
            "task_count": 40,
            "trajectory_count": 160,
            "baseline": {
                "topology": "4_agent/2_user/2_evaluator",
                "artifact_dir": str(baseline_dir),
                "start_time": baseline_start.isoformat(),
                "end_time": baseline_end.isoformat(),
                "wall_seconds": baseline_seconds,
            },
            "candidate": {
                "topology": "2_agent/2_user/4_evaluator",
                "scheduler": "dynamic_shard_queue",
                "artifact_dir": str(candidate_dir),
                "start_time": candidate_start.isoformat(),
                "end_time": candidate_end.isoformat(),
                "wall_seconds": candidate_seconds,
                "claim_workers": claim_workers,
                "server_queue_metrics": queue_metrics,
                "scheduler_locking": scheduler_locking,
            },
            "target": {
                "minimum_speedup": args.target_speedup,
                "maximum_wall_reduction": 1 - 1 / args.target_speedup,
            },
            "observed": {
                "speedup": speedup,
                "wall_reduction": wall_reduction,
            },
        }
        _atomic_write(args.output.resolve(), report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if passed else 1
    except (ABValidationError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"A/B validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
