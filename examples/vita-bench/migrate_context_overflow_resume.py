#!/usr/bin/env python3
"""Safely migrate the interrupted local-role evaluation to the overflow fix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FROM_PROTOCOL_BUNDLE = "1c53cea1de85c8df9a6ef65b82b812c12d86ffbb517ed8e404507498dba81773"
FROM_VITABENCH_SOURCE = "2e512150fd2af8db4ee80f06d7cb1c5e83248c153d58429edd1652658a27bc0c"
TO_PROTOCOL_BUNDLE = "3f8464171f5239e076abcf8c842bf132a10d5b4a0bc881015cda1797df73c3fa"
TO_VITABENCH_SOURCE = "4f31971539b7c4510732d25eb689f80a2cf7a89b1df8dfffe350d5048e0fce87"
MIGRATION_NAME = "agent_context_window_terminalization_v1"
PENDING_PAIR = ("40711001", 1)
TRIAL_SEEDS = (626729, 373753, 361454, 1567)


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_files(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def current_protocol_state(repo_root: Path, vitabench_dir: Path) -> dict[str, Any]:
    source_root = vitabench_dir / "src" / "vita"
    source_files = [
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    task_files = sorted(
        (vitabench_dir / "data" / "vita" / "domains").glob("*/tasks.json")
    )
    state = {
        "runner_protocol_version": 5,
        "runner_sha256": digest_file(
            repo_root / "examples" / "vita-bench" / "run_qwen3_5_4b_full_eval.sh"
        ),
        "model_config_sha256": digest_file(
            repo_root / "examples" / "vita-bench" / "models_qwen35_full.yaml"
        ),
        "summarizer_sha256": digest_file(
            repo_root / "examples" / "vita-bench" / "summarize_full_eval.py"
        ),
        "sft_exporter_sha256": digest_file(
            repo_root / "examples" / "vita-bench" / "export_sft_archive.py"
        ),
        "vitabench_source_sha256": digest_files(vitabench_dir, source_files),
        "vitabench_task_data_sha256": digest_files(vitabench_dir, task_files),
        "vitabench_pyproject_sha256": digest_file(vitabench_dir / "pyproject.toml"),
    }
    state["protocol_bundle_sha256"] = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return state


def task_ids(result: dict[str, Any], path: Path) -> set[str]:
    ids = {
        str(task.get("id"))
        for task in result.get("tasks", [])
        if isinstance(task, dict) and task.get("id") is not None
    }
    if not ids:
        raise RuntimeError(f"result contains no task IDs: {path}")
    return ids


def simulation_pairs(result: dict[str, Any], path: Path) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    allowed_ids = task_ids(result, path)
    for simulation in result.get("simulations", []):
        if not isinstance(simulation, dict):
            raise RuntimeError(f"result contains a malformed simulation: {path}")
        task_id = str(simulation.get("task_id"))
        trial = simulation.get("trial")
        seed = simulation.get("seed")
        if task_id not in allowed_ids or not isinstance(trial, int) or not 0 <= trial < 4:
            raise RuntimeError(f"result contains an unexpected task/trial pair: {path}")
        if seed != TRIAL_SEEDS[trial]:
            raise RuntimeError(f"result contains an unexpected trial seed: {path}")
        pair = (task_id, trial)
        if pair in pairs:
            raise RuntimeError(f"result contains a duplicate task/trial pair: {path}")
        pairs.add(pair)
    return pairs


def validate_result(path: Path, expected_missing: set[tuple[str, int]]) -> int:
    result = json.loads(path.read_text(encoding="utf-8"))
    info = result.get("info") or {}
    if (
        info.get("num_trials") != 4
        or info.get("max_steps") != 300
        or info.get("max_errors") != 10
    ):
        raise RuntimeError(f"result protocol metadata is invalid: {path}")
    ids = task_ids(result, path)
    pairs = simulation_pairs(result, path)
    expected_pairs = {(task_id, trial) for task_id in ids for trial in range(4)}
    missing = expected_pairs - pairs
    if pairs - expected_pairs or missing != expected_missing:
        raise RuntimeError(f"result completeness changed unexpectedly: {path}")
    return len(pairs)


def validate_interrupted_artifacts(artifact_dir: Path) -> int:
    if (artifact_dir / "summary.json").exists() or (artifact_dir / "summary.csv").exists():
        raise RuntimeError("refusing to migrate an evaluation that already has a summary")

    gate_files = sorted((artifact_dir / "gate").glob("*.json"))
    shard_files = sorted((artifact_dir / "shards").glob("*/shard_*.json"))
    if len(gate_files) != 4 or len(shard_files) != 28:
        raise RuntimeError("interrupted artifact file count does not match the failed run")

    for path in gate_files:
        if validate_result(path, set()) != 4:
            raise RuntimeError(f"gate result count is invalid: {path}")

    partial_path = artifact_dir / "shards" / "delivery" / "shard_02.json"
    formal_simulations = 0
    for path in shard_files:
        expected_missing = {PENDING_PAIR} if path == partial_path else set()
        formal_simulations += validate_result(path, expected_missing)
    if formal_simulations != 1119:
        raise RuntimeError("pre-migration formal simulation count is not 1119")
    return formal_simulations


def write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.migrate.{os.getpid()}")
    mode = path.stat().st_mode & 0o777
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def preserve_manifest(path: Path) -> Path:
    backup = path.with_name("manifest.pre_context_overflow_migration.json")
    if backup.exists():
        if backup.read_bytes() != path.read_bytes():
            raise RuntimeError(f"existing manifest backup differs from {path}")
        return backup
    os.link(path, backup)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return backup


def migrate(artifact_dir: Path, repo_root: Path, vitabench_dir: Path, dry_run: bool) -> None:
    manifest_path = artifact_dir / "manifest.json"
    lock_path = artifact_dir / ".run.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("artifact directory is owned by another process") from exc

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        software = manifest.get("software") or {}
        state = current_protocol_state(repo_root, vitabench_dir)
        if state["vitabench_source_sha256"] != TO_VITABENCH_SOURCE:
            raise RuntimeError("VitaBench source does not match the reviewed overflow fix")
        if state["protocol_bundle_sha256"] != TO_PROTOCOL_BUNDLE:
            raise RuntimeError("current protocol bundle does not match the reviewed target")

        migrations = manifest.get("protocol_compatible_migrations", [])
        if software.get("protocol_bundle_sha256") == TO_PROTOCOL_BUNDLE:
            if not any(item.get("name") == MIGRATION_NAME for item in migrations):
                raise RuntimeError("target bundle is present without its migration record")
            print("CONTEXT_OVERFLOW_RESUME_MIGRATION_ALREADY_APPLIED")
            return
        if manifest.get("schema_version") != 5:
            raise RuntimeError("manifest schema is not 5")
        if software.get("protocol_bundle_sha256") != FROM_PROTOCOL_BUNDLE:
            raise RuntimeError("manifest is not the reviewed failed-run predecessor")
        if software.get("vitabench_source_sha256") != FROM_VITABENCH_SOURCE:
            raise RuntimeError("predecessor VitaBench source hash is unexpected")
        for key, value in state.items():
            if key not in {"protocol_bundle_sha256", "vitabench_source_sha256"}:
                if software.get(key) != value:
                    raise RuntimeError(f"unrelated protocol input changed: {key}")

        formal_simulations = validate_interrupted_artifacts(artifact_dir)
        old_software = deepcopy(software)
        migrated_software = deepcopy(software)
        migrated_software.update(state)
        applied_at = datetime.now(timezone.utc).isoformat()
        software_history = manifest.setdefault("software_history", [])
        if not isinstance(software_history, list):
            raise RuntimeError("software_history is malformed")
        software_history.append(
            {
                "effective_until": applied_at,
                "software": old_software,
            }
        )
        if not isinstance(migrations, list):
            raise RuntimeError("protocol_compatible_migrations is malformed")
        migrations.append(
            {
                "name": MIGRATION_NAME,
                "applied_at": applied_at,
                "from_protocol_bundle_sha256": FROM_PROTOCOL_BUNDLE,
                "to_protocol_bundle_sha256": TO_PROTOCOL_BUNDLE,
                "preexisting_formal_simulations": formal_simulations,
                "pending_task_id": PENDING_PAIR[0],
                "pending_trial": PENDING_PAIR[1],
                "predecessor_manifest_backup": "manifest.pre_context_overflow_migration.json",
                "compatibility_basis": (
                    "The patch only converts an agent context-window HTTP 400, which "
                    "previously produced no SimulationRun, into an invalid-agent-message "
                    "zero-reward terminal result. Persisted simulations do not traverse "
                    "the changed branch."
                ),
            }
        )
        manifest["protocol_compatible_migrations"] = migrations
        manifest["software"] = migrated_software
        if not dry_run:
            preserve_manifest(manifest_path)
            write_json_atomically(manifest_path, manifest)
        action = "DRY_RUN_OK" if dry_run else "APPLIED"
        print(
            f"CONTEXT_OVERFLOW_RESUME_MIGRATION_{action} "
            f"formal_simulations={formal_simulations} pending={PENDING_PAIR[0]}:{PENDING_PAIR[1]}"
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=repo_root
        / "output"
        / "experiments"
        / "vitabench-qwen35-local-role-eval"
        / "artifacts-v2",
    )
    parser.add_argument("--vitabench-dir", type=Path, default=repo_root.parent / "vitabench")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(
        artifact_dir=args.artifact_dir.resolve(),
        repo_root=repo_root,
        vitabench_dir=args.vitabench_dir.resolve(),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
