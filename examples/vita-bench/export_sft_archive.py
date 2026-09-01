#!/usr/bin/env python3
"""Build lossless, provenance-linked SFT JSONL files from a VitaBench run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

from vita.data_model.message import SystemMessage, UserMessage
from vita.data_model.simulation import Results
from vita.user.base import UserState, is_valid_user_history_message
from vita.user.user_simulator import UserSimulator
from vita.utils.llm_utils import format_messages


SCHEMA = "vita-sft/v1"
JOURNAL_SCHEMA = "vita-llm-response-journal/v1"
OUTPUT_NAMES = {
    "agent": "agent.jsonl",
    "user_simulator": "user_simulator.jsonl",
    "evaluator": "evaluator.jsonl",
}
SUPPORTED_MANIFEST_SCHEMAS = frozenset({2, 3, 4, 5, 6})


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    encoded_rows = [canonical_bytes(row) + b"\n" for row in rows]
    payload = b"".join(encoded_rows)
    atomic_write(path, payload)
    return len(encoded_rows), sha256_bytes(payload)


def result_specs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for gate in manifest["gate"]:
        specs.append(
            {
                "scope": "gate",
                "suite": gate["name"],
                "domain": gate["domain"],
                "task_set": gate["task_set"],
                "task_ids": gate["task_ids"],
                "result_file": gate["result_file"],
            }
        )
    for suite in manifest["suites"]:
        for shard in suite["shards"]:
            specs.append(
                {
                    "scope": "full",
                    "suite": suite["name"],
                    "domain": suite["domain"],
                    "task_set": suite["task_set"],
                    "task_ids": shard["task_ids"],
                    "result_file": shard["result_file"],
                }
            )
    return specs


def protocol_fingerprint(manifest: dict[str, Any]) -> str:
    stable = {
        "benchmark": manifest.get("benchmark"),
        "protocol": manifest.get("protocol"),
        "agent": manifest.get("agent"),
        "user": manifest.get("user"),
        "evaluator": manifest.get("evaluator"),
        "software": manifest.get("software"),
        "execution": manifest.get("execution"),
    }
    return sha256_bytes(canonical_bytes(stable))


def make_sft_row(
    *,
    kind: str,
    messages: list[dict[str, Any]],
    tools: Any,
    generation_config: dict[str, Any],
    usage: Any,
    source: dict[str, Any],
) -> dict[str, Any]:
    content = {"kind": kind, "messages": messages, "tools": tools}
    content_sha256 = sha256_bytes(canonical_bytes(content))
    logical_coordinates = {
        "protocol_fingerprint": source.get("protocol_fingerprint"),
        "kind": kind,
        "source_type": source.get("source_type"),
        "scope": source.get("scope"),
        "suite": source.get("suite"),
        "domain": source.get("domain"),
        "task_id": source.get("task_id"),
        "trial": source.get("trial"),
        "seed": source.get("seed"),
        "requested_model": source.get("requested_model"),
    }
    if source.get("source_type") == "atomic_api_journal":
        context = source.get("context")
        if not isinstance(context, dict):
            raise ValueError("atomic journal source is missing its context")
        runner_context = context.get("runner_context")
        context_parts = (
            runner_context.split(":", 2)
            if isinstance(runner_context, str)
            else []
        )
        if (
            len(context_parts) != 3
            or not isinstance(context.get("domain"), str)
            or not isinstance(context.get("task_id"), str)
            or not isinstance(context.get("trial"), int)
            or not isinstance(context.get("seed"), int)
            or not isinstance(context.get("attempt"), int)
            or not isinstance(context.get("request_sequence"), int)
        ):
            raise ValueError("atomic journal context lacks stable call coordinates")
        logical_coordinates.update(
            {
                "scope": context_parts[0] if len(context_parts) == 3 else None,
                "suite": context_parts[1] if len(context_parts) == 3 else None,
                "domain": context.get("domain"),
                "task_id": context.get("task_id"),
                "trial": context.get("trial"),
                "seed": context.get("seed"),
                "runner_context": runner_context,
                "attempt": context.get("attempt"),
                "request_sequence": context.get("request_sequence"),
            }
        )
    elif kind == "user_simulator":
        logical_coordinates["turn_idx"] = source.get("turn_idx")
    elif kind == "evaluator":
        logical_coordinates["window_idx"] = source.get("window_idx")
    logical_id = sha256_bytes(canonical_bytes(logical_coordinates))
    example_id = sha256_bytes(f"{logical_id}\0{content_sha256}".encode())
    if source.get("source_type") == "atomic_api_journal":
        sample_status = "linked" if source.get("result_refs") else "orphaned"
    else:
        sample_status = "linked_legacy"
    return {
        "schema": SCHEMA,
        "kind": kind,
        "example_id": f"sha256:{example_id}",
        "logical_id": f"sha256:{logical_id}",
        "logical_coordinates": logical_coordinates,
        "sample_status": sample_status,
        "content_sha256": f"sha256:{content_sha256}",
        "messages": messages,
        "tools": tools,
        "generation_config": generation_config,
        "usage": usage,
        "source": source,
    }


def journal_kind(model: str, manifest: dict[str, Any]) -> str | None:
    if model == manifest["agent"]["requested"]:
        return "agent"
    if model == manifest["user"]["requested"]:
        return "user_simulator"
    if model == manifest["evaluator"]["requested"]:
        return "evaluator"
    return None


def expected_response_model(kind: str, manifest: dict[str, Any]) -> str:
    if kind == "agent":
        expected = manifest["agent"].get("served")
    elif kind == "user_simulator":
        expected = manifest["user"].get("actual")
    else:
        expected = manifest["evaluator"].get("actual")
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"manifest is missing the {kind} response model")
    return expected


def collect_result_data(
    artifact_dir: Path,
    manifest: dict[str, Any],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    journal_refs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_artifacts = []
    legacy_rows = []
    language = manifest["protocol"]["language"]
    run_fingerprint = protocol_fingerprint(manifest)

    expected_specs = {spec["result_file"]: spec for spec in result_specs(manifest)}
    candidates = set()
    for relative in expected_specs:
        path = artifact_dir / relative
        if path.is_file():
            candidates.add(path)
    for directory_name in ("gate", "shards"):
        directory = artifact_dir / directory_name
        if directory.is_dir():
            candidates.update(
                path
                for path in directory.rglob("*")
                if path.is_file() and ".json" in path.name and ".tmp." not in path.name
            )

    for path in sorted(candidates):
        relative = str(path.relative_to(artifact_dir))
        spec = expected_specs.get(relative)
        status = "invalid" if ".invalid." in path.name else "partial"
        artifact_record = {
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "status": status,
        }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            results = Results.model_validate(payload)
        except Exception as exc:
            artifact_record["parse_error"] = type(exc).__name__
            raw_artifacts.append(artifact_record)
            continue

        observed_pairs = {
            (simulation.task_id, simulation.trial, simulation.seed)
            for simulation in results.simulations
        }
        if spec is not None:
            expected_count = len(spec["task_ids"]) * manifest["protocol"]["num_trials"]
            if len(observed_pairs) == expected_count:
                status = "complete"
        artifact_record.update(
            status=status,
            simulations=len(results.simulations),
            tasks=len(results.tasks),
        )
        raw_artifacts.append(artifact_record)

        task_by_id = {task.id: task for task in results.tasks}
        suite = spec["suite"] if spec is not None else "unknown"
        domain = spec["domain"] if spec is not None else results.info.environment_info.domain_name
        scope = spec["scope"] if spec is not None else "invalid"
        for simulation_index, simulation in enumerate(results.simulations):
            base_source = {
                "protocol_fingerprint": run_fingerprint,
                "artifact": relative,
                "artifact_sha256": artifact_record["sha256"],
                "artifact_status": status,
                "scope": scope,
                "suite": suite,
                "domain": domain,
                "simulation_id": simulation.id,
                "task_id": simulation.task_id,
                "trial": simulation.trial,
                "seed": simulation.seed,
                "termination_reason": getattr(
                    simulation.termination_reason,
                    "value",
                    str(simulation.termination_reason),
                ),
            }
            for message_index, message in enumerate(simulation.messages):
                message_raw_data = getattr(message, "raw_data", None)
                raw_data = message_raw_data if isinstance(message_raw_data, dict) else {}
                journal_id = raw_data.get("_vita_journal_id")
                pointer = f"/simulations/{simulation_index}/messages/{message_index}"
                if isinstance(journal_id, str) and journal_id:
                    journal_refs[journal_id].append({**base_source, "json_pointer": pointer})
                    continue
                if not isinstance(message, UserMessage) or not raw_data:
                    continue
                response_message = raw_data.get("message")
                if not isinstance(response_message, dict):
                    continue
                task = task_by_id.get(simulation.task_id)
                if task is None:
                    continue
                persona = str(task.user_scenario.user_profile)
                instructions = str(task.instructions)
                saved_guidelines = (
                    results.info.user_info.global_simulation_guidelines
                )
                if isinstance(saved_guidelines, str) and saved_guidelines:
                    system_prompt = saved_guidelines.format(
                        persona=persona,
                        instructions=instructions,
                    )
                else:
                    system_prompt = UserSimulator(
                        persona=persona,
                        instructions=instructions,
                        language=language,
                    ).system_prompt
                visible_prefix = [
                    previous
                    for previous in simulation.messages[:message_index]
                    if is_valid_user_history_message(previous)
                ]
                state = UserState(
                    system_messages=[SystemMessage(role="system", content=system_prompt)],
                    messages=visible_prefix,
                )
                request_messages = format_messages(
                    state.system_messages + state.flip_roles()
                )
                target = deepcopy(response_message)
                target.setdefault("role", "assistant")
                source = {
                    **base_source,
                    "source_type": "legacy_result_reconstruction",
                    "json_pointer": pointer,
                    "turn_idx": message.turn_idx,
                    "requested_model": manifest["user"]["requested"],
                    "response_model": manifest["user"]["actual"],
                }
                legacy_rows.append(
                    make_sft_row(
                        kind="user_simulator",
                        messages=request_messages + [target],
                        tools=None,
                        generation_config=results.info.user_info.llm_args or {},
                        usage=message.usage,
                        source=source,
                    )
                )

            reward_info = simulation.reward_info
            windows = reward_info.window_evaluations if reward_info is not None else None
            for window_index, window in enumerate(windows or []):
                window_data = (
                    window
                    if isinstance(window, dict)
                    else window.model_dump(mode="json")
                )
                raw_data = window_data.get("assistant_message_raw_data")
                journal_id = raw_data.get("_vita_journal_id") if isinstance(raw_data, dict) else None
                pointer = (
                    f"/simulations/{simulation_index}/reward_info/"
                    f"window_evaluations/{window_index}"
                )
                if isinstance(journal_id, str) and journal_id:
                    journal_refs[journal_id].append({**base_source, "json_pointer": pointer})
                    continue
                system_prompt = window_data.get("system_prompt")
                user_prompt = window_data.get("user_prompt")
                content = window_data.get("assistant_message_content")
                usage = window_data.get("assistent_message_usage")
                if not all(isinstance(value, str) and value for value in (system_prompt, user_prompt, content)):
                    continue
                source = {
                    **base_source,
                    "source_type": "legacy_result",
                    "json_pointer": pointer,
                    "window_idx": window_data.get("window_idx", window_index + 1),
                    "requested_model": manifest["evaluator"]["requested"],
                    "response_model": manifest["evaluator"]["actual"],
                }
                legacy_rows.append(
                    make_sft_row(
                        kind="evaluator",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": content},
                        ],
                        tools=None,
                        generation_config={},
                        usage=usage,
                        source=source,
                    )
                )
    return journal_refs, raw_artifacts, legacy_rows


def export(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if manifest_path.parent != artifact_dir:
        raise SystemExit("manifest must be directly inside the artifact directory")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") not in SUPPORTED_MANIFEST_SCHEMAS:
        raise SystemExit("unsupported manifest schema")

    journal_refs, raw_artifacts, legacy_rows = collect_result_data(
        artifact_dir, manifest
    )
    rows_by_kind: dict[str, list[dict[str, Any]]] = {
        kind: [] for kind in OUTPUT_NAMES
    }
    rejected_journals = []
    seen_journal_ids = set()
    accepted_journal_ids = set()
    journal_root = artifact_dir / "api_journal"
    journal_paths = (
        sorted(path for path in journal_root.rglob("*") if path.is_file())
        if journal_root.is_dir()
        else []
    )
    for path in journal_paths:
        relative = str(path.relative_to(artifact_dir))
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema") != JOURNAL_SCHEMA:
                raise ValueError("wrong schema")
            journal_id = record.get("journal_id")
            if not isinstance(journal_id, str) or not journal_id:
                raise ValueError("journal id/path mismatch")
            final_name = f"{journal_id}.json"
            is_final = path.name == final_name
            is_temporary = path.name.startswith(f".{journal_id}.tmp.")
            if not is_final and not is_temporary:
                raise ValueError("journal id/path mismatch")
            if is_temporary:
                recovered_path = path.with_name(final_name)
                if recovered_path.exists():
                    raise ValueError("temporary journal collides with final journal")
                os.replace(path, recovered_path)
                recovered_path.chmod(0o400)
                directory_descriptor = os.open(
                    recovered_path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
                path = recovered_path
                relative = str(path.relative_to(artifact_dir))
            if journal_id in seen_journal_ids:
                raise ValueError("duplicate journal id")
            seen_journal_ids.add(journal_id)
            requested_model = record.get("requested_model")
            kind = journal_kind(requested_model, manifest)
            if kind is None:
                raise ValueError("unexpected requested model")
            response_model = record.get("response_model")
            if response_model != expected_response_model(kind, manifest):
                raise ValueError("unexpected response model")
            request = record.get("request")
            response = record.get("response")
            if not isinstance(request, dict) or not isinstance(response, dict):
                raise ValueError("request/response must be objects")
            if response.get("model") != response_model:
                raise ValueError("response model metadata mismatch")
            request_messages = request.get("messages")
            choices = response.get("choices")
            if not isinstance(request_messages, list) or not choices:
                raise ValueError("missing request messages or response choices")
            target = choices[0].get("message") if isinstance(choices[0], dict) else None
            if not isinstance(target, dict) or target.get("role") != "assistant":
                raise ValueError("missing assistant response message")
            generation_config = {
                key: value
                for key, value in request.items()
                if key not in {"messages", "model", "tools"}
            }
            source = {
                "protocol_fingerprint": protocol_fingerprint(manifest),
                "source_type": "atomic_api_journal",
                "journal_id": journal_id,
                "journal_path": relative,
                "journal_sha256": sha256_file(path),
                "context": record.get("context"),
                "requested_model": requested_model,
                "response_model": record.get("response_model"),
                "endpoint": record.get("endpoint"),
                "result_refs": journal_refs.get(journal_id, []),
            }
            rows_by_kind[kind].append(
                make_sft_row(
                    kind=kind,
                    messages=deepcopy(request_messages) + [deepcopy(target)],
                    tools=request.get("tools"),
                    generation_config=generation_config,
                    usage=response.get("usage"),
                    source=source,
                )
            )
            accepted_journal_ids.add(journal_id)
        except Exception as exc:
            rejected_journals.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
            )

    missing_journal_ids = sorted(set(journal_refs) - accepted_journal_ids)
    if missing_journal_ids:
        raise SystemExit(
            f"result artifacts reference {len(missing_journal_ids)} missing journals"
        )
    for row in legacy_rows:
        rows_by_kind[row["kind"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    file_records = []
    for kind, filename in OUTPUT_NAMES.items():
        rows = sorted(rows_by_kind[kind], key=lambda row: row["example_id"])
        line_count, digest = write_jsonl(output_dir / filename, rows)
        file_records.append(
            {
                "kind": kind,
                "path": filename,
                "lines": line_count,
                "bytes": (output_dir / filename).stat().st_size,
                "sha256": digest,
            }
        )
    rejected_count, rejected_digest = write_jsonl(
        output_dir / "rejected_journals.jsonl", rejected_journals
    )
    raw_count, raw_digest = write_jsonl(
        output_dir / "raw_artifacts.jsonl", raw_artifacts
    )

    all_rows = [row for rows in rows_by_kind.values() for row in rows]
    content_counts = Counter(row["content_sha256"] for row in all_rows)
    example_counts = Counter(row["example_id"] for row in all_rows)
    logical_contents: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        logical_contents[row["logical_id"]].add(row["content_sha256"])
    index = {
        "schema": "vita-sft-archive/v1",
        "exporter": {
            "name": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "manifest": {
            "path": str(manifest_path.relative_to(artifact_dir)),
            "sha256": sha256_bytes(manifest_bytes),
            "protocol_fingerprint": protocol_fingerprint(manifest),
        },
        "files": file_records
        + [
            {
                "kind": "rejected_journals",
                "path": "rejected_journals.jsonl",
                "lines": rejected_count,
                "bytes": (output_dir / "rejected_journals.jsonl").stat().st_size,
                "sha256": rejected_digest,
            },
            {
                "kind": "raw_artifacts",
                "path": "raw_artifacts.jsonl",
                "lines": raw_count,
                "bytes": (output_dir / "raw_artifacts.jsonl").stat().st_size,
                "sha256": raw_digest,
            },
        ],
        "counts": {
            "journal_files": len(journal_paths),
            "journal_ids": len(seen_journal_ids),
            "journal_accepted": len(accepted_journal_ids),
            "journal_rejected": len(rejected_journals),
            "journal_linked": sum(
                1
                for journal_id in accepted_journal_ids
                if journal_refs.get(journal_id)
            ),
            "journal_orphaned": sum(
                1
                for journal_id in accepted_journal_ids
                if not journal_refs.get(journal_id)
            ),
            "legacy_examples": len(legacy_rows),
            "examples": len(all_rows),
            "unique_examples": len(example_counts),
            "duplicate_example_instances": sum(
                count - 1 for count in example_counts.values() if count > 1
            ),
            "unique_content": len(content_counts),
            "exact_duplicate_examples": sum(
                count - 1 for count in content_counts.values() if count > 1
            ),
            "logical_conflicts": sum(
                1 for values in logical_contents.values() if len(values) > 1
            ),
            "raw_result_artifacts": len(raw_artifacts),
        },
    }
    atomic_write(
        output_dir / "index.json",
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
        + b"\n",
    )
    print(json.dumps(index["counts"], ensure_ascii=False, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(export(parse_args()))
