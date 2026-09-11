#!/usr/bin/env python3
"""Prepare and validate isolated tau2 domain-expert checkpoint lineages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


DOMAINS = ("airline", "retail", "telecom")
EXPERT_ITERATIONS = (109, 119, 129)
SOURCE_ITERATION = 99
TARGET_ITERATION = 129
LINEAGE_VERSION = "boundary-v2-domain-expert-v1"
SAMPLER_STATE_KEY = "tau2_domain_quota_v1"
SOURCE_GATE_VERSION = "turn-aware-rl-health-v1"


def quota_for_domain(domain: str) -> dict[str, int]:
    if domain not in DOMAINS:
        raise ValueError(f"unsupported expert domain: {domain!r}")
    return {name: 6 if name == domain else 0 for name in DOMAINS}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_sampler_state(checkpoint_root: Path, iteration: int) -> dict[str, Any]:
    path = checkpoint_root / "rollout" / f"global_dataset_state_dict_{iteration}.pt"
    if not path.is_file():
        raise ValueError(f"missing rollout sampler state: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid rollout sampler payload: {path}")
    state = (payload.get("metadata") or {}).get(SAMPLER_STATE_KEY)
    if not isinstance(state, dict):
        raise ValueError(f"missing {SAMPLER_STATE_KEY} in {path}")
    fingerprint = state.get("dataset_fingerprint")
    offsets = state.get("domain_offsets")
    epochs = state.get("domain_epochs")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"invalid dataset fingerprint in {path}")
    if set(offsets or {}) != set(DOMAINS) or set(epochs or {}) != set(DOMAINS):
        raise ValueError(f"invalid domain cursor maps in {path}")
    return {
        "dataset_fingerprint": fingerprint,
        "domain_offsets": {domain: int(offsets[domain]) for domain in DOMAINS},
        "domain_epochs": {domain: int(epochs[domain]) for domain in DOMAINS},
        "sample_group_index": int(payload.get("sample_group_index", 0)),
        "sample_index": int(payload.get("sample_index", 0)),
    }


def checkpoint_complete(root: Path, iteration: int) -> bool:
    iteration_dir = root / f"iter_{iteration:07d}"
    required = (
        iteration_dir / ".metadata",
        iteration_dir / "common.pt",
        iteration_dir / "metadata.json",
        root / "rollout" / f"global_dataset_state_dict_{iteration}.pt",
    )
    shards = list(iteration_dir.glob("__*.distcp"))
    return bool(
        all(path.is_file() and path.stat().st_size > 0 for path in required)
        and shards
        and all(path.stat().st_size > 0 for path in shards)
    )


def validate_source_gate(
    gate: dict[str, Any], *, gate_path: Path, source_root: Path
) -> None:
    root = str(source_root.resolve())
    if not (
        gate.get("status") == "pass"
        and gate.get("gate_version") == SOURCE_GATE_VERSION
        and gate.get("stage") == "iter99"
        and gate.get("checkpoint_root") == root
        and gate.get("expected_checkpoint_root") == root
        and gate.get("expected_latest") == SOURCE_ITERATION
        and gate.get("turn_credit_version") == "turn-credit-v1"
    ):
        raise ValueError(f"source Gate does not authorize immutable iter99: {gate_path}")
    latest_path = source_root / "latest_checkpointed_iteration.txt"
    latest = latest_path.read_text(encoding="utf-8").strip() if latest_path.is_file() else ""
    if latest != str(SOURCE_ITERATION) or not checkpoint_complete(source_root, SOURCE_ITERATION):
        raise ValueError(f"source root is not a complete immutable iter99: {source_root}")


def expected_lineage(
    *,
    domain: str,
    source_root: Path,
    destination_root: Path,
    source_gate_path: Path,
    source_sampler_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lineage_version": LINEAGE_VERSION,
        "expert_domain": domain,
        "domain_quota": quota_for_domain(domain),
        "source_checkpoint_root": str(source_root.resolve()),
        "destination_checkpoint_root": str(destination_root.resolve()),
        "source_iteration": SOURCE_ITERATION,
        "target_iteration": TARGET_ITERATION,
        "source_health_gate": str(source_gate_path.resolve()),
        "source_health_gate_sha256": file_sha256(source_gate_path),
        "dataset_fingerprint": source_sampler_state["dataset_fingerprint"],
        "restored_domain_cursors": {
            "domain_offsets": source_sampler_state["domain_offsets"],
            "domain_epochs": source_sampler_state["domain_epochs"],
            "sample_group_index": source_sampler_state["sample_group_index"],
            "sample_index": source_sampler_state["sample_index"],
        },
        "scheduler_horizon_override": True,
    }


def prepare_domain_expert(
    *,
    domain: str,
    source_root: Path,
    destination_root: Path,
    source_gate_path: Path,
) -> tuple[Path, dict[str, Any]]:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    gate = read_json(source_gate_path)
    validate_source_gate(gate, gate_path=source_gate_path, source_root=source_root)
    source_state = read_sampler_state(source_root, SOURCE_ITERATION)
    lineage = expected_lineage(
        domain=domain,
        source_root=source_root,
        destination_root=destination_root,
        source_gate_path=source_gate_path,
        source_sampler_state=source_state,
    )

    destination_root.mkdir(parents=True, exist_ok=True)
    lineage_path = destination_root / "DOMAIN_EXPERT_LINEAGE.json"
    if lineage_path.is_file():
        if read_json(lineage_path) != lineage:
            raise ValueError(f"domain-expert lineage marker mismatch: {lineage_path}")
    else:
        unexpected = [path.name for path in destination_root.iterdir()]
        if unexpected:
            raise ValueError(
                "destination has no lineage marker but is non-empty: "
                f"{destination_root}: {sorted(unexpected)}"
            )
        temporary = lineage_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(lineage_path)

    latest_path = destination_root / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file():
        return source_root, lineage
    raw_latest = latest_path.read_text(encoding="utf-8").strip()
    if not raw_latest.isdigit():
        raise ValueError(f"invalid domain-expert latest marker: {latest_path}")
    latest = int(raw_latest)
    if latest >= TARGET_ITERATION:
        raise ValueError(f"{domain} expert already completed at iteration {latest}")
    if latest not in EXPERT_ITERATIONS[:-1]:
        raise ValueError(
            "domain-expert resume requires iter109 or iter119; "
            f"latest={latest}"
        )
    if not checkpoint_complete(destination_root, latest):
        raise ValueError(f"incomplete domain-expert checkpoint: {destination_root}@{latest}")
    resumed = read_sampler_state(destination_root, latest)
    if resumed["dataset_fingerprint"] != source_state["dataset_fingerprint"]:
        raise ValueError("domain-expert resume dataset fingerprint differs from iter99")
    for other in DOMAINS:
        if other == domain:
            continue
        if (
            resumed["domain_offsets"][other] != source_state["domain_offsets"][other]
            or resumed["domain_epochs"][other] != source_state["domain_epochs"][other]
        ):
            raise ValueError(f"inactive {other} sampler cursor changed in {domain} expert")
    return destination_root, lineage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    args = parser.parse_args()
    try:
        load_root, _ = prepare_domain_expert(
            domain=args.domain,
            source_root=args.source_root,
            destination_root=args.destination_root,
            source_gate_path=args.source_gate,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(load_root)


if __name__ == "__main__":
    main()
