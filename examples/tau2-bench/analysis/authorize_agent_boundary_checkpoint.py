#!/usr/bin/env python3
"""Authorize a boundary-v2 checkpoint for conversion or official evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from check_agent_boundary_rl_health import (
    GATE_VERSION,
    KL_CHECK_NAMES,
    KL_WAIVER_POLICY,
    LONG200_GATE_VERSION,
    LONG200_KL_WAIVER_GATE_VERSION,
    LONG200_STAGES,
)
from check_agent_boundary_rl_pilot import (
    PROFILE,
    PROTOCOL_SIGNATURE,
    TURN_CREDIT_VERSION,
)
from check_domain_expert_health import (
    GATE_VERSION as DOMAIN_EXPERT_GATE_VERSION,
    HEALTH_POLICY as DOMAIN_EXPERT_HEALTH_POLICY,
)
from domain_expert_checkpoint import DOMAINS, EXPERT_ITERATIONS, quota_for_domain


CURVE_ITERATIONS = tuple(range(9, 100, 10))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def checkpoint_manifest(
    checkpoint_root: Path,
    iteration: int,
    *,
    reference_root: Path,
    reference_iteration: int,
) -> dict[str, Any]:
    iteration_dir = checkpoint_root / f"iter_{iteration:07d}"
    reference_dir = reference_root / f"iter_{reference_iteration:07d}"
    required = {
        ".metadata": iteration_dir / ".metadata",
        "common.pt": iteration_dir / "common.pt",
        "metadata.json": iteration_dir / "metadata.json",
        "rollout_state": (
            checkpoint_root / "rollout" / f"global_dataset_state_dict_{iteration}.pt"
        ),
    }
    reference_shards = sorted(path.name for path in reference_dir.glob("__*.distcp"))
    shards = sorted(path.name for path in iteration_dir.glob("__*.distcp"))
    required_files = {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size}
        if path.is_file()
        else {"path": str(path.resolve()), "bytes": None}
        for name, path in required.items()
    }
    nonempty_shards = {
        name: (iteration_dir / name).stat().st_size
        for name in shards
        if (iteration_dir / name).stat().st_size > 0
    }
    complete = bool(
        iteration_dir.is_dir()
        and reference_shards
        and shards == reference_shards
        and len(nonempty_shards) == len(shards)
        and all(item["bytes"] is not None and item["bytes"] > 0 for item in required_files.values())
    )
    return {
        "complete": complete,
        "iteration_dir": str(iteration_dir.resolve()),
        "required_files": required_files,
        "shards": shards,
        "reference_shards": reference_shards,
        "shard_bytes": nonempty_shards,
    }


def _gate_passed(
    gate: dict[str, Any],
    *,
    gate_version: str,
    stage: str,
    checkpoint_root: Path,
    expected_latest: int,
) -> bool:
    root = str(checkpoint_root.resolve())
    return bool(
        gate.get("status") == "pass"
        and gate.get("gate_version") == gate_version
        and gate.get("stage") == stage
        and gate.get("checkpoint_root") == root
        and gate.get("expected_checkpoint_root") == root
        and gate.get("expected_latest") == expected_latest
        and gate.get("protocol_profile") == PROFILE
        and gate.get("agent_protocol_signature") == PROTOCOL_SIGNATURE
        and gate.get("turn_credit_version") == TURN_CREDIT_VERSION
    )


def authorize_checkpoint(
    *,
    variant: str,
    iteration: int,
    checkpoint_root: Path,
    health_gate_path: Path,
    reference_root: Path,
    reference_iteration: int = 99,
    domain: str | None = None,
) -> dict[str, Any]:
    checkpoint_root = checkpoint_root.resolve()
    gate = _read(health_gate_path)
    if variant == "curve":
        if iteration not in CURVE_ITERATIONS:
            raise ValueError(f"curve iteration must be one of {CURVE_ITERATIONS}")
        if iteration == 9:
            gate_iteration = 9
        elif iteration == 19:
            gate_iteration = 19
        else:
            gate_iteration = 99
        gate_stage = f"iter{gate_iteration}"
        gate_version = GATE_VERSION
    elif variant in ("long200", "long200-kl-waiver"):
        if iteration != 199:
            raise ValueError("long200 conversion only authorizes iter199")
        gate_iteration = LONG200_STAGES["iter199"]["end"]
        gate_stage = "iter199"
        gate_version = (
            LONG200_KL_WAIVER_GATE_VERSION
            if variant == "long200-kl-waiver"
            else LONG200_GATE_VERSION
        )
    elif variant in ("domain-expert", "mixed-control"):
        if iteration not in EXPERT_ITERATIONS:
            raise ValueError(
                f"{variant} iteration must be one of {EXPERT_ITERATIONS}"
            )
        if variant == "domain-expert" and domain not in DOMAINS:
            raise ValueError("domain-expert authorization requires a valid domain")
        if variant == "mixed-control" and domain is not None:
            raise ValueError("mixed-control authorization does not accept a domain")
        gate_iteration = iteration
        gate_stage = f"{domain + '-' if domain else 'mixed-'}iter{iteration}"
        gate_version = DOMAIN_EXPERT_GATE_VERSION
    else:
        raise ValueError(f"unsupported authorization variant: {variant}")

    if not _gate_passed(
        gate,
        gate_version=gate_version,
        stage=gate_stage,
        checkpoint_root=checkpoint_root,
        expected_latest=gate_iteration,
    ):
        raise ValueError(
            f"{health_gate_path} does not authorize {checkpoint_root}@{iteration}"
        )
    if variant == "long200" and gate.get("waived_checks"):
        raise ValueError("strict long200 authorization rejects waived health checks")
    if variant == "long200-kl-waiver":
        checks = gate.get("checks") or []
        checks_by_name = {check.get("name"): check for check in checks}
        if (
            gate.get("health_policy") != KL_WAIVER_POLICY
            or gate.get("waived_checks") != list(KL_CHECK_NAMES)
            or not all(name in checks_by_name for name in KL_CHECK_NAMES)
            or not all(
                checks_by_name[name].get("enforced") is False
                for name in KL_CHECK_NAMES
            )
            or any(
                check.get("passed") is not True
                for check in checks
                if check.get("name") not in KL_CHECK_NAMES
            )
        ):
            raise ValueError("invalid long200 KL-waiver health policy or non-KL failure")
    if variant in ("domain-expert", "mixed-control"):
        expected_quota = (
            quota_for_domain(domain)
            if variant == "domain-expert"
            else {"airline": 2, "retail": 2, "telecom": 2}
        )
        checks = gate.get("checks") or []
        if (
            gate.get("variant") != variant
            or gate.get("expert_domain") != domain
            or gate.get("domain_quota") != expected_quota
            or gate.get("health_policy") != DOMAIN_EXPERT_HEALTH_POLICY
            or gate.get("kl_enforced") is not False
            or any(
                check.get("enforced") is not False
                for check in checks
                if check.get("name") in KL_CHECK_NAMES
            )
            or any(
                check.get("passed") is not True
                for check in checks
                if check.get("name") not in KL_CHECK_NAMES
            )
        ):
            raise ValueError(
                "domain-expert authorization rejects mismatched domain/quota, "
                "KL policy, or non-KL health failure"
            )
    latest_path = checkpoint_root / "latest_checkpointed_iteration.txt"
    latest_raw = latest_path.read_text(encoding="utf-8").strip() if latest_path.is_file() else ""
    if not latest_raw.isdigit() or int(latest_raw) < iteration:
        raise ValueError(f"checkpoint latest is not at least {iteration}: {latest_path}")

    save_event = {"path": str(checkpoint_root), "iteration": iteration}
    save_events = ((gate.get("training") or {}).get("save_events") or [])
    if save_event not in save_events:
        raise ValueError(
            f"health gate has no successful save event for {checkpoint_root}@{iteration}"
        )
    manifest = checkpoint_manifest(
        checkpoint_root,
        iteration,
        reference_root=reference_root,
        reference_iteration=reference_iteration,
    )
    if not manifest["complete"]:
        raise ValueError(f"checkpoint files are incomplete: {manifest}")

    return {
        "status": "pass",
        "authorization_version": "boundary-v2-checkpoint-authorization-v1",
        "variant": variant,
        "iteration": iteration,
        "expert_domain": domain,
        "checkpoint_root": str(checkpoint_root),
        "health_gate": str(health_gate_path.resolve()),
        "health_gate_sha256": file_sha256(health_gate_path),
        "health_gate_stage": gate_stage,
        "successful_save_event": save_event,
        "checkpoint_manifest": manifest,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(
            "curve",
            "long200",
            "long200-kl-waiver",
            "domain-expert",
            "mixed-control",
        ),
        required=True,
    )
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--health-gate", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--reference-iteration", type=int, default=99)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = authorize_checkpoint(
            variant=args.variant,
            iteration=args.iteration,
            checkpoint_root=args.checkpoint_root,
            health_gate_path=args.health_gate,
            reference_root=args.reference_root,
            reference_iteration=args.reference_iteration,
            domain=args.domain,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
