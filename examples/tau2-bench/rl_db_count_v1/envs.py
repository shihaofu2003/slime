"""tau2 environment helpers for per-task RL databases."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as get_airline_environment
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as get_retail_environment
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.environment import get_environment as get_telecom_environment
from tau2.environment.environment import Environment


def task_from_metadata(metadata: dict) -> Task:
    task = metadata.get("task")
    if not isinstance(task, dict):
        raise ValueError("Sample metadata must contain a task object")
    return Task.model_validate(task)


def domain_from_metadata(metadata: dict, task: Task | None = None) -> str:
    domain = metadata.get("domain")
    if isinstance(domain, str) and domain:
        return domain
    if task is not None:
        return task.id.split("_", 1)[0]
    raise ValueError("Sample metadata must contain a domain")


def db_path_from_metadata(metadata: dict) -> Path:
    db_path = metadata.get("db_path")
    if not isinstance(db_path, str) or not db_path:
        raise ValueError("Sample metadata must contain db_path")
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Task db_path does not exist: {path}")
    return path


def make_environment_constructor(
    *,
    domain: str,
    db_path: Path,
    policy_type: str = "manual",
) -> Callable[..., Environment]:
    """Return a constructor that reloads a fresh environment on every call."""

    def construct(*, solo_mode: bool = False, **_ignored) -> Environment:
        if domain == "airline":
            return get_airline_environment(
                db=FlightDB.load(db_path),
                solo_mode=solo_mode,
            )
        if domain == "retail":
            return get_retail_environment(
                db=RetailDB.load(db_path),
                solo_mode=solo_mode,
            )
        if domain == "telecom":
            return get_telecom_environment(
                db=TelecomDB.load(db_path),
                solo_mode=solo_mode,
                policy_type=policy_type,
            )
        raise ValueError(f"Unsupported tau2 domain: {domain}")

    return construct
