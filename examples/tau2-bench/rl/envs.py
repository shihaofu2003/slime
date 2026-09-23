"""tau2 environment helpers for per-task RL databases."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as get_airline_environment
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as get_retail_environment
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.environment import get_environment as get_telecom_environment
from tau2.environment.environment import Environment

SERVICE_AGENT_ROOT = Path("/mnt/afs/users/fush/projects/ServiceAgent")
BANKING_DOCUMENTS = SERVICE_AGENT_ROOT / "tau2-bench/data/tau2/domains/banking_knowledge/documents"


@lru_cache(maxsize=16)
def _database_json(path: Path) -> bytes:
    """Cache immutable dataset bytes; each environment still gets a fresh DB."""
    return path.read_bytes()


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
    task: Task | None = None,
    policy_type: str = "manual",
) -> Callable[..., Environment]:
    """Return a constructor that reloads a fresh environment on every call."""

    def construct(*, solo_mode: bool = False, **_ignored) -> Environment:
        if domain == "airline":
            return get_airline_environment(
                db=FlightDB.model_validate_json(_database_json(db_path)),
                solo_mode=solo_mode,
            )
        if domain == "retail":
            return get_retail_environment(
                db=RetailDB.model_validate_json(_database_json(db_path)),
                solo_mode=solo_mode,
            )
        if domain == "telecom":
            return get_telecom_environment(
                db=TelecomDB.load(db_path),
                solo_mode=solo_mode,
                policy_type=policy_type,
            )
        if domain in {"banking", "banking_knowledge"}:
            if task is None:
                raise ValueError("Banking environment requires the task")
            import sys
            analysis_root = Path(__file__).resolve().parents[1] / "analysis"
            if str(analysis_root) not in sys.path:
                sys.path.insert(0, str(analysis_root))
            from banking_synthetic.validate import environment_constructor

            return environment_constructor(
                documents_dir=BANKING_DOCUMENTS,
                allowed_document_ids=tuple(_banking_document_ids()),
                retrieval_variant="bm25",
                task=task,
            )(solo_mode=solo_mode)
        raise ValueError(f"Unsupported tau2 domain: {domain}")

    return construct


def _banking_document_ids() -> list[str]:
    import json
    manifest = SERVICE_AGENT_ROOT / "datasets/tau2/rl/banking_independent_synthetic/allowed_documents.json"
    return json.loads(manifest.read_text(encoding="utf-8"))["allowed_document_ids"]
