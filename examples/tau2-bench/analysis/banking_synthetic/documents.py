"""Allowed-document loading and synthetic runtime staging.

This module intentionally has no benchmark-task path or loader. The one-time
allowlist construction and the benchmark comparison live in ``isolation.py``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    DOCUMENT_CONTENT_PATTERNS,
    DOCUMENT_FAMILY_PREFIXES,
    DOCUMENT_TOPIC_PATTERNS,
    EMPTY_DB_TABLES,
    FACT_TOPIC_PATTERNS,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def empty_db() -> dict[str, Any]:
    return {name: {"data": {}, "notes": ""} for name in EMPTY_DB_TABLES}


def indexed_body_anchor(document: dict[str, Any], word_limit: int = 80) -> str:
    """Return enough indexed body text to distinguish similar product documents."""

    lines = []
    for raw_line in document["content"].splitlines():
        line = re.sub(r"^[#|*\-\s]+|[|\s]+$", "", raw_line).strip()
        if line:
            lines.append(line)
    return " ".join(" ".join(lines).split()[:word_limit])


def _decisive_fact_candidates(document: dict[str, Any]) -> list[str]:
    marketing_phrases = (
        "at rho-bank, we want",
        "one of the many conveniences",
        "we understand that occasionally",
        "we're here to help",
    )
    facts = []
    for raw_line in document["content"].splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^(?:[-*]|\d+\.)\s+", "", line).strip()
        if not line or re.fullmatch(r"[|:\-\s]+", line):
            continue
        if line.startswith("|"):
            table_fact = line.strip("| ")
            if not re.search(r"(?:\$|%|\b\d|\b(?:yes|no)\b)", table_fact, re.I):
                continue
            facts.append(table_fact)
            continue
        question_suffix = line.rsplit("?", 1)[-1].strip() if "?" in line else ""
        if question_suffix:
            candidates = [
                line
                if len(question_suffix.split()) < 3
                or not re.search(r"[A-Za-z]", question_suffix)
                else question_suffix
            ]
        else:
            candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$])", line)
        for candidate in candidates:
            candidate = candidate.strip()
            lowered = candidate.lower()
            if not candidate or len(candidate) > 240 or candidate.endswith(":"):
                continue
            if any(phrase in lowered for phrase in marketing_phrases):
                continue
            if candidate.endswith("?") and not re.search(r"(?:\$|%|\d)", candidate):
                continue
            if len(candidate.split()) < 3 or not re.search(r"[A-Za-z]", candidate):
                continue
            facts.append(candidate)
    return facts


def decisive_fact(document: dict[str, Any], category: str | None = None) -> str:
    """Return the first concrete fact that matches the scenario category."""

    facts = _decisive_fact_candidates(document)
    if category is not None:
        patterns = FACT_TOPIC_PATTERNS[category]
        for fact in facts:
            if any(re.search(pattern, fact, re.I) for pattern in patterns):
                return fact
    return facts[0] if facts else str(document["title"])


def communication_anchors(fact: str) -> tuple[str, ...]:
    """Return stable exact-substring checks for one policy fact."""

    values = tuple(
        dict.fromkeys(re.findall(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?", fact))
    )
    if values:
        return values
    return (fact.rstrip(".;"),)


@dataclass(frozen=True)
class DocumentCatalog:
    documents: dict[str, dict[str, Any]]
    allowed_ids: tuple[str, ...]

    @classmethod
    def load(cls, documents_dir: Path, allowlist_path: Path) -> "DocumentCatalog":
        manifest = read_json(allowlist_path)
        allowed_ids = tuple(manifest["allowed_document_ids"])
        documents = {}
        for document_id in allowed_ids:
            path = documents_dir / f"{document_id}.json"
            if not path.exists():
                raise FileNotFoundError(f"allowed document is missing: {path}")
            document = read_json(path)
            if document.get("id") != document_id:
                raise ValueError(f"document id/path mismatch: {path}")
            documents[document_id] = document
        if len(documents) != 480:
            raise ValueError(f"expected 480 allowed documents, found {len(documents)}")
        return cls(documents=documents, allowed_ids=allowed_ids)

    def select(
        self,
        category: str,
        offset: int,
        count: int,
        preferred_ids: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        prefixes = DOCUMENT_FAMILY_PREFIXES[category]
        family = [
            document_id
            for document_id in self.allowed_ids
            if document_id.startswith(prefixes)
        ]
        if len(family) < count:
            raise ValueError(
                f"category {category} has only {len(family)} allowlisted family documents"
            )
        for document_id in preferred_ids:
            if document_id not in self.documents:
                raise ValueError(
                    f"category {category} support document is not allowlisted: "
                    f"{document_id}"
                )
            if document_id not in family:
                raise ValueError(
                    f"category {category} support document is outside its family: "
                    f"{document_id}"
                )
        patterns = DOCUMENT_TOPIC_PATTERNS[category]
        topical = []
        for document_id in family:
            document = self.documents[document_id]
            heading = document["title"].lower()
            content_patterns = DOCUMENT_CONTENT_PATTERNS.get(category, ())
            if any(re.search(pattern, heading) for pattern in patterns) or any(
                re.search(pattern, document["content"].lower())
                for pattern in content_patterns
            ):
                topical.append(document_id)

        def rotated(values: list[str]) -> list[str]:
            if not values:
                return []
            start = offset % len(values)
            return values[start:] + values[:start]

        ordered = list(preferred_ids)
        for preferred_id in preferred_ids:
            product_prefix = preferred_id.rsplit("_", 1)[0] + "_"
            ordered.extend(
                document_id
                for document_id in rotated(family)
                if document_id.startswith(product_prefix)
                and document_id not in ordered
            )
        ordered.extend(
            document_id
            for document_id in rotated(topical)
            if document_id not in ordered
        )
        if len(ordered) < count:
            ordered.extend(
                document_id
                for document_id in rotated(family)
                if document_id not in ordered
            )
        return tuple(ordered[:count])


def stage_runtime(
    *,
    tasks: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    allowlist_path: Path,
    source_documents_dir: Path,
    source_prompts_dir: Path,
    source_user_simulator_dir: Path,
    data_root: Path,
) -> dict[str, Any]:
    domain_dir = data_root / "tau2/domains/banking_knowledge"
    if domain_dir.exists():
        raise FileExistsError(f"staging directory already exists: {domain_dir}")
    document_dir = domain_dir / "documents"
    task_dir = domain_dir / "tasks"
    document_dir.mkdir(parents=True)
    task_dir.mkdir()

    allowed = tuple(read_json(allowlist_path)["allowed_document_ids"])
    if len(allowed) != 480 or len(set(allowed)) != 480:
        raise ValueError("allowlist must contain exactly 480 unique document ids")
    for document_id in allowed:
        shutil.copy2(source_documents_dir / f"{document_id}.json", document_dir)
    shutil.copytree(source_prompts_dir, domain_dir / "prompts")
    shutil.copytree(source_user_simulator_dir, data_root / "tau2/user_simulator")
    write_json(domain_dir / "db.json", empty_db())

    contract_by_task = {item["task_id"]: item for item in contracts}
    if len(contract_by_task) != len(contracts):
        raise ValueError("contract task ids must be unique")
    if {task["id"] for task in tasks} != set(contract_by_task):
        raise ValueError("task and contract ids differ")
    for index, task in enumerate(tasks):
        write_json(task_dir / f"task_{index:06d}.json", task)

    manifest = {
        "data_origin": "independent_synthetic",
        "task_count": len(tasks),
        "document_count": len(allowed),
        "task_ids": [task["id"] for task in tasks],
    }
    write_json(domain_dir / "synthetic_manifest.json", manifest)
    return manifest
