#!/usr/bin/env python3
"""Blind semantic review of tau2 trajectories with gemini-2.5-flash."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from trajectory_patterns import (
    PATTERNS,
    PRIMARY_FAILURE_MODES,
    PROMPT_VERSION,
    build_judge_payload,
    extract_features,
    load_result_cells,
    load_tool_catalog,
    read_jsonl,
    validate_review,
)

MODEL = "openai/gemini-2.5-flash"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_PREFIXES = {
    "raw": "Qwen3-4B-Instruct-2507_user_stop_parser_0720_053557",
    "sft": "Qwen3-4B-tau2-agent-sft-multitool-iter0002413_user_stop_parser_0720_130353",
    "rl2e6": "Qwen3-4B-tau2-agent-rl-stability-lr2e6-iter0000099_user_stop_parser_0729_155923",
}

SYSTEM_PROMPT = """You are auditing a customer-service agent trajectory against the
provided domain policy, tool namespaces, task instructions, reference actions, and
assertions. Judge the actual behavior, not merely whether it resembles the reference:
references are evidence, not the only legal trajectory.

The domain policy says at most one tool call per turn. Treat every multi-tool turn as a
policy violation even if another system instruction allowed parallel calls. Distinguish
Agent tools from User/device tools using requestor and namespaces. A preceding "yes" is
not sufficient confirmation unless the agent first listed the exact action, affected
items, cost/payment method, and consequences, and the confirmation is current.

Use these positive patterns: intent_tracking, clarify_before_act, read_plan_act,
atomic_commit, progressive_diagnosis (telecom service -> data -> MMS dependencies and
post-fix verification), grounded_recovery, grounded_closeout. Also review these negative
patterns: wrong_tool_or_workflow, wrong_arguments, missing_precondition,
confirmation_violation, unrelated_or_extra_write, multi_tool_policy_violation,
tool_namespace_confusion, nonexistent_tool, repeat_or_fanout, failed_recovery,
false_success_claim, incomplete_task, premature_transfer_or_close, and the aggregate
domain_policy_compliance.

Domain checks include airline basic-economy/insurance/compensation/itinerary/payment
constraints; retail mandatory authentication, pending-vs-delivered routing, collecting
all items, and refund method; telecom Agent-vs-User tools, payment handshake,
suspend/resume conditions, and progressive diagnosis.

Return one JSON object only. Evidence must be brief and factual. Every cited turn index
must exist. For every named pattern return status pass|fail|not_applicable|uncertain,
severity critical|minor|none, turn_indices, evidence, and correct_behavior. A critical
error directly invalidates the requested outcome or violates an important policy.

Apply statuses consistently:
- intent_tracking is always applicable.
- clarify_before_act applies only when required information was initially missing.
- read_plan_act applies only when policy or current state required a read before action.
- atomic_commit applies only when the agent attempted a state change.
- progressive_diagnosis applies only to telecom fault diagnosis with dependent checks.
- grounded_recovery applies only after a tool error, blocked workflow, or user correction.
- grounded_closeout applies when the agent made a final outcome claim or closed the task.
- Every negative pattern is always applicable: use fail when the violation occurred and
  pass when it did not. Never use not_applicable for a negative pattern.
- domain_policy_compliance is always applicable.

The payload's mechanical_audit_facts are label-free facts derived from the conversation.
Treat them as authoritative for counts, tool schemas, namespaces, and error locations,
but make semantic judgments yourself. Before answering, silently check every pattern
against those facts. A fail must name an actual violation; a pass must not describe one.
Use severity critical or minor only with fail; otherwise use none.

Emit minified JSON on one line. Use at most two turn indices per pattern. Keep evidence
and correct_behavior to at most eight words each (or twelve Chinese characters).
primary_failure_mode must be exactly one of: wrong_workflow, wrong_arguments,
missing_precondition, confirmation_violation, tool_namespace_confusion,
failed_recovery, false_success_claim, incomplete_task, policy_violation, other, none.
user_simulator_error.status must be none, helped_agent, hindered_agent, or minor.
"""


def usable_secret(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return not normalized.startswith("<") and "your_key" not in normalized


def readable_http_error(body: str) -> str:
    text = re.sub(r"<style.*?</style>|<script.*?</script>", " ", body, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:1000]


def _schema_example(trajectory_id: str) -> dict[str, Any]:
    finding = {
        "status": "uncertain",
        "severity": "none",
        "turn_indices": [],
        "evidence": "",
        "correct_behavior": "",
    }
    return {
        "trajectory_id": trajectory_id,
        "prompt_version": PROMPT_VERSION,
        "patterns": {name: dict(finding) for name in PATTERNS},
        "primary_failure_mode": "none",
        "user_simulator_error": {"status": "none", "turn_indices": []},
    }


def _response_format() -> dict[str, Any]:
    finding = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pass", "fail", "not_applicable", "uncertain"],
            },
            "severity": {
                "type": "string",
                "enum": ["critical", "minor", "none"],
            },
            "turn_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "maxItems": 2,
            },
            "evidence": {"type": "string", "maxLength": 80},
            "correct_behavior": {"type": "string", "maxLength": 80},
        },
        "required": [
            "status",
            "severity",
            "turn_indices",
            "evidence",
            "correct_behavior",
        ],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "trajectory_id": {"type": "string"},
            "prompt_version": {"type": "string", "enum": [PROMPT_VERSION]},
            "patterns": {
                "type": "object",
                "additionalProperties": False,
                "properties": {pattern: finding for pattern in PATTERNS},
                "required": list(PATTERNS),
            },
            "primary_failure_mode": {
                "type": "string",
                "enum": sorted(PRIMARY_FAILURE_MODES),
            },
            "user_simulator_error": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["none", "helped_agent", "hindered_agent", "minor"],
                    },
                    "turn_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "maxItems": 2,
                    },
                },
                "required": ["status", "turn_indices"],
            },
        },
        "required": [
            "trajectory_id",
            "prompt_version",
            "patterns",
            "primary_failure_mode",
            "user_simulator_error",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "tau2_pattern_review",
            "strict": True,
            "schema": schema,
        },
    }
def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip("'\"")


def parse_model_spec(spec: str) -> tuple[str, str]:
    label, separator, prefix = spec.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected LABEL:PREFIX, got {spec!r}")
    return label.strip(), prefix.strip()


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response is not a JSON object")
    return value


class OpenAIProxy:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        request_model: str,
        timeout: int,
        retries: int,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.request_model = request_model
        self.timeout = timeout
        self.retries = retries
        self.extra_body = extra_body or {}

    def complete(self, payload: dict[str, Any], repair: str | None = None) -> str:
        schema = _schema_example(payload["trajectory_id"])
        blinded_payload = {
            key: value for key, value in payload.items() if not key.startswith("_")
        }
        user_content = (
            "Audit this blinded trajectory and emit exactly this shape (fill every pattern):\n"
            + json.dumps(schema, ensure_ascii=False)
            + "\n\nEVIDENCE:\n"
            + json.dumps(blinded_payload, ensure_ascii=False, default=str)
        )
        if repair:
            user_content += (
                "\n\nYour prior answer was invalid. Correct it without changing the audit. "
                "Minify the JSON and limit evidence/correct_behavior to four words each. "
                f"Validation errors: {repair}"
            )
        request_body = {
            "model": self.request_model,
            "temperature": 0,
            "max_tokens": 4096,
            "response_format": _response_format(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        request_body.update(self.extra_body)
        data = json.dumps(request_body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.url,
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"HTTP {exc.code}: {readable_http_error(body)}"
                )
                if exc.code not in {408, 409, 429} and exc.code < 500:
                    break
                if attempt == self.retries:
                    break
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else min(8, 2**attempt)
                except ValueError:
                    delay = min(8, 2**attempt)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(min(8, 2**attempt))
        raise RuntimeError(f"transport failed after {self.retries + 1} attempts: {last_error}")


def select_calibration(
    cells: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    paired_outcomes: dict[tuple[str, str, int], set[bool]] = defaultdict(set)
    for cell in cells:
        sim = cell["simulation"]
        record = features[
            f"{cell['model']}|{cell['domain']}|{sim.get('task_id')}|{sim.get('trial')}"
        ]
        paired_outcomes[
            (cell["domain"], str(sim.get("task_id")), sim.get("trial"))
        ].add(record["success"])
    groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        sim = cell["simulation"]
        record = features[
            f"{cell['model']}|{cell['domain']}|{sim.get('task_id')}|{sim.get('trial')}"
        ]
        groups[(cell["model"], cell["domain"], record["success"])].append(cell)

    chosen = []
    for key, group in sorted(groups.items()):
        def score(cell: dict[str, Any]) -> tuple:
            sim = cell["simulation"]
            record = features[
                f"{cell['model']}|{cell['domain']}|{sim.get('task_id')}|{sim.get('trial')}"
            ]
            known_airline_six = (
                cell["model"] == "sft"
                and cell["domain"] == "airline"
                and str(sim.get("task_id")) == "6"
                and sim.get("trial") == 0
                and record["success"]
            )
            suspicious = bool(record["false_success_candidate_turns"])
            invalid_schema = bool(
                record["invalid_tool_calls"] or record["tool_argument_schema_errors"]
            )
            namespace = bool(record["namespace_confusion_turns"])
            stage_flip = len(
                paired_outcomes[
                    (cell["domain"], str(sim.get("task_id")), sim.get("trial"))
                ]
            ) > 1
            complexity = (
                record["num_messages"]
                + 12 * record["tool_errors"]
                + 8 * record["multitool_turns"]
                + 5 * record["nonadjacent_repeated_calls"]
            )
            return (
                known_airline_six,
                stage_flip,
                invalid_schema,
                namespace,
                suspicious,
                complexity,
            )

        ranked = sorted(group, key=score, reverse=True)
        picks = [ranked[0]]
        forced_task = (
            str(ranked[0]["simulation"].get("task_id"))
            if score(ranked[0])[0]
            else None
        )
        complexity_ranked = sorted(
            group,
            key=lambda cell: features[
                f"{cell['model']}|{cell['domain']}|"
                f"{cell['simulation'].get('task_id')}|{cell['simulation'].get('trial')}"
            ]["num_messages"],
        )
        for candidate in (complexity_ranked[0], complexity_ranked[-1], *ranked[1:]):
            if forced_task and str(candidate["simulation"].get("task_id")) == forced_task:
                continue
            if candidate not in picks:
                picks.append(candidate)
            if len(picks) == 3:
                break
        if len(picks) != 3:
            raise ValueError(f"calibration cell {key} has only {len(picks)} trajectories")
        chosen.extend(picks)
    if len(chosen) != 54:
        raise ValueError(f"expected 54 calibration trajectories, selected {len(chosen)}")
    return chosen


def review_one(
    client: OpenAIProxy,
    payload: dict[str, Any],
    calibration_run: int,
) -> dict[str, Any]:
    real_id = payload["_trajectory_id"]
    valid_turns = {
        int(message["turn_index"])
        for message in payload["conversation"]
        if isinstance(message.get("turn_index"), int)
    }
    repair = None
    for format_attempt in range(2):
        text = client.complete(payload, repair=repair)
        try:
            review = _extract_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            repair = f"JSON parse error: {exc}"
            continue
        review["trajectory_id"] = real_id
        review["prompt_version"] = PROMPT_VERSION
        errors = validate_review(review, valid_turns)
        if not errors:
            review["calibration_run"] = calibration_run
            return review
        repair = "; ".join(errors)
    raise ValueError(f"{real_id}: invalid after format retry: {repair}")


def run_reviews(
    client: OpenAIProxy,
    jobs: list[tuple[dict[str, Any], int]],
    output: Path,
    concurrency: int,
) -> None:
    existing = read_jsonl(output)
    done = {
        (record["trajectory_id"], record.get("prompt_version"), record.get("calibration_run", 0))
        for record in existing
    }
    pending = [
        job
        for job in jobs
        if (job[0]["_trajectory_id"], PROMPT_VERSION, job[1]) not in done
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures = []
    with output.open("a", encoding="utf-8") as file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(review_one, client, payload, run): (payload, run)
                for payload, run in pending
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                payload, run = futures[future]
                try:
                    review = future.result()
                except Exception as exc:
                    failures.append((payload["_trajectory_id"], run, str(exc)))
                    print(f"[failed] {payload['_trajectory_id']} run={run}: {exc}", file=sys.stderr)
                    continue
                with lock:
                    file.write(json.dumps(review, ensure_ascii=False) + "\n")
                    file.flush()
                if completed % 10 == 0 or completed == len(pending):
                    print(f"completed {completed}/{len(pending)}")
    if failures:
        raise SystemExit(f"{len(failures)} reviews failed; rerun to resume")


def calibration_report(
    path: Path,
    valid_turns: dict[str, set[int]],
    expected_ids: set[str],
) -> None:
    records = read_jsonl(path)
    keys = [
        (
            record.get("trajectory_id"),
            record.get("prompt_version"),
            record.get("calibration_run", 0),
        )
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise SystemExit("calibration contains duplicate trajectory/prompt/run records")
    invalid = []
    for record in records:
        trajectory = record.get("trajectory_id")
        if trajectory not in valid_turns:
            invalid.append((trajectory, ["unknown trajectory_id"]))
            continue
        errors = validate_review(record, valid_turns[trajectory])
        if errors:
            invalid.append((trajectory, errors))
    if invalid:
        raise SystemExit(f"calibration contains invalid reviews: {invalid[:3]}")
    by_id: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_id[record["trajectory_id"]][record.get("calibration_run", 0)] = record
    base = [runs[0] for runs in by_id.values() if 0 in runs]
    repeats = [(runs[0], runs[1]) for runs in by_id.values() if 0 in runs and 1 in runs]
    agreements = []
    for first, second in repeats:
        for pattern in PATTERNS:
            agreements.append(
                first["patterns"][pattern]["status"] == second["patterns"][pattern]["status"]
            )
    consistency = sum(agreements) / len(agreements) if agreements else 0
    parse_rate = len(records) / 72
    print(
        f"calibration: unique={len(base)}/54 repeats={len(repeats)}/18 "
        f"status_consistency={consistency:.3%} parse_rate={parse_rate:.3%}"
    )
    if len(base) != 54 or len(repeats) != 18:
        raise SystemExit("calibration incomplete")
    if {review["trajectory_id"] for review in base} != expected_ids:
        raise SystemExit("calibration trajectory set does not match deterministic selection")
    if consistency < 0.90:
        raise SystemExit("calibration pattern consistency is below 90%")
    if parse_rate < 0.99:
        raise SystemExit("calibration parse rate is below 99%")
    airline_six = [
        review
        for review in base
        if "|airline|6|" in review["trajectory_id"]
        and review["trajectory_id"].startswith("sft|")
    ]
    if not airline_six or not any(
        review["patterns"]["domain_policy_compliance"]["status"] == "fail"
        for review in airline_six
    ):
        raise SystemExit("known airline task 6 policy failure was not detected")
    candidates = [
        review
        for review in base
        if review["patterns"]["false_success_claim"]["status"] == "fail"
    ]
    if not candidates:
        raise SystemExit("calibration did not detect a false-success claim")
    if not any(
        review["patterns"]["tool_namespace_confusion"]["status"] == "fail"
        for review in base
        if "|telecom|" in review["trajectory_id"]
    ):
        raise SystemExit("calibration did not detect telecom tool namespace confusion")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("probe", "calibration", "full", "validate"), required=True
    )
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model", type=parse_model_spec, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument(
        "--provider",
        choices=("auto", "openai", "openrouter"),
        default="auto",
        help="API route; auto prefers OPENROUTER_API_KEY for Gemini",
    )
    args = parser.parse_args()

    model_specs = args.model or list(DEFAULT_PREFIXES.items())
    cells = load_result_cells(
        args.sim_root, model_specs, ("airline", "retail", "telecom")
    )
    catalog = load_tool_catalog(args.tau2_root)
    feature_records = [
        extract_features(
            cell["simulation"], cell["task"], cell["model"], cell["domain"], catalog
        )
        for cell in cells
    ]
    features = {record["trajectory_id"]: record for record in feature_records}
    selected = select_calibration(cells, features)
    selected_ids = {
        f"{cell['model']}|{cell['domain']}|"
        f"{cell['simulation'].get('task_id')}|{cell['simulation'].get('trial')}"
        for cell in selected
    }
    valid_turns = {
        record["trajectory_id"]: set(record["valid_turn_indices"])
        for record in feature_records
    }
    if args.mode == "validate":
        calibration_report(args.output, valid_turns, selected_ids)
        return
    if args.env_file:
        load_dotenv(args.env_file)
    provider = args.provider
    if provider == "auto":
        provider = (
            "openrouter"
            if usable_secret(os.environ.get("OPENROUTER_API_KEY"))
            else "openai"
        )
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = OPENROUTER_BASE
        request_model = OPENROUTER_MODEL
        extra_body = {"reasoning": {"max_tokens": 512, "exclude": True}}
        missing_message = "OPENROUTER_API_KEY is required"
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_API_BASE")
        request_model = MODEL.split("/", 1)[1]
        extra_body = {}
        missing_message = "OPENAI_API_KEY and OPENAI_API_BASE are required"
    if not usable_secret(api_key) or not base_url:
        raise SystemExit(missing_message)
    if args.mode == "calibration":
        payloads = [build_judge_payload(cell, catalog) for cell in selected]
        repeat_ids = {
            payload["_trajectory_id"]
            for payload in random.Random(20260730).sample(payloads, 18)
        }
        jobs = [(payload, 0) for payload in payloads] + [
            (payload, 1) for payload in payloads if payload["_trajectory_id"] in repeat_ids
        ]
    elif args.mode == "full":
        payloads = [build_judge_payload(cell, catalog) for cell in cells]
        if len(payloads) != 1200 or len({p["_trajectory_id"] for p in payloads}) != 1200:
            raise SystemExit("full review requires exactly 1200 unique trajectories")
        jobs = [(payload, 0) for payload in payloads]
    client = OpenAIProxy(
        base_url,
        api_key,
        request_model,
        args.timeout,
        args.transport_retries,
        extra_body,
    )
    if args.mode == "probe":
        probe_payloads = [build_judge_payload(cell, catalog) for cell in selected]
        completed_probe_ids = {
            record["trajectory_id"]
            for record in read_jsonl(args.output)
            if record.get("prompt_version") == PROMPT_VERSION
            and record.get("calibration_run", 0) == 0
        }
        probe_payloads = [
            payload
            for payload in probe_payloads
            if payload["_trajectory_id"] not in completed_probe_ids
        ]
        if not probe_payloads:
            print("all calibration base trajectories are already complete")
            return
        payload = min(
            probe_payloads,
            key=lambda item: len(
                json.dumps(
                    {key: value for key, value in item.items() if not key.startswith("_")},
                    ensure_ascii=False,
                )
            ),
        )
        run_reviews(client, [(payload, 0)], args.output, 1)
        print("probe succeeded and was saved for calibration resume")
        return
    run_reviews(client, jobs, args.output, args.concurrency)
    if args.mode == "calibration":
        calibration_report(args.output, valid_turns, selected_ids)


if __name__ == "__main__":
    main()
