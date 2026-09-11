#!/usr/bin/env python3
"""Qwen3.8 deletion-only Banking simplification: prepare, judge, and export."""

import argparse
import http.client
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from build_banking_expert import (
    BankingRuntime,
    DEFAULT_ALLOWED_DOCUMENTS, DEFAULT_CONTRACTS, DEFAULT_DOCUMENTS_DIR,
    DEFAULT_TASKS, DEFAULT_TOKENIZER, _load_runtime_factory,
)
from simplify_banking_expert import (
    ROOT, SOURCE, agent_view, apply_deletion_plan, complete_sft_rows,
    judge_visible_messages, light_messages, read_rows,
)

OUTPUT = ROOT / "output/experiments/tau2-banking-simplify-qwen38-v1"
MODEL_PATH = "/mnt/afs/models/Qwen3.8-27B"
MODEL_NAME = "Qwen3.8-27B-banking-simplifier"
CONTEXT = 262144
COMPLETION = 32768
EMPTY_PLAN = {"remove_call_ids": [], "remove_assistant_text_indices": []}


class JudgeOutputError(ValueError):
    """Inference finished without a complete structured judgment."""

PROPOSE_SYSTEM = """Simplify a saved Banking Agent trajectory for non-thinking SFT.
Your only editing operations are deleting original Agent tool calls together
with their results, and deleting whole original Assistant content fields.
Return their original call IDs and message_index values. Never rewrite, add,
reorder, or merge anything. User dialogue/actions and the final Agent answer
cannot be edited. Retained tool batches remain single simultaneous decisions.

Find a short coherent successful route using the original evidence. Remove
repeated or irrelevant searches, unused observations, abandoned mistakes and
their unnecessary narration when they can be cleanly removed. Optimize the
entire trajectory, not just identical queries or adjacent duplicates. A later
search may replace several earlier ones ONLY if its query was already
justified before those earlier results. Do not bootstrap a query/tool/argument
from information found solely in a deleted or future result. Within one tool
batch, no call may rely on another call's result from that batch.

Keep the evidence needed for EVERY retained assertion, tool selection and
argument. Keep required identity verification, user confirmation, tool
discovery/unlocking, user-tool handoff, and effective business operations.
Failure alone does not make a call wrong: distinguish a reasonable lookup
failure with necessary recovery from a plainly nonexistent tool, malformed
argument, or unnecessary failed attempt. A deleted failed attempt must not
leave an apology, user reply, or recovery decision referring to something
that no longer happened. You cannot edit User replies to hide this problem.

The original final success label does not prove every Agent action was good.
Do not invent a cleaner final answer or hide incorrect retained behavior.
When no safe deletion exists, return empty deletion lists and explain why.
Use only the supplied policy, schemas, and Agent-visible observations. No
hidden user-model instructions, user tool observations, reference answers or
expert reasoning are available. Future messages cannot justify earlier actions.
Message indices refer to the original trace and may have gaps for hidden User
tool events. Do not infer facts from these gaps. Keep explanations concise and
cite the affected original message indices/call IDs. Return one function call.
"""

REVIEW_SYSTEM = """Independently assess a deletion-only Banking SFT candidate.
The source is supplied once, with a deletion plan. Construct the candidate by
removing each listed Agent call AND its result and each listed Assistant text
field. Empty Agent messages disappear. Everything else stays in original
order; surviving calls from a batch remain one simultaneous decision. Do not
use a deleted result to justify anything in the candidate.

Judge the candidate from the Agent's actual visible prefix at each retained
turn. Check evidence for all answers and tool arguments, necessary identity
verification/confirmation, discovery/unlocking and business operations, and
coherence of retained User replies and Agent correction/recovery text. Queries
must not contain facts known only from a removed or future result. Calls in a
batch cannot use that batch's results. A final success label is insufficient:
reject clearly wrong retained Agent training targets even if the database task
succeeded. A reasonable failing lookup and correct recovery are not inherently
wrong. The source may itself be flawed; do not assume it is a quality reference.

Return keep if the remaining trajectory is coherent, grounded and suitable for
Agent SFT; reject for a clear material problem; uncertain for genuinely missing
evidence. Do not reject harmless style or demand additional simplification.
Explain problems concisely with original message indices/call IDs so a deletion
plan can be revised. Never propose rewritten content. Return one function call.
"""

PROPOSAL_TOOL = {"type": "function", "function": {
    "name": "submit_deletion_plan", "description": "Select deletions from the original Agent trace.",
    "parameters": {"type": "object", "properties": {
        "remove_call_ids": {"type": "array", "items": {"type": "string"}},
        "remove_assistant_text_indices": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    }, "required": ["remove_call_ids", "remove_assistant_text_indices", "reason"], "additionalProperties": False},
}}
REVIEW_TOOL = {"type": "function", "function": {
    "name": "submit_simplification_review", "description": "Assess only the candidate produced by the deletion plan.",
    "parameters": {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["keep", "reject", "uncertain"]},
        "reason": {"type": "string"},
    }, "required": ["verdict", "reason"], "additionalProperties": False},
}}


def has_error(messages):
    return any(m["role"] == "tool" and (m.get("error") or str(m.get("content") or "").lstrip().startswith(("Error:", "Failed to ")))
               for m in messages)


def load_inputs(path):
    rows = []
    for row in read_rows(path):
        rows.append({"id": row["id"], "task_id": row["task_id"], "messages": light_messages(row["messages"]),
                     "source_reward": row["reward_info"]["reward"], "training_eligible": row["training_eligible"]})
    tasks = {t["id"]: t for t in json.loads(DEFAULT_TASKS.read_text())}
    contracts = {c["task_id"]: c for c in read_rows(DEFAULT_CONTRACTS)}
    allowed = json.loads(DEFAULT_ALLOWED_DOCUMENTS.read_text())["allowed_document_ids"]
    factory = _load_runtime_factory(documents_dir=DEFAULT_DOCUMENTS_DIR, allowed_document_ids=allowed)
    return rows, tasks, contracts, allowed, factory


def request_body(row, runtime, *, review_plan=None, feedback=None, attempt=0):
    reviewing = review_plan is not None
    payload = {"trajectory_id": row["id"], "agent_system_policy": runtime.system_prompt,
               "agent_tool_schemas": runtime.tools, "source_messages": judge_visible_messages(row["messages"])}
    if reviewing:
        payload["deletion_plan"] = {key: review_plan[key] for key in EMPTY_PLAN}
    if feedback:
        payload["previous_attempt"] = feedback
    tool = REVIEW_TOOL if reviewing else PROPOSAL_TOOL
    return {
        "model": MODEL_NAME, "messages": [
            {"role": "system", "content": REVIEW_SYSTEM if reviewing else PROPOSE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "max_tokens": COMPLETION, "seed": 300 + int(reviewing) + 2 * attempt,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "xhigh"},
        "tools": [tool], "tool_choice": {"type": "function", "function": {"name": tool["function"]["name"]}},
    }


def count_prompt(tokenizer, body):
    return len(tokenizer.apply_chat_template(body["messages"], tools=body["tools"], tokenize=True, return_dict=False,
                                             add_generation_prompt=True, **body["chat_template_kwargs"]))


def parse_response(response, tool):
    choice = response["choices"][0]
    if choice["finish_reason"] == "length":
        raise JudgeOutputError("Model output reached completion limit")
    calls = choice["message"].get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != tool["function"]["name"]:
        raise JudgeOutputError("Expected exactly one structured judge tool call")
    result = calls[0]["function"]["arguments"]
    try:
        result = json.loads(result) if isinstance(result, str) else result
    except json.JSONDecodeError as exc:
        raise JudgeOutputError(f"Incomplete/invalid judge JSON: {exc}") from exc
    if not isinstance(result, dict) or set(result) != set(tool["function"]["parameters"]["required"]):
        raise JudgeOutputError("Unexpected judge output fields")
    if not isinstance(result["reason"], str):
        raise JudgeOutputError("Judge reason must be text")
    if tool["function"]["name"] == REVIEW_TOOL["function"]["name"] and result["verdict"] not in {"keep", "reject", "uncertain"}:
        raise JudgeOutputError("Invalid review verdict")
    return result


def query(endpoint, body, tokenizer, calls_log, trajectory_id, phase, timeout):
    prompt_tokens = count_prompt(tokenizer, body)
    if prompt_tokens + body["max_tokens"] > CONTEXT:
        raise ValueError(f"Untruncated judge input exceeds context: {prompt_tokens}+{body['max_tokens']}")
    request = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    print(f"REQUEST id={trajectory_id} phase={phase} prompt_tokens={prompt_tokens}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise urllib.error.URLError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:2000]}") from exc
    message = result["choices"][0]["message"]
    # Keep outputs needed for debugging, not generated reasoning in training data.
    entry = {"id": trajectory_id, "phase": phase, "seconds": time.monotonic() - start,
             "max_tokens": body["max_tokens"], "seed": body["seed"],
             "prompt_tokens": prompt_tokens, "usage": result.get("usage"),
             "finish_reason": result["choices"][0]["finish_reason"],
             "content": message.get("content"), "tool_calls": message.get("tool_calls")}
    calls_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
    calls_log.flush()
    print(f"RESPONSE id={trajectory_id} phase={phase} seconds={entry['seconds']:.1f} usage={entry['usage']}", flush=True)
    return parse_response(result, body["tools"][0]), entry


def prepare(args, rows, tasks, contracts, factory, judge_tokenizer, train_tokenizer):
    records = []
    runtimes = {}
    for row in rows:
        contract = contracts[row["task_id"]]
        if row["task_id"] not in runtimes:
            runtimes[row["task_id"]] = factory(tasks[row["task_id"]], contract["retrieval_variant"])
        runtime = runtimes[row["task_id"]]
        native = agent_view(row["messages"], runtime.system_prompt)
        record = {"id": row["id"], "task_id": row["task_id"], "variant": runtime.retrieval_variant,
                  "has_error": has_error(row["messages"]), "task_form": contract["task_form"],
                  "source_tokens": len(train_tokenizer.apply_chat_template(native, tools=runtime.tools, tokenize=True, return_dict=False)),
                  "judge_prompt_tokens": count_prompt(judge_tokenizer, request_body(row, runtime))}
        try:
            apply_deletion_plan(row["messages"], EMPTY_PLAN)
            if row["source_reward"] != 1 or row["training_eligible"] is not True:
                raise ValueError("Source is not an eligible successful training trajectory")
        except ValueError as exc:
            record["preflight_error"] = str(exc)
        records.append(record)
        if len(records) % 100 == 0:
            print(f"PREPARED {len(records)}/{len(rows)}", flush=True)
    groups = defaultdict(list)
    for record in records:
        groups[(record["variant"], record["has_error"])].append(record["id"])
    rng = random.Random(300)
    pilot = set()
    for key in sorted(groups):
        pilot.update(rng.sample(sorted(groups[key]), 16))
    reviewed = ROOT / "output/experiments/tau2-banking-expert-sft/data/reviewed_lookup_searches.json"
    pilot.update(json.loads(reviewed.read_text()))
    pilot.update(r["id"] for r in sorted(records, key=lambda r: r["judge_prompt_tokens"], reverse=True)[:4])
    inventory = {"input": str(args.input), "model": MODEL_PATH, "tokenizer": str(args.tokenizer),
                 "context_length": CONTEXT, "max_completion_tokens": COMPLETION,
                 "pilot_ids": sorted(pilot), "records": records}
    (args.output_dir / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"prepared": len(records), "tasks": len({r['task_id'] for r in records}), "pilot": len(pilot),
                      "max_judge_prompt": max(r["judge_prompt_tokens"] for r in records),
                      "preflight_errors": sum("preflight_error" in r for r in records)}), flush=True)


def load_decisions(output_dir):
    decisions = {}
    for path in sorted((output_dir / "decisions").glob("*.jsonl")):
        for row in read_rows(path):
            # No complete last proposal means processing was incomplete, not
            # that a candidate was semantically rejected (also covers pilot logs).
            if row["status"] == "rejected" and row.get("attempts") and "plan" not in row["attempts"][-1]:
                row["status"] = "error"
            if row["finished_at"] >= decisions.get(row["id"], {}).get("finished_at", 0):
                decisions[row["id"]] = row
    return decisions


def judge_one(args, row, runtime, task, constructor, judge_tokenizer, train_tokenizer, log):
    from replay_simplified_banking import replay_candidate

    started = time.monotonic()
    attempts = []
    feedback = None
    result = {"id": row["id"], "task_id": row["task_id"], "variant": runtime.retrieval_variant, "attempts": attempts}
    for attempt in range(2):
        for key in ("plan", "replay", "review", "candidate_tokens"):
            result.pop(key, None)
        result["status"] = "rejected"
        detail = {"attempt": attempt}
        attempts.append(detail)
        try:
            body = request_body(row, runtime, feedback=feedback, attempt=attempt)
            plan, usage = query(args.endpoint, body, judge_tokenizer, log, row["id"], f"propose_{attempt}", args.timeout)
            detail["plan"] = plan
            candidate = apply_deletion_plan(row["messages"], plan)
            unknown = [c["name"] for m in candidate if m["role"] == "assistant"
                       for c in m.get("tool_calls") or [] if c["name"] not in runtime.agent_tool_names]
            if unknown:
                raise ValueError(f"Unknown retained Agent tools: {unknown}")
            native = agent_view(candidate, runtime.system_prompt)
            detail["candidate_tokens"] = len(train_tokenizer.apply_chat_template(native, tools=runtime.tools, tokenize=True, return_dict=False))
            detail["replay"] = replay_candidate(task, constructor, row["messages"], candidate)
            review, usage = query(args.endpoint, request_body(row, runtime, review_plan=plan, attempt=attempt),
                                  judge_tokenizer, log, row["id"], f"review_{attempt}", args.timeout)
            detail["review"] = review
            if review["verdict"] == "keep":
                result.update(plan=plan, replay=detail["replay"], review=review, candidate_tokens=detail["candidate_tokens"])
                if detail["candidate_tokens"] <= args.max_tokens:
                    result["status"] = "accepted"
                    break
                result["status"] = "over_cap"
                detail["failure"] = f"Complete candidate still has {detail['candidate_tokens']} tokens; SFT cap is {args.max_tokens}. Only remove more if evidence and coherence remain intact."
            else:
                detail["failure"] = review["reason"]
        except JudgeOutputError as exc:
            detail["failure"] = str(exc)
            result["status"] = "error"
        except (ValueError, KeyError, TypeError) as exc:
            detail["failure"] = str(exc)
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            detail["failure"] = f"Service error: {exc}"
            result["status"] = "error"
        feedback = {"deletion_plan": detail.get("plan"), "problem": detail["failure"]}
    result.setdefault("status", "rejected")
    result["reason"] = attempts[-1].get("failure", "Independent review and replay passed")
    result["seconds"] = time.monotonic() - started
    result["finished_at"] = time.time()
    return result


def check_model_review(args, tokenizer, log):
    """Small semantic regression cases; these are not production trajectories."""
    schemas = tuple({"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {parameter: {"type": "string"}}, "required": [parameter]},
    }} for name, parameter, description in (
        ("get_reference", "product", "Find the document reference for a public banking product."),
        ("get_fee", "reference", "Read the monthly fee using a document reference from get_reference.")))
    runtime = BankingRuntime(
        system_prompt="Public product pricing: no identity verification is needed. Use get_reference to find the document reference, then get_fee to read the monthly fee. Do not invent document references or quote a fee without its tool evidence.",
        tools=schemas, agent_tool_names=frozenset({"get_reference", "get_fee"}), retrieval_variant="test")
    messages = [
        {"role": "user", "content": "What is the monthly fee for the R17 account product?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "reference-call", "name": "get_reference", "arguments": {"product": "R17"}, "requestor": "assistant"}]},
        {"role": "tool", "id": "reference-call", "requestor": "assistant", "content": '{"reference":"R17_DOC_XZ91"}'},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "fee-call", "name": "get_fee", "arguments": {"reference": "R17_DOC_XZ91"}, "requestor": "assistant"}]},
        {"role": "tool", "id": "fee-call", "requestor": "assistant", "content": '{"monthly_fee_usd":37}'},
        {"role": "assistant", "content": "The monthly fee for the R17 account is $37."},
    ]
    results = []
    for name, removed, should_keep in [("grounded", [], True), ("missing_answer_evidence", ["fee-call"], False),
                                       ("missing_argument_source", ["reference-call"], False)]:
        row = {"id": f"control/{name}", "messages": messages}
        plan = {"remove_call_ids": removed, "remove_assistant_text_indices": []}
        review, usage = query(args.endpoint, request_body(row, runtime, review_plan=plan), tokenizer, log, row["id"], "control", args.timeout)
        results.append({"name": name, "review": review, "passed": (review["verdict"] == "keep") == should_keep,
                        "seconds": usage["seconds"]})
    directory = args.work_ids.parent if args.work_ids else args.output_dir
    (directory / "model_checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("MODEL_REVIEW_CHECKS " + json.dumps(results, ensure_ascii=False), flush=True)


def run_judge(args, rows, tasks, contracts, allowed, factory, judge_tokenizer, train_tokenizer):
    from loguru import logger
    from tau2.data_model.tasks import Task
    from replay_simplified_banking import environment_constructor

    logger.remove()
    logger.add(__import__("sys").stderr, level="ERROR")
    inventory = json.loads((args.output_dir / "inventory.json").read_text())
    # The launcher assigns pending IDs before any worker writes. Do not reread
    # other workers' live append files to reapply the same resume selection.
    if args.work_ids:
        assigned = set(json.loads(args.work_ids.read_text())[args.shard_index])
        records = [r for r in inventory["records"] if r["id"] in assigned]
    else:
        if args.num_shards != 1:
            raise ValueError("Multi-worker runs require prepared --work-ids")
        completed = load_decisions(args.output_dir)
        selected = set(inventory["pilot_ids"]) if args.stage == "pilot" else {r["id"] for r in rows}
        records = [r for r in inventory["records"] if r["id"] in selected and
                   (r["id"] not in completed or r["id"] in args.retry_id or (args.retry_errors and completed[r["id"]]["status"] == "error"))]
    records.sort(key=lambda r: r["judge_prompt_tokens"], reverse=True)
    if args.stage == "pilot" and records:
        # Exercise capacity first, then representative cases instead of only
        # extreme contexts at the beginning of the pilot.
        records = records[:1] + random.Random(300).sample(records[1:], len(records) - 1)
    by_id = {r["id"]: r for r in rows}
    (args.output_dir / "decisions").mkdir(exist_ok=True)
    (args.output_dir / "requests").mkdir(exist_ok=True)
    name = f"{args.stage}_{args.shard_index}"
    start = time.monotonic()
    counts = Counter()
    consecutive_service_errors = 0
    with (args.output_dir / "decisions" / f"{name}.jsonl").open("a") as output, (args.output_dir / "requests" / f"{name}.jsonl").open("a") as log:
        if args.stage == "pilot" and args.shard_index == 0:
            check_model_review(args, judge_tokenizer, log)
        for number, record in enumerate(records, 1):
            row = by_id[record["id"]]
            runtime = factory(tasks[row["task_id"]], contracts[row["task_id"]]["retrieval_variant"])
            task = Task.model_validate(tasks[row["task_id"]])
            constructor = environment_constructor(DEFAULT_DOCUMENTS_DIR, tuple(allowed), runtime.retrieval_variant, task)
            print(f"START {name} {number}/{len(records)} id={row['id']} prompt_tokens={record['judge_prompt_tokens']}", flush=True)
            if record.get("preflight_error"):
                result = {"id": row["id"], "task_id": row["task_id"], "variant": runtime.retrieval_variant,
                          "status": "rejected", "reason": record["preflight_error"], "finished_at": time.time(), "attempts": []}
            else:
                result = judge_one(args, row, runtime, task, constructor, judge_tokenizer, train_tokenizer, log)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            counts[result["status"]] += 1
            elapsed = time.monotonic() - start
            print(json.dumps({"worker": name, "done": number, "total": len(records), "counts": dict(counts),
                              "last_id": row["id"], "last_status": result["status"],
                              "elapsed_seconds": elapsed, "eta_seconds": elapsed / number * (len(records) - number)}), flush=True)
            consecutive_service_errors = consecutive_service_errors + 1 if result["status"] == "error" else 0
            if consecutive_service_errors >= 3:
                raise RuntimeError("Three consecutive inference failures; stop this worker and inspect model outputs/server")


def assign_work(args):
    inventory = json.loads((args.output_dir / "inventory.json").read_text())
    completed = load_decisions(args.output_dir)
    selected = set(inventory["pilot_ids"]) if args.scope == "pilot" else {r["id"] for r in inventory["records"]}
    records = [r for r in inventory["records"] if r["id"] in selected and
               (r["id"] not in completed or r["id"] in args.retry_id or (args.retry_errors and completed[r["id"]]["status"] == "error"))]
    shards = [[] for _ in range(args.num_shards)]
    loads = [0] * args.num_shards
    for record in sorted(records, key=lambda r: r["judge_prompt_tokens"], reverse=True):
        index = min(range(args.num_shards), key=loads.__getitem__)
        shards[index].append(record["id"])
        loads[index] += record["judge_prompt_tokens"]
    args.work_ids.write_text(json.dumps(shards, indent=2) + "\n")
    print(json.dumps({"shard_counts": list(map(len, shards)), "shard_prompt_tokens": loads}), flush=True)


def trajectory_metrics(messages):
    calls = [c for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    return {"agent_calls": len(calls), "searches": sum(c["name"] == "KB_search" for c in calls),
            "text_turns": sum(m["role"] == "assistant" and bool((m.get("content") or "").strip()) for m in messages),
            "multi_call_turns": sum(m["role"] == "assistant" and len(m.get("tool_calls") or []) > 1 for m in messages),
            "agent_tool_errors": sum(m["role"] == "tool" and m.get("requestor") == "assistant" and has_error([m]) for m in messages)}


def length_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "total": sum(values), "mean": sum(values) / len(values),
            "p50": values[(len(values) - 1) // 2], "p95": values[int((len(values) - 1) * .95)], "max": values[-1]}


def finalize(args, rows, tasks, contracts, factory, tokenizer):
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    inventory = json.loads((args.output_dir / "inventory.json").read_text())
    inventory_by_id = {r["id"]: r for r in inventory["records"]}
    selected = set(inventory["pilot_ids"]) if args.scope == "pilot" else set(inventory_by_id)
    decisions = load_decisions(args.output_dir)
    destination = args.output_dir / "data" / args.scope
    destination.mkdir(parents=True, exist_ok=True)
    old_indices = defaultdict(set)
    old_counts = Counter()
    old_path = ROOT / "output/experiments/tau2-banking-expert-sft/data/banking_expert_sft.jsonl"
    for row in read_rows(old_path):
        metadata = row["metadata"]
        old_indices[metadata["source_trajectory_id"]].add(metadata["target_index"])
        old_counts[metadata["target_type"]] += 1
    reports = []
    counts = Counter()
    for_stats = []
    with (destination / "banking_simplified_sft.jsonl").open("w") as sft, (destination / "simplified_trajectories.jsonl").open("w") as traces:
        for source in rows:
            if source["id"] not in selected:
                continue
            record = inventory_by_id[source["id"]]
            decision = decisions.get(source["id"], {"status": "not_processed", "reason": "No completed judge result"})
            report = {**record, "status": decision["status"], "reason": decision["reason"],
                      "candidate_tokens": decision.get("candidate_tokens"), "source_metrics": trajectory_metrics(source["messages"])}
            runtime = factory(tasks[source["task_id"]], contracts[source["task_id"]]["retrieval_variant"])
            before = agent_view(source["messages"], runtime.system_prompt)
            original_last = sum(max(1, len(m.get("tool_calls") or [])) for m in before if m["role"] == "assistant") - 1
            report["original_sft_has_final_target"] = original_last in old_indices[source["id"]]
            if decision["status"] == "accepted":
                candidate = apply_deletion_plan(source["messages"], decision["plan"])
                metadata = {"dataset_version": "banking-qwen38-deletion-only-v1", "source_trajectory_id": source["id"],
                            "task_id": source["task_id"], "domain": "banking_knowledge", "retrieval_variant": runtime.retrieval_variant}
                try:
                    output_rows = complete_sft_rows(candidate, runtime, generator, metadata, args.max_tokens)
                except ValueError as exc:
                    report.update(status="export_error", reason=str(exc))
                else:
                    report.update(changed=any(decision["plan"][key] for key in EMPTY_PLAN),
                                  candidate_metrics=trajectory_metrics(candidate), output_rows=len(output_rows))
                    for output_row in output_rows:
                        sft.write(json.dumps(output_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts[f"sft_{runtime.retrieval_variant}_{output_row['metadata']['target_type']}"] += 1
                    scores = decision["replay"]["required_scores"]
                    traces.write(json.dumps({"id": source["id"], "task_id": source["task_id"], "retrieval_variant": runtime.retrieval_variant,
                                             "messages": candidate, "reward_info": {"reward": 1.0, "reward_basis": list(scores), "reward_breakdown": scores}},
                                            ensure_ascii=False) + "\n")
                    counts["sft_rows"] += len(output_rows)
                    counts["changed_accepted"] += report["changed"]
                    counts["recovered_over_cap"] += record["source_tokens"] > args.max_tokens
                    counts["recovered_final_targets"] += not report["original_sft_has_final_target"]
                    for_stats.append(report)
            counts[report["status"]] += 1
            reports.append(report)
    groups = {}
    for variant in sorted({r["variant"] for r in reports}):
        for error in (False, True):
            group = [r for r in reports if r["variant"] == variant and r["has_error"] == error]
            kept = [r for r in group if r["status"] == "accepted"]
            groups[f"{variant}/{'with_error' if error else 'without_error'}"] = {
                "input": len(group), "accepted": len(kept), "accepted_tasks": len({r["task_id"] for r in kept}),
                "source_tokens_all": length_summary([r["source_tokens"] for r in group]),
                "source_tokens_same_accepted_ids": length_summary([r["source_tokens"] for r in kept]),
                "candidate_tokens": length_summary([r["candidate_tokens"] for r in kept]),
                "source_metrics_same_accepted_ids": dict(sum((Counter(r["source_metrics"]) for r in kept), Counter())),
                "candidate_metrics": dict(sum((Counter(r["candidate_metrics"]) for r in kept), Counter())),
            }
    requests = [r for path in sorted((args.output_dir / "requests").glob("*.jsonl")) for r in read_rows(path) if r["id"] in selected]
    stats = {"scope": args.scope, "source_trajectories": len(reports), "source_tasks": len({r["task_id"] for r in reports}),
             "accepted_tasks": len({r["task_id"] for r in for_stats}), "counts": dict(counts), "groups": groups,
             "source_tokens_all": length_summary([r["source_tokens"] for r in reports]),
             "source_tokens_same_accepted_ids": length_summary([r["source_tokens"] for r in for_stats]),
             "candidate_tokens": length_summary([r["candidate_tokens"] for r in for_stats]),
             "old_sft_rows_all": dict(old_counts), "old_sft_final_targets_in_scope": sum(r["original_sft_has_final_target"] for r in reports),
             "model_requests": len(requests), "model_request_seconds": sum(r["seconds"] for r in requests),
             "model_usage": dict(sum((Counter({k: v for k, v in (r.get("usage") or {}).items() if isinstance(v, (int, float))}) for r in requests), Counter()))}
    (destination / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reports))
    (destination / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if counts["not_processed"] or counts["error"] or counts["export_error"]:
        raise SystemExit("Results exported, but unresolved processing/export failures remain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "assign", "pilot", "full", "finalize"], required=True)
    parser.add_argument("--scope", choices=["pilot", "full"], default="full")
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--endpoint", default="http://127.0.0.1:33200/v1")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--work-ids", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--retry-id", action="append", default=[], help="Reprocess a specific source after a pipeline correction; repeat for multiple IDs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "assign":
        assign_work(args)
        return
    from transformers import AutoTokenizer

    rows, tasks, contracts, allowed, factory = load_inputs(args.input)
    train_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.stage == "finalize":
        finalize(args, rows, tasks, contracts, factory, train_tokenizer)
        return
    judge_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    if args.stage == "prepare":
        prepare(args, rows, tasks, contracts, factory, judge_tokenizer, train_tokenizer)
    else:
        run_judge(args, rows, tasks, contracts, allowed, factory, judge_tokenizer, train_tokenizer)


if __name__ == "__main__":
    main()
