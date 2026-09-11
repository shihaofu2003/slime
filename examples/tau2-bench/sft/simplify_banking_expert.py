#!/usr/bin/env python3
"""Build a conservative, complete-trajectory Banking SFT simplification.

Only remove silent KB searches whose document bodies were already returned
in the current uninterrupted Agent tool-use segment. Keep native tool batches,
all other calls, and verbatim dialogue/results. Exclude trajectories containing
tool errors instead of leaving recovery prose with its cause removed.
"""

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from build_banking_expert import _clean_user_content, _native_call


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "output/experiments/tau2-banking-expert-sft"
SOURCE = ROOT / "output/experiments/tau2-banking-independent-synthetic-v1/final-minimal/successful_trajectories.jsonl"
DOCUMENT_HEADER = re.compile(r"^\d+\. [^\n]+\n   ID: ([^\n]+)\n   Score: [^\n]+\n   Content: ", re.M)


def light_messages(messages):
    fields = {"role", "content", "tool_calls", "id", "tool_call_id", "requestor", "error"}
    return [{k: copy.deepcopy(v) for k, v in m.items() if k in fields} for m in messages]


def apply_deletion_plan(messages, plan):
    """Delete only original Agent calls/results and whole Agent text fields."""
    removed_calls = plan["remove_call_ids"]
    removed_text = plan["remove_assistant_text_indices"]
    if not isinstance(removed_calls, list) or any(not isinstance(c, str) for c in removed_calls):
        raise ValueError("remove_call_ids must be a list of strings")
    if not isinstance(removed_text, list) or any(type(i) is not int for i in removed_text):
        raise ValueError("remove_assistant_text_indices must be a list of integers")
    removed_calls, removed_text = set(removed_calls), set(removed_text)
    agent_calls = {c["id"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    text_indices = {i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("content")}
    if not removed_calls <= agent_calls or not removed_text <= text_indices:
        raise ValueError("Deletion refers to missing or non-Agent calls/text")
    last_agent = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    if not messages[last_agent].get("tool_calls") and last_agent in removed_text:
        raise ValueError("Cannot delete the original final Agent answer")
    output = []
    index = 0
    while index < len(messages):
        original = messages[index]
        if original["role"] == "tool":
            raise ValueError(f"Unpaired tool result at message {index}")
        message = light_messages([original])[0]
        calls = message.get("tool_calls") or []
        responses = messages[index + 1:index + 1 + len(calls)]
        if len(responses) != len(calls) or any(
            r["role"] != "tool" or (r.get("id") or r.get("tool_call_id")) != c["id"]
            for c, r in zip(calls, responses)
        ):
            raise ValueError(f"Unpaired source tool calls/results at message {index}")
        if index in removed_text:
            message["content"] = None
        kept = [(c, r) for c, r in zip(calls, responses) if c["id"] not in removed_calls]
        if calls:
            message["tool_calls"] = [c for c, _ in kept] or None
        if message["role"] != "assistant" or kept or (message.get("content") or "").strip():
            output.append(message)
            output.extend(light_messages([r for _, r in kept]))
        index += len(calls) + 1
    if not any(m["role"] == "assistant" for m in output):
        raise ValueError("Deletion removed every Agent target")
    return output


def judge_visible_messages(messages):
    """Original indices, but only the observations actually routed to Agent."""
    output = []
    user_call_ids = {c["id"] for m in messages if m["role"] == "user" for c in m.get("tool_calls") or []}
    for index, message in enumerate(light_messages(messages)):
        role = message["role"]
        if role == "system" or (role == "user" and message.get("tool_calls")):
            continue
        if role == "tool" and (message.get("requestor") == "user" or (message.get("id") or message.get("tool_call_id")) in user_call_ids):
            continue
        if role == "user":
            message["content"] = _clean_user_content(message.get("content"))
            if not message["content"]:
                continue
        output.append({"message_index": index, **message})
    return output


def complete_sft_rows(messages, runtime, generator, metadata, max_tokens=16384):
    """Validate the complete conversation before producing any prefix targets."""
    native = agent_view(messages, runtime.system_prompt)
    tokenizer = generator.tokenizer
    if len(tokenizer.apply_chat_template(native, tools=runtime.tools, tokenize=True, return_dict=False)) > max_tokens:
        raise ValueError("Complete trajectory exceeds SFT token cap")
    rows = []
    for index, message in enumerate(native):
        if message["role"] != "assistant":
            continue
        prefix = copy.deepcopy(native[:index + 1])
        prefix[-1]["step_loss_mask"] = 1
        ids, mask = generator.get_loss_mask(prefix, tools=runtime.tools)
        if len(ids) > max_tokens or len(ids) != len(mask):
            raise ValueError("Target length/mask mismatch")
        text = tokenizer.apply_chat_template(prefix, tools=runtime.tools, tokenize=False, return_dict=False)
        expected = text.rsplit("<|im_start|>assistant\n", 1)[1]
        supervised = [token for token, selected in zip(ids, mask) if selected]
        if tokenizer.eos_token_id not in supervised or tokenizer.decode(supervised) != expected:
            raise ValueError("Target mask does not match the complete Assistant message/EOS")
        if "<think>" in text or "</think>" in text:
            raise ValueError("Thinking markup in non-thinking SFT")
        rows.append({"messages": prefix, "tools": runtime.tools, "metadata": {
            **metadata, "target_index": len(rows),
            "target_type": "tool_call" if message.get("tool_calls") else "text",
            "total_tokens": len(ids), "max_total_tokens": max_tokens,
        }})
    return rows


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def document_bodies(content):
    content = re.sub(r"\n\n\[Timing: [^\n]+\]\s*$", "", content)
    matches = list(DOCUMENT_HEADER.finditer(content))
    if not matches or content[:matches[0].start()].strip():
        return None
    return {
        match[1]: content[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(content)].strip()
        for i, match in enumerate(matches)
    }


def simplify_messages(messages):
    result = []
    known_documents = {}
    removed = []
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") or []
        if not calls:
            result.append(copy.deepcopy(message))
            if message["role"] in {"assistant", "user"} and (message.get("content") or "").strip():
                known_documents.clear()
            index += 1
            continue
        responses = messages[index + 1:index + 1 + len(calls)]
        if len(responses) != len(calls) or any(
            response["role"] != "tool" or (response.get("id") or response.get("tool_call_id")) != call["id"]
            for call, response in zip(calls, responses)
        ):
            raise ValueError("Unpaired source tool calls/results")
        silent_agent = message["role"] == "assistant" and not (message.get("content") or "").strip()
        if not silent_agent:
            known_documents.clear()
        kept_calls, kept_responses = [], []
        # Calls in a batch were chosen together, before any of its results.
        # A surviving batch stays one Assistant message.
        for call, response in zip(calls, responses):
            documents = document_bodies(response.get("content") or "") if call["name"] == "KB_search" else None
            if silent_agent and documents and all(known_documents.get(key) == body for key, body in documents.items()):
                removed.append(call["id"])
                continue
            kept_calls.append(copy.deepcopy(call))
            kept_responses.append(copy.deepcopy(response))
            if silent_agent and documents:
                known_documents.update(documents)
        if kept_calls:
            kept_message = copy.deepcopy(message)
            kept_message["tool_calls"] = kept_calls
            result.append(kept_message)
            result.extend(kept_responses)
        index += 1 + len(calls)
    return result, removed


def reviewed_lookup(messages, keep_ids):
    """Apply a manually reviewed search selection to a lookup-only dialogue.

    Review must establish that the retained query is meaningful from the
    opening request and its unedited result supports the complete final answer.
    Intermediate search narration is removed together with the abandoned tries.
    """
    final_index = max(i for i, m in enumerate(messages) if m["role"] == "assistant" and not m.get("tool_calls"))
    output, removed, found = [], [], set()
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") or []
        if not calls:
            if message["role"] != "assistant" or index == final_index:
                output.append(copy.deepcopy(message))
            index += 1
            continue
        if message["role"] != "assistant" or any(c["name"] != "KB_search" for c in calls):
            raise ValueError("Reviewed lookup selection contains non-search actions")
        responses = messages[index + 1:index + len(calls) + 1]
        if len(responses) != len(calls) or any(r["role"] != "tool" or r.get("id") != c["id"] for c, r in zip(calls, responses)):
            raise ValueError("Reviewed lookup has unpaired tool results")
        kept = [(c, r) for c, r in zip(calls, responses) if c["id"] in keep_ids]
        removed.extend(c["id"] for c in calls if c["id"] not in keep_ids)
        if kept:
            kept_message = copy.deepcopy(message)
            kept_message["tool_calls"] = [copy.deepcopy(c) for c, _ in kept]
            kept_message["content"] = None
            output.append(kept_message)
            output.extend(copy.deepcopy(r) for _, r in kept)
            found.update(c["id"] for c, _ in kept)
        index += len(calls) + 1
    if found != set(keep_ids):
        raise ValueError("Reviewed selection references missing search calls")
    return output, removed


def agent_view(messages, system):
    native = [{"role": "system", "content": system, "step_loss_mask": 0}]
    for message in messages:
        role = message["role"]
        calls = message.get("tool_calls") or []
        if role == "system" or (role == "tool" and message.get("requestor") == "user"):
            continue
        if role == "user":
            if calls:
                continue  # User tool-call content goes to the environment, not Agent.
            content = _clean_user_content(message.get("content"))
            if content:
                native.append({"role": "user", "content": content, "step_loss_mask": 0})
        elif role == "assistant":
            content = (message.get("content") or "").strip()
            if calls:
                native.append({
                    "role": "assistant", "content": content if content.strip() else None,
                    "tool_calls": [_native_call(output_id=c["id"], name=c["name"], arguments=c["arguments"]) for c in calls],
                    "step_loss_mask": 0,
                })
            elif content.strip():
                native.append({"role": "assistant", "content": content, "step_loss_mask": 0})
        elif role == "tool":
            native.append({"role": "tool", "content": message.get("content") or "", "tool_call_id": message.get("id") or message.get("tool_call_id"), "step_loss_mask": 0})
    last = max(i for i, m in enumerate(native) if m["role"] == "assistant")
    return native[:last + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--original-sft", type=Path, default=EXPERIMENT / "data/banking_expert_sft.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT.parent / "models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reviewed-searches", type=Path, help="Only build the manually reviewed lookup trajectories listed in this JSON mapping")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    reviewed = json.loads(args.reviewed_searches.read_text()) if args.reviewed_searches else None
    runtimes, retained_indices = {}, {}
    for row in read_rows(args.original_sft):
        metadata = row["metadata"]
        key = metadata["source_trajectory_id"]
        runtimes.setdefault(key, (row["messages"][0]["content"], row["tools"], metadata["retrieval_variant"]))
        retained_indices.setdefault(key, set()).add(metadata["target_index"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    comparisons = []
    with (args.output_dir / "banking_simplified_sft.jsonl").open("w") as output, (args.output_dir / "simplified_trajectories.jsonl").open("w") as traces:
        for ordinal, trajectory in enumerate(read_rows(args.input)):
            if args.limit is not None and ordinal >= args.limit:
                break
            stats["source_trajectories"] += 1
            source_id = trajectory["id"]
            if reviewed is not None and source_id not in reviewed:
                continue
            if source_id not in runtimes:
                stats["excluded_missing_original_runtime"] += 1
                continue
            messages = trajectory["messages"]
            if any(m["role"] == "tool" and (m.get("error") or str(m.get("content") or "").lstrip().startswith(("Error:", "Failed to "))) for m in messages):
                stats["excluded_source_tool_errors"] += 1
                continue
            system, tools, variant = runtimes[source_id]
            simplified, removed = reviewed_lookup(messages, reviewed[source_id]) if reviewed is not None else simplify_messages(messages)
            before = agent_view(messages, system)
            after = agent_view(simplified, system)
            before_tokens = len(tokenizer.apply_chat_template(before, tools=tools, tokenize=True, return_dict=False))
            after_ids = tokenizer.apply_chat_template(after, tools=tools, tokenize=True, return_dict=False)
            after_tokens = len(after_ids)
            stats["eligible_trajectories"] += 1
            stats["tokens_before"] += before_tokens
            stats["tokens_after"] += after_tokens
            stats["removed_search_calls"] += len(removed)
            stats["changed_trajectories"] += bool(removed)
            stats["within_cap_before"] += before_tokens <= args.max_tokens
            if after_tokens > args.max_tokens:
                stats["excluded_complete_trajectory_over_cap"] += 1
                continue
            stats["accepted_trajectories"] += 1
            stats[f"accepted_{variant}"] += 1
            stats["accepted_changed_trajectories"] += bool(removed)
            stats["accepted_removed_search_calls"] += len(removed)
            stats["recovered_from_over_cap"] += before_tokens > args.max_tokens
            atomic_targets = sum(max(1, len(m.get("tool_calls") or [])) for m in before if m["role"] == "assistant")
            stats["accepted_missing_final_in_original_sft"] += atomic_targets - 1 not in retained_indices[source_id]
            target_indices = [i for i, m in enumerate(after) if m["role"] == "assistant"]
            for target_number, target_index in enumerate(target_indices):
                prefix = copy.deepcopy(after[:target_index + 1])
                prefix[-1]["step_loss_mask"] = 1
                ids, mask = generator.get_loss_mask(prefix, tools=tools)
                if ids[-2] != tokenizer.eos_token_id or not mask[-2]:
                    raise ValueError("Target EOS is not supervised")
                text = tokenizer.apply_chat_template(prefix, tools=tools, tokenize=False, return_dict=False)
                expected = text.rsplit("<|im_start|>assistant\n", 1)[1]
                if tokenizer.decode([token for token, selected in zip(ids, mask) if selected]) != expected:
                    raise ValueError("Target mask does not match the final Assistant message")
                row = {"messages": prefix, "tools": tools, "metadata": {
                    "dataset_version": "banking-reviewed-lookup-v1" if reviewed is not None else "banking-complete-redundancy-pruned-v1", "source_trajectory_id": source_id,
                    "task_id": trajectory["task_id"], "domain": "banking_knowledge", "retrieval_variant": variant,
                    "target_index": target_number, "target_type": "tool_call" if prefix[-1].get("tool_calls") else "text",
                    "total_tokens": len(ids), "max_total_tokens": args.max_tokens,
                }}
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["output_rows"] += 1
                stats[f"output_{variant}_{row['metadata']['target_type']}"] += 1
            # Keep source-role messages for offline environment replay; no raw thinking.
            light_messages = [{k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "id", "tool_call_id", "requestor", "error"}} for m in simplified]
            traces.write(json.dumps({"id": source_id, "task_id": trajectory["task_id"], "retrieval_variant": variant,
                                    "messages": light_messages, "reward_info": trajectory["reward_info"]}, ensure_ascii=False) + "\n")
            comparisons.append({"id": source_id, "task_id": trajectory["task_id"], "variant": variant,
                                "tokens_before": before_tokens, "tokens_after": after_tokens, "removed_call_ids": removed})
            if stats["accepted_trajectories"] % 50 == 0:
                print(dict(stats), flush=True)
    (args.output_dir / "stats.json").write_text(json.dumps(dict(stats), indent=2) + "\n")
    (args.output_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
