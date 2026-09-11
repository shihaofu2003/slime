"""CPU tests for the Banking expert-trajectory SFT formatter."""

from __future__ import annotations

import importlib.util
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


NUM_GPUS = 0

_FORMATTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "tau2-bench"
    / "sft"
    / "build_banking_expert.py"
)
_SPEC = importlib.util.spec_from_file_location("build_banking_expert", _FORMATTER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"could not load formatter module from {_FORMATTER_PATH}")
banking_expert = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = banking_expert
_SPEC.loader.exec_module(banking_expert)
sys.path.insert(0, str(_FORMATTER_PATH.parent))
import simplify_banking_expert as simplifier
import qwen_simplify_banking as qwen_simplifier
import replay_simplified_banking as replay
import benchmark_banking_concurrency as concurrency_benchmark


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
]


class BankingSimplificationTest(unittest.TestCase):
    def test_concurrency_sample_and_disjoint_process_shards(self):
        records = [{"id": f"{variant}-{error}-{i}", "variant": variant, "has_error": error,
                    "judge_prompt_tokens": (i + 1) * 1000}
                   for variant in ("bm25", "golden") for error in (False, True) for i in range(8)]
        inventory = {"records": records, "pilot_ids": [r["id"] for r in records]}
        sample = concurrency_benchmark.select_records(inventory)
        self.assertEqual(len(sample), 16)
        self.assertEqual(len({r["id"] for r in sample}), 16)
        self.assertEqual(min(r["judge_prompt_tokens"] for r in sample), 1000)
        self.assertEqual(max(r["judge_prompt_tokens"] for r in sample), 8000)
        inventory["pilot_ids"] = [r["id"] for r in sample]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inventory.json").write_text(json.dumps(inventory))
            for concurrency in (1, 2, 4):
                args = SimpleNamespace(output_dir=root, scope="pilot", retry_id=[], retry_errors=False,
                                       num_shards=concurrency, work_ids=root / "work.json")
                qwen_simplifier.assign_work(args)
                shards = json.loads(args.work_ids.read_text())
                assigned = [key for shard in shards for key in shard]
                self.assertEqual(len(shards), concurrency)
                self.assertEqual(len(assigned), len(set(assigned)))
                self.assertEqual(set(assigned), set(inventory["pilot_ids"]))

    def setUp(self):
        self.messages = [
            {"role": "user", "content": "What is the monthly fee?"},
            {"role": "assistant", "content": "I will check.", "raw_data": {"reasoning_content": "SECRET_THINKING"},
             "tool_calls": [{"id": key, "name": "KB_search", "arguments": {"query": key}, "requestor": "assistant"}
                            for key in ("a", "b", "c")]},
            *[{"role": "tool", "id": key, "content": f"Document {key}", "requestor": "assistant"} for key in ("a", "b", "c")],
            {"role": "assistant", "content": "The monthly fee is $10."},
            {"role": "user", "content": "PRIVATE_USER_TOOL_TEXT", "tool_calls": [
                {"id": "user-call", "name": "submit", "arguments": {}, "requestor": "user"}]},
            {"role": "tool", "id": "user-call", "content": "PRIVATE_USER_RESULT", "requestor": "user"},
            {"role": "user", "content": "Thank you.\n\n###STOP###"},
        ]
        self.runtime = SimpleNamespace(system_prompt="POLICY", tools=[], agent_tool_names={"KB_search"})

    def test_partial_deletion_keeps_native_batch_and_verbatim_results(self):
        original = copy.deepcopy(self.messages)
        candidate = simplifier.apply_deletion_plan(self.messages, {
            "remove_call_ids": ["b"], "remove_assistant_text_indices": []})
        self.assertEqual([c["id"] for c in candidate[1]["tool_calls"]], ["a", "c"])
        self.assertEqual(candidate[2:4], [self.messages[2], self.messages[4]])
        self.assertEqual(candidate[4:], self.messages[5:])
        self.assertEqual(self.messages, original)
        self.assertNotIn("raw_data", candidate[1])

    def test_removing_batch_keeps_its_undeleted_prose(self):
        candidate = simplifier.apply_deletion_plan(self.messages, {
            "remove_call_ids": ["a", "b", "c"], "remove_assistant_text_indices": []})
        self.assertEqual(candidate[1]["content"], "I will check.")
        self.assertFalse(candidate[1]["tool_calls"])
        self.assertEqual(candidate[2:], self.messages[5:])

    def test_removing_batch_and_prose_removes_empty_agent_message(self):
        candidate = simplifier.apply_deletion_plan(self.messages, {
            "remove_call_ids": ["a", "b", "c"], "remove_assistant_text_indices": [1]})
        self.assertEqual(candidate, [self.messages[0], *self.messages[5:]])

    def test_removing_only_mixed_prose_preserves_calls(self):
        candidate = simplifier.apply_deletion_plan(self.messages, {
            "remove_call_ids": [], "remove_assistant_text_indices": [1]})
        self.assertIsNone(candidate[1]["content"])
        self.assertEqual(candidate[1]["tool_calls"], self.messages[1]["tool_calls"])

    def test_user_events_final_answer_and_missing_ids_cannot_be_deleted(self):
        for calls, texts in [(["user-call"], []), (["absent"], []), ([], [0]), ([], [5]), ([], [999])]:
            with self.subTest(calls=calls, texts=texts), self.assertRaises(ValueError):
                simplifier.apply_deletion_plan(self.messages, {"remove_call_ids": calls, "remove_assistant_text_indices": texts})

    def test_whitespace_agent_text_is_a_valid_deletion(self):
        self.messages[1]["content"] = "\n\n"
        candidate = simplifier.apply_deletion_plan(self.messages, {
            "remove_call_ids": ["b"], "remove_assistant_text_indices": [1]})
        self.assertIsNone(candidate[1]["content"])
        self.assertEqual([c["id"] for c in candidate[1]["tool_calls"]], ["a", "c"])
        self.assertEqual(candidate[2:4], [self.messages[2], self.messages[4]])
        self.assertEqual(candidate[4:], self.messages[5:])

    def test_missing_results_are_not_silently_dropped(self):
        self.messages.pop(3)
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            simplifier.apply_deletion_plan(self.messages, qwen_simplifier.EMPTY_PLAN)

    def test_judge_sees_original_indices_without_hidden_user_events_or_reasoning(self):
        visible = simplifier.judge_visible_messages(self.messages)
        self.assertEqual([m["message_index"] for m in visible], [0, 1, 2, 3, 4, 5, 8])
        rendered = json.dumps(visible)
        for hidden in ("PRIVATE_USER_TOOL_TEXT", "PRIVATE_USER_RESULT", "SECRET_THINKING", "###STOP###"):
            self.assertNotIn(hidden, rendered)
        self.assertEqual(visible[-1]["content"], "Thank you.")

    def test_independent_review_does_not_receive_proposal_reason(self):
        plan = {**qwen_simplifier.EMPTY_PLAN, "reason": "ANCHORING_JUSTIFICATION"}
        body = qwen_simplifier.request_body({"id": "test", "messages": self.messages}, self.runtime, review_plan=plan)
        self.assertNotIn("ANCHORING_JUSTIFICATION", json.dumps(body))
        self.assertEqual(body["seed"], 301)
        payload = json.loads(body["messages"][1]["content"])
        self.assertNotIn("candidate_messages", payload)
        self.assertEqual(payload["deletion_plan"], qwen_simplifier.EMPTY_PLAN)

    def test_complete_over_cap_does_not_export_short_prefixes(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: list(range(17)))
        generator = SimpleNamespace(tokenizer=tokenizer, get_loss_mask=lambda *a, **kw: self.fail("No prefix may be exported"))
        with self.assertRaisesRegex(ValueError, "Complete trajectory"):
            simplifier.complete_sft_rows(self.messages, self.runtime, generator, {}, max_tokens=16)

    def test_transformers_dictionary_default_is_not_counted_as_two_tokens(self):
        def render(*args, return_dict=True, **kwargs):
            tokens = list(range(23))
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)} if return_dict else tokens
        tokenizer = SimpleNamespace(apply_chat_template=render)
        body = qwen_simplifier.request_body({"id": "test", "messages": self.messages}, self.runtime)
        self.assertEqual(qwen_simplifier.count_prompt(tokenizer, body), 23)
        with self.assertRaisesRegex(ValueError, "Complete trajectory"):
            simplifier.complete_sft_rows(self.messages, self.runtime, SimpleNamespace(tokenizer=tokenizer), {}, max_tokens=16)

    def test_mask_eos_and_thinking_are_checked(self):
        for decoded, ids, mask, markup in [("answer", [1, 2, 3], [0, 0, 1], False),
                                            ("wrong", [1, 2, 3], [0, 1, 1], False),
                                            ("<think>answer", [1, 2, 3], [0, 1, 1], True)]:
            tokenizer = SimpleNamespace(eos_token_id=2,
                apply_chat_template=lambda *a, tokenize=True, **kw: ids if tokenize else "<|im_start|>assistant\n" + ("<think>answer" if markup else "answer"),
                decode=lambda values: decoded)
            generator = SimpleNamespace(tokenizer=tokenizer, get_loss_mask=lambda *a, **kw: (ids, mask))
            with self.subTest(decoded=decoded), self.assertRaises(ValueError):
                simplifier.complete_sft_rows(self.messages, self.runtime, generator, {})

    def test_replay_compares_read_values_and_ignores_only_search_ranking_timing(self):
        make = lambda text: SimpleNamespace(content=text, error=False)
        a = "1. Title\n   ID: doc_a\n   Score: 9\n   Content: fee $10\n\n[Timing: 1ms]"
        b = "2. Title\n   ID: doc_a\n   Score: 1\n   Content: fee $10\n\n[Timing: 2ms]"
        self.assertTrue(replay.matching_response("KB_search", make(a), make(b)))
        self.assertFalse(replay.matching_response("KB_search", make(a), make(b.replace("$10", "$20"))))
        self.assertFalse(replay.matching_response("get_balance", make('{"balance":10}'), make('{"balance":20}')))
        self.assertFalse(replay.matching_response("unlock", make("Tool unlocked"), make("Error: not unlocked")))

    def test_semantic_rejection_is_not_overridden_by_successful_replay(self):
        args = SimpleNamespace(endpoint="unused", timeout=1, max_tokens=16384)
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3])
        plan = {**qwen_simplifier.EMPTY_PLAN, "reason": "nothing removable"}
        review = {"verdict": "reject", "reason": "Retained answer lacks earlier evidence."}
        with patch.object(qwen_simplifier, "query", side_effect=[(plan, {}), (review, {}), (plan, {}), (review, {})]), \
             patch.object(replay, "replay_candidate", return_value={"passed": True}):
            result = qwen_simplifier.judge_one(args, {"id": "test", "task_id": "task", "messages": self.messages},
                SimpleNamespace(**vars(self.runtime), retrieval_variant="bm25"), None, None, None, tokenizer, io.StringIO())
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(len(result["attempts"]), 2)


@unittest.skipUnless(os.environ.get("BANKING_SIMPLIFY_INTEGRATION") == "1", "Requires Banking artifacts and tau2/tokenizer dependencies")
class BankingSimplificationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from loguru import logger
        from transformers import AutoTokenizer
        from slime.utils.mask_utils import MultiTurnLossMaskGenerator
        logger.remove()
        cls.rows, cls.tasks, cls.contracts, cls.allowed, factory = qwen_simplifier.load_inputs(qwen_simplifier.SOURCE)
        cls.factory = staticmethod(factory)
        cls.by_id = {r["id"]: r for r in cls.rows}
        cls.tokenizer = AutoTokenizer.from_pretrained(qwen_simplifier.DEFAULT_TOKENIZER, local_files_only=True)
        cls.generator = MultiTurnLossMaskGenerator(cls.tokenizer, tokenizer_type="qwen3_full")

    def context(self, row):
        from tau2.data_model.tasks import Task
        task = Task.model_validate(self.tasks[row["task_id"]])
        variant = self.contracts[row["task_id"]]["retrieval_variant"]
        constructor = replay.environment_constructor(qwen_simplifier.DEFAULT_DOCUMENTS_DIR, tuple(self.allowed), variant, task)
        runtime = self.factory(self.tasks[row["task_id"]], variant)
        return task, constructor, runtime

    def test_all_original_trajectories_pass_sequential_replay(self):
        for number, row in enumerate(self.rows, 1):
            with self.subTest(id=row["id"]):
                task, constructor, runtime = self.context(row)
                result = replay.replay_candidate(task, constructor, row["messages"], row["messages"])
                self.assertTrue(result["passed"])
            if number % 100 == 0:
                print(f"SOURCE_REPLAY_CHECKED {number}/{len(self.rows)}", flush=True)

    def test_required_unlock_deletion_fails_with_real_initialization(self):
        row = next(r for r in self.rows if r["task_id"] == "banking_syn_v1_train_000011_l4_two_skill")
        removed = [c["id"] for m in row["messages"] if m["role"] == "assistant"
                   for c in m.get("tool_calls") or [] if c["name"] == "unlock_discoverable_agent_tool"]
        self.assertTrue(removed)
        candidate = simplifier.apply_deletion_plan(row["messages"], {"remove_call_ids": removed, "remove_assistant_text_indices": []})
        task, constructor, runtime = self.context(row)
        with self.assertRaisesRegex(ValueError, "not been unlocked"):
            replay.replay_candidate(task, constructor, row["messages"], candidate)

    def test_real_tokenizer_keeps_multicall_target_and_final_target(self):
        inventory = json.loads((qwen_simplifier.OUTPUT / "inventory.json").read_text())
        lengths = {r["id"]: r["source_tokens"] for r in inventory["records"]}
        row = next(r for r in self.rows if lengths[r["id"]] < 16000 and any(
            m["role"] == "assistant" and len(m.get("tool_calls") or []) > 1 for m in r["messages"]))
        task, constructor, runtime = self.context(row)
        output = simplifier.complete_sft_rows(row["messages"], runtime, self.generator, {})
        self.assertTrue(any(len(r["messages"][-1].get("tool_calls") or []) > 1 for r in output))
        self.assertEqual(len(output), sum(m["role"] == "assistant" for m in simplifier.agent_view(row["messages"], runtime.system_prompt)))

    def test_export_pipeline_with_lookup_fixtures_in_temporary_directory(self):
        # Fixture deletion choices only exercise export; they are never loaded
        # into production Qwen decisions or the real output dataset.
        lookup_dir = qwen_simplifier.ROOT / "output/experiments/tau2-banking-expert-sft/data/reviewed-lookup-v1"
        candidates = list(simplifier.read_rows(lookup_dir / "simplified_trajectories.jsonl"))
        real_inventory = json.loads((qwen_simplifier.OUTPUT / "inventory.json").read_text())
        selected = {r["id"] for r in candidates}
        inventory = {**real_inventory, "pilot_ids": sorted(selected),
                     "records": [r for r in real_inventory["records"] if r["id"] in selected]}
        decisions = []
        for candidate in candidates:
            source = self.by_id[candidate["id"]]
            kept_calls = {c["id"] for m in candidate["messages"] for c in m.get("tool_calls") or []}
            final_index = max(i for i, m in enumerate(source["messages"]) if m["role"] == "assistant")
            plan = {"remove_call_ids": [c["id"] for m in source["messages"] if m["role"] == "assistant"
                                        for c in m.get("tool_calls") or [] if c["id"] not in kept_calls],
                    "remove_assistant_text_indices": [i for i, m in enumerate(source["messages"]) if m["role"] == "assistant"
                                                       and (m.get("content") or "").strip() and i != final_index]}
            rebuilt = simplifier.apply_deletion_plan(source["messages"], plan)
            task, constructor, runtime = self.context(source)
            result = replay.replay_candidate(task, constructor, source["messages"], rebuilt)
            decisions.append({"id": source["id"], "status": "accepted", "reason": "UNIT_TEST_FIXTURE", "plan": plan,
                              "replay": result, "finished_at": 1,
                              "candidate_tokens": len(self.tokenizer.apply_chat_template(simplifier.agent_view(rebuilt, runtime.system_prompt),
                                  tools=runtime.tools, tokenize=True, return_dict=False))})
        with tempfile.TemporaryDirectory(prefix="banking-export-test-") as temporary:
            root = Path(temporary)
            (root / "inventory.json").write_text(json.dumps(inventory))
            (root / "decisions").mkdir()
            (root / "decisions" / "fixture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decisions))
            args = SimpleNamespace(output_dir=root, scope="pilot", max_tokens=16384)
            qwen_simplifier.finalize(args, self.rows, self.tasks, self.contracts, self.factory, self.tokenizer)
            output_dir = root / "data/pilot"
            output = list(simplifier.read_rows(output_dir / "banking_simplified_sft.jsonl"))
            self.assertEqual(len(output), 16)
            self.assertEqual({r["metadata"]["source_trajectory_id"] for r in output}, selected)
            stats = json.loads((output_dir / "stats.json").read_text())
            self.assertEqual(stats["counts"]["accepted"], 8)
            self.assertEqual(stats["counts"]["recovered_over_cap"], 4)


class BankingExpertFormatterTest(unittest.TestCase):
    def test_agent_view_drops_user_tool_events_and_terminal_stop(self) -> None:
        raw_messages = [
            {"role": "user", "content": "Please look this up."},
            {
                "role": "assistant",
                "content": "\n\n",
                "tool_calls": [
                    {
                        "id": "source-call",
                        "name": "lookup",
                        "arguments": {"key": "balance"},
                        "requestor": "assistant",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "balance: 42",
                "tool_call_id": "source-call",
                "requestor": "assistant",
            },
            {
                "role": "user",
                "content": "I will apply the customer-side action.",
                "tool_calls": [
                    {
                        "id": "user-call",
                        "name": "customer_action",
                        "arguments": {},
                        "requestor": "user",
                    }
                ],
            },
            {
                "role": "tool",
                "content": "customer action completed",
                "tool_call_id": "user-call",
                "requestor": "user",
            },
            {"role": "user", "content": "It completed successfully.\n\n###STOP###"},
            {"role": "assistant", "content": "The balance is 42."},
        ]

        rows = banking_expert.expand_agent_targets(
            raw_messages,
            system_prompt="BANKING SYSTEM",
            tools=TOOLS,
            agent_tool_names={"lookup"},
        )

        self.assertEqual(len(rows), 2)
        self.assertNotIn("target_call_index", rows[0]["metadata"])
        self.assertEqual(
            rows[0]["messages"][-1],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0001",
                        "name": "lookup",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key": "balance"}',
                        },
                    }
                ],
                "step_loss_mask": 1,
            },
        )
        second_messages = rows[1]["messages"]
        self.assertTrue(
            all(
                message.get("role") != "tool" or message.get("requestor") != "user"
                for message in second_messages
            )
        )
        self.assertEqual(
            {message.get("content") for message in second_messages if message.get("role") == "user"},
            {
                "Please look this up.",
                "It completed successfully.",
            },
        )
        self.assertEqual(
            [message["step_loss_mask"] for message in second_messages],
            [0, 0, 0, 0, 0, 1],
        )

    def test_multi_call_turn_is_expanded_with_real_result_between_targets(self) -> None:
        raw_messages = [
            {"role": "user", "content": "Search twice."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "first", "name": "lookup", "arguments": {"key": "a"}},
                    {"id": "second", "name": "lookup", "arguments": {"key": "b"}},
                ],
            },
            {"role": "tool", "content": "A", "tool_call_id": "first", "requestor": "assistant"},
            {"role": "tool", "content": "B", "tool_call_id": "second", "requestor": "assistant"},
            {"role": "user", "content": "###STOP###"},
        ]

        rows = banking_expert.expand_agent_targets(
            raw_messages,
            system_prompt="BANKING SYSTEM",
            tools=TOOLS,
            agent_tool_names={"lookup"},
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["metadata"]["target_call_index"], 0)
        self.assertEqual(rows[1]["metadata"]["target_call_index"], 1)
        self.assertEqual(
            rows[1]["messages"][-1]["tool_calls"][0]["function"]["arguments"],
            '{"key": "b"}',
        )
        self.assertEqual(
            rows[1]["messages"][-2],
            {
                "role": "tool",
                "content": "A",
                "tool_call_id": "call_0001",
                "step_loss_mask": 0,
            },
        )
        self.assertTrue(
            all(message.get("content") != "###STOP###" for message in rows[1]["messages"])
        )

    def test_unknown_agent_tool_is_rejected_instead_of_becoming_sft_target(self) -> None:
        raw_messages = [
            {"role": "user", "content": "Do it."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "bad", "name": "not_allowed", "arguments": {}}],
            },
        ]

        with self.assertRaisesRegex(banking_expert.ConversionError, "unknown Agent tool"):
            banking_expert.expand_agent_targets(
                raw_messages,
                system_prompt="BANKING SYSTEM",
                tools=TOOLS,
                agent_tool_names={"lookup"},
            )


if __name__ == "__main__":
    raise SystemExit(unittest.main())
