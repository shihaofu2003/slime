import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


NUM_GPUS = 0

SFT_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/sft"
sys.path.insert(0, str(SFT_DIR))

import filter_official_native_turns as turn_filter  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return list(range(100))


def training_row(
    target,
    *,
    dialog_id="dialog-1",
    turn_index=1,
    domain="airline",
    target_type=None,
    prefix_assistant=None,
):
    messages = [
        {
            "role": "system",
            "content": "<policy>Use grounded facts and confirm writes.</policy>",
            "step_loss_mask": 0,
        },
        {"role": "user", "content": "Please help.", "step_loss_mask": 0},
    ]
    if prefix_assistant is not None:
        messages.append(
            {
                "role": "assistant",
                "content": prefix_assistant,
                "step_loss_mask": 0,
            }
        )
        messages.append(
            {"role": "user", "content": "Please correct that.", "step_loss_mask": 0}
        )
    target = dict(target)
    target["role"] = "assistant"
    target["step_loss_mask"] = 1
    messages.append(target)
    if target_type is None:
        target_type = "tool_call" if target.get("tool_calls") else "text"
    return {
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Read state.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "metadata": {
            "domain": domain,
            "source_dialog_id": dialog_id,
            "source_turn_index": turn_index,
            "target_type": target_type,
            "correct": 1,
            "reward": 1,
        },
    }


def pass_review(input_line, pass_id, verdict, *, metadata=None):
    issues = []
    if verdict == "drop":
        issues = [
            {
                "type": "factual_grounding",
                "evidence": ["The tool result says otherwise."],
                "explanation": "The target contradicts the supplied state.",
            }
        ]
    return {
        "input_line": input_line,
        "target_id": input_line,
        "pass_id": pass_id,
        "prompt_version": turn_filter.PROMPT_VERSION,
        "model": "judge",
        "reasoning_effort": "xhigh",
        "seed": turn_filter.PASS_SEEDS[pass_id],
        "attempts": 1,
        "finish_reason": "tool_calls",
        "had_reasoning": True,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "latency_seconds": 1.0,
        "metadata": metadata
        or {
            "domain": "airline",
            "source_dialog_id": "dialog-1",
            "source_turn_index": input_line,
            "expanded_call_index": None,
            "target_type": "text",
        },
        "verdict": verdict,
        "prefix_relation": "clean",
        "issues": issues,
        "reason": f"The target is {verdict}.",
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


class TurnQualityFilterTest(unittest.TestCase):
    def test_judge_payload_blinds_labels_and_uses_xhigh(self):
        row = training_row({"content": "Grounded answer."})

        evidence = turn_filter.evidence_record(row)
        body = turn_filter.judge_request_body(17, row, model="judge", seed=300)

        self.assertEqual(set(evidence), {"messages", "tools"})
        self.assertNotIn("correct", json.dumps(evidence))
        self.assertNotIn("reward", json.dumps(evidence))
        self.assertEqual(
            body["chat_template_kwargs"],
            {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "xhigh",
            },
        )
        self.assertEqual(body["tool_choice"]["function"]["name"], "submit_turn_review")
        self.assertEqual(len(body["tools"]), 1)
        self.assertIn("target_id 17", body["messages"][-1]["content"])
        self.assertIn("Basic economy flights cannot be modified", body["messages"][0]["content"])
        self.assertIn("two-step workaround", body["messages"][0]["content"])
        self.assertIn("every concrete action", body["messages"][0]["content"])
        self.assertIn("genuinely conditional explanation", body["messages"][0]["content"])
        self.assertIn("current_time - created_at", body["messages"][0]["content"])
        self.assertEqual(turn_filter.DEFAULT_CALIBRATION[6581], "drop")
        self.assertEqual(turn_filter.DEFAULT_CALIBRATION[23976], "retain")

    def test_only_final_assistant_may_be_supervised(self):
        valid = training_row({"content": "Valid."})
        turn_filter.validate_training_row(valid, 1)

        invalid = training_row({"content": "Invalid."})
        invalid["messages"][1]["step_loss_mask"] = 1
        with self.assertRaisesRegex(ValueError, "only the final Assistant"):
            turn_filter.validate_training_row(invalid, 1)

    def test_review_response_requires_evidence_for_drop(self):
        arguments = {
            "target_id": 8,
            "verdict": "drop",
            "prefix_relation": "depends_on_prior_error",
            "issues": [],
            "reason": "Wrong.",
        }
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "reasoning_content": "Checked the evidence.",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "submit_turn_review",
                                    "arguments": json.dumps(arguments),
                                }
                            }
                        ],
                    },
                }
            ],
            "usage": {},
        }

        with self.assertRaisesRegex(turn_filter.ReviewFormatError, "requires"):
            turn_filter.parse_review_response(response, 8)

        arguments["issues"] = [
            {
                "type": "policy_workflow",
                "evidence": ["Policy forbids the action."],
                "explanation": "The target takes the forbidden action.",
            }
        ]
        response["choices"][0]["message"]["tool_calls"][0]["function"][
            "arguments"
        ] = json.dumps(arguments)
        review, finish_reason, had_reasoning, _ = turn_filter.parse_review_response(
            response, 8
        )
        self.assertEqual(review["verdict"], "drop")
        self.assertEqual(finish_reason, "tool_calls")
        self.assertTrue(had_reasoning)

    def test_merge_truth_table_keeps_every_non_unanimous_drop(self):
        for verdict_a in turn_filter.VERDICTS:
            for verdict_b in turn_filter.VERDICTS:
                final_verdict, retained, _ = turn_filter.merge_verdicts(
                    verdict_a, verdict_b
                )
                if verdict_a == verdict_b == "drop":
                    self.assertEqual((final_verdict, retained), ("drop", False))
                elif verdict_a == verdict_b == "keep":
                    self.assertEqual((final_verdict, retained), ("keep", True))
                else:
                    self.assertEqual((final_verdict, retained), ("review", True))

    def test_prepare_groups_dialogs_and_covers_each_row_once(self):
        rows = [
            training_row({"content": "First."}, dialog_id="same", turn_index=1),
            training_row({"content": "Second."}, dialog_id="other", turn_index=1),
            training_row(
                {"content": "Correction."},
                dialog_id="same",
                turn_index=2,
                prefix_assistant="Earlier error.",
            ),
            training_row(
                {"content": "Retail."}, domain="retail", dialog_id="retail", turn_index=1
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            write_jsonl(source, rows)
            manifest = turn_filter.prepare_dataset(
                input_path=source,
                output_root=root / "out",
                tokenizer=FakeTokenizer(),
                model_path=Path("/model"),
                num_shards=2,
                pilot_size=4,
                pilot_seed=1,
                calibration={},
            )
            shard_contents = [
                turn_filter.read_jsonl(root / "out/shards" / f"shard_{index:02d}.jsonl")
                for index in range(2)
            ]

        locations = {}
        for shard_id, shard in enumerate(shard_contents):
            for row in shard:
                locations[row["input_line"]] = shard_id
                self.assertNotIn("correct", row)
                self.assertNotIn("reward", row)
        self.assertEqual(set(locations), {1, 2, 3, 4})
        self.assertEqual(locations[1], locations[3])
        self.assertEqual(manifest["rows"], 4)
        self.assertEqual(manifest["num_shards"], 2)

    def test_judge_orders_records_by_domain_turn_and_prompt_length(self):
        records = [
            {
                "input_line": 1,
                "prompt_version": turn_filter.PROMPT_VERSION,
                "prompt_tokens": 900,
                "metadata": {
                    "domain": "airline",
                    "source_dialog_id": "a-late",
                    "source_turn_index": 2,
                    "expanded_call_index": None,
                    "target_type": "text",
                },
            },
            {
                "input_line": 2,
                "prompt_version": turn_filter.PROMPT_VERSION,
                "prompt_tokens": 300,
                "metadata": {
                    "domain": "retail",
                    "source_dialog_id": "r-early",
                    "source_turn_index": 0,
                    "expanded_call_index": None,
                    "target_type": "text",
                },
            },
            {
                "input_line": 3,
                "prompt_version": turn_filter.PROMPT_VERSION,
                "prompt_tokens": 800,
                "metadata": {
                    "domain": "airline",
                    "source_dialog_id": "a-early-long",
                    "source_turn_index": 0,
                    "expanded_call_index": None,
                    "target_type": "text",
                },
            },
            {
                "input_line": 4,
                "prompt_version": turn_filter.PROMPT_VERSION,
                "prompt_tokens": 600,
                "metadata": {
                    "domain": "airline",
                    "source_dialog_id": "a-early-short",
                    "source_turn_index": 0,
                    "expanded_call_index": None,
                    "target_type": "text",
                },
            },
            {
                "input_line": 5,
                "prompt_version": turn_filter.PROMPT_VERSION,
                "prompt_tokens": 700,
                "metadata": {
                    "domain": "airline",
                    "source_dialog_id": "a-middle",
                    "source_turn_index": 1,
                    "expanded_call_index": None,
                    "target_type": "text",
                },
            },
        ]
        observed = []

        def fake_review(record, *, pass_id, **_kwargs):
            observed.append((record["input_line"], pass_id))
            return pass_review(
                record["input_line"], pass_id, "keep", metadata=record["metadata"]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            write_jsonl(input_path, records)
            with patch.object(turn_filter, "request_review", side_effect=fake_review):
                turn_filter.judge_shard(
                    input_path=input_path,
                    output_a=root / "pass_a.jsonl",
                    output_b=root / "pass_b.jsonl",
                    base_url="http://localhost/v1",
                    model="judge",
                    concurrency=1,
                    timeout=1,
                    retries=0,
                )

        self.assertEqual(
            observed,
            [
                (4, "a"),
                (4, "b"),
                (3, "a"),
                (3, "b"),
                (5, "a"),
                (5, "b"),
                (1, "a"),
                (1, "b"),
                (2, "a"),
                (2, "b"),
            ],
        )

    def test_launcher_defaults_server_and_judge_concurrency_to_eight(self):
        launcher = SFT_DIR / "run_qwen38_turn_quality_filter.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls = root / "calls.log"
            fake_python = bin_dir / "python3"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "echo \"$*\" >> \"${CALL_LOG}\"\n"
                "if [[ \"${1:-}\" == \"-m\" ]]; then\n"
                "  trap 'exit 0' TERM INT\n"
                "  while true; do /bin/sleep 1; done\n"
                "fi\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            fake_nvidia_smi = bin_dir / "nvidia-smi"
            fake_nvidia_smi.write_text(
                "#!/usr/bin/env bash\nprintf '0\\n'\n", encoding="utf-8"
            )
            fake_nvidia_smi.chmod(0o755)
            model_dir = root / "model"
            model_dir.mkdir()
            output_root = root / "output"
            (output_root / "shards").mkdir(parents=True)
            (output_root / "shards/manifest.json").write_text("{}\n", encoding="utf-8")
            (output_root / "shards/shard_00.jsonl").write_text("", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "CALL_LOG": str(calls),
                    "PROJECT_ROOT": str(Path(__file__).resolve().parents[1]),
                    "TAU2_TURN_JUDGE_MODEL_PATH": str(model_dir),
                    "TAU2_TURN_FILTER_OUTPUT_ROOT": str(output_root),
                    "TAU2_TURN_FILTER_PYTHON": str(fake_python),
                    "TAU2_TURN_JUDGE_PORT_BASE": "33900",
                }
            )
            env.pop("TAU2_TURN_JUDGE_CONCURRENCY", None)
            result = subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "--stage",
                    "full",
                    "--shard-start",
                    "0",
                    "--shard-count",
                    "1",
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            invocations = calls.read_text(encoding="utf-8").splitlines()

        server_call = next(line for line in invocations if "sglang.launch_server" in line)
        judge_call = next(
            line
            for line in invocations
            if "filter_official_native_turns.py judge" in line
        )
        self.assertIn("--max-running-requests 8", server_call)
        self.assertIn("--concurrency 8", judge_call)

    def test_invalid_model_output_becomes_review_but_transport_failure_raises(self):
        record = {
            "input_line": 1,
            "metadata": {},
            **training_row({"content": "Answer."}),
        }
        invalid_response = {"choices": [{"message": {}, "finish_reason": "stop"}]}
        with patch.object(turn_filter, "_post_json", return_value=invalid_response):
            review = turn_filter.request_review(
                record,
                pass_id="a",
                endpoint="http://localhost/v1/chat/completions",
                model="judge",
                timeout=1,
                retries=0,
            )
        self.assertEqual(review["verdict"], "review")
        self.assertIn("judge_error", review)

        with patch.object(
            turn_filter,
            "_post_json",
            side_effect=turn_filter.JudgeTransportError("offline"),
        ):
            with self.assertRaises(turn_filter.JudgeTransportError):
                turn_filter.request_review(
                    record,
                    pass_id="a",
                    endpoint="http://localhost/v1/chat/completions",
                    model="judge",
                    timeout=1,
                    retries=0,
                )

    def test_pilot_gate_requires_unanimous_drop_but_keeps_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            (output_root / "shards").mkdir(parents=True)
            (output_root / "shards/pilot_manifest.json").write_text(
                json.dumps(
                    {
                        "prompt_version": turn_filter.PROMPT_VERSION,
                        "rows": [1, 2],
                        "calibration": {"1": "drop", "2": "retain"},
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                output_root / "reviews/pilot/pass_a.jsonl",
                [pass_review(1, "a", "drop"), pass_review(2, "a", "keep")],
            )
            write_jsonl(
                output_root / "reviews/pilot/pass_b.jsonl",
                [pass_review(1, "b", "drop"), pass_review(2, "b", "review")],
            )

            summary = turn_filter.check_pilot(output_root=output_root)

        self.assertEqual(summary["calibration_failures"], [])
        self.assertEqual(summary["final_verdicts"], {"drop": 1, "review": 1})

    def test_finalize_copies_raw_rows_and_does_not_cascade_dialog_drops(self):
        first = training_row({"content": "Wrong target."}, dialog_id="same", turn_index=1)
        second = training_row(
            {"content": "Corrected target."},
            dialog_id="same",
            turn_index=2,
            prefix_assistant="Wrong target.",
        )
        third = training_row({"content": "Independent target."}, dialog_id="other")
        raw_lines = [
            json.dumps(first, ensure_ascii=False, separators=(", ", ": ")) + "\n",
            json.dumps(second, ensure_ascii=False, separators=(",", ":")) + "\n",
            json.dumps(third, ensure_ascii=True, separators=(",", ": ")) + "\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_bytes("".join(raw_lines).encode("utf-8"))
            output_root = root / "out"
            (output_root / "shards").mkdir(parents=True)
            (output_root / "shards/manifest.json").write_text(
                json.dumps(
                    {
                        "prompt_version": turn_filter.PROMPT_VERSION,
                        "rows": 3,
                        "num_shards": 1,
                    }
                ),
                encoding="utf-8",
            )
            reviews_a = [
                pass_review(1, "a", "drop"),
                pass_review(2, "a", "drop"),
                pass_review(3, "a", "keep"),
            ]
            reviews_b = [
                pass_review(1, "b", "drop"),
                pass_review(2, "b", "keep"),
                pass_review(3, "b", "keep"),
            ]
            write_jsonl(output_root / "reviews/pass_a/shard_00.jsonl", reviews_a)
            write_jsonl(output_root / "reviews/pass_b/shard_00.jsonl", reviews_b)

            summary = turn_filter.finalize_dataset(
                input_path=source, output_root=output_root
            )
            filtered = output_root / "data/agent_official_native_expanded_turn_filtered.jsonl"
            expected = (raw_lines[1] + raw_lines[2]).encode("utf-8")
            self.assertEqual(filtered.read_bytes(), expected)
            validation = turn_filter.validate_filtered_dataset(
                input_path=source,
                output_path=filtered,
                decisions_path=output_root / "reviews/turn_decisions.jsonl",
            )

        self.assertEqual(summary["input_rows"], 3)
        self.assertEqual(summary["retained_rows"], 2)
        self.assertEqual(summary["dropped_rows"], 1)
        self.assertEqual(validation, {"input_rows": 3, "retained_rows": 2})

    def test_finalize_preserves_manual_retain_calibration_as_review(self):
        row = training_row({"content": "Conditionally valid target."})
        raw_line = json.dumps(
            row, ensure_ascii=False, separators=(", ", ": ")
        ) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_bytes(raw_line.encode("utf-8"))
            output_root = root / "out"
            (output_root / "shards").mkdir(parents=True)
            (output_root / "shards/manifest.json").write_text(
                json.dumps(
                    {
                        "prompt_version": turn_filter.PROMPT_VERSION,
                        "rows": 1,
                        "num_shards": 1,
                    }
                ),
                encoding="utf-8",
            )
            (output_root / "shards/pilot_manifest.json").write_text(
                json.dumps(
                    {
                        "prompt_version": turn_filter.PROMPT_VERSION,
                        "rows": [1],
                        "calibration": {"1": "retain"},
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                output_root / "reviews/pass_a/shard_00.jsonl",
                [pass_review(1, "a", "drop")],
            )
            write_jsonl(
                output_root / "reviews/pass_b/shard_00.jsonl",
                [pass_review(1, "b", "drop")],
            )

            summary = turn_filter.finalize_dataset(
                input_path=source, output_root=output_root
            )
            filtered = (
                output_root
                / "data/agent_official_native_expanded_turn_filtered.jsonl"
            )
            filtered_bytes = filtered.read_bytes()
            decisions = turn_filter.read_jsonl(
                output_root / "reviews/turn_decisions.jsonl"
            )
            review_turns = turn_filter.read_jsonl(
                output_root / "reviews/review_turns.jsonl"
            )

        self.assertEqual(filtered_bytes, raw_line.encode("utf-8"))
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["pass_a"]["verdict"], "drop")
        self.assertEqual(decisions[0]["pass_b"]["verdict"], "drop")
        self.assertEqual(decisions[0]["final_verdict"], "review")
        self.assertTrue(decisions[0]["retained"])
        self.assertEqual(
            decisions[0]["decision_basis"], "manual_calibration_retain"
        )
        self.assertEqual(review_turns, decisions)
        self.assertEqual(summary["manual_retain_overrides"], [1])
        self.assertEqual(summary["retained_rows"], 1)
        self.assertEqual(summary["dropped_rows"], 0)


if __name__ == "__main__":
    unittest.main()
