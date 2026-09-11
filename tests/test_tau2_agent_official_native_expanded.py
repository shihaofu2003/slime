import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from pydantic import BaseModel


NUM_GPUS = 0

SFT_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/sft"
ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/analysis"
sys.path.insert(0, str(SFT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from build_agent_official_native_expanded import (  # noqa: E402
    DEFAULT_GREETING,
    DomainRuntime,
    index_source_dialogs,
    iter_converted_source_rows,
)
from summarize_official_native_expanded_sft import (  # noqa: E402
    CHECKPOINTS,
    select_best,
)
from summarize_official_native_turn_filtered_sft import (  # noqa: E402
    build_ablation_report,
    render_markdown as render_turn_filtered_markdown,
)


class QueryArgs(BaseModel):
    query: str


class CountArgs(BaseModel):
    count: int


class EmptyArgs(BaseModel):
    pass


class FakeTool:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    @property
    def openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"Fixture tool {self.name}",
                "parameters": self.params.model_json_schema(),
            },
        }


def make_runtime(domain, agent_tools, user_tools=()):
    return DomainRuntime(
        domain=domain,
        policy=f"{domain.upper()} POLICY",
        system_prompt=f"{domain.upper()} SYSTEM",
        agent_tools=tuple(agent_tools),
        user_tools=tuple(user_tools),
        tool_schemas=tuple(tool.openai_schema for tool in agent_tools),
        greeting=DEFAULT_GREETING,
    )


AIRLINE = make_runtime(
    "airline",
    (
        FakeTool("lookup_flight", QueryArgs),
        FakeTool("reserve_seats", CountArgs),
        FakeTool("confirm_booking", EmptyArgs),
    ),
)
TELECOM = make_runtime(
    "telecom",
    (
        FakeTool("lookup_line", QueryArgs),
        FakeTool("add_data", CountArgs),
    ),
    (FakeTool("read_signal", EmptyArgs),),
)
RUNTIMES = {"airline": AIRLINE, "telecom": TELECOM}


def direct_call(name, arguments, call_id=None):
    call = {"name": name, "arguments": arguments}
    if call_id is not None:
        call["id"] = call_id
    return call


def openai_call(name, arguments, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def source_row(
    dialog_id,
    domain,
    messages,
    answer,
    *,
    turn_index=0,
    correct=1,
    reward=1,
):
    return {
        "messages": messages,
        "answer": answer,
        "metadata": {
            "source_dialog_id": dialog_id,
            "domain": domain,
            "turn_index": turn_index,
            "correct": correct,
            "reward": reward,
        },
    }


def base_messages(domain, user_text="Help me."):
    return [
        {"role": "system", "content": f"legacy {domain} prompt"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": user_text},
    ]


class OfficialNativeExpandedTest(unittest.TestCase):
    def convert(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            with source.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row) + "\n")
            dialogs, first_pass = index_source_dialogs(source, RUNTIMES)
            second_pass = Counter()
            converted = list(
                iter_converted_source_rows(
                    source,
                    dialogs=dialogs,
                    runtimes=RUNTIMES,
                    stats=second_pass,
                )
            )
        return converted, dialogs, first_pass, second_pass

    def test_only_consistent_double_success_dialogs_are_selected(self):
        valid = source_row(
            "airline_dialog_valid",
            "airline",
            base_messages("airline"),
            {"role": "assistant", "content": "Valid target."},
        )
        failed = source_row(
            "airline_dialog_failed",
            "airline",
            base_messages("airline"),
            {"role": "assistant", "content": "Failed target."},
            reward=0,
        )
        inconsistent_a = source_row(
            "airline_dialog_inconsistent",
            "airline",
            base_messages("airline"),
            {"role": "assistant", "content": "First label."},
        )
        inconsistent_b = source_row(
            "airline_dialog_inconsistent",
            "airline",
            base_messages("airline", "Again."),
            {"role": "assistant", "content": "Second label."},
            correct=0,
        )
        wrong_domain = source_row(
            "airline_dialog_domain_conflict",
            "airline",
            base_messages("airline"),
            {"role": "assistant", "content": "Airline."},
        )
        wrong_domain_2 = source_row(
            "airline_dialog_domain_conflict",
            "telecom",
            base_messages("telecom"),
            {"role": "assistant", "content": "Telecom."},
        )

        converted, dialogs, _, _ = self.convert(
            [valid, failed, inconsistent_a, inconsistent_b, wrong_domain, wrong_domain_2]
        )

        self.assertEqual([row["messages"][-1]["content"] for row in converted], ["Valid target."])
        self.assertEqual(dialogs["airline_dialog_valid"].labels, (1, 1))
        self.assertIn("conflicting_labels", dialogs["airline_dialog_inconsistent"].invalid_reasons)
        self.assertIn("conflicting_domain", dialogs["airline_dialog_domain_conflict"].invalid_reasons)

    def test_user_tool_call_and_result_are_hidden_but_natural_report_is_kept(self):
        messages = base_messages("telecom") + [
            {"role": "assistant", "content": "Please read the signal."},
            {
                "role": "user",
                "content": "hidden mixed text",
                "tool_calls": [direct_call("read_signal", {}, "user-1")],
            },
            {"role": "tool", "id": "user-1", "content": "-92 dBm"},
            {"role": "user", "content": "It reports minus 92 dBm."},
        ]
        row = source_row(
            "telecom-visible-report",
            "telecom",
            messages,
            {"role": "assistant", "content": "Thanks, I can continue."},
        )

        converted, _, _, _ = self.convert([row])

        self.assertEqual(len(converted), 1)
        serialized = json.dumps(converted[0])
        self.assertNotIn("read_signal", serialized)
        self.assertNotIn("-92 dBm", serialized)
        self.assertNotIn("hidden mixed text", serialized)
        self.assertIn("It reports minus 92 dBm.", serialized)
        greetings = [
            message
            for message in converted[0]["messages"]
            if message.get("role") == "assistant" and message.get("content") == DEFAULT_GREETING
        ]
        self.assertEqual(len(greetings), 1)

    def test_openai_ids_match_results_and_prefix_multicalls_are_serialized(self):
        calls = [
            openai_call("lookup_flight", {"query": "HAT1"}, "source-a"),
            openai_call("reserve_seats", {"count": 2}, "source-b"),
        ]
        first = source_row(
            "airline_dialog_openai",
            "airline",
            base_messages("airline"),
            {"role": "assistant", "content": "discard me", "tool_calls": calls},
            turn_index=4,
        )
        second = source_row(
            "airline_dialog_openai",
            "airline",
            base_messages("airline")
            + [
                {"role": "assistant", "content": "discard me", "tool_calls": calls},
                {"role": "tool", "id": "source-b", "content": "RESULT B"},
                {"role": "tool", "id": "source-a", "content": "RESULT A"},
            ],
            {"role": "assistant", "content": "Done."},
            turn_index=5,
        )

        converted, _, _, _ = self.convert([first, second])
        first_targets = [row for row in converted if row["metadata"]["source_row"] == 0]
        prefix_target = next(row for row in converted if row["metadata"]["source_row"] == 1)

        self.assertEqual(len(first_targets), 2)
        second_target_messages = first_targets[1]["messages"]
        self.assertEqual(
            [(message["role"], message.get("content")) for message in second_target_messages[-3:]],
            [("assistant", None), ("tool", "RESULT A"), ("assistant", None)],
        )
        prefix_events = [
            (
                message["role"],
                (message.get("tool_calls") or [{}])[0].get("name"),
                message.get("content"),
            )
            for message in prefix_target["messages"]
            if message["role"] in {"assistant", "tool"}
            and message.get("content") != DEFAULT_GREETING
        ]
        self.assertEqual(
            prefix_events,
            [
                ("assistant", "lookup_flight", None),
                ("tool", None, "RESULT A"),
                ("assistant", "reserve_seats", None),
                ("tool", None, "RESULT B"),
                ("assistant", None, "Done."),
            ],
        )

    def test_direct_calls_use_ordinal_fallback_and_expand_with_real_results(self):
        calls = [
            direct_call("lookup_flight", {"query": "A"}),
            direct_call("reserve_seats", {"count": 1}),
            direct_call("confirm_booking", {}),
        ]
        first = source_row(
            "airline_dialog_direct",
            "airline",
            base_messages("airline"),
            {"role": "assistant", "tool_calls": calls},
            turn_index=7,
        )
        second = source_row(
            "airline_dialog_direct",
            "airline",
            base_messages("airline")
            + [
                {"role": "assistant", "tool_calls": calls},
                {"role": "tool", "content": "FOUND"},
                {"role": "tool", "content": "RESERVED"},
                {"role": "tool", "content": "CONFIRMED"},
            ],
            {"role": "assistant", "content": "All set."},
            turn_index=8,
        )

        converted, _, _, _ = self.convert([first, second])
        targets = [row for row in converted if row["metadata"]["source_row"] == 0]

        self.assertEqual(len(targets), 3)
        self.assertEqual(
            [message.get("content") for message in targets[2]["messages"] if message["role"] == "tool"],
            ["FOUND", "RESERVED"],
        )
        final_call = targets[2]["messages"][-1]["tool_calls"][0]
        self.assertEqual(final_call["id"], "call_0003")
        self.assertEqual(final_call["name"], "confirm_booking")
        self.assertEqual(final_call["function"]["name"], "confirm_booking")
        self.assertEqual(final_call["function"]["arguments"], "{}")

    def test_openai_call_with_id_uses_position_when_result_has_no_id(self):
        call = openai_call("lookup_flight", {"query": "HAT2"}, "source-openai-id")
        first = source_row(
            "airline_dialog_result_without_id",
            "airline",
            base_messages("airline"),
            {
                "role": "assistant",
                "tool_calls": [
                    call,
                    openai_call("reserve_seats", {"count": 2}, "source-openai-id-2"),
                ],
            },
        )
        second = source_row(
            "airline_dialog_result_without_id",
            "airline",
            base_messages("airline")
            + [
                {
                    "role": "assistant",
                    "tool_calls": [
                        call,
                        openai_call("reserve_seats", {"count": 2}, "source-openai-id-2"),
                    ],
                },
                {"role": "tool", "content": "IDLESS FIRST RESULT"},
                {"role": "tool", "content": "IDLESS SECOND RESULT"},
            ],
            {"role": "assistant", "content": "Done."},
        )

        converted, dialogs, _, _ = self.convert([first, second])

        self.assertFalse(dialogs["airline_dialog_result_without_id"].invalid_reasons)
        target = next(
            row
            for row in converted
            if row["metadata"].get("source_row") == 0
            and row["metadata"].get("expanded_call_index") == 1
        )
        self.assertEqual(
            [message.get("content") for message in target["messages"] if message["role"] == "tool"],
            ["IDLESS FIRST RESULT"],
        )

    def test_missing_previous_target_result_keeps_only_the_earliest_target(self):
        row = source_row(
            "airline_dialog_missing_result",
            "airline",
            base_messages("airline"),
            {
                "role": "assistant",
                "tool_calls": [
                    direct_call("lookup_flight", {"query": "A"}),
                    direct_call("reserve_seats", {"count": 1}),
                ],
            },
        )

        converted, _, _, second_pass = self.convert([row])

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["metadata"]["expanded_call_index"], 0)
        self.assertEqual(converted[0]["metadata"]["expanded_call_count"], 2)
        self.assertEqual(second_pass["drop_missing_target_result"], 1)

    def test_invalid_agent_tools_and_arguments_drop_affected_rows(self):
        rows = [
            source_row(
                "telecom_user_tool_as_agent",
                "telecom",
                base_messages("telecom"),
                {"role": "assistant", "tool_calls": [direct_call("read_signal", {})]},
            ),
            source_row(
                "airline_missing_argument",
                "airline",
                base_messages("airline"),
                {"role": "assistant", "tool_calls": [direct_call("lookup_flight", {})]},
            ),
            source_row(
                "airline_unknown_argument",
                "airline",
                base_messages("airline"),
                {
                    "role": "assistant",
                    "tool_calls": [direct_call("lookup_flight", {"query": "A", "extra": 1})],
                },
            ),
            source_row(
                "airline_wrong_type",
                "airline",
                base_messages("airline"),
                {"role": "assistant", "tool_calls": [direct_call("reserve_seats", {"count": "2"})]},
            ),
        ]

        converted, _, _, second_pass = self.convert(rows)

        self.assertEqual(converted, [])
        self.assertEqual(second_pass["drop_invalid_agent_tool"], 1)
        self.assertEqual(second_pass["drop_invalid_tool_arguments"], 3)

    def test_missing_user_report_and_orphan_result_drop_affected_rows(self):
        no_report = source_row(
            "telecom_no_report",
            "telecom",
            base_messages("telecom")
            + [
                {
                    "role": "user",
                    "content": "",
                    "tool_calls": [direct_call("read_signal", {}, "user-2")],
                },
                {"role": "tool", "id": "user-2", "content": "-100 dBm"},
            ],
            {"role": "assistant", "content": "I should not see this."},
        )
        orphan = source_row(
            "airline_orphan",
            "airline",
            base_messages("airline") + [{"role": "tool", "id": "orphan", "content": "bad"}],
            {"role": "assistant", "content": "Also invalid."},
        )

        converted, dialogs, _, second_pass = self.convert([no_report, orphan])

        self.assertEqual(converted, [])
        self.assertEqual(second_pass["drop_missing_user_report"], 1)
        self.assertIn("unpaired_tool_result", dialogs["airline_orphan"].invalid_reasons)

    def test_only_final_assistant_is_supervised_and_metadata_is_minimal(self):
        row = source_row(
            "airline_dialog_masks",
            "airline",
            base_messages("airline") + [{"role": "assistant", "content": "Prefix."}, {"role": "user", "content": "Continue."}],
            {
                "role": "assistant",
                "content": "text must be discarded",
                "reasoning": "hidden",
                "tool_calls": [direct_call("lookup_flight", {"query": "A"})],
            },
            turn_index=9,
        )

        converted, _, _, _ = self.convert([row])

        self.assertEqual(len(converted), 1)
        output = converted[0]
        self.assertIsNone(output["messages"][-1]["content"])
        self.assertNotIn("text must be discarded", json.dumps(output))
        self.assertNotIn("reasoning", json.dumps(output))
        self.assertEqual(
            [message["step_loss_mask"] for message in output["messages"]],
            [0] * (len(output["messages"]) - 1) + [1],
        )
        self.assertEqual(
            output["metadata"],
            {
                "domain": "airline",
                "source_dialog_id": "airline_dialog_masks",
                "source_row": 0,
                "source_turn_index": 9,
                "target_type": "tool_call",
                "correct": 1,
                "reward": 1,
            },
        )

    def test_shell_wrapper_resolves_python_from_service_agent_root_when_snapshotted(self):
        wrapper = SFT_DIR / "build_agent_official_native_expanded.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "job-snapshot" / "build.sh"
            snapshot.parent.mkdir()
            shutil.copy2(wrapper, snapshot)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            call_log = root / "calls.txt"
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            service_root = root / "service-agent"
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "CALL_LOG": str(call_log),
                    "SERVICE_AGENT_ROOT": str(service_root),
                }
            )

            subprocess.run(
                ["bash", str(snapshot)],
                cwd=root,
                env=environment,
                check=True,
            )

            calls = call_log.read_text(encoding="utf-8").splitlines()
        converter = service_root / "slime/examples/tau2-bench/sft/build_agent_official_native_expanded.py"
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call.startswith(str(converter)) for call in calls))


class OfficialNativeExpandedWorkflowTest(unittest.TestCase):
    expected_checkpoints = (
        ("iter_0000399", 400),
        ("iter_0000799", 800),
        ("iter_0001199", 1200),
        ("iter_0001599", 1600),
        ("iter_0001897", 1898),
        ("iter_0001999", 2000),
        ("iter_0002399", 2400),
        ("iter_0002799", 2800),
        ("iter_0003199", 3200),
        ("iter_0003599", 3600),
        ("iter_0003795", 3796),
    )
    turn_filtered_checkpoints = (
        ("iter_0000399", 400),
        ("iter_0000799", 800),
        ("iter_0001199", 1200),
        ("iter_0001599", 1600),
        ("iter_0001855", 1856),
        ("iter_0001999", 2000),
        ("iter_0002399", 2400),
        ("iter_0002799", 2800),
        ("iter_0003199", 3200),
        ("iter_0003599", 3600),
        ("iter_0003711", 3712),
    )

    def _stubbed_environment(self, root):
        capture_path = root / "environment.txt"
        python_log = root / "python.txt"
        fake_bin = root / "bin"
        fake_bin.mkdir()
        fake_bash = fake_bin / "bash"
        fake_bash.write_text(
            "#!/bin/sh\nenv > \"${ENV_CAPTURE_PATH}\"\n",
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)
        fake_python = fake_bin / "python3"
        fake_python.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"${PYTHON_LOG_PATH}\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        service_root = root / "service-agent"
        project_root = service_root / "slime"
        environment = dict(os.environ)
        for key in (
            "CHECKPOINT_ROOT",
            "MODEL_ROOT",
            "PROJECT_ROOT",
            "SERVICE_AGENT_ROOT",
        ):
            environment.pop(key, None)
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "ENV_CAPTURE_PATH": str(capture_path),
                "PYTHON_LOG_PATH": str(python_log),
                "SERVICE_AGENT_ROOT": str(service_root),
                "PROJECT_ROOT": str(project_root),
            }
        )
        return service_root, project_root, environment, capture_path, python_log

    def _read_environment(self, path):
        return dict(
            line.split("=", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )

    def test_training_wrapper_pins_full_and_smoke_recipes(self):
        wrapper = SFT_DIR / "run_qwen3_4b_instruct_2507_sft_official_native_expanded.sh"
        cases = (
            (
                "smoke",
                32,
                "agent_official_native_expanded_longest32.jsonl",
                "Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_longest32_smoke_v2_20260903",
                "1",
            ),
            (
                "full",
                30376,
                "agent_official_native_expanded.jsonl",
                "Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903",
                "2",
            ),
        )
        for mode, rows, data_basename, save_basename, epochs in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                service_root, project_root, environment, capture, python_log = (
                    self._stubbed_environment(root)
                )
                data_dir = (
                    project_root
                    / "output/experiments/tau2-sft-official-native-expanded/data"
                )
                data_dir.mkdir(parents=True)
                (data_dir / data_basename).write_text(
                    "{}\n" * rows,
                    encoding="utf-8",
                )

                subprocess.run(
                    ["/bin/bash", str(wrapper), mode],
                    env=environment,
                    check=True,
                )
                captured = self._read_environment(capture)

                self.assertEqual(Path(captured["SFT_DATA_PATH"]).name, data_basename)
                self.assertEqual(Path(captured["SAVE_DIR"]).name, save_basename)
                self.assertEqual(captured["NUM_GPUS"], "8")
                self.assertEqual(captured["ROLLOUT_BATCH_SIZE"], "16")
                self.assertEqual(captured["GLOBAL_BATCH_SIZE"], "16")
                self.assertEqual(captured["NUM_EPOCH"], epochs)
                self.assertEqual(captured["START_ROLLOUT_ID"], "0")
                self.assertEqual(captured["SAVE_INTERVAL"], "400")
                self.assertEqual(captured["MAX_TOKENS_PER_GPU"], "16384")
                self.assertEqual(captured["LR"], "1e-5")
                self.assertEqual(captured["MIN_LR"], "1e-6")
                self.assertEqual(captured["LOSS_MASK_TYPE"], "qwen3_full")
                self.assertEqual(captured["TOOL_KEY"], "tools")
                self.assertEqual(captured["CONTEXT_PARALLEL_SIZE"], "1")
                self.assertEqual(
                    captured["HF_CHECKPOINT"],
                    str(service_root / "models/Qwen3-4B-Instruct-2507"),
                )
                original_torch_dist = (
                    service_root / "models/Qwen3-4B-Instruct-2507_torch_dist"
                )
                self.assertEqual(captured["TORCH_DIST_DIR"], str(original_torch_dist))
                self.assertEqual(captured["LOAD_DIR"], str(original_torch_dist))
                validator = python_log.read_text(encoding="utf-8")
                self.assertIn("build_agent_official_native_expanded.py validate", validator)
                self.assertIn("--max-tokens 16384", validator)

    def test_turn_filtered_training_wrapper_preserves_processed_recipe(self):
        wrapper = (
            SFT_DIR
            / "run_qwen3_4b_instruct_2507_sft_official_native_expanded_turn_filtered.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_root, project_root, environment, capture, python_log = (
                self._stubbed_environment(root)
            )
            data_dir = (
                project_root
                / "output/experiments/tau2-sft-turn-quality-qwen38-xhigh-v1/data"
            )
            data_dir.mkdir(parents=True)
            data_path = data_dir / "agent_official_native_expanded_turn_filtered.jsonl"
            data_path.write_text("{}\n" * 29701, encoding="utf-8")

            subprocess.run(
                ["/bin/bash", str(wrapper)],
                env=environment,
                check=True,
            )
            captured = self._read_environment(capture)

            self.assertEqual(captured["SFT_DATA_PATH"], str(data_path))
            self.assertEqual(
                Path(captured["SAVE_DIR"]).name,
                "Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_turn_filtered_20260908",
            )
            self.assertEqual(captured["NUM_GPUS"], "8")
            self.assertEqual(captured["ROLLOUT_BATCH_SIZE"], "16")
            self.assertEqual(captured["GLOBAL_BATCH_SIZE"], "16")
            self.assertEqual(captured["NUM_EPOCH"], "2")
            self.assertEqual(captured["START_ROLLOUT_ID"], "0")
            self.assertEqual(captured["SAVE_INTERVAL"], "400")
            self.assertEqual(captured["MAX_TOKENS_PER_GPU"], "16384")
            self.assertEqual(captured["LR"], "1e-5")
            self.assertEqual(captured["MIN_LR"], "1e-6")
            self.assertEqual(captured["TRAIN_SEED"], "1234")
            self.assertEqual(captured["LOSS_MASK_TYPE"], "qwen3_full")
            self.assertEqual(captured["TOOL_KEY"], "tools")
            self.assertEqual(captured["CONTEXT_PARALLEL_SIZE"], "1")
            original_torch_dist = (
                service_root / "models/Qwen3-4B-Instruct-2507_torch_dist"
            )
            self.assertEqual(captured["LOAD_DIR"], str(original_torch_dist))
            validator = python_log.read_text(encoding="utf-8")
            self.assertIn("build_agent_official_native_expanded.py validate", validator)
            self.assertIn("--max-tokens 16384", validator)

    def test_turn_filtered_training_wrapper_resumes_latest_checkpoint(self):
        wrapper = (
            SFT_DIR
            / "run_qwen3_4b_instruct_2507_sft_official_native_expanded_turn_filtered.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_root, project_root, environment, capture, _ = (
                self._stubbed_environment(root)
            )
            data_dir = (
                project_root
                / "output/experiments/tau2-sft-turn-quality-qwen38-xhigh-v1/data"
            )
            data_dir.mkdir(parents=True)
            (data_dir / "agent_official_native_expanded_turn_filtered.jsonl").write_text(
                "{}\n" * 29701,
                encoding="utf-8",
            )
            checkpoint_root = (
                service_root
                / "checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_turn_filtered_20260908"
            )
            checkpoint_root.mkdir(parents=True)
            (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
                "799",
                encoding="utf-8",
            )

            subprocess.run(
                ["/bin/bash", str(wrapper), "resume"],
                env=environment,
                check=True,
            )
            captured = self._read_environment(capture)

            self.assertEqual(captured["LOAD_DIR"], str(checkpoint_root))
            self.assertNotIn("START_ROLLOUT_ID", captured)

    def test_turn_filtered_training_wrapper_rejects_incomplete_dataset(self):
        wrapper = (
            SFT_DIR
            / "run_qwen3_4b_instruct_2507_sft_official_native_expanded_turn_filtered.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, project_root, environment, _, _ = self._stubbed_environment(root)
            data_dir = (
                project_root
                / "output/experiments/tau2-sft-turn-quality-qwen38-xhigh-v1/data"
            )
            data_dir.mkdir(parents=True)
            (data_dir / "agent_official_native_expanded_turn_filtered.jsonl").write_text(
                "{}\n" * 29700,
                encoding="utf-8",
            )

            result = subprocess.run(
                ["/bin/bash", str(wrapper)],
                env=environment,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 29701 rows, found 29700", result.stderr)

    def test_checkpoint_list_and_hf_output_mapping_are_exact(self):
        self.assertEqual(CHECKPOINTS, self.expected_checkpoints)
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "scripts/convert_tau2_agent_sft_official_native_expanded_to_hf.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_root, _, environment, capture, _ = self._stubbed_environment(root)
            run_root = (
                service_root
                / "checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903"
            )
            for checkpoint, _ in self.expected_checkpoints:
                subprocess.run(
                    ["/bin/bash", str(wrapper), "full", checkpoint],
                    env=environment,
                    check=True,
                )
                captured = self._read_environment(capture)
                expected_output = (
                    run_root / "final_hf"
                    if checkpoint == "iter_0003795"
                    else run_root / f"{checkpoint}_hf"
                )
                self.assertEqual(captured["ITER_DIR"], str(run_root / checkpoint))
                self.assertEqual(captured["OUTPUT_DIR"], str(expected_output))
                self.assertEqual(
                    captured["ORIGIN_HF_DIR"],
                    str(service_root / "models/Qwen3-4B-Instruct-2507"),
                )

    def test_turn_filtered_conversion_maps_each_training_checkpoint(self):
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "scripts/convert_tau2_agent_sft_official_native_expanded_turn_filtered_to_hf.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_root, _, environment, capture, _ = self._stubbed_environment(root)
            run_root = (
                service_root
                / "checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_turn_filtered_20260908"
            )
            for checkpoint, _ in self.turn_filtered_checkpoints:
                subprocess.run(
                    ["/bin/bash", str(wrapper), checkpoint],
                    env=environment,
                    check=True,
                )
                captured = self._read_environment(capture)
                expected_output = (
                    run_root / "final_hf"
                    if checkpoint == "iter_0003711"
                    else run_root / f"{checkpoint}_hf"
                )
                self.assertEqual(captured["ITER_DIR"], str(run_root / checkpoint))
                self.assertEqual(captured["OUTPUT_DIR"], str(expected_output))
                self.assertEqual(
                    captured["ORIGIN_HF_DIR"],
                    str(service_root / "models/Qwen3-4B-Instruct-2507"),
                )

    def test_eval_wrapper_maps_every_checkpoint_and_pins_protocol(self):
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "examples/tau2-bench/eval/official/models"
            / "run_tau2_agent_sft_official_native_expanded_checkpoint.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_root, _, environment, capture, _ = self._stubbed_environment(root)
            run_root = (
                service_root
                / "checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903"
            )
            for checkpoint, _ in self.expected_checkpoints:
                hf_basename = (
                    "final_hf"
                    if checkpoint == "iter_0003795"
                    else f"{checkpoint}_hf"
                )
                hf_dir = run_root / hf_basename
                hf_dir.mkdir(parents=True, exist_ok=True)
                (hf_dir / "config.json").write_text("{}\n", encoding="utf-8")
                subprocess.run(
                    ["/bin/bash", str(wrapper), "full", checkpoint, "300"],
                    env=environment,
                    check=True,
                )
                captured = self._read_environment(capture)
                self.assertEqual(captured["MODEL_PATH"], str(hf_dir))
                self.assertEqual(captured["AGENT_EVAL_MODE"], "official-native")
                self.assertEqual(captured["AGENT_TOOL_CALL_PARSER"], "qwen")
                self.assertEqual(captured["DOMAINS"], "airline,retail,telecom")
                self.assertEqual(
                    captured["DOMAIN_CONCURRENCY"],
                    "airline:2,retail:2,telecom:5",
                )
                self.assertEqual(captured["NUM_TASKS"], "")
                self.assertEqual(captured["NUM_TRIALS"], "4")
                self.assertEqual(captured["SEED"], "300")
                self.assertEqual(captured["MAX_STEPS"], "200")
                self.assertEqual(captured["AGENT_TEMPERATURE"], "0.6")
                self.assertEqual(captured["AGENT_TOP_P"], "1.0")
                self.assertEqual(captured["AGENT_MAX_TOKENS"], "1200")

    def test_turn_filtered_eval_maps_curve_and_preserves_processed_protocol(self):
        wrapper = (
            Path(__file__).resolve().parents[1]
            / "examples/tau2-bench/eval/official/models"
            / "run_tau2_agent_sft_official_native_expanded_turn_filtered_checkpoint.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_root, project_root, environment, capture, _ = (
                self._stubbed_environment(root)
            )
            run_root = (
                service_root
                / "checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_turn_filtered_20260908"
            )
            for checkpoint, _ in self.turn_filtered_checkpoints:
                hf_basename = (
                    "final_hf"
                    if checkpoint == "iter_0003711"
                    else f"{checkpoint}_hf"
                )
                hf_dir = run_root / hf_basename
                hf_dir.mkdir(parents=True, exist_ok=True)
                (hf_dir / "config.json").write_text("{}\n", encoding="utf-8")

                subprocess.run(
                    ["/bin/bash", str(wrapper), checkpoint, "300"],
                    env=environment,
                    check=True,
                )
                captured = self._read_environment(capture)

                self.assertEqual(captured["MODEL_PATH"], str(hf_dir))
                self.assertEqual(
                    captured["EXPERIMENT_DIR"],
                    str(
                        project_root
                        / "output/experiments/tau2-sft-official-native-expanded-turn-filtered"
                    ),
                )
                self.assertEqual(
                    captured["EVAL_LABEL"], f"checkpoint-{checkpoint}"
                )
                self.assertEqual(captured["AGENT_EVAL_MODE"], "official-native")
                self.assertEqual(captured["AGENT_TOOL_CALL_PARSER"], "qwen")
                self.assertEqual(captured["DOMAINS"], "airline,retail,telecom")
                self.assertEqual(
                    captured["DOMAIN_CONCURRENCY"],
                    "airline:2,retail:2,telecom:5",
                )
                self.assertEqual(captured["NUM_TASKS"], "")
                self.assertEqual(captured["NUM_TRIALS"], "4")
                self.assertEqual(captured["SEED"], "300")
                self.assertEqual(captured["MAX_STEPS"], "200")
                self.assertEqual(captured["AGENT_TEMPERATURE"], "0.6")
                self.assertEqual(captured["AGENT_TOP_P"], "1.0")
                self.assertEqual(captured["AGENT_MAX_TOKENS"], "1200")

    def test_smoke_conversion_and_eval_use_the_two_update_checkpoint(self):
        conversion = (
            Path(__file__).resolve().parents[1]
            / "scripts/convert_tau2_agent_sft_official_native_expanded_to_hf.sh"
        )
        evaluation = (
            Path(__file__).resolve().parents[1]
            / "examples/tau2-bench/eval/official/models"
            / "run_tau2_agent_sft_official_native_expanded_checkpoint.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_root, _, environment, capture, _ = self._stubbed_environment(root)
            subprocess.run(
                ["/bin/bash", str(conversion), "smoke", "iter_0000001"],
                env=environment,
                check=True,
            )
            captured = self._read_environment(capture)
            smoke_root = (
                service_root
                / "checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_longest32_smoke_v2_20260903"
            )
            self.assertEqual(captured["ITER_DIR"], str(smoke_root / "iter_0000001"))
            self.assertEqual(captured["OUTPUT_DIR"], str(smoke_root / "final_hf"))
            (smoke_root / "final_hf").mkdir(parents=True)
            (smoke_root / "final_hf/config.json").write_text("{}\n", encoding="utf-8")
            subprocess.run(
                ["/bin/bash", str(evaluation), "smoke", "iter_0000001", "300"],
                env=environment,
                check=True,
            )
            captured = self._read_environment(capture)
            self.assertEqual(captured["MODEL_PATH"], str(smoke_root / "final_hf"))
            self.assertEqual(captured["NUM_TASKS"], "1")
            self.assertEqual(captured["NUM_TRIALS"], "1")
            self.assertEqual(captured["EVAL_LABEL"], "smoke-e2e")

    def test_selection_ties_use_pass_power_four_then_earlier_checkpoint(self):
        models = {
            checkpoint: {
                "overall": {"pass_at_1": 0.1, "pass_power_4": 0.0}
            }
            for checkpoint, _ in self.expected_checkpoints
        }
        models["iter_0000799"]["overall"].update(
            pass_at_1=0.5, pass_power_4=0.2
        )
        models["iter_0001199"]["overall"].update(
            pass_at_1=0.5, pass_power_4=0.2
        )
        self.assertEqual(select_best(models), "iter_0000799")

        models["iter_0001199"]["overall"]["pass_power_4"] = 0.3
        self.assertEqual(select_best(models), "iter_0001199")

    def test_turn_filtered_report_selects_checkpoint_and_compares_processed(self):
        def model(pass_at_1, pass_at_4_any, pass_power_4):
            metrics = {
                "pass_at_1": pass_at_1,
                "pass_at_4_any": pass_at_4_any,
                "pass_power_4": pass_power_4,
                "action_accuracy": 0.6,
                "db_accuracy": 0.4,
            }
            return {
                "overall": dict(metrics),
                "domains": {
                    domain: dict(metrics)
                    for domain in ("airline", "retail", "telecom")
                },
            }

        seed300 = {
            checkpoint: model(0.2, 0.4, 0.1)
            for checkpoint, _ in self.turn_filtered_checkpoints
        }
        seed300["iter_0000799"] = model(0.6, 0.8, 0.2)
        seed300["iter_0001199"] = model(0.6, 0.8, 0.3)
        seed301 = {
            "iter_0001199": model(0.5, 0.7, 0.2),
            "iter_0003711": model(0.4, 0.6, 0.1),
        }
        processed = {
            "300": model(0.55, 0.8, 0.25),
            "301": model(0.45, 0.7, 0.15),
        }

        report = build_ablation_report(
            seed300=seed300,
            seed301=seed301,
            processed=processed,
        )

        self.assertEqual(report["selected_checkpoint"], "iter_0001199")
        comparison = report["selected_vs_processed"]
        self.assertAlmostEqual(
            comparison["two_seed_mean"]["turn_filtered"]["overall"]["pass_at_1"],
            0.55,
        )
        self.assertAlmostEqual(
            comparison["two_seed_mean"]["processed"]["overall"]["pass_at_1"],
            0.5,
        )
        self.assertAlmostEqual(
            comparison["turn_filtered_minus_processed"]["mean"]["overall"]["pass_at_1"],
            0.05,
        )
        markdown = render_turn_filtered_markdown(report)
        self.assertIn("`iter_0001199`", markdown)
        self.assertIn("Turn-filtered minus Processed", markdown)
        self.assertIn("+5.00pp", markdown)

    def test_turn_filtered_report_separates_equal_update_and_final_comparisons(self):
        def model(pass_at_1):
            metrics = {
                "pass_at_1": pass_at_1,
                "pass_at_4_any": 0.4,
                "pass_power_4": 0.1,
                "action_accuracy": 0.6,
                "db_accuracy": 0.4,
            }
            return {
                "overall": dict(metrics),
                "domains": {
                    domain: dict(metrics)
                    for domain in ("airline", "retail", "telecom")
                },
            }

        seed300 = {
            checkpoint: model(0.2)
            for checkpoint, _ in self.turn_filtered_checkpoints
        }
        seed300["iter_0001199"] = model(0.8)
        seed300["iter_0003599"] = model(0.3)
        seed300["iter_0003711"] = model(0.4)
        seed301 = {
            "iter_0001199": model(0.7),
            "iter_0003599": model(0.5),
            "iter_0003711": model(0.6),
        }
        processed_final = {"300": model(0.9), "301": model(0.8)}
        processed_equal_update = {"300": model(0.4), "301": model(0.6)}

        try:
            report = build_ablation_report(
                seed300=seed300,
                seed301=seed301,
                processed=processed_final,
                processed_matched=processed_equal_update,
            )
        except TypeError as exc:
            self.fail(f"equal-update comparison input is unsupported: {exc}")

        self.assertAlmostEqual(
            report["equal_update_vs_processed"]["two_seed_mean"]
            ["turn_filtered"]["overall"]["pass_at_1"],
            0.4,
        )
        self.assertAlmostEqual(
            report["equal_update_vs_processed"]["two_seed_mean"]
            ["processed"]["overall"]["pass_at_1"],
            0.5,
        )
        self.assertAlmostEqual(
            report["equal_epoch_final_vs_processed"]["two_seed_mean"]
            ["turn_filtered"]["overall"]["pass_at_1"],
            0.5,
        )
        markdown = render_turn_filtered_markdown(report)
        equal_update_heading = markdown.index("Equal-update comparison")
        final_heading = markdown.index("Equal-epoch final comparison")
        self.assertLess(equal_update_heading, final_heading)
        self.assertIn("`iter_0003599` versus `iter_0003599`", markdown)
        self.assertIn("`iter_0003711` versus `iter_0003795`", markdown)

    def test_turn_filtered_report_supports_seed300_selection_before_confirmation(self):
        def model(pass_at_1):
            metrics = {
                "pass_at_1": pass_at_1,
                "pass_at_4_any": 0.4,
                "pass_power_4": 0.1,
                "action_accuracy": 0.6,
                "db_accuracy": 0.4,
            }
            return {
                "overall": dict(metrics),
                "domains": {
                    domain: dict(metrics)
                    for domain in ("airline", "retail", "telecom")
                },
            }

        seed300 = {
            checkpoint: model(0.3)
            for checkpoint, _ in self.turn_filtered_checkpoints
        }
        seed300["iter_0002399"] = model(0.5)
        processed = {"300": model(0.4), "301": model(0.4)}

        report = build_ablation_report(
            seed300=seed300,
            seed301={},
            processed=processed,
        )

        self.assertEqual(report["selected_checkpoint"], "iter_0002399")
        self.assertIsNone(report["selected_vs_processed"])
        markdown = render_turn_filtered_markdown(report)
        self.assertIn("`iter_0002399`", markdown)
        self.assertNotIn("Turn-filtered minus Processed", markdown)

    def test_turn_filtered_summary_cli_exposes_curve_and_confirmation_inputs(self):
        script = (
            ANALYSIS_DIR / "summarize_official_native_turn_filtered_sft.py"
        )

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("--summary", result.stdout)
        self.assertIn("--confirmation", result.stdout)
        self.assertIn("--processed-seed300", result.stdout)
        self.assertIn("--processed-seed301", result.stdout)
        self.assertIn("--processed-matched-seed300", result.stdout)
        self.assertIn("--processed-matched-seed301", result.stdout)
        self.assertIn("--json-output", result.stdout)
        self.assertIn("--markdown-output", result.stdout)

    def test_turn_filtered_summary_shell_uses_repo_python_entrypoint(self):
        wrapper = (
            ANALYSIS_DIR / "run_official_native_turn_filtered_sft_summary.sh"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            call_log = root / "python.txt"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" > \"${PYTHON_LOG_PATH}\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            service_root = root / "service-agent"
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "PYTHON_LOG_PATH": str(call_log),
                    "SERVICE_AGENT_ROOT": str(service_root),
                }
            )

            subprocess.run(
                ["/bin/bash", str(wrapper), "--summary", "iter_0000399=x.json"],
                env=environment,
                check=True,
            )

            call = call_log.read_text(encoding="utf-8")
            self.assertTrue(
                call.startswith(
                    str(
                        service_root
                        / "slime/examples/tau2-bench/analysis/summarize_official_native_turn_filtered_sft.py"
                    )
                )
            )
            self.assertIn("--summary iter_0000399=x.json", call)


if __name__ == "__main__":
    unittest.main()
