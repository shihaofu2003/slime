import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


NUM_GPUS = 0

OFFICIAL_EVAL_DIR = (
    Path(__file__).resolve().parents[1] / "examples/tau2-bench/eval/official"
)
SHARED_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/shared"
ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "examples/tau2-bench/analysis"
sys.path.insert(0, str(OFFICIAL_EVAL_DIR))
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

import run_eval  # noqa: E402
from tau2_async_eval_fixture import (  # noqa: E402
    evaluate_borrowing,
    evaluate_domain,
)


class Tau2OfficialAsyncEvalTest(unittest.TestCase):
    def _capture_wrapper_environment(
        self,
        wrapper,
        *,
        arguments=None,
        overrides=None,
    ):
        if not wrapper.exists():
            self.fail(f"missing evaluation wrapper: {wrapper}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            capture_path = tmp_path / "environment.txt"
            bash_stub = tmp_path / "bash"
            bash_stub.write_text(
                "#!/bin/sh\nenv > \"${ENV_CAPTURE_PATH}\"\n",
                encoding="utf-8",
            )
            bash_stub.chmod(0o755)
            environment = dict(os.environ)
            for key in (
                "AGENT_EVAL_MODE",
                "AGENT_EXTRA_BODY_JSON",
                "AGENT_MAX_TOKENS",
                "AGENT_TOOL_CALL_PARSER",
                "DOMAIN_CONCURRENCY",
                "DOMAINS",
                "EVAL_LABEL",
                "EXPERIMENT_DIR",
                "MODEL_NAME",
                "MODEL_PATH",
                "NUM_TASKS",
                "NUM_TRIALS",
                "RETRIEVAL_CONFIG",
            ):
                environment.pop(key, None)
            environment.update(
                {
                    "PATH": f"{tmp_path}:{environment['PATH']}",
                    "ENV_CAPTURE_PATH": str(capture_path),
                    "RUN_STAMP": "wrapper-test",
                    **(overrides or {}),
                }
            )

            subprocess.run(
                ["/bin/bash", str(wrapper), *(arguments or [])],
                cwd=OFFICIAL_EVAL_DIR.parents[3],
                env=environment,
                check=True,
            )

            return dict(
                line.split("=", 1)
                for line in capture_path.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )

    def _run_eval_shell(self, *, overrides=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bin_path = tmp_path / "bin"
            bin_path.mkdir()
            capture_path = tmp_path / "python-invocations.tsv"
            capture_path.touch()
            model_path = tmp_path / "model"
            model_path.mkdir()
            python_stub = bin_path / "python3"
            python_stub.write_text(
                """#!/usr/bin/env bash
printf '%s' "${1:-}" >> "${PYTHON_CAPTURE_PATH}"
for argument in "${@:2}"; do
  printf '\t%s' "${argument}" >> "${PYTHON_CAPTURE_PATH}"
done
printf '\n' >> "${PYTHON_CAPTURE_PATH}"
if [[ "${1:-}" == "-m" && "${2:-}" == "sglang.launch_server" ]]; then
  exec /bin/sleep 600
fi
exit 0
""",
                encoding="utf-8",
            )
            python_stub.chmod(0o755)

            environment = dict(os.environ)
            for key in (
                "AGENT_EVAL_MODE",
                "AGENT_LLM",
                "AGENT_PROTOCOL_PROFILE",
                "AGENT_REPLICA_CUDA_GROUPS",
                "AGENT_SERVED_MODEL_NAME",
                "AGENT_SGLANG_EXTRA_ARGS",
                "AGENT_TOOL_CALL_PARSER",
                "SGLANG_EXTRA_ARGS",
                "USER_REPLICA_CUDA_GROUPS",
            ):
                environment.pop(key, None)
            environment.update(
                {
                    "PATH": f"{bin_path}:{environment['PATH']}",
                    "PYTHON_CAPTURE_PATH": str(capture_path),
                    "SERVICE_AGENT_ROOT": str(OFFICIAL_EVAL_DIR.parents[4]),
                    "PROJECT_ROOT": str(OFFICIAL_EVAL_DIR.parents[3]),
                    "MODEL_PATH": str(model_path),
                    "MODEL_NAME": "test-agent",
                    "SUMMARY_OUTPUT": str(tmp_path / "summary.json"),
                    "RUN_STAMP": "shell-test",
                    "DOMAINS": "airline",
                    "NUM_TASKS": "1",
                    "NUM_TRIALS": "1",
                    "USER_SGLANG": "0",
                    "USER_API_BASE": "http://user.test/v1",
                    "USER_API_KEY": "user-key",
                    "SGLANG_CLEANUP_TIMEOUT_SECONDS": "1",
                    **(overrides or {}),
                }
            )
            completed = subprocess.run(
                ["/bin/bash", str(OFFICIAL_EVAL_DIR / "run_eval.sh")],
                cwd=OFFICIAL_EVAL_DIR.parents[3],
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
            )
            invocations = [
                line.split("\t")
                for line in capture_path.read_text(encoding="utf-8").splitlines()
            ]
            return completed, invocations

    @staticmethod
    def _empty_eval_jobs():
        domain_summary = {
            "pass_metrics": {
                "tasks": 0,
                "simulations": 0,
                "pass_at_1": 0.0,
                "pass_at_4_any": 0.0,
                "pass_power_4": 0.0,
            },
            "metrics": {},
            "single_call": {},
            "timing": {
                "wall_seconds": 0.0,
                "trajectory_seconds": 0.0,
                "other": {"seconds": 0.0},
            },
        }
        return {
            "airline": {
                "summary": domain_summary,
                "timing_samples": {"agent": [], "user": []},
            }
        }

    def _run_main_and_read_summary(self, *extra_args):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "summary.json"
            argv = [
                "run_eval.py",
                "--domains",
                "airline",
                "--save-prefix",
                "test",
                "--summary-output",
                str(summary_path),
                "--agent-llm",
                "test-agent",
                "--user-llm",
                "test-user",
                *extra_args,
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(
                    run_eval,
                    "_run_domain_jobs",
                    return_value=(self._empty_eval_jobs(), []),
                ):
                    with mock.patch("builtins.print"):
                        run_eval.main()
            return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_official_native_is_the_default_agent_and_uses_upstream_prompt(self):
        from tau2.agent.llm_agent import AGENT_INSTRUCTION, LLMAgent, SYSTEM_PROMPT

        args = run_eval._build_parser().parse_args(
            [
                "--save-prefix",
                "test",
                "--agent-llm",
                "test-agent",
                "--user-llm",
                "test-user",
            ]
        )

        self.assertEqual(args.agent_eval_mode, "official-native")
        self.assertEqual(args.agent, "llm_agent")
        domain_policy = "Only make changes after the user confirms."
        agent = LLMAgent(
            tools=[],
            domain_policy=domain_policy,
            llm="openai/test-agent",
        )
        self.assertEqual(
            agent.system_prompt,
            SYSTEM_PROMPT.format(
                agent_instruction=AGENT_INSTRUCTION,
                domain_policy=domain_policy,
            ),
        )

    def test_official_generate_sends_native_tools_with_auto_choice(self):
        from tau2.data_model.message import SystemMessage, UserMessage
        from tau2.environment.tool import as_tool
        from tau2.utils import llm_utils

        def lookup_order(order_id: str) -> str:
            """Look up an order by identifier."""
            return order_id

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="The order is ready.",
                        tool_calls=None,
                    ),
                )
            ],
            to_dict=lambda: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "The order is ready.",
                            "tool_calls": None,
                        },
                    }
                ]
            },
        )
        tool = as_tool(lookup_order)
        messages = [
            SystemMessage(role="system", content="Follow policy."),
            UserMessage(role="user", content="Where is order A-1?"),
        ]

        with mock.patch.object(
            llm_utils,
            "completion",
            return_value=response,
        ) as completion:
            with mock.patch.object(llm_utils, "get_response_cost", return_value=0.0):
                with mock.patch.object(
                    llm_utils,
                    "get_response_usage",
                    return_value={},
                ):
                    with mock.patch.object(llm_utils, "_write_llm_log"):
                        llm_utils.generate(
                            model="openai/test-agent",
                            messages=messages,
                            tools=[tool],
                        )

        request = completion.call_args.kwargs
        self.assertEqual(request["tool_choice"], "auto")
        self.assertEqual(len(request["tools"]), 1)
        self.assertEqual(request["tools"][0]["type"], "function")
        self.assertEqual(
            request["tools"][0]["function"]["name"],
            "lookup_order",
        )
        self.assertNotIn("tools", request["messages"][0])

    def test_official_generate_parses_multiple_assistant_tool_calls(self):
        from tau2.data_model.message import SystemMessage, UserMessage
        from tau2.utils import llm_utils

        raw_tool_calls = [
            SimpleNamespace(
                id="call_order",
                function=SimpleNamespace(
                    name="lookup_order",
                    arguments='{"order_id": "A-1"}',
                ),
            ),
            SimpleNamespace(
                id="call_customer",
                function=SimpleNamespace(
                    name="lookup_customer",
                    arguments='{"customer_id": "C-2"}',
                ),
            ),
        ]
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=raw_tool_calls,
                    ),
                )
            ],
            to_dict=lambda: {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                        },
                    }
                ]
            },
        )
        messages = [
            SystemMessage(role="system", content="Follow policy."),
            UserMessage(role="user", content="Check both records."),
        ]

        with mock.patch.object(llm_utils, "completion", return_value=response):
            with mock.patch.object(llm_utils, "get_response_cost", return_value=0.0):
                with mock.patch.object(
                    llm_utils,
                    "get_response_usage",
                    return_value={},
                ):
                    with mock.patch.object(llm_utils, "_write_llm_log"):
                        message = llm_utils.generate(
                            model="openai/test-agent",
                            messages=messages,
                        )

        self.assertEqual(
            [tool_call.model_dump() for tool_call in message.tool_calls],
            [
                {
                    "id": "call_order",
                    "name": "lookup_order",
                    "arguments": {"order_id": "A-1"},
                    "requestor": "assistant",
                },
                {
                    "id": "call_customer",
                    "name": "lookup_customer",
                    "arguments": {"customer_id": "C-2"},
                    "requestor": "assistant",
                },
            ],
        )

    def test_official_agent_history_keeps_assistant_calls_and_all_tool_results(self):
        import tau2.agent.llm_agent as llm_agent_module
        from tau2.data_model.message import (
            AssistantMessage,
            MultiToolMessage,
            ToolCall,
            ToolMessage,
            UserMessage,
        )
        from tau2.utils.llm_utils import to_litellm_messages

        first_response = AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_order",
                    name="lookup_order",
                    arguments={"order_id": "A-1"},
                ),
                ToolCall(
                    id="call_customer",
                    name="lookup_customer",
                    arguments={"customer_id": "C-2"},
                ),
            ],
        )
        responses = iter(
            [
                first_response,
                AssistantMessage(role="assistant", content="Both records match."),
            ]
        )
        captured_histories = []

        def fake_generate(**kwargs):
            captured_histories.append(list(kwargs["messages"]))
            return next(responses)

        agent = llm_agent_module.LLMAgent(
            tools=[],
            domain_policy="Follow policy.",
            llm="openai/test-agent",
        )
        state = agent.get_init_state()
        with mock.patch.object(
            llm_agent_module,
            "generate",
            side_effect=fake_generate,
        ):
            _, state = agent.generate_next_message(
                UserMessage(role="user", content="Check both records."),
                state,
            )
            agent.generate_next_message(
                MultiToolMessage(
                    role="tool",
                    tool_messages=[
                        ToolMessage(
                            id="call_order",
                            role="tool",
                            content="order-result",
                        ),
                        ToolMessage(
                            id="call_customer",
                            role="tool",
                            content="customer-result",
                        ),
                    ],
                ),
                state,
            )

        history = to_litellm_messages(captured_histories[1])
        self.assertEqual(
            [message["role"] for message in history],
            ["system", "user", "assistant", "tool", "tool"],
        )
        self.assertEqual(
            [call["id"] for call in history[2]["tool_calls"]],
            ["call_order", "call_customer"],
        )
        self.assertEqual(history[3]["tool_call_id"], "call_order")
        self.assertEqual(history[3]["content"], "order-result")
        self.assertEqual(history[4]["tool_call_id"], "call_customer")
        self.assertEqual(history[4]["content"], "customer-result")

    def test_eval_shell_maps_official_native_to_openai_chat_completions(self):
        completed, invocations = self._run_eval_shell()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        launch = next(
            invocation
            for invocation in invocations
            if invocation[:2] == ["-m", "sglang.launch_server"]
        )
        self.assertEqual(
            launch[launch.index("--served-model-name") + 1],
            "tau2-agent",
        )
        self.assertEqual(
            launch[launch.index("--tool-call-parser") + 1],
            "qwen",
        )
        evaluation = next(
            invocation
            for invocation in invocations
            if invocation and invocation[0].endswith("/run_eval.py")
        )
        self.assertEqual(
            evaluation[evaluation.index("--agent-eval-mode") + 1],
            "official-native",
        )
        self.assertEqual(
            evaluation[evaluation.index("--agent") + 1],
            "llm_agent",
        )
        self.assertEqual(
            evaluation[evaluation.index("--agent-llm") + 1],
            "tau2-agent",
        )
        self.assertEqual(
            evaluation[evaluation.index("--agent-api-base") + 1],
            "http://127.0.0.1:30000/v1",
        )

    def test_eval_shell_keeps_legacy_custom_generate_transport(self):
        completed, invocations = self._run_eval_shell(
            overrides={
                "AGENT_EVAL_MODE": "legacy-custom",
                "AGENT_PROTOCOL_PROFILE": "current-single",
            }
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        launch = next(
            invocation
            for invocation in invocations
            if invocation[:2] == ["-m", "sglang.launch_server"]
        )
        self.assertNotIn("--served-model-name", launch)
        evaluation = next(
            invocation
            for invocation in invocations
            if invocation and invocation[0].endswith("/run_eval.py")
        )
        self.assertEqual(
            evaluation[evaluation.index("--agent-eval-mode") + 1],
            "legacy-custom",
        )
        self.assertEqual(
            evaluation[evaluation.index("--agent") + 1],
            "slime_sglang_agent",
        )
        self.assertEqual(
            evaluation[evaluation.index("--agent-api-base") + 1],
            "http://127.0.0.1:30000/generate",
        )

    def test_official_mode_rejects_custom_protocol_profile(self):
        argv = [
            "run_eval.py",
            "--domains",
            "airline",
            "--save-prefix",
            "test",
            "--agent-llm",
            "test-agent",
            "--user-llm",
            "test-user",
            "--agent-protocol-profile",
            "current-single",
        ]
        stderr = io.StringIO()

        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(
                run_eval,
                "_run_domain_jobs",
                return_value=(self._empty_eval_jobs(), []),
            ):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        run_eval.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "official-native does not allow --agent-protocol-profile",
            stderr.getvalue(),
        )

    def test_summary_records_official_native_agent_semantics(self):
        summary = self._run_main_and_read_summary()

        self.assertEqual(summary["agent_eval_mode"], "official-native")
        self.assertEqual(summary["agent"], "llm_agent")
        self.assertIsNone(summary["agent_protocol_profile"])

    def test_summary_records_legacy_custom_agent_semantics(self):
        summary = self._run_main_and_read_summary(
            "--agent-eval-mode",
            "legacy-custom",
            "--agent",
            "slime_sglang_agent",
            "--agent-protocol-profile",
            "strict-single-v1",
        )

        self.assertEqual(summary["agent_eval_mode"], "legacy-custom")
        self.assertEqual(summary["agent"], "slime_sglang_agent")
        self.assertEqual(summary["agent_protocol_profile"], "strict-single-v1")

    def test_historical_custom_agent_wrappers_pin_legacy_mode(self):
        wrappers = [
            *sorted(
                (OFFICIAL_EVAL_DIR / "models").glob(
                    "run_full_tau2_agent*.sh"
                )
            ),
            *sorted(
                (OFFICIAL_EVAL_DIR / "models").glob("run_full_this_sft*.sh")
            ),
            OFFICIAL_EVAL_DIR
            / "models/run_full_qwen3-4b-tau2-grpo-v1.sh",
            OFFICIAL_EVAL_DIR
            / "models/run_full_qwen3-4b-tau2-grpo-v1_user_sft.sh",
            OFFICIAL_EVAL_DIR
            / "models/run_full_qwen3-4b-instruct-2507_dependency_safe_multi_user_stop_parser.sh",
            OFFICIAL_EVAL_DIR
            / "models/run_full_qwen3.5-4b_nonthinking_single_call_user_stop_parser.sh",
        ]

        for wrapper in wrappers:
            with self.subTest(wrapper=wrapper.name):
                self.assertIn(
                    'export AGENT_EVAL_MODE="legacy-custom"',
                    wrapper.read_text(encoding="utf-8"),
                )

    def test_qwen35_official_wrapper_selects_qwen3_coder_parser(self):
        wrapper = OFFICIAL_EVAL_DIR / "models/run_full_qwen3.5-4b.sh"

        captured = self._capture_wrapper_environment(wrapper)

        self.assertEqual(captured["AGENT_TOOL_CALL_PARSER"], "qwen3_coder")

    def test_async_wrapper_exports_elastic_two_agent_three_user_topology(self):
        wrapper = (
            OFFICIAL_EVAL_DIR
            / "models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            capture_path = tmp_path / "environment.txt"
            bash_stub = tmp_path / "bash"
            bash_stub.write_text(
                "#!/bin/sh\nenv > \"${ENV_CAPTURE_PATH}\"\n",
                encoding="utf-8",
            )
            bash_stub.chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{tmp_path}:{environment['PATH']}",
                    "ENV_CAPTURE_PATH": str(capture_path),
                    "RUN_STAMP": "topology-test",
                }
            )

            subprocess.run(
                ["/bin/bash", str(wrapper)],
                cwd=OFFICIAL_EVAL_DIR.parents[3],
                env=environment,
                check=True,
            )

            captured = dict(
                line.split("=", 1)
                for line in capture_path.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )

        self.assertEqual(captured["AGENT_REPLICA_CUDA_GROUPS"], "0;1")
        self.assertEqual(
            captured["USER_REPLICA_CUDA_GROUPS"],
            "2,3;4,5;6,7",
        )
        self.assertEqual(
            captured["DOMAIN_CONCURRENCY"],
            "airline:2,retail:2,telecom:5",
        )
        self.assertEqual(captured["GLOBAL_CONCURRENCY"], "9")
        self.assertEqual(captured["BORROW_COMPLETED_DOMAIN_SLOTS"], "1")
        self.assertEqual(captured["AGENT_ROUTER_POLICY"], "cache_aware")
        self.assertEqual(captured["USER_ROUTER_POLICY"], "round_robin")

    def test_qwen36_user_uses_qwen3_coder_tool_call_parser(self):
        wrapper = (
            OFFICIAL_EVAL_DIR
            / "models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
        )

        captured = self._capture_wrapper_environment(wrapper)

        user_args = captured["USER_SGLANG_EXTRA_ARGS"].split()
        parser_index = user_args.index("--tool-call-parser")
        self.assertEqual(user_args[parser_index + 1], "qwen3_coder")

    def test_qwen36_user_is_nonthinking_and_text_only(self):
        wrapper = (
            OFFICIAL_EVAL_DIR
            / "models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
        )

        captured = self._capture_wrapper_environment(wrapper)

        self.assertEqual(
            captured["USER_EXTRA_BODY_JSON"],
            '{"chat_template_kwargs":{"enable_thinking":false}}',
        )
        self.assertIn(
            "--language-only",
            captured["USER_SGLANG_EXTRA_ARGS"].split(),
        )

    def test_async_wrapper_preserves_explicit_domain_overrides(self):
        wrapper = (
            OFFICIAL_EVAL_DIR
            / "models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
        )
        captured = self._capture_wrapper_environment(
            wrapper,
            overrides={
                "AGENT_MAX_TOKENS": "8192",
                "DOMAINS": "airline,retail,telecom,banking_knowledge",
                "DOMAIN_CONCURRENCY": (
                    "airline:1,retail:2,telecom:2,banking_knowledge:4"
                ),
                "EXPERIMENT_DIR": "/tmp/four-domain-results",
                "MODEL_NAME": "Qwen3.5-4B-test",
                "MODEL_PATH": "/tmp/Qwen3.5-4B",
            },
        )

        self.assertEqual(
            captured["DOMAINS"],
            "airline,retail,telecom,banking_knowledge",
        )
        self.assertEqual(
            captured["DOMAIN_CONCURRENCY"],
            "airline:1,retail:2,telecom:2,banking_knowledge:4",
        )
        self.assertEqual(captured["EXPERIMENT_DIR"], "/tmp/four-domain-results")
        self.assertEqual(captured["MODEL_PATH"], "/tmp/Qwen3.5-4B")
        self.assertEqual(captured["MODEL_NAME"], "Qwen3.5-4B-test")
        self.assertEqual(captured["AGENT_MAX_TOKENS"], "8192")

    def test_four_domain_wrapper_exports_smoke_budget(self):
        wrapper = (
            OFFICIAL_EVAL_DIR
            / "models/run_qwen3_4b_qwen36_user_async_four_domain.sh"
        )
        captured = self._capture_wrapper_environment(
            wrapper,
            arguments=["smoke"],
        )

        self.assertEqual(
            captured["DOMAINS"],
            "airline,retail,telecom,banking_knowledge",
        )
        self.assertEqual(
            captured["DOMAIN_CONCURRENCY"],
            "airline:1,retail:2,telecom:2,banking_knowledge:4",
        )
        self.assertEqual(captured["GLOBAL_CONCURRENCY"], "9")
        self.assertEqual(captured["RETRIEVAL_CONFIG"], "bm25")
        self.assertEqual(captured["NUM_TASKS"], "3")
        self.assertEqual(captured["NUM_TRIALS"], "1")
        self.assertEqual(captured["EVAL_LABEL"], "four-domain-smoke-bm25")

    def test_four_domain_wrapper_exports_full_budget(self):
        wrapper = (
            OFFICIAL_EVAL_DIR
            / "models/run_qwen3_4b_qwen36_user_async_four_domain.sh"
        )
        captured = self._capture_wrapper_environment(
            wrapper,
            arguments=["full"],
        )

        self.assertEqual(captured["NUM_TASKS"], "")
        self.assertEqual(captured["NUM_TRIALS"], "4")
        self.assertEqual(captured["EVAL_LABEL"], "four-domain-full-bm25")

    def test_qwen35_four_domain_wrapper_selects_agent_thinking_mode(self):
        wrapper = (
            OFFICIAL_EVAL_DIR
            / "models/run_qwen3_5_4b_qwen36_user_async_four_domain.sh"
        )
        model_path = OFFICIAL_EVAL_DIR.parents[4] / "models/Qwen3.5-4B"

        for mode, enable_thinking in (
            ("thinking", True),
            ("nonthinking", False),
        ):
            with self.subTest(mode=mode):
                captured = self._capture_wrapper_environment(
                    wrapper,
                    arguments=[mode],
                )

                self.assertEqual(captured["MODEL_PATH"], str(model_path))
                self.assertEqual(
                    captured["MODEL_NAME"],
                    f"Qwen3.5-4B-{mode}-qwen36-user-async",
                )
                self.assertEqual(captured["AGENT_EVAL_MODE"], "official-native")
                self.assertEqual(
                    captured["AGENT_TOOL_CALL_PARSER"],
                    "qwen3_coder",
                )
                self.assertEqual(captured["AGENT_MAX_TOKENS"], "8192")
                self.assertEqual(
                    json.loads(captured["AGENT_EXTRA_BODY_JSON"]),
                    {
                        "chat_template_kwargs": {
                            "enable_thinking": enable_thinking,
                        }
                    },
                )
                self.assertEqual(
                    captured["EVAL_LABEL"],
                    f"four-domain-full-bm25-{mode}",
                )

    def test_tau2_batch_runner_respects_external_slot_semaphore(self):
        import tau2.runner.batch as batch_runner
        from tau2.data_model.simulation import (
            SimulationRun,
            TerminationReason,
            TextRunConfig,
        )
        from tau2.runner.helpers import get_tasks

        slot_semaphore = threading.Semaphore(2)
        first_pair_started = threading.Barrier(2)
        state_lock = threading.Lock()
        state = {"active": 0, "max_active": 0, "call_index": 0}

        def fake_run_single_task(config, task, *, seed=None, **kwargs):
            with state_lock:
                index = state["call_index"]
                state["call_index"] += 1
                state["active"] += 1
                state["max_active"] = max(
                    state["max_active"], state["active"]
                )
            try:
                if index < 2:
                    first_pair_started.wait(timeout=1)
                return SimulationRun(
                    id=f"simulation-{task.id}",
                    task_id=task.id,
                    start_time="2026-09-01T00:00:00Z",
                    end_time="2026-09-01T00:00:00Z",
                    duration=0.0,
                    termination_reason=TerminationReason.USER_STOP,
                    messages=[],
                    seed=seed,
                )
            finally:
                with state_lock:
                    state["active"] -= 1

        config = TextRunConfig(
            domain="mock",
            agent="llm_agent",
            user="user_simulator",
            llm_agent="unused-agent",
            llm_args_agent={},
            llm_user="unused-user",
            llm_args_user={},
            num_trials=1,
            max_steps=5,
            max_errors=1,
            max_concurrency=1,
            max_retries=0,
        )
        base_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        tasks = [
            base_task.model_copy(update={"id": f"slot-test-{index}"})
            for index in range(3)
        ]

        with mock.patch.object(
            batch_runner,
            "run_single_task",
            side_effect=fake_run_single_task,
        ):
            results = batch_runner.run_tasks(
                config,
                tasks,
                console_display=False,
                executor_max_workers=3,
                slot_semaphore=slot_semaphore,
            )

        self.assertEqual(len(results.simulations), 3)
        self.assertEqual(state["max_active"], 2)

    def test_parallel_domain_jobs_run_outside_parent_and_preserve_order(self):
        if not hasattr(run_eval, "_run_domain_jobs"):
            self.fail("run_eval._run_domain_jobs is not implemented")

        domains = ["telecom", "airline", "retail"]
        results, events = run_eval._run_domain_jobs(
            domains,
            {"parent_pid": os.getpid()},
            parallel=True,
            evaluator=evaluate_domain,
            domain_concurrency={"telecom": 5, "airline": 2, "retail": 2},
            global_concurrency=9,
            borrow_completed_slots=False,
        )

        self.assertEqual(list(results), domains)
        self.assertEqual(
            [result["domain"] for result in results.values()],
            domains,
        )
        self.assertTrue(
            all(result["pid"] != os.getpid() for result in results.values())
        )
        self.assertEqual(
            [result["domain_concurrency"] for result in results.values()],
            [5, 2, 2],
        )
        self.assertEqual(events, [])

    def test_completed_domains_redistribute_slots_and_preserve_global_limit(self):
        domains = ["airline", "retail", "telecom"]
        results, events = run_eval._run_domain_jobs(
            domains,
            {
                "parent_pid": os.getpid(),
                "domain_delays": {
                    "airline": 0.0,
                    "retail": 0.1,
                    "telecom": 0.2,
                },
            },
            parallel=True,
            evaluator=evaluate_domain,
            domain_concurrency={"airline": 2, "retail": 2, "telecom": 5},
            global_concurrency=9,
            borrow_completed_slots=True,
        )

        self.assertEqual(list(results), domains)
        self.assertEqual(
            [event["completed_domain"] for event in events],
            ["airline", "retail"],
        )
        self.assertEqual(
            events[0]["domain_concurrency"],
            {"airline": 0, "retail": 3, "telecom": 6},
        )
        self.assertEqual(
            events[1]["domain_concurrency"],
            {"airline": 0, "retail": 0, "telecom": 9},
        )
        self.assertTrue(
            all(
                sum(event["domain_concurrency"].values()) == 9
                for event in events
            )
        )

    def test_released_slot_wakes_a_blocked_domain_worker(self):
        results, events = run_eval._run_domain_jobs(
            ["airline", "retail"],
            {"parent_pid": os.getpid()},
            parallel=True,
            evaluator=evaluate_borrowing,
            domain_concurrency={"airline": 1, "retail": 1},
            global_concurrency=2,
            borrow_completed_slots=True,
        )

        self.assertEqual(results["retail"]["max_active"], 2)
        self.assertEqual(
            events[0]["domain_concurrency"],
            {"airline": 0, "retail": 2},
        )

    def test_domain_failure_propagates_instead_of_becoming_completion(self):
        with self.assertRaisesRegex(RuntimeError, "fixture failure: airline"):
            run_eval._run_domain_jobs(
                ["airline", "retail"],
                {"parent_pid": os.getpid(), "fail_domain": "airline"},
                parallel=True,
                evaluator=evaluate_domain,
                domain_concurrency={"airline": 1, "retail": 1},
                global_concurrency=2,
                borrow_completed_slots=True,
            )

    def test_parallel_domains_cli_flag_is_opt_in(self):
        parser = run_eval._build_parser()
        required = [
            "--save-prefix",
            "test",
            "--agent-llm",
            "agent",
            "--user-llm",
            "user",
        ]

        default_args = parser.parse_args(required)
        if not hasattr(default_args, "parallel_domains"):
            self.fail("run_eval --parallel-domains is not implemented")
        self.assertFalse(default_args.parallel_domains)
        self.assertTrue(parser.parse_args([*required, "--parallel-domains"]).parallel_domains)

    def test_domain_concurrency_configuration_is_opt_in(self):
        parser = run_eval._build_parser()
        required = [
            "--save-prefix",
            "test",
            "--agent-llm",
            "agent",
            "--user-llm",
            "user",
        ]

        default_args = parser.parse_args(required)
        self.assertIsNone(default_args.domain_concurrency)
        self.assertIsNone(default_args.global_concurrency)
        self.assertFalse(default_args.borrow_completed_domain_slots)

        configured = parser.parse_args(
            [
                *required,
                "--domain-concurrency",
                "airline:2,retail:2,telecom:5",
                "--global-concurrency",
                "9",
                "--borrow-completed-domain-slots",
            ]
        )
        self.assertEqual(
            configured.domain_concurrency,
            "airline:2,retail:2,telecom:5",
        )
        self.assertEqual(configured.global_concurrency, 9)
        self.assertTrue(configured.borrow_completed_domain_slots)

    def test_banking_retrieval_configuration_is_explicit(self):
        parser = run_eval._build_parser()
        required = [
            "--save-prefix",
            "test",
            "--agent-llm",
            "agent",
            "--user-llm",
            "user",
        ]

        default_args = parser.parse_args(required)
        self.assertTrue(hasattr(default_args, "retrieval_config"))
        self.assertIsNone(default_args.retrieval_config)
        self.assertIsNone(default_args.retrieval_config_kwargs_json)

        configured = parser.parse_args(
            [
                *required,
                "--domains",
                "airline,retail,telecom,banking_knowledge",
                "--retrieval-config",
                "bm25",
                "--retrieval-config-kwargs-json",
                '{"top_k": 8}',
            ]
        )
        self.assertEqual(configured.retrieval_config, "bm25")
        self.assertEqual(
            configured.retrieval_config_kwargs_json,
            '{"top_k": 8}',
        )

    def test_banking_domain_requires_retrieval_config_before_evaluation(self):
        argv = [
            "run_eval.py",
            "--domains",
            "banking_knowledge",
            "--save-prefix",
            "test",
            "--agent-llm",
            "agent",
            "--user-llm",
            "user",
        ]
        domain_summary = {
            "pass_metrics": {
                "tasks": 0,
                "simulations": 0,
                "pass_at_1": 0.0,
                "pass_at_4_any": 0.0,
                "pass_power_4": 0.0,
            },
            "metrics": {},
            "single_call": {},
            "timing": {
                "wall_seconds": 0.0,
                "trajectory_seconds": 0.0,
                "other": {"seconds": 0.0},
            },
        }
        jobs = {
            "banking_knowledge": {
                "summary": domain_summary,
                "timing_samples": {"agent": [], "user": []},
            }
        }

        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(
                run_eval,
                "_run_domain_jobs",
                return_value=(jobs, []),
            ):
                with self.assertRaises(SystemExit) as raised:
                    run_eval.main()

        self.assertEqual(raised.exception.code, 2)

    def test_banking_retrieval_config_reaches_environment_and_run_config(self):
        captured_environment_kwargs = {}
        captured_run_config = {}

        class FakeEnvironment:
            def get_user_tools(self):
                return []

        class FakeRegistry:
            def get_env_constructor(self, domain):
                self.domain = domain

                def constructor(**kwargs):
                    captured_environment_kwargs.update(kwargs)
                    return FakeEnvironment()

                return constructor

        def fake_text_run_config(**kwargs):
            captured_run_config.update(kwargs)
            return SimpleNamespace()

        def fake_run_domain(config):
            return {"simulations": []}

        sglang_agent = ModuleType("sglang_agent")
        register_slime_sglang_agent = mock.Mock()
        sglang_agent.register_slime_sglang_agent = register_slime_sglang_agent
        timed_user = ModuleType("timed_user")
        timed_user.register_timed_user_simulator = lambda: None
        tau2 = ModuleType("tau2")
        tau2.__path__ = []
        tau2_data_model = ModuleType("tau2.data_model")
        tau2_data_model.__path__ = []
        tau2_simulation = ModuleType("tau2.data_model.simulation")
        tau2_simulation.TextRunConfig = fake_text_run_config
        tau2_metrics = ModuleType("tau2.metrics")
        tau2_metrics.__path__ = []
        tau2_agent_metrics = ModuleType("tau2.metrics.agent_metrics")
        tau2_agent_metrics.compute_metrics = lambda results: {}
        tau2_registry = ModuleType("tau2.registry")
        tau2_registry.registry = FakeRegistry()
        tau2_runner = ModuleType("tau2.runner")
        tau2_runner.run_domain = fake_run_domain
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = object

        parser = run_eval._build_parser()
        args = parser.parse_args(
            [
                "--domains",
                "banking_knowledge",
                "--save-prefix",
                "test",
                "--agent-llm",
                "agent",
                "--user-llm",
                "user",
                "--retrieval-config",
                "bm25",
                "--retrieval-config-kwargs-json",
                '{"top_k": 8}',
            ]
        )
        payload = {
            "args": vars(args),
            "agent_args": {},
            "user_args": {},
            "selected_protocol_profile": None,
            "retrieval_config": "bm25",
            "retrieval_config_kwargs": {"top_k": 8},
        }

        with mock.patch.dict(
            sys.modules,
            {
                "sglang_agent": sglang_agent,
                "timed_user": timed_user,
                "tau2": tau2,
                "tau2.data_model": tau2_data_model,
                "tau2.data_model.simulation": tau2_simulation,
                "tau2.metrics": tau2_metrics,
                "tau2.metrics.agent_metrics": tau2_agent_metrics,
                "tau2.registry": tau2_registry,
                "tau2.runner": tau2_runner,
                "transformers": transformers,
            },
        ):
            with mock.patch.object(
                run_eval,
                "analyze_namespace_trajectories",
                return_value={},
            ):
                with mock.patch.object(
                    run_eval,
                    "analyze_single_call_attempts",
                    return_value={},
                ):
                    result = run_eval._evaluate_domain(
                        payload,
                        "banking_knowledge",
                    )

        self.assertEqual(
            captured_environment_kwargs,
            {
                "retrieval_variant": "bm25",
                "retrieval_kwargs": {"top_k": 8},
            },
        )
        self.assertEqual(captured_run_config["retrieval_config"], "bm25")
        self.assertEqual(
            captured_run_config["retrieval_config_kwargs"],
            {"top_k": 8},
        )
        self.assertEqual(result["summary"]["retrieval_config"], "bm25")
        self.assertEqual(
            result["summary"]["retrieval_config_kwargs"],
            {"top_k": 8},
        )
        register_slime_sglang_agent.assert_not_called()

    def test_explicit_zero_global_concurrency_is_rejected(self):
        argv = [
            "run_eval.py",
            "--save-prefix",
            "test",
            "--agent-llm",
            "agent",
            "--user-llm",
            "user",
            "--global-concurrency",
            "0",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as raised:
                run_eval.main()

        self.assertEqual(raised.exception.code, 2)

    def test_domain_concurrency_parser_preserves_selected_domain_order(self):
        self.assertEqual(
            run_eval._parse_domain_concurrency(
                "airline:2,retail:2,telecom:5",
                ["telecom", "airline", "retail"],
                default_concurrency=3,
            ),
            {"telecom": 5, "airline": 2, "retail": 2},
        )
        self.assertEqual(
            run_eval._parse_domain_concurrency(
                None,
                ["airline", "retail", "telecom"],
                default_concurrency=3,
            ),
            {"airline": 3, "retail": 3, "telecom": 3},
        )
        with self.assertRaisesRegex(ValueError, "missing.*telecom"):
            run_eval._parse_domain_concurrency(
                "airline:2,retail:2",
                ["airline", "retail", "telecom"],
                default_concurrency=3,
            )

    def test_released_slots_use_remaining_domain_weights(self):
        initial = {"airline": 2, "retail": 2, "telecom": 5}
        self.assertEqual(
            run_eval._allocate_released_slots(
                2,
                ["retail", "telecom"],
                initial,
            ),
            {"retail": 1, "telecom": 1},
        )
        self.assertEqual(
            run_eval._allocate_released_slots(3, ["telecom"], initial),
            {"telecom": 3},
        )
        self.assertEqual(
            run_eval._allocate_released_slots(5, [], initial),
            {},
        )

    def _model_request_module(self):
        try:
            import model_request
        except ModuleNotFoundError:
            self.fail("official eval model_request module is not implemented")
        return model_request

    def test_agent_sampling_params_map_trial_seed_to_sglang_sampling_seed(self):
        model_request = self._model_request_module()

        params = model_request.build_sglang_sampling_params(
            {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_tokens": 1200,
                "seed": 314159,
                "frequency_penalty": 0.2,
                "extra_body": {"top_k": 20},
            },
            eos_token="<eos>",
        )

        self.assertEqual(
            params,
            {
                "temperature": 0.6,
                "top_p": 0.9,
                "max_new_tokens": 1200,
                "presence_penalty": 0.0,
                "stop": ["<eos>"],
                "sampling_seed": 314159,
                "frequency_penalty": 0.2,
                "top_k": 20,
            },
        )

    def test_request_timing_preserves_existing_raw_data(self):
        model_request = self._model_request_module()

        raw_data = model_request.with_request_timing(
            {"text": "response", "provider": "sglang"},
            participant="agent",
            elapsed_seconds=1.25,
        )

        self.assertEqual(
            raw_data,
            {
                "text": "response",
                "provider": "sglang",
                "tau2_eval_timing": {
                    "participant": "agent",
                    "elapsed_seconds": 1.25,
                },
            },
        )

    def test_timed_user_preserves_response_and_records_user_latency(self):
        module_path = OFFICIAL_EVAL_DIR / "timed_user.py"
        if not module_path.exists():
            self.fail("official eval timed_user.py is not implemented")

        class FakeUserSimulator:
            def _generate_next_message(self, message, state):
                return SimpleNamespace(raw_data={"provider": "litellm"})

        class FakeRegistry:
            def __init__(self):
                self.users = {}

            def get_users(self):
                return list(self.users)

            def register_user(self, constructor, name):
                self.users[name] = constructor

        tau2_registry = ModuleType("tau2.registry")
        tau2_registry.registry = FakeRegistry()
        tau2_user = ModuleType("tau2.user.user_simulator")
        tau2_user.UserSimulator = FakeUserSimulator
        spec = importlib.util.spec_from_file_location("timed_user_test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {
                "tau2.registry": tau2_registry,
                "tau2.user.user_simulator": tau2_user,
            },
        ):
            spec.loader.exec_module(module)

        response = module.TimedUserSimulator()._generate_next_message(None, None)

        self.assertEqual(response.raw_data["provider"], "litellm")
        self.assertEqual(
            response.raw_data["tau2_eval_timing"]["participant"], "user"
        )
        self.assertGreaterEqual(
            response.raw_data["tau2_eval_timing"]["elapsed_seconds"], 0.0
        )
        module.register_timed_user_simulator()
        module.register_timed_user_simulator()
        self.assertEqual(
            tau2_registry.registry.users,
            {module.USER_NAME: module.TimedUserSimulator},
        )

    def test_timing_summary_separates_agent_user_and_other_time(self):
        if not hasattr(run_eval, "_timing_summary"):
            self.fail("run_eval._timing_summary is not implemented")

        results = {
            "simulations": [
                {
                    "duration": 10.0,
                    "messages": [
                        {
                            "role": "assistant",
                            "raw_data": {
                                "tau2_eval_timing": {
                                    "participant": "agent",
                                    "elapsed_seconds": 2.0,
                                }
                            },
                        },
                        {
                            "role": "user",
                            "raw_data": {
                                "tau2_eval_timing": {
                                    "participant": "user",
                                    "elapsed_seconds": 3.0,
                                }
                            },
                        },
                    ],
                },
                {
                    "duration": 6.0,
                    "messages": [
                        {
                            "role": "assistant",
                            "raw_data": {
                                "tau2_eval_timing": {
                                    "participant": "agent",
                                    "elapsed_seconds": 1.0,
                                }
                            },
                        }
                    ],
                },
            ]
        }

        summary, samples = run_eval._timing_summary(results, wall_seconds=8.0)

        self.assertEqual(samples, {"agent": [2.0, 1.0], "user": [3.0]})
        self.assertEqual(
            summary,
            {
                "wall_seconds": 8.0,
                "trajectory_seconds": 16.0,
                "parallelism_factor": 2.0,
                "agent": {
                    "calls": 2,
                    "seconds": 3.0,
                    "mean_seconds": 1.5,
                    "p50_seconds": 1.5,
                    "p95_seconds": 1.95,
                    "trajectory_share": 0.1875,
                },
                "user": {
                    "calls": 1,
                    "seconds": 3.0,
                    "mean_seconds": 3.0,
                    "p50_seconds": 3.0,
                    "p95_seconds": 3.0,
                    "trajectory_share": 0.1875,
                },
                "other": {
                    "seconds": 10.0,
                    "trajectory_share": 0.625,
                },
            },
        )

    def test_timing_summary_reads_official_agent_generation_time(self):
        results = {
            "simulations": [
                {
                    "duration": 5.0,
                    "messages": [
                        {
                            "role": "assistant",
                            "generation_time_seconds": 1.25,
                            "raw_data": {"provider": "litellm"},
                        },
                        {
                            "role": "user",
                            "generation_time_seconds": 9.0,
                            "raw_data": {
                                "tau2_eval_timing": {
                                    "participant": "user",
                                    "elapsed_seconds": 2.0,
                                }
                            },
                        },
                    ],
                }
            ]
        }

        summary, samples = run_eval._timing_summary(results, wall_seconds=5.0)

        self.assertEqual(samples, {"agent": [1.25], "user": [2.0]})
        self.assertEqual(summary["agent"]["seconds"], 1.25)
        self.assertEqual(summary["user"]["seconds"], 2.0)
        self.assertEqual(summary["other"]["seconds"], 1.75)

    def test_combined_timing_recomputes_percentiles_from_all_domains(self):
        if not hasattr(run_eval, "_combined_timing_summary"):
            self.fail("run_eval._combined_timing_summary is not implemented")

        domain_timings = {
            "airline": {
                "trajectory_seconds": 10.0,
                "other": {"seconds": 4.0},
            },
            "retail": {
                "trajectory_seconds": 20.0,
                "other": {"seconds": 5.0},
            },
        }
        domain_samples = {
            "airline": {"agent": [1.0, 2.0], "user": [3.0]},
            "retail": {"agent": [4.0], "user": [5.0, 6.0]},
        }

        summary = run_eval._combined_timing_summary(
            domain_timings,
            domain_samples,
            wall_seconds=15.0,
        )

        self.assertEqual(summary["wall_seconds"], 15.0)
        self.assertEqual(summary["trajectory_seconds"], 30.0)
        self.assertEqual(summary["parallelism_factor"], 2.0)
        self.assertEqual(
            summary["agent"],
            {
                "calls": 3,
                "seconds": 7.0,
                "mean_seconds": 7.0 / 3.0,
                "p50_seconds": 2.0,
                "p95_seconds": 3.8,
                "trajectory_share": 7.0 / 30.0,
            },
        )
        self.assertEqual(
            summary["user"],
            {
                "calls": 3,
                "seconds": 14.0,
                "mean_seconds": 14.0 / 3.0,
                "p50_seconds": 5.0,
                "p95_seconds": 5.9,
                "trajectory_share": 14.0 / 30.0,
            },
        )
        self.assertEqual(
            summary["other"],
            {"seconds": 9.0, "trajectory_share": 0.3},
        )

    def test_overall_summary_preserves_official_weighting_and_adds_timing(self):
        if not hasattr(run_eval, "_overall_summary"):
            self.fail("run_eval._overall_summary is not implemented")

        domains = {
            "airline": {
                "pass_metrics": {
                    "tasks": 1,
                    "simulations": 4,
                    "pass_at_1": 0.25,
                    "pass_at_4_any": 1.0,
                    "pass_power_4": 0.0,
                },
                "metrics": {
                    "total_read_actions": 2,
                    "correct_read_actions": 1,
                    "total_write_actions": 2,
                    "correct_write_actions": 1,
                    "db_match_count": 1,
                    "db_mismatch_count": 1,
                    "termination_max_steps": 1,
                },
                "single_call": {
                    "trajectory_count": 4,
                    "trajectories_with_multi_call_output": 1,
                    "multi_call_output_turns": 2,
                    "attempted_calls_in_multi_call_outputs": 3,
                    "parsed_multi_call_turns": 1,
                    "single_call_protocol_error_turns": 1,
                },
            },
            "retail": {
                "pass_metrics": {
                    "tasks": 2,
                    "simulations": 8,
                    "pass_at_1": 0.5,
                    "pass_at_4_any": 0.5,
                    "pass_power_4": 0.5,
                },
                "metrics": {
                    "total_read_actions": 3,
                    "correct_read_actions": 3,
                    "total_write_actions": 1,
                    "correct_write_actions": 0,
                    "db_match_count": 2,
                    "db_mismatch_count": 0,
                    "termination_max_steps": 2,
                },
                "single_call": {
                    "trajectory_count": 8,
                    "trajectories_with_multi_call_output": 0,
                    "multi_call_output_turns": 0,
                    "attempted_calls_in_multi_call_outputs": 0,
                    "parsed_multi_call_turns": 0,
                    "single_call_protocol_error_turns": 0,
                },
            },
        }
        timing = {"wall_seconds": 12.0}

        overall = run_eval._overall_summary(domains, timing=timing)

        self.assertEqual(overall["tasks"], 3)
        self.assertEqual(overall["simulations"], 12)
        self.assertAlmostEqual(overall["pass_at_1"], 5.0 / 12.0)
        self.assertAlmostEqual(overall["pass_at_4_any"], 2.0 / 3.0)
        self.assertAlmostEqual(overall["pass_power_4"], 1.0 / 3.0)
        self.assertEqual(overall["single_call"]["trajectory_count"], 12)
        self.assertEqual(
            overall["single_call"]["attempted_calls_in_multi_call_outputs"], 3
        )
        self.assertEqual(overall["diagnostics"]["action_accuracy"], 5.0 / 8.0)
        self.assertEqual(overall["diagnostics"]["db_accuracy"], 3.0 / 4.0)
        self.assertEqual(overall["diagnostics"]["max_steps"], 3)
        self.assertIs(overall["timing"], timing)

    def test_infrastructure_errors_make_the_eval_process_fail(self):
        domains = {
            "airline": {"metrics": {"infra_error_count": 2}},
            "retail": {"metrics": {"infra_error_count": 0}},
            "telecom": {"metrics": {"infra_error_count": 1}},
        }

        with self.assertRaisesRegex(RuntimeError, "3 infrastructure errors"):
            run_eval._require_no_infrastructure_errors(domains)


if __name__ == "__main__":
    unittest.main()
