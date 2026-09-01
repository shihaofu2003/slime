import sys
import importlib.util
import os
import subprocess
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
