from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


NUM_GPUS = 0

ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "examples/tau2-bench/sft"
ANALYSIS_DIR = ROOT / "examples/tau2-bench/analysis"
SHARED_DIR = ROOT / "examples/tau2-bench/shared"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SFT_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SHARED_DIR))

from build_raw_all_sft import _domain, build_dataset  # noqa: E402
from audit_raw_processed_sft import (  # noqa: E402
    build_paired_eval,
    call_parts,
    has_complete_prefix_pairing,
    paired_bootstrap_ci,
)
from summarize_official_native_expanded_sft import load_checkpoint  # noqa: E402
from summarize_raw_processed_vanilla_grpo import compact_training  # noqa: E402
from slime.rollout import prompt_answer_sft_rollout  # noqa: E402
from protocol_profiles import (  # noqa: E402
    PROTOCOL_OFFICIAL_NATIVE,
    domain_policy_for_profile,
    protocol_signature,
)


class RecordingMaskGenerator:
    def __init__(self):
        self.calls = []

    def get_loss_mask(self, messages, tools=None):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        token_count = sum(
            len(str(message.get("content") or ""))
            + 3 * len(message.get("tool_calls") or [])
            for message in messages
        )
        token_count = max(token_count, 2)
        return list(range(token_count)), [0] * (token_count - 2) + [1, 1]

    def get_response_lengths(self, masks):
        return [len(mask) - mask.index(1) if 1 in mask else 0 for mask in masks]


def source_row(domain, content, *, correct, reward, tool_calls=None):
    return {
        "messages": [
            {"role": "system", "content": "S"},
            {
                "role": "assistant",
                "content": "H",
                "reasoning": "historical hidden reasoning",
                "tool_calls": [
                    {"name": "history_call", "arguments": {"value": 1}},
                    {"name": "history_call_two", "arguments": {"value": 2}},
                ],
            },
        ],
        "answer": {
            "role": "assistant",
            "content": content,
            "thinking": "target hidden thinking",
            "tool_calls": tool_calls or [],
        },
        "metadata": {
            "source_dialog_id": f"{domain}_dialog_fixture",
            "correct": correct,
            "reward": reward,
        },
        "untouched_top_level": {"source": "raw"},
    }


class RawAllDataTest(unittest.TestCase):
    def test_domain_falls_back_to_the_unchanged_system_policy(self):
        row = source_row("unknown", "answer", correct=1, reward=1)
        row["metadata"]["source_dialog_id"] = "473"
        row["messages"][0]["content"] = "<policy>\n# Telecom Agent Policy\n</policy>"
        self.assertEqual(_domain(row), "telecom")

    def test_filter_drops_only_over_cap_and_preserves_raw_rows(self):
        rows = [
            source_row("airline", "A", correct=1, reward=1),
            source_row(
                "retail",
                "BBBB",
                correct=0,
                reward=0,
                tool_calls=[
                    {"name": "first", "arguments": {"x": 1}},
                    {"name": "second", "arguments": {"y": 2}},
                ],
            ),
            source_row("telecom", "X" * 30, correct=1, reward=1),
        ]
        rows[0]["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "unchanged_schema",
                    "parameters": {"type": "object"},
                },
            }
        ]
        rows[0]["messages"].extend(
            [
                {
                    "role": "user",
                    "content": "",
                    "tool_calls": [
                        {"name": "device_call", "arguments": {"enabled": True}}
                    ],
                },
                {"role": "tool", "name": "device_call", "content": "device result"},
            ]
        )
        original_rows = copy.deepcopy(rows)
        source_lines = [
            json.dumps(row, ensure_ascii=False, separators=(", ", ": ")) + "\n"
            for row in rows
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "full.jsonl"
            smoke = root / "smoke.jsonl"
            stats = root / "stats.json"
            source.write_text("".join(source_lines), encoding="utf-8")
            generator = RecordingMaskGenerator()

            summary = build_dataset(
                input_path=source,
                output_path=output,
                smoke_output_path=smoke,
                stats_output_path=stats,
                max_tokens=25,
                smoke_size=2,
                mask_generator=generator,
            )

            self.assertEqual(output.read_text(encoding="utf-8"), "".join(source_lines[:2]))
            self.assertEqual(
                {json.loads(line)["metadata"]["source_dialog_id"] for line in smoke.read_text().splitlines()},
                {"airline_dialog_fixture", "retail_dialog_fixture"},
            )
            self.assertEqual(summary["source_rows"], 3)
            self.assertEqual(summary["kept_rows"], 2)
            self.assertEqual(summary["dropped_over_cap"], 1)
            self.assertEqual(summary["kept_labels"], {"double_success": 1, "has_failure": 1})
            self.assertEqual(summary["domains"]["telecom"]["dropped_over_cap"], 1)
            self.assertEqual(json.loads(stats.read_text()), summary)

        self.assertEqual(rows, original_rows)
        self.assertEqual(generator.calls[0][1], rows[0]["tools"])
        rendered_messages, rendered_tools = generator.calls[1]
        self.assertIsNone(rendered_tools)
        self.assertEqual(rendered_messages[-1]["content"], "BBBB")
        self.assertEqual(rendered_messages[-1]["tool_calls"], rows[1]["answer"]["tool_calls"])
        self.assertEqual(rendered_messages[-1]["step_loss_mask"], 1)
        self.assertTrue(
            all(message["step_loss_mask"] == 0 for message in rendered_messages[:-1])
        )
        self.assertNotIn("thinking", rendered_messages[-1])
        self.assertNotIn("reasoning", rendered_messages[1])

    def test_prompt_answer_rollout_masks_history_and_keeps_answer_shape(self):
        generator = RecordingMaskGenerator()
        sample = SimpleNamespace(
            prompt=[
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "old",
                    "thinking": "hidden old thought",
                    "tool_calls": [
                        {"name": "old_call", "arguments": {"value": 1}}
                    ],
                },
            ],
            label={
                "role": "assistant",
                "content": "final",
                "reasoning": "hidden target thought",
                "tool_calls": [
                    {"name": "one", "arguments": {"x": 1}},
                    {"name": "two", "arguments": {"y": 2}},
                ],
            },
            metadata={},
        )

        class Buffer:
            def get_samples(self, count):
                self.count = count
                return [[sample]]

        old_state = (
            prompt_answer_sft_rollout.TOKENIZER,
            prompt_answer_sft_rollout.MASK_GENERATOR,
            prompt_answer_sft_rollout.SAMPLE_PRINTED,
        )
        try:
            prompt_answer_sft_rollout.TOKENIZER = object()
            prompt_answer_sft_rollout.MASK_GENERATOR = generator
            prompt_answer_sft_rollout.SAMPLE_PRINTED = False
            groups = prompt_answer_sft_rollout.generate_rollout(
                SimpleNamespace(
                    rollout_global_dataset=True,
                    rollout_batch_size=1,
                    hf_checkpoint="unused",
                    loss_mask_type="qwen3_full",
                ),
                0,
                Buffer(),
            )
        finally:
            (
                prompt_answer_sft_rollout.TOKENIZER,
                prompt_answer_sft_rollout.MASK_GENERATOR,
                prompt_answer_sft_rollout.SAMPLE_PRINTED,
            ) = old_state

        self.assertIs(groups[0][0], sample)
        self.assertGreater(sample.response_length, 0)
        self.assertTrue(all(sample.loss_mask))
        messages, _ = generator.calls[0]
        self.assertEqual([message["step_loss_mask"] for message in messages], [0, 0, 1])
        self.assertNotIn("thinking", messages[1])
        self.assertNotIn("reasoning", messages[2])
        self.assertEqual(messages[2]["content"], "final")
        self.assertEqual(len(messages[2]["tool_calls"]), 2)


class RawAllWorkflowTest(unittest.TestCase):
    def _environment(self, root):
        fake_bin = root / "bin"
        fake_bin.mkdir()
        capture = root / "environment.txt"
        fake_bash = fake_bin / "bash"
        fake_bash.write_text("#!/bin/sh\nenv > \"${ENV_CAPTURE_PATH}\"\n", encoding="utf-8")
        fake_bash.chmod(0o755)
        service_root = root / "service-agent"
        project_root = service_root / "slime"
        environment = dict(os.environ)
        for key in ("CHECKPOINT_ROOT", "MODEL_ROOT", "PROJECT_ROOT", "SERVICE_AGENT_ROOT"):
            environment.pop(key, None)
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "ENV_CAPTURE_PATH": str(capture),
                "SERVICE_AGENT_ROOT": str(service_root),
                "PROJECT_ROOT": str(project_root),
            }
        )
        return service_root, project_root, environment, capture

    def _read_environment(self, path):
        return dict(
            line.split("=", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )

    def test_training_wrapper_pins_smoke_and_full_recipes(self):
        wrapper = SFT_DIR / "run_qwen3_4b_instruct_2507_sft_raw_all_max16384.sh"
        cases = (
            (
                "smoke",
                32,
                "raw_all_max16384_longest32.jsonl",
                "Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_longest32_smoke_20260903",
                "1",
                "2",
            ),
            (
                "full",
                32548,
                "raw_all_max16384.jsonl",
                "Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903",
                "2",
                "4068",
            ),
        )
        for mode, row_count, basename, save_basename, epochs, save_interval in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                service_root, project_root, environment, capture = self._environment(root)
                data_dir = project_root / "output/experiments/tau2-sft-raw-all-max16384/data"
                data_dir.mkdir(parents=True)
                (data_dir / basename).write_text("{}\n" * row_count, encoding="utf-8")

                subprocess.run(["/bin/bash", str(wrapper), mode], env=environment, check=True)
                captured = self._read_environment(capture)

                self.assertEqual(Path(captured["SFT_DATA_PATH"]).name, basename)
                self.assertEqual(Path(captured["SAVE_DIR"]).name, save_basename)
                self.assertEqual(captured["NUM_GPUS"], "8")
                self.assertEqual(captured["ROLLOUT_BATCH_SIZE"], "16")
                self.assertEqual(captured["GLOBAL_BATCH_SIZE"], "16")
                self.assertEqual(captured["NUM_EPOCH"], epochs)
                self.assertEqual(captured["SAVE_INTERVAL"], save_interval)
                self.assertEqual(captured["START_ROLLOUT_ID"], "0")
                self.assertEqual(captured["MAX_TOKENS_PER_GPU"], "16384")
                self.assertEqual(captured["LOSS_MASK_TYPE"], "qwen3_full")
                self.assertEqual(captured["LABEL_KEY"], "answer")
                self.assertEqual(
                    captured["ROLLOUT_FUNCTION_PATH"],
                    "slime.rollout.prompt_answer_sft_rollout.generate_rollout",
                )
                original = service_root / "models/Qwen3-4B-Instruct-2507_torch_dist"
                self.assertEqual(captured["LOAD_DIR"], str(original))
                self.assertEqual(captured["TORCH_DIST_DIR"], str(original))
                self.assertEqual((row_count // 16) * int(epochs), 2 if mode == "smoke" else 4068)

    def test_conversion_maps_only_final_checkpoint(self):
        wrapper = ROOT / "scripts/convert_tau2_agent_sft_raw_all_final_to_hf.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service_root, _, environment, capture = self._environment(root)
            subprocess.run(["/bin/bash", str(wrapper)], env=environment, check=True)
            captured = self._read_environment(capture)
            run_root = service_root / "checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903"
            self.assertEqual(captured["ITER_DIR"], str(run_root / "iter_0004067"))
            self.assertEqual(captured["OUTPUT_DIR"], str(run_root / "final_hf"))

    def test_eval_wrapper_is_final_only_three_domain_seed_300_or_301(self):
        wrapper = (
            ROOT
            / "examples/tau2-bench/eval/official/models/run_tau2_agent_sft_raw_all_final.sh"
        )
        for seed in ("300", "301"):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                service_root, _, environment, capture = self._environment(root)
                final_hf = (
                    service_root
                    / "checkpoints"
                    / "Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903"
                    / "final_hf"
                )
                final_hf.mkdir(parents=True)
                (final_hf / "config.json").write_text("{}\n", encoding="utf-8")

                subprocess.run(["/bin/bash", str(wrapper), seed], env=environment, check=True)
                captured = self._read_environment(capture)
                self.assertEqual(captured["MODEL_PATH"], str(final_hf))
                self.assertEqual(captured["EVAL_LABEL"], "final")
                self.assertEqual(captured["AGENT_EVAL_MODE"], "official-native")
                self.assertEqual(captured["DOMAINS"], "airline,retail,telecom")
                self.assertEqual(captured["NUM_TASKS"], "")
                self.assertEqual(captured["NUM_TRIALS"], "4")
                self.assertEqual(captured["SEED"], seed)
                self.assertEqual(captured["MAX_STEPS"], "200")
                self.assertEqual(captured["AGENT_TEMPERATURE"], "0.6")
                self.assertEqual(captured["AGENT_TOP_P"], "1.0")
                self.assertEqual(captured["AGENT_MAX_TOKENS"], "1200")

    def test_common_runner_exposes_optional_rollout_and_label_interfaces(self):
        runner = (SFT_DIR / "run_qwen3_4b_instruct_2507_sft.sh").read_text(encoding="utf-8")
        self.assertIn(
            'ROLLOUT_FUNCTION_PATH="${ROLLOUT_FUNCTION_PATH:-slime.rollout.sft_rollout.generate_rollout}"',
            runner,
        )
        self.assertIn('LABEL_KEY="${LABEL_KEY:-}"', runner)
        self.assertIn('--rollout-function-path "${ROLLOUT_FUNCTION_PATH}"', runner)
        self.assertIn('SFT_ARGS+=(--label-key "${LABEL_KEY}")', runner)


class RawAllEvaluationValidationTest(unittest.TestCase):
    def _result(self, domain, task_count, *, seed=300):
        simulations = [
            {
                "task_id": f"{domain}-{task}",
                "trial": trial,
                "reward_info": {"reward": float(trial == 0)},
                "termination_reason": "user_stop",
                "messages": [],
            }
            for task in range(task_count)
            for trial in range(4)
        ]
        return {
            "info": {
                "num_trials": 4,
                "max_steps": 200,
                "seed": seed,
                "agent_info": {
                    "implementation": "llm_agent",
                    "llm": "openai/raw-all",
                    "llm_args": {
                        "temperature": 0.6,
                        "top_p": 1.0,
                        "max_tokens": 1200,
                    },
                },
                "user_info": {
                    "implementation": "slime_timed_user_simulator",
                    "llm": "openai/Qwen3.6-27B-tau2-user-nonthinking",
                    "llm_args": {"temperature": 0.0, "max_tokens": 512},
                },
            },
            "simulations": simulations,
        }

    def test_summary_validator_requires_100_tasks_and_unique_400_trajectories(self):
        task_counts = {"airline": 20, "retail": 40, "telecom": 40}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domains = {}
            for domain, task_count in task_counts.items():
                result_path = root / f"{domain}.json"
                result_path.write_text(
                    json.dumps(self._result(domain, task_count)), encoding="utf-8"
                )
                domains[domain] = {
                    "results_file": result_path.name,
                    "pass_metrics": {
                        "tasks": task_count,
                        "simulations": task_count * 4,
                        "pass_at_1": 0.25,
                        "pass_at_4_any": 1.0,
                        "pass_power_4": 0.0,
                    },
                    "metrics": {
                        "infra_error_count": 0,
                        "total_read_actions": 0,
                        "correct_read_actions": 0,
                        "total_write_actions": 0,
                        "correct_write_actions": 0,
                        "db_match_count": 0,
                        "db_mismatch_count": 0,
                    },
                    "diagnostics": {"action_accuracy": 0.5, "db_accuracy": 0.5},
                    "timing": {"wall_seconds": 60.0},
                }
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "task_split_name": "test",
                        "num_trials": 4,
                        "num_tasks": None,
                        "seed": 300,
                        "max_steps": 200,
                        "agent_eval_mode": "official-native",
                        "agent": "llm_agent",
                        "domains": domains,
                        "overall": {
                            "tasks": 100,
                            "simulations": 400,
                            "pass_at_1": 0.25,
                            "pass_at_4_any": 1.0,
                            "pass_power_4": 0.0,
                            "diagnostics": {
                                "action_accuracy": 0.5,
                                "db_accuracy": 0.5,
                                "infrastructure_errors": 0,
                            },
                            "timing": {"wall_seconds": 120.0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            runtimes = {
                domain: SimpleNamespace(agent_tools_by_name={}) for domain in task_counts
            }

            result = load_checkpoint(
                "raw-all-final",
                summary_path,
                results_root=root,
                expected_seed=300,
                runtimes=runtimes,
                expected_agent_llm="openai/raw-all",
            )

            self.assertEqual(result["overall"]["pass_at_1"], 0.25)
            self.assertEqual(result["overall"]["terminations"], {"user_stop": 400})

            airline_path = root / "airline.json"
            airline = json.loads(airline_path.read_text(encoding="utf-8"))
            airline["simulations"][1]["trial"] = 0
            airline_path.write_text(json.dumps(airline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate or malformed task trial"):
                load_checkpoint(
                    "raw-all-final",
                    summary_path,
                    results_root=root,
                    expected_seed=300,
                    runtimes=runtimes,
                    expected_agent_llm="openai/raw-all",
                )


class VanillaGRPOWorkflowTest(unittest.TestCase):
    training_wrapper = (
        ROOT
        / "examples/tau2-bench/rl"
        / "run_qwen3_4b_instruct_2507_tau2_rl_sft_raw_vs_processed_vanilla_grpo.sh"
    )
    converter = ROOT / "scripts/convert_tau2_agent_rl_raw_vs_processed_vanilla_grpo_to_hf.sh"
    evaluator = (
        ROOT
        / "examples/tau2-bench/eval/official/models"
        / "run_tau2_agent_rl_raw_vs_processed_vanilla_grpo.sh"
    )

    sft_roots = {
        "raw-all": "Qwen3-4B-Instruct-2507_tau2_agent_sft_raw_all_max16384_20260903",
        "processed": "Qwen3-4B-Instruct-2507_tau2_agent_sft_official_native_expanded_20260903",
    }
    rl_roots = {
        "raw-all": "Qwen3-4B-Instruct-2507_tau2_agent_rl_raw_all_vanilla_grpo_qwen36_20260903",
        "processed": "Qwen3-4B-Instruct-2507_tau2_agent_rl_official_native_expanded_vanilla_grpo_qwen36_20260903",
    }

    def _environment(self, root):
        fake_bin = root / "bin"
        fake_bin.mkdir()
        capture = root / "environment.txt"
        fake_bash = fake_bin / "bash"
        fake_bash.write_text(
            '#!/bin/sh\nenv > "${ENV_CAPTURE_PATH}"\n',
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)
        service_root = root / "service-agent"
        project_root = service_root / "slime"
        environment = dict(os.environ)
        for key in (
            "CHECKPOINT_ROOT",
            "MODEL_ROOT",
            "PROJECT_ROOT",
            "SERVICE_AGENT_ROOT",
            "WANDB_GROUP",
            "WANDB_PROJECT",
        ):
            environment.pop(key, None)
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "ENV_CAPTURE_PATH": str(capture),
                "SERVICE_AGENT_ROOT": str(service_root),
                "PROJECT_ROOT": str(project_root),
            }
        )
        return service_root, project_root, environment, capture

    def _prepare_sft(self, service_root, arm):
        sft_root = service_root / "checkpoints" / self.sft_roots[arm]
        (sft_root / "final_hf").mkdir(parents=True)
        (sft_root / "final_hf/config.json").write_text("{}\n", encoding="utf-8")
        (sft_root / "latest_checkpointed_iteration.txt").write_text(
            "4067\n" if arm == "raw-all" else "3795\n",
            encoding="utf-8",
        )
        return sft_root

    def _read_environment(self, path):
        return dict(
            line.split("=", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )

    def test_official_native_profile_is_unsigned_and_uses_tau2_base_prompt(self):
        policy = "domain policy unchanged"
        resolved, replaced = domain_policy_for_profile(
            policy,
            PROTOCOL_OFFICIAL_NATIVE,
        )
        self.assertEqual(resolved, policy)
        self.assertFalse(replaced)
        self.assertIsNone(protocol_signature(PROTOCOL_OFFICIAL_NATIVE))

        agent_source = (
            ROOT / "examples/tau2-bench/rl/agent.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "if self.protocol_profile == PROTOCOL_OFFICIAL_NATIVE:\n"
            "            return super().system_prompt",
            agent_source,
        )
        preflight = (
            ROOT / "examples/tau2-bench/rl/test_rollout_logic.py"
        ).read_text(encoding="utf-8")
        self.assertIn('for domain in ("airline", "retail", "telecom"):', preflight)
        self.assertIn("official-native prompt differs from Tau2 runtime", preflight)
        self.assertIn("training view lost native tool results", preflight.replace("{active_profile} ", ""))
        self.assertIn("signature, hash, or summary", preflight)

    def test_training_wrapper_pins_the_paired_vanilla_recipe(self):
        captures = {}
        for arm in ("raw-all", "processed"):
            for mode in ("smoke", "train100"):
                with self.subTest(arm=arm, mode=mode), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    service_root, _, environment, capture = self._environment(root)
                    sft_root = self._prepare_sft(service_root, arm)
                    subprocess.run(
                        ["/bin/bash", str(self.training_wrapper), arm, mode],
                        env=environment,
                        check=True,
                    )
                    captured = self._read_environment(capture)
                    captures[(arm, mode)] = captured

                    self.assertEqual(captured["HF_CHECKPOINT"], str(sft_root / "final_hf"))
                    self.assertEqual(captured["REF_LOAD"], str(sft_root))
                    expected_save_name = self.rl_roots[arm]
                    if mode == "smoke":
                        expected_save_name = expected_save_name.replace(
                            "_20260903",
                            "_smoke_20260903",
                        )
                    self.assertEqual(Path(captured["SAVE_DIR"]).name, expected_save_name)
                    self.assertEqual(captured["TAU2_AGENT_PROTOCOL_PROFILE"], "official-native")
                    self.assertEqual(captured["LOSS_MASK_TYPE"], "qwen3_full")
                    self.assertEqual(captured["TAU2_TURN_CREDIT_VERSION"], "")
                    self.assertEqual(captured["USE_REWARD_SHAPING"], "0")
                    self.assertEqual(captured["TAU2_REPLACE_ZERO_SIGNAL_GROUPS"], "0")
                    self.assertEqual(captured["DATA_SOURCE_PATH"], "filters.DomainQuotaDataSource")
                    self.assertEqual(captured["TAU2_RL_DOMAIN_QUOTA"], "airline:1,retail:2,telecom:2")
                    self.assertEqual(captured["ROLLOUT_BATCH_SIZE"], "5")
                    self.assertEqual(captured["N_SAMPLES_PER_PROMPT"], "8")
                    self.assertEqual(captured["GLOBAL_BATCH_SIZE"], "40")
                    self.assertEqual(captured["NUM_ROLLOUT"], "1" if mode == "smoke" else "100")
                    self.assertEqual(captured["SAVE_INTERVAL"], "1" if mode == "smoke" else "10")
                    self.assertEqual(captured["TRAIN_SEED"], "1234")
                    self.assertEqual(captured["ROLLOUT_SEED"], "42")
                    self.assertEqual(captured["LR"], "2e-6")
                    self.assertEqual(captured["KL_LOSS_TYPE"], "k2")
                    self.assertEqual(captured["KL_LOSS_COEF"], "0")
                    self.assertEqual(captured["KL_COEF"], "0")
                    self.assertEqual(captured["ENTROPY_COEF"], "0")
                    self.assertEqual(captured["EPS_CLIP"], "0.2")
                    self.assertEqual(captured["EPS_CLIP_HIGH"], "0.2")
                    self.assertEqual(captured["ROLLOUT_TEMPERATURE"], "1.0")
                    self.assertEqual(captured["ROLLOUT_TOP_P"], "1.0")
                    self.assertEqual(captured["AGENT_MAX_TOKENS"], "1200")
                    self.assertEqual(captured["TAU2_MAX_STEPS"], "200")
                    self.assertEqual(captured["TAU2_RL_MAX_TRAIN_TOKENS"], "16384")
                    self.assertEqual(captured["TAU2_RL_MAX_ROLLOUT_RETRIES"], "2")
                    self.assertEqual(captured["RAY_GPUS"], "4")
                    self.assertEqual(captured["AGENT_CUDA_VISIBLE_DEVICES"], "0,1,2,3")
                    self.assertEqual(captured["USER_CUDA_VISIBLE_DEVICES"], "4,5")
                    self.assertEqual(captured["USER_TP"], "2")
                    self.assertEqual(captured["TAU2_USER_TEMPERATURE"], "0.0")
                    self.assertEqual(captured["TAU2_USER_TOP_P"], "1.0")
                    self.assertEqual(captured["TAU2_USER_MAX_TOKENS"], "512")
                    self.assertEqual(
                        json.loads(captured["TAU2_USER_EXTRA_BODY_JSON"]),
                        {"chat_template_kwargs": {"enable_thinking": False}},
                    )

        paired_keys = (
            "TAU2_AGENT_PROTOCOL_PROFILE",
            "LOSS_MASK_TYPE",
            "TAU2_RL_DOMAIN_QUOTA",
            "ROLLOUT_BATCH_SIZE",
            "N_SAMPLES_PER_PROMPT",
            "GLOBAL_BATCH_SIZE",
            "TRAIN_SEED",
            "ROLLOUT_SEED",
            "LR",
            "KL_LOSS_TYPE",
            "KL_LOSS_COEF",
            "KL_COEF",
            "ENTROPY_COEF",
            "EPS_CLIP",
            "EPS_CLIP_HIGH",
            "ROLLOUT_TEMPERATURE",
            "AGENT_MAX_TOKENS",
            "TAU2_MAX_STEPS",
            "TAU2_USER_TEMPERATURE",
            "TAU2_USER_TOP_P",
            "TAU2_USER_MAX_TOKENS",
            "TAU2_USER_EXTRA_BODY_JSON",
        )
        for mode in ("smoke", "train100"):
            self.assertEqual(
                {key: captures[("raw-all", mode)][key] for key in paired_keys},
                {key: captures[("processed", mode)][key] for key in paired_keys},
            )

        wrapper_source = self.training_wrapper.read_text(encoding="utf-8")
        for forbidden in (
            "custom-reward-post-process",
            "custom-advantage-function",
            "on-policy-distillation",
            "use-tis",
        ):
            self.assertNotIn(forbidden, wrapper_source)
        base_source = (
            ROOT / "examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
        ).read_text(encoding="utf-8")
        for variable in ("TAU2_USER_TOP_P", "TAU2_USER_EXTRA_BODY_JSON"):
            self.assertIn(variable, base_source)

    def test_conversion_and_evaluation_wrappers_map_the_fixed_checkpoints(self):
        for arm in ("raw-all", "processed"):
            for iteration in (9, 39, 69, 99):
                with self.subTest(kind="convert", arm=arm, iteration=iteration), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    service_root, _, environment, capture = self._environment(root)
                    self._prepare_sft(service_root, arm)
                    run_root = service_root / "checkpoints" / self.rl_roots[arm]
                    run_root.mkdir(parents=True)
                    (run_root / "latest_checkpointed_iteration.txt").write_text("99\n", encoding="utf-8")
                    subprocess.run(
                        ["/bin/bash", str(self.converter), arm, str(iteration)],
                        env=environment,
                        check=True,
                    )
                    captured = self._read_environment(capture)
                    checkpoint = f"iter_{iteration:07d}"
                    self.assertEqual(captured["ITER_DIR"], str(run_root / checkpoint))
                    self.assertEqual(captured["OUTPUT_DIR"], str(run_root / f"{checkpoint}_hf"))
                    self.assertEqual(
                        captured["ORIGIN_HF_DIR"],
                        str(service_root / "checkpoints" / self.sft_roots[arm] / "final_hf"),
                    )

            with self.subTest(kind="eval", arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                service_root, project_root, environment, capture = self._environment(root)
                run_root = service_root / "checkpoints" / self.rl_roots[arm]
                hf_dir = run_root / "iter_0000099_hf"
                hf_dir.mkdir(parents=True)
                (hf_dir / "config.json").write_text("{}\n", encoding="utf-8")
                subprocess.run(
                    ["/bin/bash", str(self.evaluator), arm, "99", "301"],
                    env=environment,
                    check=True,
                )
                captured = self._read_environment(capture)
                self.assertEqual(captured["MODEL_PATH"], str(hf_dir))
                self.assertEqual(captured["EXPERIMENT_DIR"], str(project_root / "output/experiments/tau2-rl-sft-raw-vs-processed-vanilla-grpo"))
                self.assertEqual(captured["AGENT_EVAL_MODE"], "official-native")
                self.assertEqual(captured["DOMAINS"], "airline,retail,telecom")
                self.assertEqual(captured["DOMAIN_CONCURRENCY"], "airline:2,retail:2,telecom:5")
                self.assertEqual(captured["NUM_TASKS"], "")
                self.assertEqual(captured["NUM_TRIALS"], "4")
                self.assertEqual(captured["SEED"], "301")
                self.assertEqual(captured["MAX_STEPS"], "200")
                self.assertEqual(captured["AGENT_TEMPERATURE"], "0.6")
                self.assertEqual(captured["AGENT_TOP_P"], "1.0")
                self.assertEqual(captured["AGENT_MAX_TOKENS"], "1200")

    def test_training_summary_uses_coefficient_zero_k2_drift(self):
        def stats(value):
            return {
                "updates": 100,
                "min": value,
                "max": value,
                "mean": value,
                "final": value,
                "final_step": 99,
            }

        def series(value):
            return [{"step": step, "value": value} for step in range(100)]

        metrics = {
            "train/loss": stats(0.0),
            "train/grad_norm": stats(0.5),
            "train/global_batch_size": stats(40.0),
            "train/kl_loss": stats(0.125),
            "rollout/kl": stats(0.0),
            "rollout/raw_reward": stats(0.4),
            "rollout/truncated_ratio": stats(0.0),
        }
        diagnostic = {
            "run_log_summary": {
                "observed_steps": list(range(100)),
                "metric_parse_error_lines": {},
                "metrics": metrics,
                "sparse_count_totals": {
                    "rollout/domain_quota/airline/accepted": 100.0,
                    "rollout/domain_quota/retail/accepted": 200.0,
                    "rollout/domain_quota/telecom/accepted": 200.0,
                },
                "zero_std_count_totals": {"rollout/zero_std/count_0.0": 25.0},
                "series": {
                    "train/loss": series(0.0),
                    "train/grad_norm": series(0.5),
                    "train/global_batch_size": series(40.0),
                    "train/kl_loss": series(0.125),
                    "rollout/kl": series(0.0),
                    "rollout/raw_reward": series(0.4),
                    "rollout/truncated_ratio": series(0.0),
                    "rollout/domain_quota/airline/accepted": series(1.0),
                    "rollout/domain_quota/retail/accepted": series(2.0),
                    "rollout/domain_quota/telecom/accepted": series(2.0),
                    "rollout/zero_std/count_0.0": series(0.25),
                },
            },
            "trajectory_summary": {
                "policy_signal_groups": {},
                "sampling": {
                    "retried_trajectories": 200,
                    "permanently_too_long": 40,
                },
                "trajectory_counts": {"dumped": 4000},
                "termination": {},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            path.write_text(json.dumps(diagnostic), encoding="utf-8")
            summary = compact_training(path, "raw-all")

        self.assertEqual(summary["kl_drift"]["mean"], 0.125)
        self.assertEqual(summary["curve"][99]["kl_drift"], 0.125)
        self.assertEqual(summary["retry_rate"], 0.05)
        self.assertEqual(summary["permanently_overlong_rate"], 0.01)


class RawProcessedQualityAuditTest(unittest.TestCase):
    def test_native_call_parsing_and_prefix_pairing(self):
        self.assertEqual(
            call_parts(
                {
                    "name": "lookup",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"item_id": "1"}',
                    },
                }
            ),
            ("lookup", {"item_id": "1"}, True),
        )
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_0001"}]},
            {"role": "tool", "tool_call_id": "call_0001"},
            {"role": "assistant", "tool_calls": [{"id": "call_0002"}]},
        ]
        self.assertTrue(has_complete_prefix_pairing(messages))
        self.assertFalse(has_complete_prefix_pairing([messages[0], messages[2]]))

    def test_two_seed_task_cluster_bootstrap(self):
        keys = [(domain, f"{domain}-task") for domain in ("airline", "retail", "telecom")]
        raw = {
            seed: {key: (0.0, 0.0, 0.0, 0.0) for key in keys}
            for seed in ("300", "301")
        }
        processed = copy.deepcopy(raw)
        for seed in processed:
            processed[seed][keys[0]] = (1.0, 0.0, 0.0, 0.0)

        rows = build_paired_eval(
            raw=raw,
            processed=processed,
            samples=100,
            seed=7,
        )
        overall = next(
            row
            for row in rows
            if row["scope"] == "overall" and row["metric"] == "pass_at_1"
        )
        self.assertAlmostEqual(overall["delta"], 1 / 12)
        self.assertEqual(
            (overall["tasks_better"], overall["tasks_tied"], overall["tasks_worse"]),
            (1, 2, 0),
        )
        first = paired_bootstrap_ci([0.25, 0.0, 0.0], samples=100, seed=7)
        second = paired_bootstrap_ci([0.25, 0.0, 0.0], samples=100, seed=7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
