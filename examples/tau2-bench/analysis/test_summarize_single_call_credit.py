#!/usr/bin/env python3

import unittest
from pathlib import Path
from unittest.mock import patch

from summarize_single_call_credit import (
    EXPECTED_TASKS,
    LABELS,
    SEEDS,
    build_report,
    decide,
)


def _models(*, v1, l000, l010):
    def row(values):
        pass_at_1, pass_at_4_any, pass_power_4, db_accuracy, action_accuracy = values
        return {
            "overall": {
                "pass_at_1": pass_at_1,
                "pass_at_4_any": pass_at_4_any,
                "pass_power_4": pass_power_4,
                "db_accuracy": db_accuracy,
                "action_accuracy": action_accuracy,
            }
        }

    return {
        "sft": row((0.1, 0.1, 0.1, 0.1, 0.1)),
        "v1-matched": row(v1),
        "v2-l000": row(l000),
        "v2-l010": row(l010),
    }


def _summary(seed, pass_value, correct_actions, total_actions):
    domains = {}
    for domain, tasks in EXPECTED_TASKS.items():
        domains[domain] = {
            "results_file": f"missing/seed{seed}/{domain}/results.json",
            "pass_metrics": {
                "tasks": tasks,
                "simulations": 4 * tasks,
                "pass_at_1": pass_value,
                "pass_at_4_any": pass_value,
                "pass_power_4": pass_value,
            },
            "metrics": {
                "infra_error_count": 0,
                "correct_read_actions": correct_actions,
                "correct_write_actions": 0,
                "total_read_actions": total_actions,
                "total_write_actions": 0,
                "db_match_count": correct_actions,
                "db_mismatch_count": total_actions - correct_actions,
                "termination_user_stop": 4 * tasks - seed + 299,
                "termination_max_steps": seed - 299,
                "agent_error_tags_by_severity": {
                    "minor": {"fixture_rule": seed - 299}
                },
            },
            "single_call": {
                "trajectory_count": 4 * tasks,
                "single_call_protocol_error_turns": seed - 300,
            },
            "namespace": {
                "trajectory_count": 4 * tasks,
                "affected_trajectory_count": seed - 300,
                "raw_attempt_count": seed - 300,
                "parsed_call_count": 0,
                "executed_call_count": 0,
                "namespace_attributed_termination_count": 0,
            },
        }
    return {
        "task_split_name": "test",
        "num_trials": 4,
        "seed": seed,
        "max_steps": 200,
        "max_errors": 10,
        "agent_protocol_profile": "strict-single-v1",
        "domains": domains,
        "overall": {
            "tasks": 100,
            "simulations": 400,
            "pass_at_1": pass_value,
            "pass_at_4_any": pass_value,
            "pass_power_4": pass_value,
        },
    }


class DecisionTest(unittest.TestCase):
    def test_strict_primary_winner_ignores_pass_power_4(self):
        models = _models(
            v1=(0.30, 0.40, 0.90, 0.5, 0.5),
            l000=(0.31, 0.39, 0.90, 0.5, 0.5),
            l010=(0.32, 0.41, 0.00, 0.1, 0.1),
        )
        decision = decide(models)
        self.assertEqual(decision["selected_arm"], "v2-l010")
        self.assertEqual(decision["pass_power_4_role"], "report_only")

    def test_mixed_primary_result_retains_v1(self):
        decision = decide(
            _models(
                v1=(0.30, 0.40, 0.10, 0.5, 0.5),
                l000=(0.31, 0.39, 0.10, 0.9, 0.9),
                l010=(0.29, 0.41, 0.10, 0.9, 0.9),
            )
        )
        self.assertEqual(decision["selected_arm"], "v1-matched")
        self.assertEqual(
            decision["reason"], "mixed_or_no_strict_primary_winner_retain_v1"
        )

    def test_exact_primary_and_db_tie_uses_action_accuracy(self):
        decision = decide(
            _models(
                v1=(0.20, 0.20, 0.10, 0.9, 0.9),
                l000=(0.30, 0.40, 0.10, 0.5, 0.4),
                l010=(0.30, 0.40, 0.90, 0.5, 0.6),
            )
        )
        self.assertEqual(decision["selected_arm"], "v2-l010")
        self.assertEqual(
            decision["reason"],
            "exact_primary_and_db_tie_broken_by_action_accuracy",
        )

    def test_selection_policy_edge_cases(self):
        sft_high = _models(
            v1=(0.20, 0.20, 0.10, 0.5, 0.5),
            l000=(0.30, 0.30, 0.10, 0.5, 0.5),
            l010=(0.25, 0.25, 0.10, 0.5, 0.5),
        )
        sft_high["sft"]["overall"].update(
            {"pass_at_1": 0.99, "pass_at_4_any": 0.99}
        )
        cases = [
            (
                "sft_is_baseline_only",
                sft_high,
                "v2-l000",
                "strictly_higher_pass_at_1_and_pass_at_4_any_than_both_other_arms",
            ),
            (
                "db_breaks_exact_leading_primary_tie",
                _models(
                    v1=(0.20, 0.20, 0.10, 0.9, 0.9),
                    l000=(0.30, 0.40, 0.10, 0.6, 0.1),
                    l010=(0.30, 0.40, 0.10, 0.5, 0.9),
                ),
                "v2-l000",
                "exact_primary_pass_tie_broken_by_db_accuracy",
            ),
            (
                "unresolved_exact_tie_retains_v1",
                _models(
                    v1=(0.30, 0.40, 0.10, 0.5, 0.5),
                    l000=(0.30, 0.40, 0.90, 0.5, 0.5),
                    l010=(0.30, 0.40, 0.00, 0.5, 0.5),
                ),
                "v1-matched",
                "exact_tie_unresolved_retain_v1",
            ),
            (
                "nonleading_exact_tie_does_not_use_diagnostics",
                _models(
                    v1=(0.30, 0.40, 0.10, 0.1, 0.1),
                    l000=(0.30, 0.40, 0.10, 0.9, 0.9),
                    l010=(0.31, 0.39, 0.10, 0.1, 0.1),
                ),
                "v1-matched",
                "mixed_or_no_strict_primary_winner_retain_v1",
            ),
        ]
        for name, models, expected_arm, expected_reason in cases:
            with self.subTest(name=name):
                decision = decide(models)
                self.assertEqual(decision["selected_arm"], expected_arm)
                self.assertEqual(decision["reason"], expected_reason)

    def test_bootstrap_result_does_not_change_decision(self):
        records = {
            label: {
                seed: {
                    "path": Path(f"/tmp/{label}-seed{seed}.json"),
                    "summary": _summary(seed, pass_value, 1, 2),
                }
                for seed in SEEDS
            }
            for label, pass_value in {
                "sft": 0.90,
                "v1-matched": 0.20,
                "v2-l000": 0.25,
                "v2-l010": 0.30,
            }.items()
        }
        bootstrap = {
            "status": "available",
            "used_for_decision": False,
            "rows": [
                {
                    "comparison": "v2-l010-v1-matched",
                    "metric": "pass_at_1",
                    "ci95": [-0.5, 0.5],
                    "ci_excludes_zero": False,
                }
            ],
        }
        with patch(
            "summarize_single_call_credit.build_paired_bootstrap",
            return_value=bootstrap,
        ):
            report = build_report(
                records=records,
                results_root=None,
                bootstrap_samples=10,
                bootstrap_seed=7,
            )
        self.assertEqual(report["decision"]["selected_arm"], "v2-l010")
        self.assertIs(report["paired_bootstrap"], bootstrap)


class SummaryTest(unittest.TestCase):
    def test_two_seed_means_and_existing_diagnostics_are_preserved(self):
        records = {
            label: {
                300: {
                    "path": Path(f"/tmp/{label}-seed300.json"),
                    "summary": _summary(300, 0.2, 1, 1),
                },
                301: {
                    "path": Path(f"/tmp/{label}-seed301.json"),
                    "summary": _summary(301, 0.4, 0, 3),
                },
            }
            for label in LABELS
        }
        report = build_report(
            records=records,
            results_root=None,
            bootstrap_samples=10,
            bootstrap_seed=7,
        )
        overall = report["models"]["sft"]["overall"]
        self.assertAlmostEqual(overall["pass_at_1"], 0.3)
        self.assertAlmostEqual(overall["action_accuracy"], 0.5)
        self.assertAlmostEqual(overall["db_accuracy"], 0.5)
        self.assertEqual(overall["terminations"]["termination_max_steps"], 9)
        self.assertEqual(
            overall["rule_diagnostics"]["agent_error_tags_by_severity"]
            ["minor"]["fixture_rule"],
            9,
        )
        self.assertEqual(overall["single_call"]["single_call_protocol_error_turns"], 3)
        self.assertEqual(overall["namespace"]["raw_attempt_count"], 3)
        self.assertEqual(report["paired_bootstrap"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
