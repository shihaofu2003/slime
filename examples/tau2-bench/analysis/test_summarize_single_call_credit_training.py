#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from summarize_single_call_credit_training import (
    build_summary,
    discover_inputs,
    render_markdown,
)


def _turn(critique, *, stored=None, span=None):
    value = {"critique": critique}
    if stored is not None:
        value["reallocation_modifier"] = stored
    if span is not None:
        value["response_span"] = span
    return value


def _row(
    group,
    reward,
    *,
    domain="retail",
    turns=None,
    actions=None,
    counts=None,
    components=None,
    status="completed",
    termination="user_stop",
    removed=False,
    loss_mask=None,
    **metadata,
):
    row = {
        "group_index": group,
        "reward": reward,
        "remove_sample": removed,
        "status": status,
        "metadata": {
            "tau2_domain": domain,
            "raw_reward": reward,
            "tau2_termination_reason": termination,
            "tau2_turn_credit_version": "turn-credit-v2",
            "tau2_turn_credits": turns or [],
            "tau2_field_reward_signals": {
                "action_details": actions or [],
                "counts": counts or {},
                "components": components or {},
            },
            "partial_components": components or {},
            **metadata,
        },
    }
    if loss_mask is not None:
        row["loss_mask"] = loss_mask
    return row


class CreditTrainingSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_trajectory_and_v2_modifier_aggregates(self):
        path = self.root / "trajectories" / "stage-a.jsonl"
        actions = [
            {"tool_name_match": True, "arguments": {"x": True}},
            {"tool_name_match": True, "arguments": {"x": True, "y": False}},
            {"tool_name_match": True, "arguments": {"x": False}},
            {"tool_name_match": False, "arguments": {"x": False}},
        ]
        rows = [
            _row(
                0,
                0.0,
                turns=[_turn(1, stored=0.5), _turn(-1, stored=-0.5)],
                actions=actions,
                counts={"malformed_json": 1, "wrong_argument_fields": 2},
                components={"db": 0.0, "env_assertion": 0.5, "communicate": 1.0},
            ),
            _row(0, 1.0, turns=[_turn(0), _turn(-1)]),
            _row(1, 0.0, turns=[_turn(1), _turn(0)]),
            _row(
                1,
                0.0,
                turns=[_turn(0)],
                removed=True,
                status="truncated",
                termination="max_steps",
                tau2_dropped_too_long=True,
                tau2_permanently_too_long=True,
                tau2_rollout_attempts=3,
                tau2_domain_quota_replacement=True,
            ),
        ]
        self._write_jsonl(path, rows)

        zero_weight = build_summary(
            arm="v2-l000",
            trajectory_paths=[path],
            log_paths=[],
            reallocation_weight=0.0,
        )
        point_one_weight = build_summary(
            arm="v2-l010",
            trajectory_paths=[path],
            log_paths=[],
            reallocation_weight=0.1,
        )
        summary = zero_weight["trajectory_summary"]
        self.assertEqual(summary["trajectory_counts"]["dumped"], 4)
        self.assertEqual(summary["group_counts"]["dumped"], 2)
        self.assertEqual(summary["binary_outcome_groups"]["zero_variance"], 1)
        self.assertEqual(
            summary["turn_credit"]["raw_directional_active_trajectories"], 3
        )
        self.assertEqual(summary["turn_credit"]["modifier_active_trajectories"], 2)
        self.assertEqual(
            summary["turn_credit"]["weighted_local_channel_active_trajectories"],
            0,
        )
        self.assertEqual(
            summary["turn_credit"][
                "weighted_local_l1_budget_per_modifier_active_trajectory"
            ],
            0.0,
        )
        self.assertEqual(
            summary["turn_credit"][
                "weighted_local_l1_budget_across_dumped_modifier_active_trajectories"
            ],
            0.0,
        )
        point_one_credit = point_one_weight["trajectory_summary"]["turn_credit"]
        self.assertEqual(
            point_one_credit["weighted_local_channel_active_trajectories"], 2
        )
        self.assertEqual(
            point_one_credit["weighted_local_channel_activation_rate"], 0.5
        )
        self.assertEqual(
            point_one_credit[
                "weighted_local_l1_budget_per_modifier_active_trajectory"
            ],
            0.1,
        )
        self.assertAlmostEqual(
            point_one_credit[
                "weighted_local_l1_budget_across_dumped_modifier_active_trajectories"
            ],
            0.2,
        )
        zero_markdown = render_markdown(zero_weight)
        self.assertIn("Weighted local-channel activation | 0 (0)", zero_markdown)
        self.assertNotIn("Effective activation", zero_markdown)
        self.assertEqual(summary["turn_credit"]["complementary_neutral_turns"], 1)
        self.assertEqual(summary["turn_credit"]["stored_modifier_mismatches"], 0)
        self.assertEqual(
            summary["reference_actions"]["tiers"],
            {
                "exact": 1,
                "partial_arguments": 1,
                "tool_name_only": 1,
                "unmatched": 1,
            },
        )
        self.assertEqual(summary["rule_errors"]["wrong_argument_fields"]["events"], 2)
        self.assertEqual(summary["termination"]["max_steps"], 1)
        self.assertEqual(summary["sampling"]["extra_rollout_attempts"], 2)

    def test_weighted_local_per_token_advantage_scale(self):
        path = self.root / "trajectories" / "scale.jsonl"
        self._write_jsonl(
            path,
            [
                _row(
                    0,
                    0.0,
                    turns=[
                        _turn(1, stored=0.5, span=[0, 1]),
                        _turn(-1, stored=-0.5, span=[1, 4]),
                    ],
                    loss_mask=[1, 1, 1, 1],
                ),
                _row(
                    0,
                    0.0,
                    turns=[
                        _turn(1, stored=0.5, span=[0, 1]),
                        _turn(-1, stored=-0.5, span=[1, 4]),
                    ],
                    loss_mask=[1, 1, 1, 1],
                    removed=True,
                ),
            ],
        )

        no_weight = build_summary(
            arm="v1-matched", trajectory_paths=[path], log_paths=[]
        )["trajectory_summary"]["turn_credit"]
        zero_weight = build_summary(
            arm="v2-l000",
            trajectory_paths=[path],
            log_paths=[],
            reallocation_weight=0.0,
        )["trajectory_summary"]["turn_credit"]
        point_one_summary = build_summary(
            arm="v2-l010",
            trajectory_paths=[path],
            log_paths=[],
            reallocation_weight=0.1,
        )
        point_one = point_one_summary["trajectory_summary"]["turn_credit"]

        self.assertIsNone(no_weight["weighted_local_per_token_advantage"])
        zero_scale = zero_weight["weighted_local_per_token_advantage"]
        self.assertEqual(zero_scale["token_count"], 4)
        self.assertEqual(zero_scale["signed_min"], 0.0)
        self.assertEqual(zero_scale["signed_max"], 0.0)
        self.assertEqual(zero_scale["abs_p90"], 0.0)
        self.assertEqual(zero_scale["abs_p99"], 0.0)
        self.assertEqual(zero_scale["abs_max"], 0.0)

        point_one_scale = point_one["weighted_local_per_token_advantage"]
        self.assertEqual(point_one_scale["reconstructed_trajectories"], 1)
        self.assertEqual(point_one_scale["turn_count"], 2)
        self.assertEqual(point_one_scale["token_count"], 4)
        self.assertAlmostEqual(point_one_scale["signed_min"], -1 / 15)
        self.assertAlmostEqual(point_one_scale["signed_max"], 0.2)
        self.assertAlmostEqual(point_one_scale["abs_p90"], 0.16)
        self.assertAlmostEqual(point_one_scale["abs_p99"], 0.196)
        self.assertAlmostEqual(point_one_scale["abs_max"], 0.2)
        self.assertIn("before dynamic filtering", point_one_scale["view"])
        markdown = render_markdown(point_one_summary)
        self.assertIn("Weighted local per-token signed min / max", markdown)
        self.assertIn("before dynamic filtering", markdown)

    def test_policy_signal_groups_exclude_rejected_and_count_local_rescue(self):
        path = self.root / "trajectories" / "policy-signal.jsonl"
        active_turns = [_turn(1), _turn(-1)]
        self._write_jsonl(
            path,
            [
                _row(0, 0.0),
                _row(0, 1.0),
                _row(1, 0.0, turns=active_turns),
                _row(1, 0.0),
                _row(2, 0.0),
                _row(2, 0.0),
                _row(3, 0.0, turns=active_turns),
                _row(3, 0.0, removed=True),
            ],
        )

        with_local = build_summary(
            arm="v2-l010",
            trajectory_paths=[path],
            log_paths=[],
            reallocation_weight=0.1,
        )["trajectory_summary"]["policy_signal_groups"]
        without_local = build_summary(
            arm="v2-l000",
            trajectory_paths=[path],
            log_paths=[],
            reallocation_weight=0.0,
        )["trajectory_summary"]["policy_signal_groups"]

        self.assertEqual(with_local["trainable_candidates"], 3)
        self.assertEqual(with_local["outcome_signal"], 1)
        self.assertEqual(with_local["local_signal"], 1)
        self.assertEqual(with_local["rescued_zero_variance"], 1)
        self.assertEqual(with_local["zero_signal"], 1)
        self.assertAlmostEqual(with_local["zero_signal_rate"], 1 / 3)
        self.assertEqual(without_local["rescued_zero_variance"], 0)
        self.assertEqual(without_local["zero_signal"], 2)

    def test_policy_signal_groups_use_latest_resume_rollout_attempt(self):
        path = self.root / "trajectories" / "resume.jsonl"
        self._write_jsonl(
            path,
            [
                _row(0, 0.0, tau2_domain_quota_rollout_id=20),
                _row(0, 0.0, tau2_domain_quota_rollout_id=20),
                _row(1, 0.0, tau2_domain_quota_rollout_id=21),
                _row(1, 1.0, tau2_domain_quota_rollout_id=21),
                _row(0, 0.0, tau2_domain_quota_rollout_id=20),
                _row(0, 1.0, tau2_domain_quota_rollout_id=20),
            ],
        )

        groups = build_summary(
            arm="resume",
            trajectory_paths=[path],
            log_paths=[],
            reallocation_weight=0.1,
        )["trajectory_summary"]["policy_signal_groups"]

        self.assertEqual(groups["trainable_candidates"], 2)
        self.assertEqual(groups["outcome_signal"], 2)
        self.assertEqual(groups["zero_signal"], 0)

    def test_log_metrics_fill_sparse_perf_counts_with_zero(self):
        trajectory = self.root / "trajectories" / "stage-a.jsonl"
        self._write_jsonl(trajectory, [_row(0, 0.0)])
        log = self.root / "jobs" / "run_fixture.log"
        log.parent.mkdir(parents=True)
        log.write_text(
            "\x1b[36mworker\x1b[0m perf 0: "
            "{'rollout/truncated_ratio': 0.25, "
            "'rollout/zero_std/count_0.0': 2, "
            "'rollout/dynamic_filter/drop_turn_credit_v2_zero_signal': 1, "
            "'rollout/domain_quota/airline/accepted': 1, "
            "'rollout/domain_quota/retail/accepted': 2, "
            "'rollout/domain_quota/telecom/accepted': 2}\n"
            "worker rollout 0: {'rollout/raw_reward': 0.5, 'rollout/truncated': 0.25}\n"
            "worker step 0: {'train/loss': 0.2, 'train/kl_loss': 0.01, "
            "'train/grad_norm': 1.0, 'train/global_batch_size': 40}\n"
            "worker perf 1: {'rollout/truncated_ratio': 0.5}\n"
            "worker rollout 1: {'rollout/raw_reward': 0.75, 'rollout/truncated': 0.5}\n"
            "worker step 1: {'train/loss': 0.1, 'train/kl_loss': 0.03, "
            "'train/grad_norm': 2.0, 'train/global_batch_size': 40}\n"
            "worker step 2: {'train/loss': nan}\n",
            encoding="utf-8",
        )

        result = build_summary(
            arm="fixture", trajectory_paths=[trajectory], log_paths=[log]
        )
        logs = result["run_log_summary"]
        self.assertEqual(logs["observed_updates"], 2)
        self.assertEqual(logs["metric_parse_error_lines"][str(log.resolve())], [7])
        self.assertEqual(logs["metrics"]["train/loss"]["final"], 0.1)
        self.assertEqual(logs["metrics"]["train/loss"]["max"], 0.2)
        self.assertEqual(
            logs["series"]["train/loss"],
            [{"step": 0, "value": 0.2}, {"step": 1, "value": 0.1}],
        )
        self.assertEqual(logs["metrics"]["train/kl_loss"]["mean"], 0.02)
        self.assertEqual(logs["metrics"]["train/global_batch_size"]["updates"], 2)
        self.assertEqual(logs["metrics"]["train/global_batch_size"]["min"], 40)
        self.assertEqual(logs["metrics"]["train/global_batch_size"]["max"], 40)
        sparse = logs["metrics"]["rollout/dynamic_filter/drop_turn_credit_v2_zero_signal"]
        self.assertEqual(sparse["updates"], 2)
        self.assertEqual(sparse["mean"], 0.5)
        self.assertEqual(sparse["final"], 0.0)
        self.assertEqual(
            logs["sparse_count_totals"][
                "rollout/dynamic_filter/drop_turn_credit_v2_zero_signal"
            ],
            1.0,
        )
        self.assertEqual(
            logs["sparse_count_totals"]["rollout/domain_quota/airline/accepted"],
            1.0,
        )
        self.assertEqual(
            logs["sparse_count_totals"]["rollout/domain_quota/retail/accepted"],
            2.0,
        )
        self.assertEqual(
            logs["sparse_count_totals"]["rollout/domain_quota/telecom/accepted"],
            2.0,
        )
        markdown = render_markdown(result)
        self.assertIn("Accepted-update log metrics", markdown)
        self.assertIn("v2-l010", render_markdown({**result, "arm": "v2-l010"}))

    def test_arm_directory_discovers_only_trajectory_subtree_jsonl(self):
        trajectory = self.root / "trajectories" / "stage-a.jsonl"
        unrelated = self.root / "data" / "prompts.jsonl"
        log = self.root / "jobs" / "job" / "run_now.log"
        self._write_jsonl(trajectory, [_row(0, 0.0)])
        self._write_jsonl(unrelated, [{"prompt": "not a trajectory"}])
        log.parent.mkdir(parents=True)
        log.write_text("", encoding="utf-8")
        trajectories, logs = discover_inputs(
            arm_dir=self.root, trajectory_inputs=[], log_inputs=[]
        )
        self.assertEqual(trajectories, [trajectory.resolve()])
        self.assertEqual(logs, [log.resolve()])


if __name__ == "__main__":
    unittest.main()
