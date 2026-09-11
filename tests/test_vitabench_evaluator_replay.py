"""CPU contracts for the VitaBench evaluator thinking A/B replay."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAYER_PATH = REPO_ROOT / "examples" / "vita-bench" / "replay_evaluator_thinking_ab.py"


def _load_replayer():
    spec = importlib.util.spec_from_file_location("vita_evaluator_thinking_replay", REPLAYER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _states(decision: bool, justification: str = "state") -> list[dict]:
    return [
        {
            "rubric_idx": "rubric_0",
            "rubric": "the assistant completed the requested action",
            "justification": justification,
            "meetExpectation": decision,
        }
    ]


def _user_prompt(states: list[dict], window: str = "[1] assistant: done") -> str:
    return (
        "\n# Input\n<window_content>\n"
        f"{window}\n"
        "</window_content>\n\n<current_rubrics>\n"
        f"{json.dumps(states, ensure_ascii=False, indent=2)}\n"
        "</current_rubrics>\n"
    )


def _raw_window(index: int, input_decision: bool, output_decision: bool) -> dict:
    states = _states(input_decision)
    results = _states(output_decision, justification=f"window {index}")
    return {
        "window_idx": index,
        "system_prompt": f"system window {index}",
        "user_prompt": _user_prompt(states, window=f"[{index}] assistant: done"),
        "assistant_message_content": json.dumps(results),
        "assistant_message_raw_data": {"_vita_journal_id": f"{'a' * 31}{index}"},
        "assistent_message_usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "validation_attempts": 1,
        "validation_failures": [],
        "expected_rubric_count": 1,
        "validated_rubric_count": 1,
    }


def _simulation(*, windows: list[dict], reward: int, task_id: str = "task-0") -> dict:
    final = None
    if windows:
        final_decision = json.loads(windows[-1]["assistant_message_content"])[0]["meetExpectation"]
        final = [
            {
                "nl_rubric": "the assistant completed the requested action",
                "met": final_decision,
                "justification": "final",
            }
        ]
    return {
        "id": f"simulation-{task_id}",
        "task_id": task_id,
        "trial": 0,
        "seed": 300,
        "termination_reason": "agent_stop" if windows else "invalid_agent_message",
        "reward_info": {
            "reward": reward,
            "nl_rubrics": final,
            "reward_breakdown": {"NL_ASSERTION": float(reward)},
            "window_evaluations": windows or None,
        },
    }


def _write_source(tmp_path: Path, simulation: dict) -> Path:
    artifact = tmp_path / "source"
    shard = artifact / "shards" / "delivery" / "shard_00.json"
    shard.parent.mkdir(parents=True)
    shard.write_text(json.dumps({"simulations": [simulation]}), encoding="utf-8")
    manifest = {
        "protocol": {
            "shard_count": 1,
            "trajectory_count": 1,
            "task_count": 1,
            "num_trials": 1,
        }
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifact / "summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return artifact


def _result(module, window, mode: str, user_prompt: str, decision: bool) -> dict:
    return {
        "schema": module.REPLAY_SCHEMA,
        "mode": mode,
        "session_id": "test-session",
        "window_key": window.key,
        "trajectory_key": window.trajectory_key,
        "source_prompt_sha256": window.source_prompt_sha256,
        "request_prompt_sha256": module.prompt_sha256(window.system_prompt, user_prompt),
        "enable_thinking": False,
        "semantic_attempts": [
            {
                "attempt": 1,
                "endpoint": "http://localhost:32000/v1/chat/completions",
                "request_seconds": 1.0,
                "pool_wait_seconds": 0.0,
                "http_failures": [],
                "content": json.dumps(_states(decision)),
                "reasoning_content": "",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "reasoning_tokens": 0,
                    "total_tokens": 15,
                },
            }
        ],
        "results": _states(decision),
    }


def test_current_rubrics_round_trip_changes_only_the_state_block():
    module = _load_replayer()
    original = _user_prompt(_states(False), window="[7] assistant: unchanged")
    replacement = _states(True, justification="candidate state")
    updated = module.replace_current_rubrics(original, replacement)

    assert module.extract_current_rubrics(updated, "prompt") == tuple(replacement)
    assert "[7] assistant: unchanged" in updated
    assert updated.split(module.CURRENT_RUBRICS_OPEN)[0] == original.split(module.CURRENT_RUBRICS_OPEN)[0]
    assert updated.split(module.CURRENT_RUBRICS_CLOSE)[1] == original.split(module.CURRENT_RUBRICS_CLOSE)[1]


@pytest.mark.parametrize(
    "results,match",
    [
        ([], "nonempty"),
        (_states(True) + _states(False), "order/coverage"),
        ([{"rubric_idx": "rubric_0", "meetExpectation": "true"}], "boolean"),
        ([{"rubric_idx": "rubric_1", "meetExpectation": True}], "order/coverage"),
    ],
)
def test_result_validation_requires_exact_order_coverage_and_booleans(results, match):
    module = _load_replayer()
    with pytest.raises(module.ReplayError, match=match):
        module.validate_evaluator_results(results, _states(False), "candidate")


def test_source_loader_proves_the_final_window_reward_contract(tmp_path: Path):
    module = _load_replayer()
    artifact = _write_source(tmp_path, _simulation(windows=[_raw_window(1, False, True)], reward=1))
    dataset = module.load_source_dataset(artifact)

    assert len(dataset.trajectories) == 1
    assert dataset.trajectories[0].baseline_success is True
    assert module.source_counts(dataset)["baseline_validation_attempt_histogram"] == {"1": 1}

    shard_path = artifact / "shards" / "delivery" / "shard_00.json"
    corrupt = json.loads(shard_path.read_text(encoding="utf-8"))
    corrupt["simulations"][0]["reward_info"]["nl_rubrics"][0]["met"] = False
    shard_path.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(module.ReplayError, match="final persisted evaluator window"):
        module.load_source_dataset(artifact)


def test_frozen_and_chained_modes_answer_different_estimands(tmp_path: Path):
    module = _load_replayer()
    raw_windows = [_raw_window(1, False, True), _raw_window(2, True, True)]
    artifact = _write_source(tmp_path, _simulation(windows=raw_windows, reward=1))
    dataset = module.load_source_dataset(artifact)
    trajectory = dataset.trajectories[0]
    first, second = trajectory.windows
    output = tmp_path / "replay"

    module.atomic_write_json(
        module.result_path(output, "frozen", first),
        _result(module, first, "frozen", first.user_prompt, False),
    )
    module.atomic_write_json(
        module.result_path(output, "frozen", second),
        _result(module, second, "frozen", second.user_prompt, True),
    )

    chained_first_prompt = first.user_prompt
    chained_first_result = _result(module, first, "chained", chained_first_prompt, False)
    module.atomic_write_json(module.result_path(output, "chained", first), chained_first_result)
    chained_states = module.update_states(first.source_states, chained_first_result["results"])
    chained_second_prompt = module.replace_current_rubrics(second.user_prompt, chained_states)
    module.atomic_write_json(
        module.result_path(output, "chained", second),
        _result(module, second, "chained", chained_second_prompt, False),
    )

    _, frozen_success, _ = module.load_mode_results(dataset, output, "frozen", dataset.trajectories)
    _, chained_success, _ = module.load_mode_results(dataset, output, "chained", dataset.trajectories)

    assert frozen_success[trajectory.key] is True
    assert chained_success[trajectory.key] is False
    assert chained_second_prompt != second.user_prompt

    frozen_report = module.analyze_mode(
        dataset,
        output,
        "frozen",
        dataset.trajectories,
        baseline_evaluator_mean_seconds=125.5,
        baseline_total_critical_path_hours=1.0,
        bootstrap_samples=20,
    )
    chained_report = module.analyze_mode(
        dataset,
        output,
        "chained",
        dataset.trajectories,
        baseline_evaluator_mean_seconds=125.5,
        baseline_total_critical_path_hours=1.0,
        bootstrap_samples=20,
    )
    assert frozen_report["scores"]["delta"]["avg_at_k"] == 0
    assert chained_report["scores"]["delta"]["avg_at_k"] == -1
    assert frozen_report["operational_pass"] is True


def test_result_overlay_falls_back_to_immutable_base(tmp_path: Path):
    module = _load_replayer()
    raw_windows = [_raw_window(1, False, False), _raw_window(2, False, True)]
    artifact = _write_source(tmp_path, _simulation(windows=raw_windows, reward=1))
    dataset = module.load_source_dataset(artifact)
    trajectory = dataset.trajectories[0]
    first, second = trajectory.windows
    base = tmp_path / "base"
    overlay = tmp_path / "overlay"

    first_result = _result(module, first, "chained", first.user_prompt, False)
    module.atomic_write_json(module.result_path(base, "chained", first), first_result)
    current_states = module.update_states(first.source_states, first_result["results"])
    second_prompt = module.replace_current_rubrics(second.user_prompt, current_states)
    module.atomic_write_json(
        module.result_path(overlay, "chained", second),
        _result(module, second, "chained", second_prompt, True),
    )

    results, successes, _ = module.load_mode_results(
        dataset,
        overlay,
        "chained",
        dataset.trajectories,
        base,
    )

    assert set(results) == {first.key, second.key}
    assert successes[trajectory.key] is True
    assert not module.result_path(overlay, "chained", first).exists()


def test_score_metrics_scopes_global_success_map_to_selected_trajectories():
    module = _load_replayer()
    selected = module.TrajectoryRecord(
        suite="delivery",
        task_id="task-a",
        trial=0,
        seed=300,
        simulation_id="simulation-a",
        termination_reason="agent_stop",
        baseline_success=True,
        baseline_final_rubrics=(True,),
        windows=(),
    )
    other = module.TrajectoryRecord(
        suite="instore",
        task_id="task-b",
        trial=0,
        seed=300,
        simulation_id="simulation-b",
        termination_reason="agent_stop",
        baseline_success=False,
        baseline_final_rubrics=(False,),
        windows=(),
    )

    metrics = module.score_metrics([selected], {selected.key: True, other.key: False})

    assert metrics["trajectories"] == 1
    assert metrics["successful_trajectories"] == 1
    assert metrics["avg_at_k"] == 1


def test_semantic_retry_keeps_thinking_disabled_and_adds_corrective_turn(monkeypatch):
    module = _load_replayer()
    captured_requests = []

    def fake_post(_endpoint, request_data, **_kwargs):
        captured_requests.append(request_data)
        valid = len(captured_requests) == 2
        content = json.dumps(_states(True)) if valid else "not json"
        return (
            {
                "model": module.DEFAULT_MODEL,
                "id": f"response-{len(captured_requests)}",
                "created": 1,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "reasoning_tokens": 0,
                    "total_tokens": 15,
                },
            },
            [],
            0.1,
        )

    monkeypatch.setattr(module, "post_with_retries", fake_post)
    attempts, results = module.evaluate_prompt(
        pool=module.EndpointPool(["http://localhost:32000"], 1),
        model=module.DEFAULT_MODEL,
        system_prompt="system",
        user_prompt=_user_prompt(_states(False)),
        expected_states=_states(False),
        max_tokens=8192,
        max_semantic_attempts=3,
        max_http_retries=5,
        timeout_seconds=600,
    )

    assert len(attempts) == 2
    assert results[0]["meetExpectation"] is True
    assert all(request["chat_template_kwargs"] == {"enable_thinking": False} for request in captured_requests)
    assert len(captured_requests[1]["messages"]) == 2
    assert "previous response was invalid" in captured_requests[1]["messages"][-1]["content"]
    assert "unquoted JSON boolean" in captured_requests[1]["messages"][-1]["content"]
    assert captured_requests[1]["max_tokens"] == 1024
    assert "not json" not in captured_requests[1]["messages"][-1]["content"]


def test_terminal_semantic_failure_is_atomically_journaled(tmp_path: Path, monkeypatch):
    module = _load_replayer()
    artifact = _write_source(tmp_path, _simulation(windows=[_raw_window(1, False, True)], reward=1))
    window = module.load_source_dataset(artifact).trajectories[0].windows[0]

    def fake_post(_endpoint, _request_data, **_kwargs):
        invalid = _states(True)
        invalid[0]["meetExpectation"] = "true"
        return (
            {
                "model": module.DEFAULT_MODEL,
                "id": "invalid-response",
                "created": 1,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(invalid)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "reasoning_tokens": 0,
                    "total_tokens": 15,
                },
            },
            [],
            0.1,
        )

    monkeypatch.setattr(module, "post_with_retries", fake_post)
    output = tmp_path / "replay"
    with pytest.raises(module.ReplayError, match="remained invalid after 3 semantic attempts"):
        module.run_one_window(
            output_dir=output,
            mode="chained",
            window=window,
            user_prompt=window.user_prompt,
            expected_states=window.source_states,
            pool=module.EndpointPool(["http://localhost:32000"], 1),
            model=module.DEFAULT_MODEL,
            max_tokens=8192,
            max_semantic_attempts=3,
            max_http_retries=5,
            timeout_seconds=600,
            session_id="failure-test",
        )

    failure = module.read_json(module.failure_path(output, "chained", window))
    assert failure["schema"] == module.FAILURE_SCHEMA
    assert failure["status"] == "failed"
    assert len(failure["semantic_attempts"]) == 3
    assert all(attempt["validation_error"] for attempt in failure["semantic_attempts"])


def test_pilot_selection_is_deterministic_and_balanced():
    module = _load_replayer()
    trajectories = []
    for suite in module.SUITES:
        for index in range(5):
            window = module.WindowRecord(
                suite=suite,
                task_id=f"task-{index}",
                trial=0,
                seed=300,
                simulation_id=f"{suite}-{index}",
                window_idx=1,
                system_prompt="system",
                user_prompt=_user_prompt(_states(False)),
                source_states=tuple(_states(False)),
                baseline_results=tuple(_states(False)),
                baseline_usage={"prompt_tokens": 1, "completion_tokens": 1},
                baseline_content=json.dumps(_states(False)),
                baseline_journal_id=None,
                baseline_validation_attempts=1,
                baseline_validation_failures=(),
            )
            trajectories.append(
                module.TrajectoryRecord(
                    suite=suite,
                    task_id=f"task-{index}",
                    trial=0,
                    seed=300,
                    simulation_id=f"{suite}-{index}",
                    termination_reason="agent_stop",
                    baseline_success=False,
                    baseline_final_rubrics=(False,),
                    windows=(window,),
                )
            )

    first = module.select_pilot_tasks(trajectories, 2)
    second = module.select_pilot_tasks(list(reversed(trajectories)), 2)

    assert [item.key for item in first] == [item.key for item in second]
    assert {suite: sum(item.suite == suite for item in first) for suite in module.SUITES} == {
        suite: 2 for suite in module.SUITES
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
