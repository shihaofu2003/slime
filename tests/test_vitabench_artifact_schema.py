"""CPU-only contracts for local VitaBench manifest schema support."""

from __future__ import annotations

import ast
import concurrent.futures
import errno
import fcntl
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
VITA_EXAMPLE = REPO_ROOT / "examples" / "vita-bench"


def _load_summarizer():
    path = VITA_EXAMPLE / "summarize_full_eval.py"
    spec = importlib.util.spec_from_file_location("vita_full_eval_summarizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_scheduler():
    path = VITA_EXAMPLE / "eval_scheduler.py"
    spec = importlib.util.spec_from_file_location("vita_eval_scheduler", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_flock_adapter():
    path = VITA_EXAMPLE / "afs_flock_retry" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("vita_afs_flock_retry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ab_comparator():
    path = VITA_EXAMPLE / "compare_local_topology_ab.py"
    spec = importlib.util.spec_from_file_location("vita_ab_comparator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assigned_frozenset(path: Path, name: str) -> frozenset[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
            and len(node.value.args) == 1
        ):
            break
        return frozenset(ast.literal_eval(node.value.args[0]))
    raise AssertionError(f"{path} does not define {name} as a literal frozenset")


def _local_manifest(schema_version: int) -> dict:
    suites = ("delivery", "instore", "ota", "cross_domain")
    return {
        "schema_version": schema_version,
        "protocol": {
            "num_trials": 4,
            "seed": 300,
            "shard_size": 10,
            "temperature": 0,
            "evaluation_type": "trajectory",
            "enable_think": True,
            "language": "chinese",
            "max_steps": 300,
            "max_errors": 10,
            "suite_count": 4,
            "task_count": 400,
            "trajectory_count": 1600,
            "shard_count": 40,
            "max_concurrency": 4,
        },
        "execution": {
            "role_backend": "local",
            "suite_parallelism": 4,
            "per_suite_max_concurrency": 4,
            "aggregate_max_concurrency": 16,
            "remote_api_concurrency_limit": 8,
            "api_response_journal_schema": "vita-llm-response-journal/v1",
            "journaled_models": ["agent", "user", "evaluator"],
            "gpu_binding": "four_agent_plus_four_role_replicas",
            "suite_workers": [
                {"suite": suite, "gpu_slot": index, "port": 30000 + index} for index, suite in enumerate(suites)
            ],
        },
        "agent": {"requested": "agent", "sglang_instance_count": 4},
        "user": {"requested": "user"},
        "evaluator": {"requested": "evaluator"},
        "local_role_serving": {
            "user_instance_count": 2,
            "evaluator_instance_count": 2,
        },
    }


def _schema6_local_manifest() -> dict:
    manifest = _local_manifest(5)
    manifest["schema_version"] = 6
    manifest["execution"] = {
        "role_backend": "local",
        "suite_parallelism": 4,
        "scheduler": "dynamic_shard_queue",
        "run_mode": "full",
        "selected_shard_index": None,
        "worker_count": 4,
        "per_worker_max_concurrency": 4,
        "aggregate_max_concurrency": 16,
        "remote_api_concurrency_limit": 8,
        "api_response_journal_schema": "vita-llm-response-journal/v1",
        "journaled_models": ["agent", "user", "evaluator"],
        "gpu_binding": "eight_single_gpu_role_instances",
        "queue_lock": {
            "primitive": "fcntl.flock",
            "retry_errno_names": ["EACCES", "EAGAIN"],
            "retry_interval_seconds": 0.05,
            "retry_timeout_seconds": 120.0,
        },
        "worker_endpoints": [
            {
                "worker_id": worker_id,
                "agent_instance": worker_id % 2,
                "agent_gpu_slot": worker_id % 2,
                "agent_port": 30000 + worker_id % 2,
                "user_instance": worker_id % 2,
                "user_gpu_slot": 2 + worker_id % 2,
                "user_port": 31000 + worker_id % 2,
                "evaluator_instance": worker_id,
                "evaluator_gpu_slot": 4 + worker_id,
                "evaluator_port": 31002 + worker_id,
            }
            for worker_id in range(4)
        ],
    }
    manifest["agent"] = {
        "requested": "agent",
        "sglang_instance_count": 2,
        "sglang_max_running_requests": 16,
        "dtype": "bfloat16",
    }
    manifest["gate"] = [
        {"name": suite, "worker_id": index}
        for index, suite in enumerate(("delivery", "instore", "ota", "cross_domain"))
    ]
    manifest["local_role_serving"] = {
        "agent_instance_count": 2,
        "user_instance_count": 2,
        "evaluator_instance_count": 4,
        "total_instance_count": 8,
        "context_length": 32768,
        "dtype": "bfloat16",
        "user_sglang_max_running_requests": 8,
        "evaluator_sglang_max_running_requests": 4,
        "agent_ports": [30000, 30001],
        "user_ports": [31000, 31001],
        "evaluator_ports": [31002, 31003, 31004, 31005],
    }
    return manifest


def _metadata_manifest(model: str, score: float | None, available: bool) -> dict:
    return {
        "schema_version": 6,
        "benchmark": "VITA-Bench",
        "model": model,
        "agent": {
            "requested": model,
            "served": model,
            "response_model_family": model,
        },
        "user": {"model": "user"},
        "evaluator": {
            "model": "evaluator",
            "provider": "local",
            "official_comparable": False,
            "selection_reason": "local evaluator",
        },
        "official_reference": {
            "model": model,
            "available": available,
            "score": score,
            "metric": "four-suite macro Avg@4",
            "source_url": f"https://huggingface.co/Qwen/{model}",
            "strictly_comparable": False,
        },
    }


def _write_run_plan(path: Path, shard_count: int) -> None:
    lines = []
    for index in range(shard_count):
        lines.append(
            "\t".join(
                (
                    "shard",
                    "delivery",
                    "delivery",
                    "delivery",
                    str(index),
                    f"shards/delivery/shard_{index:02d}.json",
                    f"task-{index}",
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_schema_acceptance_is_consistent_across_artifact_tools():
    expected = frozenset({2, 3, 4, 5, 6})
    for filename in ("summarize_full_eval.py", "export_sft_archive.py"):
        assert _assigned_frozenset(VITA_EXAMPLE / filename, "SUPPORTED_MANIFEST_SCHEMAS") == expected


def test_local_vitabench_role_thinking_defaults_are_explicit():
    config = yaml.safe_load((VITA_EXAMPLE / "models_qwen35_full.yaml").read_text(encoding="utf-8"))
    models = {model["name"]: model for model in config["models"]}

    assert models["Qwen3.5-4B"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert models["Qwen3-4B-Instruct-2507"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert models["Qwen3.6-27B-user"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert models["Qwen3.6-27B-evaluator"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_local_nonthinking_protocol_uses_a_fresh_locked_artifact_namespace():
    runner = (VITA_EXAMPLE / "run_qwen3_5_4b_full_eval.sh").read_text(encoding="utf-8")

    assert 'RUNNER_PROTOCOL_VERSION="8"' in runner
    assert "artifacts-v4" in runner
    assert "artifacts-v4-gate" in runner
    assert "artifacts-v4-ab-shard${AB_SHARD_INDEX}" in runner
    assert 'expected_evaluator_thinking = False if role_backend == "local" else None' in runner

    manifest_evaluator = runner.rsplit('    "evaluator": {', maxsplit=1)[1].split('    "software": {', maxsplit=1)[0]
    assert '"thinking_mode": False if os.environ["ROLE_BACKEND"] == "local" else None' in manifest_evaluator


def test_qwen3_instruct_vitabench_wrapper_uses_hf_and_nonthinking():
    wrapper = (VITA_EXAMPLE / "run_qwen3_4b_instruct_2507_full_eval_local_roles.sh").read_text(encoding="utf-8")

    assert '/models/Qwen3-4B-Instruct-2507"' in wrapper
    assert "Qwen3-4B-Instruct-2507_torch_dist" not in wrapper
    assert "export VITA_AGENT_ENABLE_THINKING=false" in wrapper
    assert "export VITA_AGENT_TOOL_CALL_PARSER=qwen25" in wrapper
    assert "vitabench-qwen3-4b-instruct-2507-full-eval/artifacts" in wrapper


@pytest.mark.parametrize(
    ("model", "score", "available"),
    [
        ("Qwen3.5-4B", 22.0, True),
        ("Qwen3-4B-Instruct-2507", None, False),
    ],
)
def test_manifest_metadata_supports_scored_and_unscored_agents(model, score, available):
    summarizer = _load_summarizer()
    summarizer._validate_manifest_metadata(_metadata_manifest(model, score, available))


def test_manifest_metadata_rejects_score_for_unavailable_reference():
    summarizer = _load_summarizer()
    manifest = _metadata_manifest("Qwen3-4B-Instruct-2507", 22.0, False)

    with pytest.raises(
        summarizer.CorruptResultsError,
        match="unavailable official reference must have a null score",
    ):
        summarizer._validate_manifest_metadata(manifest)


def test_summarizer_accepts_local_parallel_manifests():
    summarizer = _load_summarizer()
    for schema_version in (4, 5):
        protocol = summarizer._manifest_protocol(_local_manifest(schema_version))
        assert protocol["trajectory_count"] == 1600
    protocol = summarizer._manifest_protocol(_schema6_local_manifest())
    assert protocol["trajectory_count"] == 1600


def test_summarizer_rejects_unknown_manifest_schema():
    summarizer = _load_summarizer()
    try:
        summarizer._manifest_protocol(_local_manifest(7))
    except summarizer.CorruptResultsError as exc:
        assert "unsupported manifest schema" in str(exc)
    else:
        raise AssertionError("schema 7 unexpectedly passed validation")


def test_summarizer_rejects_partial_schema6_run_modes():
    summarizer = _load_summarizer()
    manifest = _schema6_local_manifest()
    manifest["execution"]["run_mode"] = "ab"
    manifest["execution"]["selected_shard_index"] = 0
    with pytest.raises(summarizer.CorruptResultsError, match="only schema-6 full runs"):
        summarizer._manifest_protocol(manifest)


def test_summarizer_rejects_schema6_without_native_queue_lock_policy():
    summarizer = _load_summarizer()
    manifest = _schema6_local_manifest()
    del manifest["execution"]["queue_lock"]
    with pytest.raises(summarizer.CorruptResultsError, match="queue lock policy"):
        summarizer._manifest_protocol(manifest)


def test_local_topology_has_eight_unique_services_and_expected_worker_routes():
    scheduler = _load_scheduler()
    topology = scheduler.build_local_topology(
        [str(index) for index in range(8)],
        worker_count=4,
        agent_instance_count=2,
        user_instance_count=2,
        evaluator_instance_count=4,
        agent_port_base=30000,
        role_port_base=31000,
    )
    assert topology["scheduler"] == "dynamic_shard_queue"
    assert topology["queue_lock"] == {
        "primitive": "fcntl.flock",
        "retry_errno_names": ["EACCES", "EAGAIN"],
        "retry_interval_seconds": 0.05,
        "retry_timeout_seconds": 120.0,
    }
    assert topology["instance_count"] == 8
    assert len(topology["agent_instances"]) == 2
    assert len(topology["user_instances"]) == 2
    assert len(topology["evaluator_instances"]) == 4
    routes = topology["worker_endpoints"]
    assert [route["agent_instance"] for route in routes] == [0, 1, 0, 1]
    assert [route["user_instance"] for route in routes] == [0, 1, 0, 1]
    assert [route["evaluator_instance"] for route in routes] == [0, 1, 2, 3]
    ports = {
        instance["port"]
        for key in ("agent_instances", "user_instances", "evaluator_instances")
        for instance in topology[key]
    }
    assert len(ports) == 8


@pytest.mark.parametrize(
    ("selectors", "counts"),
    [
        ([str(index) for index in range(7)], (4, 2, 2, 4)),
        (["0", "1", "2", "3", "4", "5", "6", "6"], (4, 2, 2, 4)),
        ([str(index) for index in range(8)], (4, 4, 2, 2)),
    ],
)
def test_local_topology_rejects_wrong_gpu_or_role_counts(selectors, counts):
    scheduler = _load_scheduler()
    with pytest.raises(scheduler.SchedulerError):
        scheduler.build_local_topology(
            selectors,
            worker_count=counts[0],
            agent_instance_count=counts[1],
            user_instance_count=counts[2],
            evaluator_instance_count=counts[3],
            agent_port_base=30000,
            role_port_base=31000,
        )


def test_dynamic_queue_claims_each_shard_at_most_once(tmp_path: Path):
    scheduler = _load_scheduler()
    run_plan = tmp_path / "run_plan.tsv"
    state = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    _write_run_plan(run_plan, 24)
    scheduler.initialize_queue(
        run_plan,
        state,
        lock,
        worker_count=4,
        completion_checker=lambda _entry: "partial",
    )

    def claim(index: int) -> str:
        completed = subprocess.run(
            [
                sys.executable,
                str(VITA_EXAMPLE / "eval_scheduler.py"),
                "claim",
                "--state",
                str(state),
                "--lock",
                str(lock),
                "--worker-id",
                str(index % 4),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        fields = completed.stdout.strip().split("\t")
        return f"{fields[1]}:{fields[4]}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        claimed = list(executor.map(claim, range(24)))
    assert len(claimed) == 24
    assert len(set(claimed)) == 24


def test_dynamic_queue_restart_skips_complete_and_reclaims_incomplete(tmp_path: Path):
    scheduler = _load_scheduler()
    run_plan = tmp_path / "run_plan.tsv"
    state = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    _write_run_plan(run_plan, 3)
    scheduler.initialize_queue(
        run_plan,
        state,
        lock,
        worker_count=4,
        completion_checker=lambda _entry: "partial",
    )
    first = scheduler.claim_next_shard(state, lock, worker_id=0)
    second = scheduler.claim_next_shard(state, lock, worker_id=1)
    assert first is not None and first.key == "delivery:0"
    assert second is not None and second.key == "delivery:1"

    restarted = scheduler.initialize_queue(
        run_plan,
        state,
        lock,
        worker_count=4,
        completion_checker=lambda entry: "complete" if entry.key == first.key else "partial",
    )
    assert restarted["skipped_complete"] == ["delivery:0"]
    reclaimed = scheduler.claim_next_shard(state, lock, worker_id=2)
    assert reclaimed is not None and reclaimed.key == second.key


def test_dynamic_queue_can_select_one_shard_per_suite(tmp_path: Path):
    scheduler = _load_scheduler()
    run_plan = tmp_path / "run_plan.tsv"
    state = tmp_path / "queue.json"
    lock = tmp_path / "queue.lock"
    lines = []
    for suite in ("delivery", "instore", "ota", "cross_domain"):
        for shard_index in (0, 1):
            domain = "delivery,instore,ota" if suite == "cross_domain" else suite
            lines.append(
                "\t".join(
                    (
                        "shard",
                        suite,
                        domain,
                        suite,
                        str(shard_index),
                        f"shards/{suite}/shard_{shard_index:02d}.json",
                        f"{suite}-task-{shard_index}",
                    )
                )
            )
    run_plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    queue = scheduler.initialize_queue(
        run_plan,
        state,
        lock,
        worker_count=4,
        completion_checker=lambda _entry: "partial",
        shard_index=1,
    )
    assert queue["selected_shard_index"] == 1
    assert [entry["suite"] for entry in queue["available"]] == [
        "delivery",
        "instore",
        "ota",
        "cross_domain",
    ]
    assert all(entry["shard_index"] == 1 for entry in queue["available"])


def test_afs_flock_adapter_retries_eagain_until_lock_is_acquired():
    adapter = _load_flock_adapter()
    calls = []
    sleeps = []
    acquired = object()

    def afs_flock(_file_descriptor, _operation):
        calls.append(1)
        if len(calls) < 4:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        return acquired

    result = adapter.flock_with_retry(
        afs_flock,
        123,
        fcntl.LOCK_EX,
        timeout_seconds=1,
        sleep_function=sleeps.append,
        monotonic_function=lambda: 0,
    )
    assert result is acquired
    assert len(calls) == 4
    assert sleeps == [adapter.RETRY_INTERVAL_SECONDS] * 3


def test_native_scheduler_retries_eagain_until_lock_is_acquired():
    scheduler = _load_scheduler()
    calls = []
    sleeps = []
    acquired = object()

    def afs_flock(_file_descriptor, _operation):
        calls.append(1)
        if len(calls) < 4:
            raise BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable")
        return acquired

    result = scheduler.flock_with_retry(
        afs_flock,
        123,
        fcntl.LOCK_EX,
        timeout_seconds=1,
        sleep_function=sleeps.append,
        monotonic_function=lambda: 0,
    )
    assert result is acquired
    assert len(calls) == 4
    assert sleeps == [scheduler.FLOCK_RETRY_INTERVAL_SECONDS] * 3


def test_ab_comparator_accepts_manifest_locked_native_queue_lock(tmp_path: Path):
    comparator = _load_ab_comparator()
    manifest = _schema6_local_manifest()
    scheduler_sha256 = hashlib.sha256((VITA_EXAMPLE / "eval_scheduler.py").read_bytes()).hexdigest()
    manifest["software"] = {"eval_scheduler_sha256": scheduler_sha256}
    locking = comparator._validate_scheduler_locking(
        tmp_path,
        manifest,
        VITA_EXAMPLE,
    )
    assert locking == {
        "mode": "native_scheduler",
        "policy": manifest["execution"]["queue_lock"],
        "scheduler_sha256": scheduler_sha256,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
