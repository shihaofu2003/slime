"""CPU checks for asynchronous OPD gradients, metrics and launcher settings."""

from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    for function in functions:
        function.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), path, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("student,behavior,teacher", [
    ([0.8, 0.2], [0.5, 0.5], [0.8, 0.2]),
    ([0.6, 0.4], [0.5, 0.5], [0.8, 0.2]),
    ([0.6, 0.4], [0.6, 0.4], [0.8, 0.2]),
])
def test_current_opd_with_tis_matches_exact_kl_gradient(student, behavior, teacher):
    ns = {"torch": torch, "mpu": SimpleNamespace(is_pipeline_last_stage=lambda: True)}
    load_functions("slime/utils/ppo_utils.py", {"get_grpo_returns", "compute_policy_loss"}, ns)
    load_functions("slime/backends/megatron_utils/loss.py", {
        "compute_advantages_and_returns", "apply_opd_kl_to_advantages", "vanilla_tis_function",
    }, ns)
    p, b, q = [torch.tensor(x, dtype=torch.float64) for x in (student, behavior, teacher)]
    args = SimpleNamespace(use_rollout_logprobs=False, kl_coef=0, custom_advantage_function_path=None,
                           advantage_estimator="grpo", use_opd=True, opd_type="sglang", opd_kl_coef=1,
                           normalize_advantages=False, tis_clip=2, tis_clip_low=0)
    data = {"log_probs": [p.log()], "rollout_log_probs": [b.log()], "teacher_log_probs": [q.log()],
            "rewards": [0.0], "loss_masks": [torch.ones_like(p)], "response_lengths": [2], "total_lengths": [3]}
    ns["compute_advantages_and_returns"](args, data)
    logits = p.log().requires_grad_()
    logp = logits.log_softmax(-1)
    loss, _ = ns["compute_policy_loss"](p.log() - logp, data["advantages"][0], 0.2, 0.2)
    loss, _, metrics = ns["vanilla_tis_function"](
        args, pg_loss=loss, train_log_probs=[p.log()], rollout_log_probs=[b.log()], loss_masks=data["loss_masks"],
    )
    exact = (logp.exp() * (logp - q.log())).sum()
    expected = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    actual = torch.autograd.grad((b * loss).sum(), logits)[0]
    assert metrics["tis_clipfrac"].sum() == 0
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    if student == teacher:
        torch.testing.assert_close(actual, torch.zeros_like(actual))
    else:
        assert actual.norm() > 0  # K=1 and zero task reward still have an OPD signal.


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("partitions", [[[0, 1, 2, 3]], [[0, 2], [1, 3]], [[0, 1, 3], [2]]])
def test_domain_metrics_preserve_mask_and_rollout_weights_across_dp_cp(monkeypatch, cp_size, partitions):
    all_masks = [torch.tensor(x) for x in ([1, 0], [1, 1, 1], [1, 1], [1])]
    all_current = [torch.full((len(x),), -1.0) for x in all_masks]
    all_diffs = [torch.tensor(x) for x in ([0.2, 100.0], [0.4, 0.6, 0.8], [0.5, 0.5], [-0.2])]
    all_teacher = [p - diff for p, diff in zip(all_current, all_diffs, strict=True)]
    all_behavior = [x.clone() for x in all_current]
    all_behavior[0][0] -= torch.tensor(4.0).log()  # Clipped TIS on one valid token.
    domains = ["airline", "airline", "retail", "airline"]
    denoms = [4.0, 4.0, 2.0, 1.0]  # First two segments belong to the same trajectory.
    gathered = []
    for partition in partitions:
        for cp_rank in range(cp_size):
            mpu = SimpleNamespace(is_pipeline_last_stage=lambda: True, get_tensor_model_parallel_rank=lambda: 0,
                                  get_context_parallel_world_size=lambda: cp_size,
                                  get_context_parallel_rank=lambda cp_rank=cp_rank: cp_rank)
            cp = load_functions("slime/backends/megatron_utils/cp_utils.py",
                                {"get_sum_of_sample_mean", "get_logits_and_tokens_offset_with_cp"},
                                {"torch": torch, "mpu": mpu})
            monkeypatch.setitem(sys.modules, "megatron.core", SimpleNamespace(mpu=mpu))
            monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.cp_utils", SimpleNamespace(**cp))
            monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.data", SimpleNamespace(
                gather_log_data=lambda _prefix, _args, _step, metrics: gathered.append(metrics)))
            ns = load_functions("examples/tau2-bench/opd/tau2_opd.py", {"log_training_metrics"},
                                {"_DOMAIN_ENV_KEYS": {"airline": "", "retail": ""}})

            def chunk(values, index, cp=cp):
                value = values[index]
                if cp_size == 1:
                    return value
                offsets = cp["get_logits_and_tokens_offset_with_cp"](len(value) + 3, len(value))[3]
                return torch.cat([value[max(0, start - 3):max(0, end - 3)] for start, end in offsets])

            data = {"metadata": [{"domain": domains[i]} for i in partition],
                    "total_lengths": [len(all_masks[i]) + 3 for i in partition],
                    "response_lengths": [len(all_masks[i]) for i in partition],
                    "loss_masks": [all_masks[i] for i in partition],
                    "rollout_mask_sums": torch.tensor([denoms[i] for i in partition]),
                    "log_probs": [chunk(all_current, i) for i in partition],
                    "rollout_log_probs": [chunk(all_behavior, i) for i in partition],
                    "teacher_log_probs": [chunk(all_teacher, i) for i in partition]}
            ns["log_training_metrics"](SimpleNamespace(tis_clip=2, tis_clip_low=0), 0, data)
            assert data["metadata"] == [{"domain": domains[i]} for i in partition]
    assert all(set(row) == set(gathered[0]) for row in gathered)
    reduced = {key: sum(row[key][0] for row in gathered) / sum(row[key][1] for row in gathered)
               for key in gathered[0]}
    assert reduced["opd/airline/sampled_current_teacher_log_ratio"] == pytest.approx(0.15)
    assert reduced["opd/airline/tis_weighted_current_teacher_log_ratio"] == pytest.approx(0.175)
    assert reduced["opd/airline/tis_clipfrac"] == pytest.approx(0.125)
    assert reduced["opd/retail/sampled_current_teacher_log_ratio"] == pytest.approx(0.5)
    assert reduced["opd/retail/tis_clipfrac"] == 0


def test_post_update_metrics_exclude_masked_tokens():
    captured = {}
    mpu = SimpleNamespace(is_pipeline_last_stage=lambda: True, get_tensor_model_parallel_rank=lambda: 0,
                          get_context_parallel_world_size=lambda: 1,
                          get_data_parallel_world_size=lambda **_: 1)
    ns = {"torch": torch, "mpu": mpu,
          "gather_log_data": lambda _prefix, _args, _step, metrics: captured.update(metrics)}
    load_functions("slime/backends/megatron_utils/cp_utils.py",
                   {"get_sum_of_sample_mean", "rollout_log_metric_contribution"}, ns)
    load_functions("slime/backends/megatron_utils/data.py", {"log_opd_post_update"}, ns)
    before = torch.tensor([-1.0, -99.0, -2.0])
    data = {"log_probs": [before], "rollout_log_probs": [before.clone()],
            "loss_masks": [torch.tensor([1, 0, 1])], "response_lengths": [3], "total_lengths": [5],
            "rollout_mask_sums": torch.tensor([2.0]), "global_batch_sizes": [1]}
    ns["log_opd_post_update"](9, SimpleNamespace(tis_clip=2, tis_clip_low=0), data,
                              [torch.tensor([-0.9, -10.0, -2.3])])
    assert captured["post_update_logprob_abs_diff"] == pytest.approx((0.2, 1.0))
    assert captured["post_update_tis_weighted_k2"] == pytest.approx((0.025, 1.0))
    torch.testing.assert_close(data["log_probs"][0], before)


def test_generic_rollout_logger_skips_domain_metadata():
    captured = {}
    mpu = SimpleNamespace(is_pipeline_last_stage=lambda: True, get_tensor_model_parallel_rank=lambda: 0,
                          get_context_parallel_world_size=lambda: 1,
                          get_data_parallel_world_size=lambda **_: 1)
    ns = {"torch": torch, "mpu": mpu,
          "gather_log_data": lambda _prefix, _args, _step, metrics: captured.update(metrics)}
    load_functions("slime/backends/megatron_utils/cp_utils.py",
                   {"get_sum_of_sample_mean", "rollout_log_metric_contribution"}, ns)
    load_functions("slime/backends/megatron_utils/data.py", {"log_rollout_data"}, ns)
    args = SimpleNamespace(ci_test=False, log_multi_turn=False, log_passrate=False, log_correct_samples=False)
    data = {"log_probs": [torch.tensor([-1.0])], "loss_masks": [torch.tensor([1])],
            "response_lengths": [1], "total_lengths": [2], "rollout_mask_sums": torch.tensor([1.0]),
            "global_batch_sizes": [1], "metadata": [{"domain": "airline"}]}
    ns["log_rollout_data"](0, args, data)
    assert "metadata" not in captured
    assert captured["log_probs"] == pytest.approx((-1.0, 1.0))


def test_training_forward_delivers_opd_log_ratio_to_loss(monkeypatch):
    monkeypatch.delenv("ENABLE_ROUTING_REPLAY", raising=False)
    args = SimpleNamespace(data_pad_size_multiplier=1, allgather_cp=False, enable_mtp_training=False)
    ns = {"os": os, "partial": partial, "args": args, "num_microbatches": 1, "step_global_batch_size": 1,
          "_with_rollout_top_p_token_keys": lambda _args, keys: keys,
          "get_batch": lambda data, keys, *_: {**{k: data[k] for k in keys if k in data}, "full_loss_masks": None},
          "loss_function": lambda _args, batch, *_: batch["opd_reverse_kl"]}
    load_functions("slime/backends/megatron_utils/model.py", {"forward_step"}, ns)
    values = [torch.tensor([0.25])]
    data = {"tokens": torch.tensor([1, 2]), "packed_seq_params": None,
            "multimodal_train_inputs": None, "opd_reverse_kl": values}
    logits, loss = ns["forward_step"](data, lambda **_: torch.tensor(0.0))
    assert loss(logits) is values


@pytest.mark.parametrize("interval,step,expected", [(0, 9, False), (10, 8, False), (10, 9, True)])
def test_post_update_forward_runs_after_optimizer_at_requested_interval(interval, step, expected):
    events = []
    before, after = torch.tensor([-1.0]), torch.tensor([-0.8])
    ns = {"get_data_iterator": lambda data: [], "inverse_timer": lambda *_: nullcontext(),
          "timer": lambda *_: nullcontext(), "log_rollout_data": lambda *_: None,
          "train": lambda *_: events.append("optimizer"),
          "log_opd_post_update": lambda _id, _args, data, post: events.append((data["log_probs"], post)),
          "train_dump_utils": SimpleNamespace(save_debug_train_data=lambda *_args, **_kwargs: None),
          "log_perf_data": lambda *_args, **_kwargs: None}
    load_functions("slime/backends/megatron_utils/actor.py", {"train_actor"}, ns)
    args = SimpleNamespace(use_rollout_routing_replay=False, compute_advantages_and_returns=False,
                           use_routing_replay=False, use_opd=True, opd_post_update_log_interval=interval,
                           ref_update_interval=None)

    def forward(*_args, **_kwargs):
        assert events == ["optimizer"]
        return {"post_update_log_probs": [after]}

    actor = SimpleNamespace(args=args, rollout_data_postprocess=None, compute_log_prob=forward,
                            model=None, optimizer=None, opt_param_scheduler=None,
                            prof=SimpleNamespace(step=lambda **_: None),
                            weights_backuper=SimpleNamespace(backup=lambda _: None),
                            weight_updater=SimpleNamespace(pop_metrics=lambda: {}))
    ns["train_actor"](actor, step, {"log_probs": [before], "num_microbatches": [1], "global_batch_sizes": [1]})
    assert len(events) == (2 if expected else 1)
    if expected:
        assert events[1] == ([before], [after])


def test_teacher_scores_and_domain_tags_survive_segment_conversion():
    import math
    from collections.abc import Mapping
    from slime.utils.types import Sample

    ns = load_functions("examples/tau2-bench/opd/tau2_opd.py",
                        {"post_process_rewards", "extract_teacher_log_probs", "_task_reward", "_sample_domain"},
                        {"math": math, "Mapping": Mapping, "Sample": Sample})
    sample = Sample(index=1, rollout_id=1, tokens=[10, 11], response_length=1, reward=0.0,
                    metadata={"domain": "airline", "tau2_opd_task_reward": 1.0})
    sample.training_segments = [
        {"tokens": [10, 11], "response_length": 1, "loss_mask": [1], "rollout_log_probs": [-0.1]},
        {"tokens": [20, 21, 22, 23], "response_length": 3, "loss_mask": [1, 0, 1],
         "rollout_log_probs": [-0.2, 0.0, -0.3]},
    ]
    sample.metadata["tau2_opd_segment_responses"] = [
        {"meta_info": {"input_token_logprobs": [[None, 10], [-0.4, 11]]}},
        {"meta_info": {"input_token_logprobs": [[None, 20], [-0.5, 21], [-0.6, 22], [-0.7, 23]]}},
    ]
    load_functions("slime/ray/rollout.py", {"_convert_samples_to_train_data"}, ns)
    args = SimpleNamespace()
    owner = SimpleNamespace(args=args, custom_convert_samples_to_train_data_func=None,
                            _post_process_rewards=lambda samples: ns["post_process_rewards"](args, samples))
    data = ns["_convert_samples_to_train_data"](owner, [sample])
    assert data["metadata"] == [{"domain": "airline"}, {"domain": "airline"}]
    assert data["teacher_log_probs"] == [[-0.4], [-0.5, -0.6, -0.7]]
    assert data["loss_masks"] == [[1], [1, 0, 1]]
    assert data["rollout_mask_sums"] == [3, 3]
    assert data["rewards"] == [0.0, 0.0]
    assert data["trajectory_rewards"] == [1.0]


@pytest.mark.parametrize("overrides", [{}, {"TAU2_MAX_BUFFERED_GROUPS": "-1", "TAU2_OPD_LR": "5e-6",
                                            "TAU2_OPD_RUN_TAG": "control", "TAU2_ENVIRONMENT_WORKERS": "1",
                                            "TAU2_MAX_POLICY_LAG": "1", "OPD_POST_UPDATE_LOG_INTERVAL": "0"},
                                       {"TAU2_OPD_DOMAINS": "airline,retail,telecom,banking",
                                        "TRAIN_SEED": "1235", "ROLLOUT_SEED": "43",
                                        "TAU2_OPD_NUM_UPDATES": "160"}])
def test_opd_launcher_defaults_and_overrides(tmp_path, overrides):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    for name in ("python3", "curl"):
        path = fakebin / name
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    project = tmp_path / "project"
    runner = project / "examples/tau2-bench/rl/run_qwen3_4b_instruct_2507_tau2_rl.sh"
    runner.parent.mkdir(parents=True)
    capture = tmp_path / "capture.json"
    runner.write_text(
        f"#!/bin/bash\n{shlex.quote(sys.executable)} - <<'PY'\n"
        "import json, os\nfrom pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(dict(os.environ)))\nPY\n"
    )
    data = tmp_path / "data"
    data.mkdir()
    domains = overrides.get("TAU2_OPD_DOMAINS", "airline,retail").split(",")
    for domain in domains:
        (data / f"{domain}_train.jsonl").touch()
    endpoint = tmp_path / "teachers.env"
    endpoint.write_text("".join(f"TAU2_OPD_{d.upper()}_URL=http://{d}/generate\n" for d in domains))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TAU2_", "OPD_"))}
    env.update(PROJECT_ROOT=str(project), SERVICE_AGENT_ROOT=str(tmp_path), DATA_DIR=str(data),
               AIRLINE_EXPERT_MODEL=str(data), RETAIL_EXPERT_MODEL=str(data), HF_CHECKPOINT=str(data), REF_LOAD=str(data),
               TELECOM_EXPERT_MODEL=str(data), BANKING_EXPERT_MODEL=str(data),
               TEACHER_ENDPOINT_FILE=str(endpoint), TAU2_USER_API_BASE="http://user/v1",
               CAPTURE=str(capture), PATH=str(fakebin) + os.pathsep + os.environ["PATH"], **overrides)
    subprocess.run(["bash", str(ROOT / "examples/tau2-bench/opd/run_tau2_opd_airline_retail.sh"), "pilot", "student"],
                   env=env, check=True, capture_output=True, text=True, timeout=10)
    settings = json.loads(capture.read_text())
    assert settings["OPD_USE_BEHAVIOR_LOGPROBS"] == "0"
    assert settings["TAU2_MAX_BUFFERED_GROUPS"] == overrides.get("TAU2_MAX_BUFFERED_GROUPS", "32")
    assert settings["LR"] == overrides.get("TAU2_OPD_LR", "2e-6")
    assert settings["TAU2_ENVIRONMENT_WORKERS"] == overrides.get("TAU2_ENVIRONMENT_WORKERS", "4")
    assert settings["TAU2_MAX_POLICY_LAG"] == overrides.get("TAU2_MAX_POLICY_LAG", "-1")
    assert settings["OPD_POST_UPDATE_LOG_INTERVAL"] == overrides.get("OPD_POST_UPDATE_LOG_INTERVAL", "10")
    assert settings["NUM_ROLLOUT"] == overrides.get("TAU2_OPD_NUM_UPDATES", "60")
    for domain in domains:
        assert settings[f"TAU2_OPD_{domain.upper()}_URL"] == f"http://{domain}/generate"
    if "TRAIN_SEED" in overrides:
        assert settings["TRAIN_SEED"] == overrides["TRAIN_SEED"]
        assert settings["ROLLOUT_SEED"] == overrides["ROLLOUT_SEED"]
    assert settings["SAVE_DIR"].endswith(f"pilot-{overrides.get('TAU2_OPD_RUN_TAG', 'current-bounded-lr2e6')}/checkpoints")


@pytest.mark.parametrize("tag,iteration", [(None, None), ("control", 39), ("", 19)])
def test_opd_evaluation_selects_recipe_checkpoint(tmp_path, tag, iteration):
    wrapper = tmp_path / "examples/tau2-bench/eval/official/models/run_full_qwen3_4b_qwen36_user_async_timed.sh"
    wrapper.parent.mkdir(parents=True)
    capture = tmp_path / "eval.json"
    wrapper.write_text(
        f"#!/bin/bash\n{shlex.quote(sys.executable)} - <<'PY'\n"
        "import json, os\nfrom pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(dict(os.environ)))\nPY\n"
    )
    selected_tag = "current-bounded-lr2e6" if tag is None else tag
    selected_iteration = 59 if iteration is None else iteration
    suffix = f"-{selected_tag}" if selected_tag else ""
    model = (tmp_path / "output/experiments/tau2-opd-airline-retail-pilot"
             / f"pilot{suffix}/checkpoints/iter_{selected_iteration:07d}_hf")
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("TAU2_", "OPD_", "MODEL_", "EXPERIMENT_"))}
    env.update(PROJECT_ROOT=str(tmp_path), SERVICE_AGENT_ROOT=str(tmp_path),
               USER_ENDPOINT_FILE=str(tmp_path / "unused.env"), TAU2_USER_API_BASE="http://user/v1", CAPTURE=str(capture))
    if tag is not None:
        env["TAU2_OPD_RUN_TAG"] = tag
    if iteration is not None:
        env["TAU2_OPD_EVAL_ITERATION"] = str(iteration)
    subprocess.run(["bash", str(ROOT / "examples/tau2-bench/opd/run_official_eval_external_user.sh"), "opd", "301"],
                   env=env, check=True, capture_output=True, text=True, timeout=10)
    settings = json.loads(capture.read_text())
    assert settings["MODEL_PATH"] == str(model)
    assert settings["EVAL_LABEL"] == f"opd{suffix}-iter{selected_iteration}"
    assert settings["SEED"] == "301"
    assert settings["NUM_TRIALS"] == "4"
    assert settings["MAX_STEPS"] == "200"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
