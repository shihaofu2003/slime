from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPD_DIR = Path(__file__).resolve().parent
RL_DIR = OPD_DIR.parent / "rl"
sys.path.insert(0, str(OPD_DIR))
sys.path.insert(0, str(RL_DIR))

from slime.utils.types import Sample  # noqa: E402

import filters  # noqa: E402
import continuous  # noqa: E402
import prepare_airline_retail_data as prepare_data  # noqa: E402
import tau2_opd  # noqa: E402
import rollout  # noqa: E402
from slime.rollout.agent_tokens import expand_training_segments  # noqa: E402


def _sample(*, domain="airline", tokens=None, response_length=2, reward=None, remove_sample=False):
    sample = Sample(
        prompt="task",
        tokens=list(tokens or [10, 11, 12]),
        response_length=response_length,
        reward=reward,
        remove_sample=remove_sample,
        metadata={"domain": domain, "tau2_opd_task_reward": 0.75},
    )
    return sample


def test_domain_routing_and_full_on_policy_payload(monkeypatch):
    sample = _sample(tokens=[10, 11, 12, 13])
    assert tau2_opd.domain_teacher_url("airline", {"TAU2_OPD_AIRLINE_URL": "http://air"}) == "http://air"
    assert tau2_opd.domain_teacher_url("retail", {"TAU2_OPD_RETAIL_URL": "http://retail"}) == "http://retail"
    with pytest.raises(ValueError, match="Unsupported OPD teacher domain"):
        tau2_opd.domain_teacher_url("unknown", {})

    payload = tau2_opd.build_teacher_payload(sample)
    assert payload["input_ids"] == [10, 11, 12, 13]
    assert payload["sampling_params"]["max_new_tokens"] == 0
    assert payload["return_logprob"] is True
    assert payload["logprob_start_len"] == 0


@pytest.mark.parametrize("domain,env_key", [
    ("airline", "TAU2_OPD_AIRLINE_URL"),
    ("retail", "TAU2_OPD_RETAIL_URL"),
    ("telecom", "TAU2_OPD_TELECOM_URL"),
    ("banking", "TAU2_OPD_BANKING_URL"),
    ("banking_knowledge", "TAU2_OPD_BANKING_URL"),
])
def test_reward_func_routes_teacher_request(monkeypatch, domain, env_key):
    sample = _sample(domain=domain, tokens=[10, 11, 12])
    seen = {}

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return {"meta_info": {"input_token_logprobs": [[None, 0], [-0.1, 10], [-0.2, 11], [-0.3, 12]]}}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def post(self, url, *, json):
            seen.update(url=url, payload=json)
            return Response()

    monkeypatch.setattr(tau2_opd.aiohttp, "ClientSession", lambda **_: Session())
    args = SimpleNamespace(
        tau2_opd_airline_url="http://unused",
        tau2_opd_retail_url="http://unused",
    )
    monkeypatch.setenv(env_key, "http://expert/generate")
    result = asyncio.run(tau2_opd.reward_func(args, sample))
    assert seen["url"] == "http://expert/generate"
    assert result["tau2_opd_domain"] == domain
    assert seen["payload"]["input_ids"] == sample.tokens
    assert result["meta_info"]["input_token_logprobs"][0][0] is None


def test_reward_func_propagates_teacher_request_error(monkeypatch):
    sample = _sample()

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def raise_for_status(self):
            raise RuntimeError("teacher unavailable")

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def post(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(tau2_opd.aiohttp, "ClientSession", lambda **_: Session())
    with pytest.raises(RuntimeError, match="teacher unavailable"):
        asyncio.run(tau2_opd.reward_func(SimpleNamespace(), sample))


def test_logprob_alignment_drops_placeholder_and_requires_full_length():
    reward = {
        "meta_info": {
            "input_token_logprobs": [[None, 0], [-0.1, 10], [-0.2, 11], [-0.3, 12]],
        }
    }
    assert tau2_opd.extract_teacher_log_probs(reward, tokens_length=3, response_length=2) == [-0.2, -0.3]
    with pytest.raises(ValueError, match="teacher input log-prob length"):
        tau2_opd.extract_teacher_log_probs(reward, tokens_length=5, response_length=2)


def test_removed_sample_skips_teacher_request(monkeypatch):
    sample = _sample(tokens=[1], response_length=1, remove_sample=True)

    class Session:
        def __init__(self, **_):
            raise AssertionError("removed samples must not call the teacher")

    monkeypatch.setattr(tau2_opd.aiohttp, "ClientSession", Session)
    result = asyncio.run(tau2_opd.reward_func(SimpleNamespace(), sample))
    assert result["tau2_opd_skipped"] is True
    assert result["meta_info"]["input_token_logprobs"] == [[0.0, 0]]


def test_post_process_preserves_task_reward_and_zeroes_training_reward():
    sample = _sample(
        tokens=[10, 11, 12],
        response_length=2,
        reward={"meta_info": {"input_token_logprobs": [[None, 0], [-0.1, 10], [-0.2, 11], [-0.3, 12]]}},
    )
    raw, rewards = tau2_opd.post_process_rewards(SimpleNamespace(), [sample])
    assert raw == [0.75]
    assert rewards == [0.0]
    assert sample.teacher_log_probs == [-0.2, -0.3]
    assert sample.reward == 0.0
    assert sample.metadata["raw_reward"] == 0.75


def test_post_process_accepts_producer_stashed_teacher_response():
    sample = _sample(tokens=[10, 11, 12], response_length=2, reward=0.0)
    sample.metadata["tau2_opd_teacher_response"] = {
        "meta_info": {"input_token_logprobs": [[None, 0], [-0.1, 10], [-0.2, 11], [-0.3, 12]]}
    }
    tau2_opd.post_process_rewards(SimpleNamespace(), [sample])
    assert sample.teacher_log_probs == [-0.2, -0.3]
    assert sample.reward == 0.0


def test_k1_zero_reward_group_is_kept(monkeypatch):
    monkeypatch.setenv("TAU2_REPLACE_ZERO_SIGNAL_GROUPS", "0")
    monkeypatch.setenv("TAU2_DROP_UNIFORM_OUTCOME_GROUPS", "0")
    sample = _sample(tokens=[1, 2], response_length=1, reward=0.0)
    assert filters.drop_zero_std_or_unsampleable(SimpleNamespace(), [sample]).keep


def test_generate_opd_saves_task_reward_and_clears_training_reward(monkeypatch):
    sample = _sample(tokens=[1, 2], response_length=1, reward=None)

    async def fake_generate(_args, _sample, _sampling_params, **_kwargs):
        _sample.reward = 0.5
        return _sample

    monkeypatch.setattr(rollout, "generate", fake_generate)
    result = asyncio.run(rollout.generate_opd(SimpleNamespace(), sample, {}, evaluation=False))
    assert result.reward is None
    assert result.metadata["tau2_opd_task_reward"] == 0.5
    assert result.metadata["raw_reward"] == 0.5


def test_async_producer_fetches_teacher_before_group_metrics(monkeypatch):
    sample = _sample(tokens=[1, 2], response_length=1, reward=0.25)
    monkeypatch.setenv("TAU2_OPD_PURE", "1")

    async def fake_reward(_args, _sample):
        return {"meta_info": {"input_token_logprobs": [[None, 0], [-0.1, 1]]}}

    monkeypatch.setattr(tau2_opd, "reward_func", fake_reward)
    result = continuous._apply_opd_teacher(SimpleNamespace(), sample)
    assert result.reward == 0.0
    assert result.metadata["tau2_opd_task_reward"] == 0.25
    assert result.metadata["tau2_opd_teacher_response"]["meta_info"]


def test_segmented_opd_scores_each_context_and_preserves_alignment(monkeypatch):
    sample = _sample(tokens=[10, 11, 12], response_length=1, reward=0.25)
    sample.index = 25
    sample.training_segments = [
        {"tokens": [10, 11, 12], "response_length": 1,
         "loss_mask": [1], "rollout_log_probs": [-0.1]},
        {"tokens": [10, 21, 22, 23, 24, 25], "response_length": 3,
         "loss_mask": [1, 0, 1], "rollout_log_probs": [-0.2, 0.0, -0.3]},
    ]
    requested = []

    async def fake_request(url, payload):
        requested.append((url, payload["input_ids"]))
        tokens = payload["input_ids"]
        return {"meta_info": {"input_token_logprobs":
                [[None, tokens[0]]] + [[-float(t), t] for t in tokens[1:]]}}

    monkeypatch.setenv("TAU2_OPD_PURE", "1")
    monkeypatch.setenv("TAU2_OPD_AIRLINE_URL", "http://air/generate")
    monkeypatch.setattr(tau2_opd, "_request_teacher", fake_request)
    continuous._apply_opd_teacher(SimpleNamespace(), sample)
    assert requested == [("http://air/generate", s["tokens"]) for s in sample.training_segments]
    raw, rewards = tau2_opd.post_process_rewards(SimpleNamespace(), [sample])
    expanded, expanded_raw, expanded_rewards = expand_training_segments([sample], raw, rewards)
    assert [p.teacher_log_probs for p in expanded] == [[-12.0], [-23.0, -24.0, -25.0]]
    assert [len(p.teacher_log_probs) for p in expanded] == [p.response_length for p in expanded]
    assert [p.loss_mask for p in expanded] == [[1], [1, 0, 1]]
    assert [p.rollout_id for p in expanded] == [25, 25]
    assert expanded_raw == [0.25, 0.25]
    assert expanded_rewards == [0.0, 0.0]


def test_segmented_opd_rejects_missing_or_short_segment_scores():
    sample = _sample(reward={"tau2_opd_segment_responses": []})
    sample.training_segments = [{"tokens": [10, 11, 12], "response_length": 2}]
    with pytest.raises(ValueError):
        tau2_opd.post_process_rewards(SimpleNamespace(), [sample])
    sample.reward["tau2_opd_segment_responses"] = [{"meta_info": {"input_token_logprobs": [[None, 10]]}}]
    with pytest.raises(ValueError, match="teacher input log-prob length"):
        tau2_opd.post_process_rewards(SimpleNamespace(), [sample])


def _processed_row(task_id: str, domain: str, db_path: Path) -> dict:
    return {
        "prompt": task_id,
        "metadata": {"domain": domain, "db_path": str(db_path), "task": {"id": task_id}},
    }


@pytest.mark.parametrize("domains", [("airline", "retail"), ("airline", "retail", "telecom", "banking")])
def test_merge_sources_validates_and_preserves_domain_counts(tmp_path, domains):
    sources, rows = [], []
    for domain in domains:
        db = tmp_path / f"{domain}.json"
        db.write_text("{}")
        source = tmp_path / f"{domain}.jsonl"
        row = _processed_row(f"{domain}_1", domain, db)
        source.write_text(json.dumps(row) + "\n")
        sources.append(source)
        rows.append(row)
    output = tmp_path / "mixed.jsonl"
    assert prepare_data.merge_sources(sources, output, required_domains=domains) == {d: 1 for d in domains}
    assert [json.loads(line) for line in output.read_text().splitlines()] == rows


def test_merge_sources_rejects_empty_requested_domain(tmp_path):
    db = tmp_path / "db.json"
    db.write_text("{}")
    source = tmp_path / "airline.jsonl"
    source.write_text(json.dumps(_processed_row("airline_1", "airline", db)) + "\n")
    with pytest.raises(ValueError, match="missing requested domains: banking"):
        prepare_data.merge_sources([source], tmp_path / "out.jsonl", required_domains=("airline", "banking"))


def test_merge_sources_rejects_domain_task_mismatch(tmp_path):
    db = tmp_path / "db.json"
    db.write_text("{}")
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(_processed_row("retail_1", "airline", db)) + "\n")
    with pytest.raises(ValueError, match="domain does not match task id"):
        prepare_data.merge_sources([source], tmp_path / "out.jsonl")
