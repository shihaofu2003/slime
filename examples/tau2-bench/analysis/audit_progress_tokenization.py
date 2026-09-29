"""Replay saved smoke tokenization in the job image, without generating new trajectories."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.simulation import SimulationRun
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

from agent_contract import openai_tool_schemas
from envs import make_environment_constructor
from rollout import _fill_sample_from_simulation


def audit(stage_dir, checkpoint_dir):
    os.environ["TAU2_TURN_CREDIT_VERSION"] = "progress-rtg-v1"
    tokenizer = AutoTokenizer.from_pretrained(os.environ["HF_CHECKPOINT"])
    generator = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type="qwen3_full")
    with (stage_dir / "credit.jsonl").open() as stream:
        accepted = {row["sample_index"]: row for row in map(json.loads, stream)}
    counts = Counter()
    examples = []
    with (stage_dir / "trajectories/smoke.jsonl").open() as stream:
        for row in map(json.loads, stream):
            if row["sample_index"] not in accepted:
                continue
            metadata = row["metadata"]
            environment = make_environment_constructor(domain=metadata["tau2_domain"], db_path=Path(metadata["db_path"]))()
            prompt = LLMAgent.system_prompt.fget(SimpleNamespace(domain_policy=environment.get_policy()))
            sample = Sample(metadata=metadata)
            _fill_sample_from_simulation(
                generator=generator, sample=sample, simulation=SimulationRun.model_validate(row["simulation"]),
                reward_value=row["reward"], system_prompt=prompt, protocol_profile="official-native",
                native_tools=openai_tool_schemas(environment.get_tools()),
                field_reward_signals=metadata["tau2_field_reward_signals"],
            )
            assert sample.loss_mask == row["loss_mask"]
            assert sample.response_length == row["response_length"]
            old_turns = accepted[row["sample_index"]]["turns"]
            turns = sample.metadata["tau2_turn_credits"]
            assert [(t["simulation_message_index"], t["response_span"]) for t in turns] == [
                (t["simulation_message_index"], t["response_span"]) for t in old_turns]
            counts["trajectories"] += 1
            for turn in turns:
                diagnostic = turn["raw_rerender"]
                counts["turns"] += 1
                counts["body_mismatches"] += not diagnostic["body_equal"]
                if not diagnostic["body_equal"] and len(examples) < 5:
                    examples.append({"sample_index": row["sample_index"], "source_index": turn["simulation_message_index"], **diagnostic})
    state = torch.load(checkpoint_dir / "rollout/global_dataset_state_dict_0.pt", weights_only=False)
    domain_state = state["metadata"]["tau2_domain_quota_v1"]
    assert domain_state["domain_offsets"] == {"airline": 1, "retail": 2, "telecom": 3}
    assert state["sample_group_index"] == 6 and state["sample_index"] == 48
    result = {"diagnostic_version": 2, "counts": dict(counts), "mismatch_examples": examples,
              "saved_domain_state": domain_state, "sample_group_index": state["sample_group_index"],
              "sample_index": state["sample_index"]}
    (stage_dir / "tokenization_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.stage_dir, args.checkpoint_dir)
