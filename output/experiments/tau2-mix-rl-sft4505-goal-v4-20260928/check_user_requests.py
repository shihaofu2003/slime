"""Paired User-only probes with the existing simulator and recorded Agent context."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RL = ROOT / "examples/tau2-bench/rl"
sys.path[:0] = [str(ROOT), str(RL), str(ROOT / "output/experiments/tau2-areal-async-rl/dependencies/tau2/src")]
os.environ.setdefault("TAU2_DATA_DIR", str(ROOT.parent / "tau2-bench/data"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from loguru import logger
logger.remove()
from tau2.data_model.tasks import Task
from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import is_valid_user_history_message


def main():
    directory = Path(__file__).resolve().parent
    previous = ROOT / "output/experiments/tau2-mix-rl-sft4505-reward-v3-20260927"
    examples = json.loads((previous / "domain_diagnosis_20260928/examples.json").read_text())
    sources = {
        "v3": {row["metadata"]["task"]["id"]: row["metadata"] for row in map(json.loads, (previous / "data/train.jsonl").open())},
        "v4": {row["metadata"]["task"]["id"]: row["metadata"] for row in map(json.loads, (directory / "data/train.jsonl").open())},
    }
    roles = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
    cases = [(task_id, index, arm) for task_id, index in (("retail_265", 24), ("retail_541", 0), ("retail_347", 26)) for arm in sources]

    def run(case):
        task_id, index, arm = case
        example = next(e for e in examples if e["task"] == task_id)
        task = Task.model_validate(sources[arm][task_id]["task"])
        messages = [roles[m["role"]].model_validate({k: v for k, v in m.items() if k != "index"}) for m in example["messages"]]
        history = [m for m in messages[:index] if is_valid_user_history_message(m)]
        user = UserSimulator(llm="openai/Qwen3.6-27B-tau2-user-nonthinking", instructions=str(task.user_scenario),
                             tools=None, llm_args=dict(api_base="http://10.119.98.94:30000/v1", api_key="EMPTY", temperature=0,
                                                      top_p=1, max_tokens=512, extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
        user.set_seed(42)
        response, state = user.generate_next_message(messages[index], user.get_init_state(history))
        result = dict(task=task_id, arm=arm, agent_message_index=index, agent_message=messages[index].content,
                      user_response=response.content)
        if task_id == "retail_265" and arm == "v4":
            result["followups"] = []
            for content in (
                "Order #W8367851 has been returned and will be refunded to your Visa ending in 8923. Is there anything else you need?",
                "Order #W5712077 has also been returned, with the refund to your gift card ending in 2736. All the keyboard and return changes are complete. Anything else?",
            ):
                followup, state = user.generate_next_message(AssistantMessage(role="assistant", content=content), state)
                result["followups"].append(dict(agent=content, user=followup.content))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, cases))
    (directory / "user_probe_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
