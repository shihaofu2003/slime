"""Measure initial Agent context using its real system prompt and tool schemas."""

import json
from pathlib import Path
from types import SimpleNamespace

from verify_goals import ROOT, goal
from agent import TrainableSGLangAgent


def main():
    directory = Path(__file__).resolve().parent
    goal.setup_worker()
    checkpoint = ROOT.parent / "checkpoints/Qwen3-4B-Instruct-2507_tau2_agent_sft_areal3_banking_simplified_full_20260920/iter_0004505_hf"
    rows = []
    for line in (directory / "data/train.jsonl").open():
        metadata = json.loads(line)["metadata"]
        if metadata["domain"] != "banking":
            continue
        task = goal.task_from_metadata(metadata)
        environment = goal.make_environment_constructor(domain="banking", db_path=Path(metadata["db_path"]), task=task)()
        agent = TrainableSGLangAgent(tools=environment.get_tools(), domain_policy=environment.get_policy(), llm=str(checkpoint),
                                    llm_args={"api_base": "http://unused", "protocol_profile": "official-native"}, domain="banking")
        messages = task.initial_state.message_history or [] if task.initial_state else []
        chat = agent._chat_messages(SimpleNamespace(messages=messages))
        tokens = agent.tokenizer.apply_chat_template(chat, tools=agent.native_tools, tokenize=True, return_dict=False,
                                                    add_generation_prompt=True, enable_thinking=False)
        rows.append(dict(task=task.id, evidence_mode=task.retrieval_variant, initial_tokens=len(tokens)))
    (directory / "initial_prompt_tokens.json").write_text(json.dumps(rows, indent=2) + "\n")
    by_task = {row["task"]: row["initial_tokens"] for row in rows}
    checks_path = directory / "banking_reference_checks.jsonl"
    checks = [json.loads(line) for line in checks_path.open()]
    for check in checks:
        check["initial_tokens"] = by_task[check["task"]]
    checks_path.write_text("".join(json.dumps(row) + "\n" for row in checks))
    summary_path = directory / "reference_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["initial_tokens_max"] = max(by_task.values())
    summary["initial_context_over_16k"] = [task for task, n in by_task.items() if n >= 16384]
    summary["prompt_measurement"] = "Real TrainableSGLangAgent official-native prompt/tools and SFT4505 tokenizer"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("initial_tokens_max", "initial_context_over_16k", "prompt_measurement")}, indent=2))


if __name__ == "__main__":
    main()
