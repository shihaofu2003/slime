#!/usr/bin/env python3
"""Build and validate the Tau2 Banking task-decomposition curriculum."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from atomic_gap_analysis import extract_banking_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVICE_AGENT_ROOT = PROJECT_ROOT.parent
TAU2_ROOT = SERVICE_AGENT_ROOT / "tau2-bench"
DEFAULT_TASKS_DIR = TAU2_ROOT / "data/tau2/domains/banking_knowledge/tasks"
DEFAULT_AGGREGATE_TASKS = TAU2_ROOT / "data/tau2/domains/banking_knowledge/tasks.json"
DEFAULT_DOCUMENTS_DIR = TAU2_ROOT / "data/tau2/domains/banking_knowledge/documents"
DEFAULT_TOOLS_FILE = TAU2_ROOT / "src/tau2/domains/banking_knowledge/tools.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/experiments/tau2-banking-task-curriculum"

HISTORICAL_RUNS = {
    "qwen3_a": PROJECT_ROOT
    / "output/experiments/tau2-eval-official-native-full/eval/four-domain-full-bm25"
    / "seed300_0902_044845_summary.json",
    "qwen3_b": PROJECT_ROOT
    / "output/experiments/tau2-eval-official-native-full/eval/four-domain-full-bm25"
    / "seed300_0902_060330_summary.json",
    "qwen35_nonthinking": PROJECT_ROOT
    / "output/experiments/tau2-eval-qwen35-official-native-four-domain/eval"
    / "four-domain-full-bm25-nonthinking/seed300_0902_054605_summary.json",
}

CATEGORIES = {
    "信用卡选择与申请资格": [1, 2, 3, 6, 7, 8, 23, 24, 25],
    "身份验证与资料变更": [4, 5],
    "信用卡推荐奖励": [10, 14, 15, 16],
    "知识缺失与异常转人工": [12, 32, 33, 34, 35],
    "返现与奖励核算": [17, 18, 19, 20, 21, 22, 26, 27, 28, 29],
    "交易争议、购买保护与补卡": [31, 36, 37, 38, 39, 40, 41, 53, 54],
    "信用卡保留、销户与竞品比较": [43, 44, 45, 46, 47, 48, 49],
    "信用额度调整": [50, 51, 52],
    "账户推荐、开户、销户与入金": list(range(55, 72)),
    "ATM 费用与旅行支票账户": list(range(72, 77)),
    "卡片遗失或被盗": list(range(77, 82)),
    "借记卡交易争议": list(range(82, 87)),
    "借记卡拒付与 PIN 锁定": list(range(87, 93)),
    "储蓄利息核算": list(range(93, 98)),
    "账户推荐奖励": list(range(98, 103)),
}

CATEGORY_DETAILS = {
    "信用卡选择与申请资格": {
        "product_families": ["credit_card"],
        "operation_modes": ["recommend", "apply", "refuse_if_ineligible"],
        "reasoning": ["extract", "compare", "filter", "classify"],
        "primary_capabilities": ["跨文档比较", "约束过滤", "动作选择"],
    },
    "身份验证与资料变更": {
        "product_families": ["customer_profile"],
        "operation_modes": ["verify", "update", "escalate"],
        "reasoning": ["classify", "exception_check"],
        "primary_capabilities": ["身份验证", "前置条件检查", "拒答与转人工"],
    },
    "信用卡推荐奖励": {
        "product_families": ["credit_card_referral"],
        "operation_modes": ["check_eligibility", "refer", "escalate"],
        "reasoning": ["extract", "filter", "classify"],
        "primary_capabilities": ["单文档事实抽取", "前置条件检查", "用户工具协调"],
    },
    "知识缺失与异常转人工": {
        "product_families": ["support_escalation"],
        "operation_modes": ["refuse", "escalate"],
        "reasoning": ["classify", "exception_check"],
        "primary_capabilities": ["拒答与转人工", "动作顺序"],
    },
    "返现与奖励核算": {
        "product_families": ["credit_card_rewards", "credit_card_transaction"],
        "operation_modes": ["calculate", "dispute", "correct"],
        "reasoning": ["extract", "calculate", "classify"],
        "primary_capabilities": ["数值计算", "实体—状态绑定", "参数构造"],
    },
    "交易争议、购买保护与补卡": {
        "product_families": ["credit_card_transaction", "credit_card_replacement"],
        "operation_modes": ["investigate", "dispute", "replace"],
        "reasoning": ["extract", "classify", "exception_check"],
        "primary_capabilities": ["实体—状态绑定", "动作选择", "参数构造"],
    },
    "信用卡保留、销户与竞品比较": {
        "product_families": ["credit_card_retention", "credit_card_closure"],
        "operation_modes": ["check_eligibility", "retain", "close"],
        "reasoning": ["compare", "filter", "exception_check"],
        "primary_capabilities": ["前置条件检查", "跨文档比较", "动作顺序"],
    },
    "信用额度调整": {
        "product_families": ["credit_limit"],
        "operation_modes": ["check_eligibility", "approve", "deny"],
        "reasoning": ["calculate", "classify", "exception_check"],
        "primary_capabilities": ["前置条件检查", "数值计算", "动作顺序"],
    },
    "账户推荐、开户、销户与入金": {
        "product_families": ["checking_account", "savings_account"],
        "operation_modes": ["recommend", "open", "fund", "close"],
        "reasoning": ["compare", "filter", "calculate", "exception_check"],
        "primary_capabilities": ["跨文档比较", "约束过滤", "动作顺序"],
    },
    "ATM 费用与旅行支票账户": {
        "product_families": ["checking_account", "atm_fee"],
        "operation_modes": ["calculate", "credit", "recommend"],
        "reasoning": ["calculate", "compare", "classify"],
        "primary_capabilities": ["数值计算", "实体—状态绑定", "跨文档比较"],
    },
    "卡片遗失或被盗": {
        "product_families": ["debit_card", "credit_card"],
        "operation_modes": ["freeze", "close", "replace", "dispute"],
        "reasoning": ["classify", "exception_check"],
        "primary_capabilities": ["子任务覆盖", "实体—状态绑定", "动作顺序"],
    },
    "借记卡交易争议": {
        "product_families": ["debit_card_transaction"],
        "operation_modes": ["investigate", "dispute", "replace"],
        "reasoning": ["classify", "exception_check"],
        "primary_capabilities": ["实体—状态绑定", "前置条件检查", "参数构造"],
    },
    "借记卡拒付与 PIN 锁定": {
        "product_families": ["debit_card", "pin"],
        "operation_modes": ["diagnose", "reset", "replace", "escalate"],
        "reasoning": ["classify", "calculate", "exception_check"],
        "primary_capabilities": ["观察后恢复", "实体—状态绑定", "动作选择"],
    },
    "储蓄利息核算": {
        "product_families": ["savings_account", "interest"],
        "operation_modes": ["calculate", "credit", "report"],
        "reasoning": ["extract", "calculate", "compare"],
        "primary_capabilities": ["数值计算", "跨文档比较", "参数构造"],
    },
    "账户推荐奖励": {
        "product_families": ["bank_account_referral"],
        "operation_modes": ["compare", "check_eligibility", "refer"],
        "reasoning": ["compare", "filter", "calculate"],
        "primary_capabilities": ["跨文档比较", "约束过滤", "前置条件检查"],
    },
}

ALL_CAPABILITIES = {
    "查询构造",
    "检索覆盖",
    "文档相关性判断",
    "单文档事实抽取",
    "跨文档比较",
    "约束过滤",
    "数值计算",
    "用户信息补全",
    "身份验证",
    "实体—状态绑定",
    "工具发现",
    "动作选择",
    "参数构造",
    "前置条件检查",
    "动作顺序",
    "用户工具协调",
    "观察后恢复",
    "子任务覆盖",
    "拒答与转人工",
}

EXPECTED_AGGREGATE_DRIFT = {
    "task_027",
    "task_046",
    "task_048",
    "task_053",
    "task_056",
    "task_061",
    "task_062",
    "task_081",
    "task_083",
    "task_084",
    "task_085",
    "task_088",
    "task_102",
}

SECONDARY_CATEGORIES = {
    "task_047": ["信用卡选择与申请资格"],
    "task_048": ["信用卡选择与申请资格"],
    "task_053": ["信用额度调整"],
    "task_054": ["信用额度调整"],
    **{f"task_{number:03d}": ["信用卡选择与申请资格"] for number in range(58, 70)},
    "task_080": ["交易争议、购买保护与补卡"],
    "task_081": ["交易争议、购买保护与补卡", "知识缺失与异常转人工"],
    "task_102": ["身份验证与资料变更"],
}

PILOT_SOURCES = {
    "task_001": "信用卡选择与申请资格",
    "task_031": "交易争议、购买保护与补卡",
    "task_071": "账户推荐、开户、销户与入金",
    "task_036": "交易争议、购买保护与补卡",
}
PILOT_VARIANTS = (
    "evidence_given",
    "retrieval_only",
    "decision_only",
    "single_action",
    "two_skill_composition",
    "full_composition",
)

# These runtime tasks remain useful inventory rows, but their current reference
# action chains do not replay cleanly against the current Banking environment.
EXPANSION_EXCLUDED_SOURCES = {
    "task_051": "A credit-limit request already exists before the reference write.",
    "task_077": "The reference chain names an agent tool unavailable in the current environment.",
    "task_083": "Four dispute calls omit the current customer_max_liability_amount argument.",
}

WORKFLOW_ONLY_ACTIONS = {
    "log_verification",
    "unlock_discoverable_agent_tool",
    "give_discoverable_user_tool",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_runtime_tasks(tasks_dir: Path) -> dict[str, dict[str, Any]]:
    tasks = {}
    for path in sorted(tasks_dir.glob("task_*.json")):
        task = read_json(path)
        if task["id"] in tasks:
            raise ValueError(f"duplicate runtime task id: {task['id']}")
        tasks[task["id"]] = task
    return tasks


def load_documents(documents_dir: Path) -> dict[str, dict[str, Any]]:
    documents = {}
    for path in sorted(documents_dir.glob("*.json")):
        document = read_json(path)
        if document["id"] in documents:
            raise ValueError(f"duplicate document id: {document['id']}")
        documents[document["id"]] = document
    return documents


def task_id(number: int) -> str:
    return f"task_{number:03d}"


def category_lookup() -> dict[str, str]:
    result = {}
    for category, numbers in CATEGORIES.items():
        for number in numbers:
            identifier = task_id(number)
            if identifier in result:
                raise ValueError(f"task appears in two categories: {identifier}")
            result[identifier] = category
    return result


def normalize_snapshot(value: Any) -> Any:
    """Remove serialization-only defaults before comparing task snapshots."""

    if isinstance(value, dict):
        return {
            key: normalize_snapshot(item)
            for key, item in value.items()
            if item is not None and key != "annotations"
        }
    if isinstance(value, list):
        return [normalize_snapshot(item) for item in value]
    return value


def changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    left = normalize_snapshot(left)
    right = normalize_snapshot(right)
    if type(left) is not type(right):
        return [prefix or "/"]
    if isinstance(left, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}/{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix]
        paths = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(changed_paths(left_item, right_item, f"{prefix}/{index}"))
        return paths
    return [] if left == right else [prefix]


def parse_tool_types(tools_file: Path) -> dict[str, str]:
    """Read ToolType decorators without importing the Tau2 environment."""

    tree = ast.parse(tools_file.read_text(encoding="utf-8"))
    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            function_name = decorator.func.id if isinstance(decorator.func, ast.Name) else None
            if function_name not in {"is_tool", "is_discoverable_tool"}:
                continue
            tool_type = decorator.args[0]
            if isinstance(tool_type, ast.Attribute) and tool_type.attr in {"READ", "WRITE", "GENERIC", "THINK"}:
                result[node.name] = tool_type.attr.lower()
    return result


def effective_action(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = action["name"]
    arguments = action.get("arguments") or {}
    if name == "call_discoverable_agent_tool":
        inner_name = arguments["agent_tool_name"]
        inner_arguments = json.loads(arguments.get("arguments") or "{}")
        return inner_name, inner_arguments
    if name == "call_discoverable_user_tool":
        inner_name = arguments["discoverable_tool_name"]
        inner_arguments = json.loads(arguments.get("arguments") or "{}")
        return inner_name, inner_arguments
    return name, arguments


def concise_intent(task: dict[str, Any]) -> str:
    description = task.get("description") or {}
    notes = str(description.get("notes") or "").strip()
    if notes:
        first = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", notes))[0]
        return first[:500].strip()

    instructions = (task.get("user_scenario") or {}).get("instructions") or ""
    if isinstance(instructions, dict):
        instructions = " ".join(
            str(instructions.get(key) or "") for key in ("reason_for_call", "task_instructions")
        )
    text = re.sub(r"[*#]", "", str(instructions))
    text = re.sub(r"\s+", " ", text).strip()
    for marker in ("Your goal:", "You're looking for", "You want to"):
        if marker.lower() in text.lower():
            start = text.lower().index(marker.lower())
            sentences = re.split(r"(?<=[.!?])\s+", text[start:])
            return " ".join(sentences[:3])[:700].strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        if re.search(r"\b(?:want|wants|need|needs|looking)\s+to\b", sentence, re.IGNORECASE):
            return sentence[:500].strip()
    return sentences[0][:500].strip()


def normalized_document_line(raw_line: str) -> str | None:
    line = raw_line.strip().lstrip("-* ").replace("**", "").strip()
    if line.startswith("|") and line.endswith("|"):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2 and not all(set(cell) <= {"-", ":"} for cell in cells):
            line = f"{cells[0]}: {'; '.join(cells[1:])}"
    if not line or line.startswith("#") or re.fullmatch(r"[|:\- ]+", line):
        return None
    if len(line) < 12 or len(line) > 350:
        return None
    return line


def salient_document_fact(
    document: dict[str, Any],
    task_terms: set[str],
    priority_terms: set[str] | None = None,
) -> str:
    """Select one concrete source sentence for the task's evidence inventory."""

    candidates = []
    for raw_line in str(document.get("content") or "").splitlines():
        line = normalized_document_line(raw_line)
        if line is None:
            continue
        terms = set(re.findall(r"[a-z0-9_]+", line.lower()))
        terms |= {term[:-1] for term in terms if len(term) > 3 and term.endswith("s")}
        overlap = len(terms & task_terms)
        priority_overlap = len(terms & (priority_terms or set()))
        concrete = int(bool(re.search(r"\d|\$|%|yes|no|must|required|eligible|fee|apy", line.lower())))
        candidates.append((10 * priority_overlap + overlap, concrete, -len(line), line))
    if not candidates:
        return document["title"]
    return max(candidates)[3]


def document_fact_locator(document: dict[str, Any], fact: str) -> str:
    """Describe one source location precisely without repeating the target line."""

    heading = "document start"
    prose_index = 0
    bullet_index = 0
    for raw_line in str(document.get("content") or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("# ").strip()
            prose_index = 0
            bullet_index = 0
            continue
        line = normalized_document_line(raw_line)
        if line is None:
            continue
        if stripped.startswith("|"):
            first_cell = stripped.strip("|").split("|", 1)[0].strip()
            locator = f"the table row whose first cell is `{first_cell}`"
        elif stripped.startswith(("-", "*")):
            bullet_index += 1
            locator = f"bullet {bullet_index} under section `{heading}`"
        else:
            prose_index += 1
            locator = f"prose line {prose_index} under section `{heading}`"
        if line == fact:
            return locator
    return "the document title"


def comparison_documents(category: str, required: list[str], documents: dict[str, dict[str, Any]]) -> list[str]:
    comparison_categories = {
        "信用卡选择与申请资格",
        "信用卡保留、销户与竞品比较",
        "账户推荐、开户、销户与入金",
        "ATM 费用与旅行支票账户",
        "储蓄利息核算",
        "账户推荐奖励",
    }
    if category not in comparison_categories:
        return []
    result = []
    for document_id in required:
        title = documents[document_id]["title"].lower()
        if "(general)" not in document_id and "internal:" not in title:
            result.append(document_id)
        elif "promotion" in title:
            result.append(document_id)
    return result


def target_entities(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_types = {
        "transaction_id": "transaction",
        "credit_card_account_id": "credit_card_account",
        "account_id": "bank_account",
        "source_account_id": "bank_account",
        "destination_account_id": "bank_account",
        "debit_card_id": "debit_card",
        "card_id": "card",
        "card_type": "credit_card_product",
        "account_class": "bank_account_product",
        "referral_id": "referral",
    }
    values: dict[str, set[str]] = {}
    for action in actions:
        _, arguments = effective_action(action)
        for key, value in arguments.items():
            kind = key_types.get(key)
            if kind and value is not None:
                values.setdefault(kind, set()).add(str(value))
    return [{"type": kind, "count": len(items)} for kind, items in sorted(values.items())]


def action_summary(action: dict[str, Any]) -> dict[str, Any]:
    effective_name, effective_arguments = effective_action(action)
    return {
        "action_id": action["action_id"],
        "requestor": action.get("requestor", "assistant"),
        "name": effective_name,
        "arguments": effective_arguments,
    }


def graph_for_task(task: dict[str, Any]) -> dict[str, Any]:
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    nodes = []
    evidence_id = f"{task['id']}:evidence"
    decision_id = f"{task['id']}:decision"
    nodes.append(
        {
            "id": evidence_id,
            "kind": "evidence",
            "description": "Collect all benchmark-declared evidence needed by the task.",
            "depends_on": [],
            "source_refs": [f"document:{item}" for item in task.get("required_documents") or []],
        }
    )
    nodes.append(
        {
            "id": decision_id,
            "kind": "decision",
            "description": concise_intent(task),
            "depends_on": [evidence_id],
            "source_refs": [],
        }
    )
    previous = decision_id
    for action in actions:
        node_id = f"{task['id']}:action:{action['action_id']}"
        effective_name, _ = effective_action(action)
        nodes.append(
            {
                "id": node_id,
                "kind": "action",
                "description": f"{action.get('requestor', 'assistant')} executes {effective_name}.",
                "depends_on": [previous],
                "source_refs": [f"action:{action['action_id']}"],
            }
        )
        previous = node_id
    nodes.append(
        {
            "id": f"{task['id']}:state:final",
            "kind": "state",
            "description": "The configured reward criteria reach their reference end state.",
            "depends_on": [previous],
            "source_refs": [],
        }
    )
    return {"nodes": nodes}


def resolve_results_file(summary_path: Path) -> Path:
    summary = read_json(summary_path)
    relative = Path(summary["domains"]["banking_knowledge"]["results_file"])
    if relative.is_absolute():
        return relative
    return TAU2_ROOT / relative


def historical_run_metrics(
    task: dict[str, Any], result: dict[str, Any], *, label: str, source_path: Path
) -> dict[str, Any]:
    result_tasks = {item["id"]: item for item in result["tasks"]}
    historical_task = result_tasks[task["id"]]
    differences = changed_paths(historical_task, task)
    simulations = [item for item in result["simulations"] if item["task_id"] == task["id"]]
    successes = 0
    search_calls = 0
    document_hits = 0
    document_opportunities = 0
    matched_actions = 0
    action_opportunities = 0
    db_matches = 0
    db_checks = 0
    for simulation in simulations:
        reward_info = simulation.get("reward_info") or {}
        successes += int(bool(reward_info.get("reward")))
        features = extract_banking_features(simulation, historical_task)
        search_calls += len(features["retrieval_events"])
        document_hits += len(features["required_document_hits"])
        document_opportunities += len(features["required_documents"])
        for check in reward_info.get("action_checks") or []:
            action_opportunities += 1
            matched_actions += int(bool(check.get("action_match")))
        db_check = reward_info.get("db_check") or {}
        if db_check.get("db_match") is not None:
            db_checks += 1
            db_matches += int(bool(db_check["db_match"]))
    return {
        "label": label,
        "source": str(source_path.relative_to(SERVICE_AGENT_ROOT)),
        "compatible_with_runtime_task": not differences,
        "changed_paths": differences,
        "trials": len(simulations),
        "successes": successes,
        "search_calls": search_calls,
        "gold_document_hits": document_hits,
        "gold_document_opportunities": document_opportunities,
        "gold_document_coverage": (
            round(document_hits / document_opportunities, 6) if document_opportunities else None
        ),
        "matched_reference_actions": matched_actions,
        "reference_action_opportunities": action_opportunities,
        "db_matches": db_matches,
        "db_checks": db_checks,
    }


def load_historical_results(run_summaries: dict[str, Path]) -> dict[str, tuple[Path, dict[str, Any]]]:
    results = {}
    for label, summary_path in run_summaries.items():
        results_path = resolve_results_file(summary_path)
        results[label] = (results_path, read_json(results_path))
    return results


def build_annotation(
    task: dict[str, Any],
    category: str,
    documents: dict[str, dict[str, Any]],
    tool_types: dict[str, str],
) -> dict[str, Any]:
    details = CATEGORY_DETAILS[category]
    criteria = task.get("evaluation_criteria") or {}
    actions = criteria.get("actions") or []
    summarized_actions = [action_summary(action) for action in actions]
    assistant_read = 0
    assistant_write = 0
    assistant_workflow = 0
    user_actions = 0
    dynamic_agent_tools = []
    dynamic_user_tools = []
    business_actions = []
    for action in actions:
        effective_name, _ = effective_action(action)
        if action.get("requestor", "assistant") == "user":
            user_actions += 1
        else:
            action_kind = tool_types.get(effective_name, tool_types.get(action["name"], "generic"))
            if action_kind == "read":
                assistant_read += 1
            elif action_kind == "write":
                assistant_write += 1
            else:
                assistant_workflow += 1
        if action["name"] == "call_discoverable_agent_tool" and effective_name not in dynamic_agent_tools:
            dynamic_agent_tools.append(effective_name)
        if action["name"] == "call_discoverable_user_tool" and effective_name not in dynamic_user_tools:
            dynamic_user_tools.append(effective_name)
        if action["name"] not in WORKFLOW_ONLY_ACTIONS:
            business_actions.append(action_summary(action))

    required = list(task.get("required_documents") or [])
    instruction_text = json.dumps((task.get("user_scenario") or {}).get("instructions") or "", ensure_ascii=False)
    source_text = f"{concise_intent(task)} {instruction_text}".lower()
    supporting_capabilities = ["查询构造", "检索覆盖", "文档相关性判断", "单文档事实抽取"]
    if any(action["name"] == "log_verification" for action in actions):
        supporting_capabilities.append("身份验证")
    if dynamic_agent_tools:
        supporting_capabilities.append("工具发现")
    if dynamic_user_tools:
        supporting_capabilities.append("用户工具协调")
    if len(business_actions) > 1:
        supporting_capabilities.append("子任务覆盖")
    supporting_capabilities = [
        item for item in dict.fromkeys(supporting_capabilities) if item not in details["primary_capabilities"]
    ]

    description_notes = str((task.get("description") or {}).get("notes") or "")
    fact_context = f"{description_notes} {concise_intent(task)} {instruction_text}"
    fact_terms = set(re.findall(r"[a-z0-9_]+", fact_context.lower()))
    fact_terms |= {term[:-1] for term in fact_terms if len(term) > 3 and term.endswith("s")}
    fact_terms -= {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "bank",
        "card",
        "customer",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "rho",
        "the",
        "their",
        "to",
        "tool",
        "user",
        "with",
    }
    final_action_name, final_action_arguments = effective_action(actions[-1])
    fact_terms.update(re.findall(r"[a-z0-9_]+", final_action_name.lower()))
    final_action_text = json.dumps(list(final_action_arguments.values()), ensure_ascii=False)
    final_action_terms = set(re.findall(r"[a-z0-9_]+", final_action_text.lower()))
    facts = []
    for index, document_id in enumerate(required, start=1):
        facts.append(
            {
                "id": f"fact_{index:02d}",
                "statement": salient_document_fact(
                    documents[document_id],
                    fact_terms,
                    final_action_terms,
                ),
                "source_document_ids": [document_id],
            }
        )

    if "lie" in source_text or "incorrect" in source_text or "mistaken" in source_text:
        pressure = "misstated_claims"
    elif "persist" in source_text or "repeated" in source_text or "push back" in source_text:
        pressure = "repeated_requests"
    else:
        pressure = "none"

    return {
        "task_id": task["id"],
        "business": {
            "primary_category": category,
            "secondary_categories": SECONDARY_CATEGORIES.get(task["id"], []),
            "intent": concise_intent(task),
            "product_families": details["product_families"],
            "operation_modes": details["operation_modes"],
            "target_entities": target_entities(actions),
        },
        "evidence": {
            "gold_document_ids": required,
            "decisive_document_ids": required,
            "decisive_set_status": "conservative_upper_bound_from_benchmark_gold",
            "comparison_document_ids": comparison_documents(category, required, documents),
            "required_facts": facts,
            "gold_document_count": len(required),
            "decisive_document_count": len(required),
            "comparison_document_count": len(comparison_documents(category, required, documents)),
            "fact_count": len(facts),
        },
        "reasoning": {
            "operations": details["reasoning"],
            "primary_capabilities": details["primary_capabilities"],
            "supporting_capabilities": supporting_capabilities,
        },
        "interaction": {
            "requires_clarification": any(
                marker in source_text for marker in ("clarif", "vague", "ask for", "when asked")
            ),
            "requires_identity_verification": any(action["name"] == "log_verification" for action in actions),
            "has_progressive_disclosure": any(
                marker in source_text for marker in ("only mention", "when asked", "do not provide")
            ),
            "hidden_information": [
                label
                for label, present in (
                    ("verification_fields", any(action["name"] == "log_verification" for action in actions)),
                    (
                        "progressively_disclosed_constraints",
                        any(marker in source_text for marker in ("only mention", "when asked", "do not provide")),
                    ),
                    (
                        "ambiguous_entity_or_transaction_details",
                        any(marker in source_text for marker in ("clarif", "vague")),
                    ),
                )
                if present
            ],
            "user_pressure": pressure,
            "user_tool_coordination": bool(dynamic_user_tools),
        },
        "workflow": {
            "subtask_count": max(1, len(business_actions)),
            "assistant_read_actions": assistant_read,
            "assistant_write_actions": assistant_write,
            "assistant_workflow_actions": assistant_workflow,
            "user_actions": user_actions,
            "dynamic_agent_tools": dynamic_agent_tools,
            "dynamic_user_tools": dynamic_user_tools,
            "reference_actions": summarized_actions,
        },
        "success": {
            "reward_basis": criteria.get("reward_basis") or [],
            "checkpoints": [
                {
                    "id": f"checkpoint_{index:02d}",
                    "description": f"Complete {action['name']} with the reference entity and arguments.",
                    "action_refs": [action["action_id"]],
                }
                for index, action in enumerate(business_actions, start=1)
            ],
            "final_environment_assertions": criteria.get("env_assertions") or [],
            "required_communication": criteria.get("communicate_info") or [],
            "natural_language_assertions": criteria.get("nl_assertions") or [],
            "terminal_condition": "All configured reward-basis components match the reference task outcome.",
        },
        "graph": graph_for_task(task),
    }


def build_catalog(
    tasks: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    aggregate_tasks: dict[str, dict[str, Any]],
    tool_types: dict[str, str],
    historical_results: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    categories = category_lookup()
    annotations = []
    catalog = []
    for identifier, task in sorted(tasks.items()):
        annotation = build_annotation(task, categories[identifier], documents, tool_types)
        annotations.append(annotation)
        aggregate_differences = changed_paths(aggregate_tasks[identifier], task)
        history = {
            label: historical_run_metrics(task, result, label=label, source_path=source_path)
            for label, (source_path, result) in historical_results.items()
        }
        row = copy.deepcopy(annotation)
        row["source"] = {
            "runtime_file": str((DEFAULT_TASKS_DIR / f"{identifier}.json").relative_to(SERVICE_AGENT_ROOT)),
            "aggregate_matches_runtime": not aggregate_differences,
            "aggregate_changed_paths": aggregate_differences,
        }
        row["historical"] = history
        catalog.append(row)
    return annotations, catalog


def action_to_initialization(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_type": action.get("requestor", "assistant"),
        "func_name": action["name"],
        "arguments": copy.deepcopy(action.get("arguments") or {}),
    }


def pilot_id(source_id: str, variant: str) -> str:
    return f"banking_curriculum_{source_id}_{variant}"


def response_format_placeholder(info: str) -> str:
    """Keep the requested format visible without exposing its gold value."""

    if ":" not in info:
        return "<answer>"
    label = info.split(":", 1)[0].strip()
    value_names = {
        "annual fee": "amount",
        "cash back": "rate",
        "decision action": "tool_name",
        "document": "document_id",
        "fact": "verbatim_fact",
        "recommendation": "product_name",
    }
    return f"{label}: <{value_names.get(label.lower(), 'answer')}>"


def agent_facing_request(instructions: str, communicate_info: list[str] | None) -> str:
    """Turn a scenario directive into a first user request without leaking labels."""

    request = instructions
    for info in sorted(communicate_info or [], key=len, reverse=True):
        request = request.replace(info, response_format_placeholder(info))
    return request


CURRICULUM_USER_FOLLOWUP_INSTRUCTIONS = (
    "The initial User message is the entire synthetic task; do not start or continue any other customer "
    "goal from the source scenario. Do not call a customer tool. Let the assistant use any tools it needs. "
    "After the assistant gives its first substantive natural-language answer or confirms completion, "
    "respond exactly with ###STOP###, whether that answer is correct or incorrect. If the assistant asks "
    "a necessary clarification, answer only from the initial message and introduce no new request."
)

CURRICULUM_USER_TOOL_INSTRUCTIONS = (
    "The initial User message is the entire synthetic task; do not start or continue any other customer "
    "goal from the source scenario. Use only the customer-owned tool explicitly assigned by that message, "
    "with exactly its resolved payload and only at the stated trigger. Never call any other customer tool. "
    "After that tool returns, respond exactly with ###STOP###. If the assistant gives a final answer without "
    "satisfying the stated trigger, respond exactly with ###STOP### instead of introducing another request."
)


def make_pilot_task(
    source: dict[str, Any],
    *,
    variant: str,
    instructions: str,
    required_documents: list[str],
    actions: list[dict[str, Any]] | None,
    communicate_info: list[str] | None,
    reward_basis: list[str],
    initialization_actions: list[dict[str, Any]] | None = None,
    user_tools: list[str] | None = None,
) -> dict[str, Any]:
    initial_source = source.get("initial_state") or {}
    initial_user_request = agent_facing_request(instructions, communicate_info)
    return {
        "id": pilot_id(source["id"], variant),
        "description": {
            "purpose": f"Banking curriculum {variant} derived from {source['id']}",
            "relevant_policies": None,
            "notes": (
                f"Controlled {variant} benchmark diagnostic; excluded from training. "
                "Source task remains unchanged."
            ),
        },
        "user_scenario": {
            "persona": None,
            "instructions": (
                CURRICULUM_USER_TOOL_INSTRUCTIONS
                if user_tools
                else CURRICULUM_USER_FOLLOWUP_INSTRUCTIONS
            ),
        },
        "initial_state": {
            "initialization_data": copy.deepcopy(initial_source.get("initialization_data")),
            "initialization_actions": copy.deepcopy(initialization_actions),
            "message_history": [{"role": "user", "content": initial_user_request}],
        },
        "evaluation_criteria": {
            "actions": copy.deepcopy(actions),
            "env_assertions": None,
            "communicate_info": communicate_info,
            "nl_assertions": None,
            "reward_basis": reward_basis,
        },
        "required_documents": required_documents,
        "user_tools": user_tools if user_tools is not None else [],
    }


def user_tools_for_actions(
    source: dict[str, Any], actions: list[dict[str, Any]]
) -> list[str]:
    """Expose only customer tools that the bounded task actually evaluates."""

    expected = {
        action["name"]
        for action in actions
        if action.get("requestor", "assistant") == "user"
    }
    return [name for name in source.get("user_tools") or [] if name in expected]


def resolved_operation_goal(name: str, arguments: dict[str, Any]) -> str:
    operation = name.replace("_", " ")
    payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return f'the customer needs the "{operation}" operation with resolved fields `{payload}`'


def search_action(identifier: str, query: str) -> dict[str, Any]:
    return {
        "name": "KB_search",
        "arguments": {"query": query},
        "requestor": "assistant",
        "action_id": f"{identifier}_search",
        "compare_args": [],
    }


def contract(
    task: dict[str, Any],
    *,
    source_id: str,
    category: str,
    variant: str,
    level: str,
    retrieval_variant: str,
    capabilities: list[str],
    given_context: list[str],
    required_documents: list[str],
    reference_steps: list[dict[str, Any]],
    reference_messages: list[str],
    expected_changed_tables: list[str],
) -> dict[str, Any]:
    criteria = task["evaluation_criteria"]
    return {
        "task_id": task["id"],
        "source_task_id": source_id,
        "data_origin": "benchmark_derived_diagnostic",
        "training_eligible": False,
        "business_category": category,
        "variant": variant,
        "level": level,
        "retrieval_variant": retrieval_variant,
        "trained_capabilities": capabilities,
        "given_context": given_context,
        "success_contract": {
            "required_document_ids": required_documents,
            "required_response_facts": criteria.get("communicate_info") or [],
            "required_actions": criteria.get("actions") or [],
            "expected_changed_tables": expected_changed_tables,
        },
        "reference_steps": reference_steps,
        "reference_messages": reference_messages,
    }


def _answer_and_retrieval_pilots(
    source: dict[str, Any],
    *,
    category: str,
    evidence_documents: list[str],
    evidence_prompt: str,
    evidence_facts: list[str],
    retrieval_query: str,
    retrieval_documents: list[str],
    decision_documents: list[str],
    decision_prompt: str,
    decision_facts: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = []
    contracts = []

    task = make_pilot_task(
        source,
        variant="evidence_given",
        instructions=evidence_prompt,
        required_documents=evidence_documents,
        actions=None,
        communicate_info=evidence_facts,
        reward_basis=["COMMUNICATE"],
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="evidence_given",
            level="L1",
            retrieval_variant="golden_retrieval",
            capabilities=["单文档事实抽取"],
            given_context=["decisive_evidence"],
            required_documents=evidence_documents,
            reference_steps=[],
            reference_messages=["; ".join(evidence_facts)],
            expected_changed_tables=[],
        )
    )

    identifier = pilot_id(source["id"], "retrieval_only")
    retrieval_action = search_action(identifier, retrieval_query)
    retrieval_facts = [f"Document: {item}" for item in retrieval_documents]
    task = make_pilot_task(
        source,
        variant="retrieval_only",
        instructions=(
            f"Search the Banking knowledge base for the policy needed to answer this request: {retrieval_query}. "
            "Do not perform a banking operation. Return each decisive document as `Document: <document_id>`."
        ),
        required_documents=retrieval_documents,
        actions=[retrieval_action],
        communicate_info=retrieval_facts,
        reward_basis=["ACTION", "COMMUNICATE"],
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="retrieval_only",
            level="L2",
            retrieval_variant="bm25",
            capabilities=["查询构造", "检索覆盖", "文档相关性判断"],
            given_context=["business_question"],
            required_documents=retrieval_documents,
            reference_steps=[retrieval_action],
            reference_messages=retrieval_facts,
            expected_changed_tables=[],
        )
    )

    task = make_pilot_task(
        source,
        variant="decision_only",
        instructions=decision_prompt,
        required_documents=decision_documents,
        actions=None,
        communicate_info=decision_facts,
        reward_basis=["COMMUNICATE"],
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="decision_only",
            level="L3",
            retrieval_variant="golden_retrieval",
            capabilities=["约束过滤", "动作选择"],
            given_context=["decisive_evidence", "relevant_customer_state"],
            required_documents=decision_documents,
            reference_steps=[],
            reference_messages=["; ".join(decision_facts)],
            expected_changed_tables=[],
        )
    )
    return tasks, contracts


def build_task_001_pilots(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category = PILOT_SOURCES[source["id"]]
    required = list(source["required_documents"])
    tasks, contracts = _answer_and_retrieval_pilots(
        source,
        category=category,
        evidence_documents=["doc_credit_cards_gold_rewards_card_001"],
        evidence_prompt=(
            "Use the provided Gold Rewards Card document. Report exactly `Annual fee: $0.00` and "
            "`Cash back: 2.5%`. Do not apply for a card."
        ),
        evidence_facts=["Annual fee: $0.00", "Cash back: 2.5%"],
        retrieval_query="Gold Rewards Card annual fee cash back Rho-Bank+ subscription requirement",
        retrieval_documents=["doc_credit_cards_gold_rewards_card_001"],
        decision_documents=required[:2],
        decision_prompt=(
            "Sarah Bosch earns $100,000, has Rho-Bank+, wants the highest general cash back, and rejects "
            "any annual fee. Compare the provided Gold and Silver card evidence and return one line: "
            "`Recommendation: Gold Rewards Card`. Do not apply."
        ),
        decision_facts=["Recommendation: Gold Rewards Card"],
    )

    source_action = copy.deepcopy(source["evaluation_criteria"]["actions"][0])
    task = make_pilot_task(
        source,
        variant="single_action",
        instructions=(
            "The correct recommendation is the Gold Rewards Card. After the agent confirms exactly "
            "`Recommendation: Gold Rewards Card`, immediately apply using your available application tool with "
            "name Sarah Bosch, annual income 100000, and Rho-Bank+ set to true; then end the conversation."
        ),
        required_documents=[],
        actions=[source_action],
        communicate_info=["Recommendation: Gold Rewards Card"],
        reward_basis=["DB", "COMMUNICATE"],
        user_tools=["apply_for_credit_card"],
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="single_action",
            level="L3",
            retrieval_variant="golden_retrieval",
            capabilities=["用户工具协调", "参数构造"],
            given_context=["final_recommendation"],
            required_documents=[],
            reference_steps=[source_action],
            reference_messages=["Recommendation: Gold Rewards Card"],
            expected_changed_tables=["credit_card_applications"],
        )
    )

    identifier = pilot_id(source["id"], "two_skill_composition")
    search = search_action(identifier, "personal credit cards annual fee general cash back Rho-Bank+ Gold Silver Bronze")
    task = make_pilot_task(
        source,
        variant="two_skill_composition",
        instructions=(
            "Sarah Bosch has Rho-Bank+, rejects annual fees, and wants the highest cash back on everyday purchases. "
            "Search the knowledge base, compare the eligible cards, and return one line: "
            "`Recommendation: Gold Rewards Card`. Do not apply."
        ),
        required_documents=required,
        actions=[search],
        communicate_info=["Recommendation: Gold Rewards Card"],
        reward_basis=["ACTION", "COMMUNICATE"],
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="two_skill_composition",
            level="L4",
            retrieval_variant="bm25",
            capabilities=["检索覆盖", "约束过滤"],
            given_context=["customer_constraints"],
            required_documents=required,
            reference_steps=[search],
            reference_messages=["Recommendation: Gold Rewards Card"],
            expected_changed_tables=[],
        )
    )
    return _append_full_composition(
        source,
        category,
        "L4",
        ["credit_card_applications"],
        tasks,
        contracts,
    )


def build_task_031_pilots(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category = PILOT_SOURCES[source["id"]]
    tasks, contracts = _answer_and_retrieval_pilots(
        source,
        category=category,
        evidence_documents=["doc_credit_cards_credit_cards_(general)_015"],
        evidence_prompt=(
            "Use the provided provisional-credit policy. For a goods/services-not-as-described dispute, return "
            "exactly `Provisional credit: not eligible`."
        ),
        evidence_facts=["Provisional credit: not eligible"],
        retrieval_query="credit card dispute provisional credit goods services not as described eligibility",
        retrieval_documents=["doc_credit_cards_credit_cards_(general)_015"],
        decision_documents=[
            "doc_credit_cards_credit_cards_(general)_014",
            "doc_credit_cards_credit_cards_(general)_015",
        ],
        decision_prompt=(
            "The customer received a cheaper room than booked, contacted the merchant twice, wants a full refund, "
            "and wants the card kept active. Use the provided dispute policies and return exactly "
            "`Dispute reason: goods_services_not_as_described` and `Provisional credit: not eligible`."
        ),
        decision_facts=[
            "Dispute reason: goods_services_not_as_described",
            "Provisional credit: not eligible",
        ],
    )
    actions = source["evaluation_criteria"]["actions"]
    target = copy.deepcopy(actions[4])
    prerequisites = [action_to_initialization(item) for item in actions[:4]]
    task = make_pilot_task(
        source,
        variant="single_action",
        instructions=(
            "Identity verification, card-digit retrieval, and tool discovery are complete. File the Marriott Hotels "
            "dispute now using transaction txn_adea68821a1d, card digits 5320, the supplied customer record, "
            "goods_services_not_as_described, full_refund, keep_active, and provisional credit false."
        ),
        required_documents=[],
        actions=[target],
        communicate_info=None,
        reward_basis=["DB", "ACTION"],
        initialization_actions=prerequisites,
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="single_action",
            level="L3",
            retrieval_variant="golden_retrieval",
            capabilities=["参数构造", "动作选择"],
            given_context=["verification", "card_digits", "dispute_decision", "unlocked_tool"],
            required_documents=[],
            reference_steps=[target],
            reference_messages=[],
            expected_changed_tables=["transaction_disputes"],
        )
    )

    init = [action_to_initialization(actions[0]), action_to_initialization(actions[3])]
    steps = [copy.deepcopy(actions[index]) for index in (1, 2, 4)]
    task = make_pilot_task(
        source,
        variant="two_skill_composition",
        instructions=(
            "The customer is verified and the dispute tool is unlocked. Give Fatima the card-digits tool, let her "
            "retrieve digits for cc_890389b165_silver, then file the already-classified Marriott dispute with the "
            "reference customer and transaction details."
        ),
        required_documents=list(source["required_documents"]),
        actions=steps,
        communicate_info=None,
        reward_basis=["DB", "ACTION"],
        initialization_actions=init,
        user_tools=["call_discoverable_user_tool"],
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="two_skill_composition",
            level="L4",
            retrieval_variant="golden_retrieval",
            capabilities=["用户工具协调", "参数构造"],
            given_context=["verification", "dispute_decision", "unlocked_dispute_tool"],
            required_documents=list(source["required_documents"]),
            reference_steps=steps,
            reference_messages=[],
            expected_changed_tables=["user_discoverable_tools", "user_discoverable_tool_calls", "transaction_disputes"],
        )
    )
    return _append_full_composition(
        source,
        category,
        "L5",
        ["transaction_disputes"],
        tasks,
        contracts,
    )


def build_task_036_pilots(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category = PILOT_SOURCES[source["id"]]
    replacement_doc = ["doc_credit_cards_credit_card_replacements_001"]
    tasks, contracts = _answer_and_retrieval_pilots(
        source,
        category=category,
        evidence_documents=replacement_doc,
        evidence_prompt=(
            "Use the provided replacement-card policy for a Silver Rewards Card replaced for suspected fraud. "
            "Return exactly `Expedited shipping fee: $10.00` and `Expedited shipping: recommended`."
        ),
        evidence_facts=["Expedited shipping fee: $10.00", "Expedited shipping: recommended"],
        retrieval_query="replacement credit card suspected fraud expedited shipping Silver Rewards fee",
        retrieval_documents=replacement_doc,
        decision_documents=replacement_doc,
        decision_prompt=(
            "Two unrecognized transactions indicate card fraud. Fatima wants delivery to 2500 Woodward Avenue, "
            "Suite 300, Detroit, MI 48201 and accepts fast shipping. Return exactly "
            "`Replacement reason: fraud_suspected`, `Shipping: expedited`, and "
            "`Ship to: 2500 Woodward Avenue, Suite 300, Detroit, MI 48201`."
        ),
        decision_facts=[
            "Replacement reason: fraud_suspected",
            "Shipping: expedited",
            "Ship to: 2500 Woodward Avenue, Suite 300, Detroit, MI 48201",
        ],
    )
    actions = source["evaluation_criteria"]["actions"]
    target = copy.deepcopy(actions[2])
    prerequisites = [action_to_initialization(item) for item in actions[:2]]
    task = make_pilot_task(
        source,
        variant="single_action",
        instructions=(
            "The customer is verified, fraud is confirmed, and the replacement tool is unlocked. Order a replacement "
            "for cc_890389b165_silver to 2500 Woodward Avenue, Suite 300, Detroit, MI 48201 with reason "
            "fraud_suspected and expedited shipping enabled."
        ),
        required_documents=[],
        actions=[target],
        communicate_info=None,
        reward_basis=["DB", "ACTION"],
        initialization_actions=prerequisites,
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="single_action",
            level="L3",
            retrieval_variant="golden_retrieval",
            capabilities=["参数构造", "动作选择"],
            given_context=["verification", "fraud_decision", "shipping_preference", "unlocked_tool"],
            required_documents=[],
            reference_steps=[target],
            reference_messages=[],
            expected_changed_tables=["credit_card_accounts", "credit_card_orders"],
        )
    )

    read_action = {
        "name": "get_credit_card_transactions_by_user",
        "arguments": {"user_id": "890389b165"},
        "requestor": "assistant",
        "action_id": f"{pilot_id(source['id'], 'two_skill_composition')}_read_transactions",
    }
    steps = [read_action, target]
    task = make_pilot_task(
        source,
        variant="two_skill_composition",
        instructions=(
            "Fatima is verified and the replacement tool is unlocked. Inspect her credit-card transactions, identify "
            "Electronics Express Miami and GamerZone LA as the two unrecognized merchants, report both names, then "
            "order the expedited fraud replacement to her work address."
        ),
        required_documents=replacement_doc,
        actions=steps,
        communicate_info=["Electronics Express Miami", "GamerZone LA"],
        reward_basis=["DB", "ACTION", "COMMUNICATE"],
        initialization_actions=prerequisites,
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="two_skill_composition",
            level="L4",
            retrieval_variant="golden_retrieval",
            capabilities=["实体—状态绑定", "动作选择"],
            given_context=["verification", "shipping_preference", "unlocked_tool"],
            required_documents=replacement_doc,
            reference_steps=steps,
            reference_messages=["Electronics Express Miami; GamerZone LA"],
            expected_changed_tables=["credit_card_accounts", "credit_card_orders"],
        )
    )
    return _append_full_composition(
        source,
        category,
        "L5",
        ["credit_card_accounts", "credit_card_orders"],
        tasks,
        contracts,
    )


def build_task_071_pilots(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category = PILOT_SOURCES[source["id"]]
    active_checking = "doc_bank_accounts_bank_accounts_(general)_013"
    expired_checking = "doc_bank_accounts_bank_accounts_(general)_014"
    decision_documents = [active_checking, expired_checking]
    tasks, contracts = _answer_and_retrieval_pilots(
        source,
        category=category,
        evidence_documents=[active_checking],
        evidence_prompt=(
            "Use the provided November 2025 business-checking promotion. Return exactly "
            "`Promotion dates: 11/01/2025 to 11/30/2025` and `Priority: Sky Blue`."
        ),
        evidence_facts=["Promotion dates: 11/01/2025 to 11/30/2025", "Priority: Sky Blue"],
        retrieval_query="active November 2025 business checking account promotional priority Sky Blue Lime Green",
        retrieval_documents=[active_checking],
        decision_documents=decision_documents,
        decision_prompt=(
            "Sky Blue and Lime Green have both already been confirmed to meet the customer's hard requirements. "
            "Compare the provided October and November promotion notices for 11/14/2025 and return exactly "
            "`Recommendation: Sky Blue`. Do not open the account."
        ),
        decision_facts=["Recommendation: Sky Blue"],
    )
    actions = source["evaluation_criteria"]["actions"]
    target_checking = copy.deepcopy(actions[4])
    prerequisites = [action_to_initialization(item) for item in actions[:4]]
    task = make_pilot_task(
        source,
        variant="single_action",
        instructions=(
            "Yumi Tanaka is verified and eligible, account state has been checked, the opening tool is unlocked, and "
            "Sky Blue is the selected product. Open exactly one Sky Blue business checking account for yt71c9e4f2."
        ),
        required_documents=[],
        actions=[target_checking],
        communicate_info=None,
        reward_basis=["DB", "ACTION"],
        initialization_actions=prerequisites,
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="single_action",
            level="L3",
            retrieval_variant="golden_retrieval",
            capabilities=["参数构造", "动作选择"],
            given_context=["verification", "eligibility", "product_decision", "unlocked_tool"],
            required_documents=[],
            reference_steps=[target_checking],
            reference_messages=[],
            expected_changed_tables=["accounts"],
        )
    )

    task = make_pilot_task(
        source,
        variant="two_skill_composition",
        instructions=(
            "Yumi is verified, Sky Blue and Lime Green both meet her hard requirements, and the opening tool is "
            "unlocked. Compare the provided October and November promotion notices for 11/14/2025, report "
            "`Recommendation: Sky Blue`, and open that business checking account."
        ),
        required_documents=decision_documents,
        actions=[target_checking],
        communicate_info=["Recommendation: Sky Blue"],
        reward_basis=["DB", "ACTION", "COMMUNICATE"],
        initialization_actions=prerequisites,
    )
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="two_skill_composition",
            level="L4",
            retrieval_variant="golden_retrieval",
            capabilities=["跨文档比较", "动作选择"],
            given_context=["verification", "eligibility", "unlocked_tool"],
            required_documents=decision_documents,
            reference_steps=[target_checking],
            reference_messages=["Recommendation: Sky Blue"],
            expected_changed_tables=["accounts"],
        )
    )
    return _append_full_composition(
        source,
        category,
        "L6",
        ["accounts"],
        tasks,
        contracts,
    )


def _append_full_composition(
    source: dict[str, Any],
    category: str,
    level: str,
    expected_changed_tables: list[str],
    tasks: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task = copy.deepcopy(source)
    task.pop("annotations", None)
    task["id"] = pilot_id(source["id"], "full_composition")
    task["description"] = copy.deepcopy(task.get("description") or {})
    task["description"]["purpose"] = f"Banking curriculum full composition derived from {source['id']}"
    tasks.append(task)
    reference_steps = copy.deepcopy((source.get("evaluation_criteria") or {}).get("actions") or [])
    contracts.append(
        contract(
            task,
            source_id=source["id"],
            category=category,
            variant="full_composition",
            level=level,
            retrieval_variant="bm25",
            capabilities=CATEGORY_DETAILS[category]["primary_capabilities"],
            given_context=[],
            required_documents=list(source.get("required_documents") or []),
            reference_steps=reference_steps,
            reference_messages=[],
            expected_changed_tables=expected_changed_tables,
        )
    )
    return tasks, contracts


def build_pilots(tasks: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    builders = {
        "task_001": build_task_001_pilots,
        "task_031": build_task_031_pilots,
        "task_036": build_task_036_pilots,
        "task_071": build_task_071_pilots,
    }
    pilot_tasks = []
    pilot_contracts = []
    for source_id in PILOT_SOURCES:
        source_tasks, source_contracts = builders[source_id](tasks[source_id])
        pilot_tasks.extend(source_tasks)
        pilot_contracts.extend(source_contracts)
    return pilot_tasks, pilot_contracts


def full_curriculum_level(source: dict[str, Any]) -> str:
    document_count = len(source.get("required_documents") or [])
    action_count = len((source.get("evaluation_criteria") or {}).get("actions") or [])
    if document_count <= 4 and action_count <= 4:
        return "L4"
    if document_count <= 8 and action_count <= 16:
        return "L5"
    return "L6"


def build_systematic_variants(
    source: dict[str, Any],
    annotation: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project one clean runtime source into six conservative curriculum tasks."""

    source_id = source["id"]
    category = annotation["business"]["primary_category"]
    required_documents = list(source.get("required_documents") or [])
    source_actions = list((source.get("evaluation_criteria") or {}).get("actions") or [])
    source_reward_basis = list((source.get("evaluation_criteria") or {}).get("reward_basis") or [])
    if not required_documents or not source_actions:
        raise ValueError(f"{source_id}: systematic expansion requires evidence and reference actions")
    final_action_name, final_action_arguments = effective_action(source_actions[-1])
    operation_goal = resolved_operation_goal(final_action_name, final_action_arguments)

    tasks = []
    contracts = []
    context_text = json.dumps(
        {
            "description": source.get("description"),
            "instructions": (source.get("user_scenario") or {}).get("instructions"),
        },
        ensure_ascii=False,
    ).lower()
    action_text = json.dumps(list(final_action_arguments.values()), ensure_ascii=False).lower()
    context_terms = set(re.findall(r"[a-z0-9_]+", context_text))
    context_terms.update(re.findall(r"[a-z0-9_]+", final_action_name.lower()))
    action_terms = set(re.findall(r"[a-z0-9_]+", action_text))

    def evidence_relevance(item: dict[str, Any]) -> tuple[int, int]:
        document_id = item["source_document_ids"][0]
        candidate_text = f"{item['statement']} {documents[document_id]['title']}".lower()
        candidate_terms = set(re.findall(r"[a-z0-9_]+", candidate_text))
        return (
            10 * len(candidate_terms & action_terms) + len(candidate_terms & context_terms),
            -required_documents.index(document_id),
        )

    evidence_row = max(annotation["evidence"]["required_facts"], key=evidence_relevance)
    evidence_document = evidence_row["source_document_ids"][0]
    evidence_fact = evidence_row["statement"]
    evidence_locator = document_fact_locator(documents[evidence_document], evidence_fact)
    fact_label = f"Fact: {evidence_fact}"

    task = make_pilot_task(
        source,
        variant="evidence_given",
        instructions=(
            f"For this resolved customer goal, {operation_goal}. "
            f"Use the provided document `{evidence_document}` and copy {evidence_locator} verbatim. "
            f"Return exactly `{fact_label}`. "
            "Do not search or perform a banking operation."
        ),
        required_documents=[evidence_document],
        actions=None,
        communicate_info=[fact_label],
        reward_basis=["COMMUNICATE"],
    )
    task["id"] = f"banking_scale_{source_id}_evidence_given"
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source_id,
            category=category,
            variant="evidence_given",
            level="L1",
            retrieval_variant="golden_retrieval",
            capabilities=["单文档事实抽取"],
            given_context=["decisive_evidence"],
            required_documents=[evidence_document],
            reference_steps=[],
            reference_messages=[fact_label],
            expected_changed_tables=[],
        )
    )

    retrieval_query = f"{documents[evidence_document]['title']} {evidence_fact}"
    search = search_action(
        f"banking_scale_{source_id}_retrieval_only",
        retrieval_query,
    )
    document_label = f"Document: {evidence_document}"
    task = make_pilot_task(
        source,
        variant="retrieval_only",
        instructions=(
            f"Search the Banking knowledge base for the policy topic "
            f"`{documents[evidence_document]['title']}`. "
            f"Return exactly `{document_label}` and do not perform a banking operation."
        ),
        required_documents=[evidence_document],
        actions=[search],
        communicate_info=[document_label],
        reward_basis=["ACTION", "COMMUNICATE"],
    )
    task["id"] = f"banking_scale_{source_id}_retrieval_only"
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source_id,
            category=category,
            variant="retrieval_only",
            level="L2",
            retrieval_variant="bm25",
            capabilities=["查询构造", "文档相关性判断"],
            given_context=["business_question"],
            required_documents=[evidence_document],
            reference_steps=[search],
            reference_messages=[document_label],
            expected_changed_tables=[],
        )
    )

    final_action = copy.deepcopy(source_actions[-1])
    decision_label = f"Decision action: {final_action_name}"
    decision_documents = required_documents[:2]
    task = make_pilot_task(
        source,
        variant="decision_only",
        instructions=(
            "All prerequisite workflow steps are complete. "
            f"For the resolved customer goal, {operation_goal}. "
            f"Use the provided evidence and return exactly `{decision_label}`. "
            "Do not execute the action."
        ),
        required_documents=decision_documents,
        actions=None,
        communicate_info=[decision_label],
        reward_basis=["COMMUNICATE"],
    )
    task["id"] = f"banking_scale_{source_id}_decision_only"
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source_id,
            category=category,
            variant="decision_only",
            level="L3",
            retrieval_variant="golden_retrieval",
            capabilities=["动作选择"],
            given_context=["decisive_evidence", "resolved_customer_context"],
            required_documents=decision_documents,
            reference_steps=[],
            reference_messages=[decision_label],
            expected_changed_tables=[],
        )
    )

    action_payload = json.dumps(
        {
            "requestor": final_action.get("requestor", "assistant"),
            "name": final_action_name,
            "arguments": final_action_arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    action_reward_basis = (["DB"] if "DB" in source_reward_basis else []) + ["ACTION"]
    task = make_pilot_task(
        source,
        variant="single_action",
        instructions=(
            "All preceding source-task steps and decisions are complete. Execute the one remaining reference "
            f"operation using this resolved payload: `{action_payload}`."
        ),
        required_documents=[],
        actions=[final_action],
        communicate_info=None,
        reward_basis=action_reward_basis,
        initialization_actions=[action_to_initialization(action) for action in source_actions[:-1]] or None,
        user_tools=user_tools_for_actions(source, [final_action]),
    )
    task["id"] = f"banking_scale_{source_id}_single_action"
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source_id,
            category=category,
            variant="single_action",
            level="L3",
            retrieval_variant="golden_retrieval",
            capabilities=["参数构造", "动作选择"],
            given_context=["completed_prerequisites", "resolved_action"],
            required_documents=[],
            reference_steps=[final_action],
            reference_messages=[],
            expected_changed_tables=[],
        )
    )

    suffix_size = min(2, len(source_actions))
    suffix_actions = copy.deepcopy(source_actions[-suffix_size:])
    suffix_payload = json.dumps(
        [action_summary(action) for action in suffix_actions],
        ensure_ascii=False,
        sort_keys=True,
    )
    task = make_pilot_task(
        source,
        variant="two_skill_composition",
        instructions=(
            f"Use the provided evidence to confirm `{decision_label}`, report that exact line, then execute the "
            f"remaining reference operation sequence: `{suffix_payload}`."
        ),
        required_documents=decision_documents,
        actions=suffix_actions,
        communicate_info=[decision_label],
        reward_basis=action_reward_basis + ["COMMUNICATE"],
        initialization_actions=[action_to_initialization(action) for action in source_actions[:-suffix_size]] or None,
        user_tools=user_tools_for_actions(source, suffix_actions),
    )
    task["id"] = f"banking_scale_{source_id}_two_skill_composition"
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source_id,
            category=category,
            variant="two_skill_composition",
            level="L4",
            retrieval_variant="golden_retrieval",
            capabilities=["动作选择", "动作顺序" if suffix_size == 2 else "参数构造"],
            given_context=["completed_prerequisites"],
            required_documents=decision_documents,
            reference_steps=suffix_actions,
            reference_messages=[decision_label],
            expected_changed_tables=[],
        )
    )

    task = copy.deepcopy(source)
    task.pop("annotations", None)
    task["id"] = f"banking_scale_{source_id}_full_composition"
    task["description"] = copy.deepcopy(task.get("description") or {})
    task["description"]["purpose"] = f"Banking curriculum full composition derived from {source_id}"
    tasks.append(task)
    contracts.append(
        contract(
            task,
            source_id=source_id,
            category=category,
            variant="full_composition",
            level=full_curriculum_level(source),
            retrieval_variant="bm25",
            capabilities=annotation["reasoning"]["primary_capabilities"],
            given_context=[],
            required_documents=required_documents,
            reference_steps=copy.deepcopy(source_actions),
            reference_messages=[],
            expected_changed_tables=[],
        )
    )
    return tasks, contracts


def build_expanded_curriculum(
    runtime_tasks: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curated_tasks, curated_contracts = build_pilots(runtime_tasks)
    curated_task_by_id = {task["id"]: task for task in curated_tasks}
    curated_by_source: dict[str, list[dict[str, Any]]] = {}
    for item in curated_contracts:
        curated_by_source.setdefault(item["source_task_id"], []).append(item)

    expanded_tasks = []
    expanded_contracts = []
    variant_order = {variant: index for index, variant in enumerate(PILOT_VARIANTS)}
    for source_id, source in sorted(runtime_tasks.items()):
        if source_id in EXPANSION_EXCLUDED_SOURCES:
            continue
        if source_id in curated_by_source:
            source_contracts = sorted(curated_by_source[source_id], key=lambda item: variant_order[item["variant"]])
            expanded_contracts.extend(copy.deepcopy(source_contracts))
            expanded_tasks.extend(copy.deepcopy(curated_task_by_id[item["task_id"]]) for item in source_contracts)
            continue
        source_tasks, source_contracts = build_systematic_variants(
            source,
            annotations[source_id],
            documents,
        )
        expanded_tasks.extend(source_tasks)
        expanded_contracts.extend(source_contracts)
    return expanded_tasks, expanded_contracts


def validate_expanded_curriculum(
    expanded_tasks: list[dict[str, Any]],
    expanded_contracts: list[dict[str, Any]],
    runtime_tasks: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors = []
    task_by_id = {task["id"]: task for task in expanded_tasks}
    contract_by_id = {item["task_id"]: item for item in expanded_contracts}
    expected_sources = set(runtime_tasks) - set(EXPANSION_EXCLUDED_SOURCES)
    expected_count = len(expected_sources) * len(PILOT_VARIANTS)
    if len(task_by_id) != expected_count or len(expanded_tasks) != expected_count:
        errors.append(
            f"expected {expected_count} unique expanded tasks, found {len(task_by_id)}/{len(expanded_tasks)}"
        )
    if set(task_by_id) != set(contract_by_id) or len(contract_by_id) != len(expanded_contracts):
        errors.append("expanded task and contract ids differ or contain duplicates")

    by_source = Counter(item["source_task_id"] for item in expanded_contracts)
    if set(by_source) != expected_sources:
        errors.append("expanded source ids differ from clean runtime source ids")
    for source_id in expected_sources:
        variants = {
            item["variant"] for item in expanded_contracts if item["source_task_id"] == source_id
        }
        if variants != set(PILOT_VARIANTS):
            errors.append(f"{source_id}: expanded variants differ: {sorted(variants)}")

    categories = category_lookup()
    for identifier, task in task_by_id.items():
        item = contract_by_id[identifier]
        source_id = item["source_task_id"]
        if source_id not in runtime_tasks:
            errors.append(f"{identifier}: unknown source task {source_id}")
            continue
        if item["business_category"] != categories[source_id]:
            errors.append(f"{identifier}: category differs from source inventory")
        missing = set(task.get("required_documents") or []) - set(documents)
        if missing:
            errors.append(f"{identifier}: missing documents {sorted(missing)}")
        criteria = task.get("evaluation_criteria") or {}
        if not criteria.get("reward_basis"):
            errors.append(f"{identifier}: empty reward basis")
        reference_text = "\n".join(item["reference_messages"]).lower().replace(",", "")
        for fact in item["success_contract"]["required_response_facts"]:
            if fact.lower().replace(",", "") not in reference_text:
                errors.append(f"{identifier}: reference message misses `{fact}`")
        for action in criteria.get("actions") or []:
            if not any(
                action["name"] == step["name"]
                and action.get("requestor", "assistant") == step.get("requestor", "assistant")
                and action.get("arguments", {}) == step.get("arguments", {})
                for step in item["reference_steps"]
            ):
                errors.append(f"{identifier}: evaluation action {action['action_id']} absent from reference steps")

        variant = item["variant"]
        document_count = len(task.get("required_documents") or [])
        action_count = len(criteria.get("actions") or [])
        message_history = (task.get("initial_state") or {}).get("message_history")
        if variant != "full_composition":
            exposed_user_tools = set(task.get("user_tools") or [])
            expected_user_tools = {
                action["name"]
                for action in criteria.get("actions") or []
                if action.get("requestor", "assistant") == "user"
            }
            if exposed_user_tools != expected_user_tools:
                errors.append(
                    f"{identifier}: exposed user tools {sorted(exposed_user_tools)} differ from "
                    f"evaluated user actions {sorted(expected_user_tools)}"
                )
            if (
                not isinstance(message_history, list)
                or len(message_history) != 1
                or message_history[0].get("role") != "user"
                or not message_history[0].get("content")
            ):
                errors.append(f"{identifier}: atomic task must seed one user request")
            else:
                for fact in item["success_contract"]["required_response_facts"]:
                    if fact in message_history[0]["content"]:
                        errors.append(f"{identifier}: initial user request leaks `{fact}`")
        if variant in {"evidence_given", "retrieval_only"} and document_count != 1:
            errors.append(f"{identifier}: {variant} must use exactly one document")
        if variant == "retrieval_only" and (
            action_count != 1 or criteria["actions"][0]["name"] != "KB_search"
        ):
            errors.append(f"{identifier}: retrieval-only task must contain one KB_search")
        if variant == "decision_only" and (document_count > 2 or action_count):
            errors.append(f"{identifier}: decision-only task exceeds its controlled boundary")
        if variant == "single_action" and (document_count or action_count != 1):
            errors.append(f"{identifier}: single-action task exceeds its controlled boundary")
        if variant == "two_skill_composition" and (document_count > 4 or not 1 <= action_count <= 3):
            errors.append(f"{identifier}: two-skill task exceeds its controlled boundary")
        if variant == "full_composition":
            expected = copy.deepcopy(runtime_tasks[source_id])
            expected.pop("annotations", None)
            expected["id"] = identifier
            expected["description"] = copy.deepcopy(expected.get("description") or {})
            expected["description"]["purpose"] = (
                f"Banking curriculum full composition derived from {source_id}"
            )
            if normalize_snapshot(task) != normalize_snapshot(expected):
                errors.append(f"{identifier}: full composition differs from its runtime source")

    source_category_counts = Counter(categories[source_id] for source_id in expected_sources)
    return errors, {
        "source_tasks": len(expected_sources),
        "hand_curated_source_tasks": len(set(PILOT_SOURCES) & expected_sources),
        "systematic_source_tasks": len(expected_sources - set(PILOT_SOURCES)),
        "excluded_sources": EXPANSION_EXCLUDED_SOURCES,
        "expanded_tasks": len(expanded_tasks),
        "expanded_contracts": len(expanded_contracts),
        "business_categories": len(source_category_counts),
        "source_tasks_by_category": dict(source_category_counts),
        "tasks_by_variant": dict(Counter(item["variant"] for item in expanded_contracts)),
        "tasks_by_level": dict(Counter(item["level"] for item in expanded_contracts)),
    }


def graph_errors(row: dict[str, Any]) -> list[str]:
    errors = []
    nodes = row["graph"]["nodes"]
    node_by_id = {node["id"]: node for node in nodes}
    if len(node_by_id) != len(nodes):
        errors.append(f"{row['task_id']}: duplicate graph node")
        return errors
    for node in nodes:
        if node["kind"] not in {"evidence", "decision", "action", "state"}:
            errors.append(f"{row['task_id']}: invalid node kind {node['kind']}")
        for dependency in node["depends_on"]:
            if dependency not in node_by_id:
                errors.append(f"{row['task_id']}: missing dependency {dependency}")

    visiting = set()
    visited = set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            errors.append(f"{row['task_id']}: graph cycle at {identifier}")
            return
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in node_by_id[identifier]["depends_on"]:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in node_by_id:
        visit(identifier)
    return errors


def validate_catalog(
    catalog: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors = []
    if len(tasks) != 97:
        errors.append(f"expected 97 runtime tasks, found {len(tasks)}")
    if len(documents) != 698:
        errors.append(f"expected 698 documents, found {len(documents)}")
    if len(catalog) != len(tasks):
        errors.append(f"catalog has {len(catalog)} rows for {len(tasks)} tasks")
    if {row["task_id"] for row in catalog} != set(tasks):
        errors.append("catalog task ids differ from runtime task ids")

    category_counts = Counter(row["business"]["primary_category"] for row in catalog)
    expected_category_counts = {category: len(numbers) for category, numbers in CATEGORIES.items()}
    if dict(category_counts) != expected_category_counts:
        errors.append(f"category counts differ: {dict(category_counts)}")

    aggregate_drift = set()
    history_totals = {label: 0 for label in HISTORICAL_RUNS}
    incompatible = set()
    for row in catalog:
        identifier = row["task_id"]
        evidence = row["evidence"]
        gold = set(evidence["gold_document_ids"])
        decisive = set(evidence["decisive_document_ids"])
        comparison = set(evidence["comparison_document_ids"])
        missing = gold - set(documents)
        if missing:
            errors.append(f"{identifier}: missing documents {sorted(missing)}")
        if not decisive <= gold:
            errors.append(f"{identifier}: decisive documents are not a subset of gold")
        if not comparison <= gold:
            errors.append(f"{identifier}: comparison documents are not a subset of gold")
        capabilities = set(row["reasoning"]["primary_capabilities"] + row["reasoning"]["supporting_capabilities"])
        if not capabilities <= ALL_CAPABILITIES:
            errors.append(f"{identifier}: unknown capabilities {sorted(capabilities - ALL_CAPABILITIES)}")
        if not row["business"]["intent"]:
            errors.append(f"{identifier}: empty intent")
        if not row["success"]["checkpoints"]:
            errors.append(f"{identifier}: no success checkpoints")
        errors.extend(graph_errors(row))
        if not row["source"]["aggregate_matches_runtime"]:
            aggregate_drift.add(identifier)
        for label, metrics in row["historical"].items():
            history_totals[label] += metrics["successes"]
            if not metrics["compatible_with_runtime_task"]:
                incompatible.add(identifier)

    if aggregate_drift != EXPECTED_AGGREGATE_DRIFT:
        errors.append(f"aggregate drift differs: {sorted(aggregate_drift)}")
    if history_totals != {"qwen3_a": 13, "qwen3_b": 14, "qwen35_nonthinking": 15}:
        errors.append(f"historical success totals differ: {history_totals}")
    if incompatible != {"task_058", "task_059"}:
        errors.append(f"historical incompatibility set differs: {sorted(incompatible)}")

    doc_counts = [row["evidence"]["gold_document_count"] for row in catalog]
    action_counts = [len(row["workflow"]["reference_actions"]) for row in catalog]
    return errors, {
        "runtime_tasks": len(tasks),
        "documents": len(documents),
        "category_counts": dict(category_counts),
        "aggregate_drift_tasks": sorted(aggregate_drift),
        "historical_incompatible_tasks": sorted(incompatible),
        "historical_successes": history_totals,
        "gold_documents": {
            "mean": round(sum(doc_counts) / len(doc_counts), 2),
            "median": sorted(doc_counts)[len(doc_counts) // 2],
            "max": max(doc_counts),
        },
        "reference_actions": {
            "mean": round(sum(action_counts) / len(action_counts), 2),
            "median": sorted(action_counts)[len(action_counts) // 2],
            "max": max(action_counts),
        },
    }


def validate_pilots(
    pilot_tasks: list[dict[str, Any]],
    pilot_contracts: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors = []
    task_by_id = {task["id"]: task for task in pilot_tasks}
    contract_by_id = {item["task_id"]: item for item in pilot_contracts}
    if len(task_by_id) != 24 or len(pilot_tasks) != 24:
        errors.append(f"expected 24 unique pilot tasks, found {len(task_by_id)}/{len(pilot_tasks)}")
    if set(task_by_id) != set(contract_by_id) or len(contract_by_id) != len(pilot_contracts):
        errors.append("pilot task and contract ids differ or contain duplicates")

    by_source = Counter(item["source_task_id"] for item in pilot_contracts)
    if by_source != Counter({source_id: 6 for source_id in PILOT_SOURCES}):
        errors.append(f"pilot source counts differ: {dict(by_source)}")
    for source_id in PILOT_SOURCES:
        variants = {
            item["variant"] for item in pilot_contracts if item["source_task_id"] == source_id
        }
        if variants != set(PILOT_VARIANTS):
            errors.append(f"{source_id}: variants differ: {sorted(variants)}")

    for identifier, task in task_by_id.items():
        contract_row = contract_by_id[identifier]
        missing = set(task.get("required_documents") or []) - set(documents)
        if missing:
            errors.append(f"{identifier}: missing documents {sorted(missing)}")
        if task["evaluation_criteria"]["reward_basis"] == []:
            errors.append(f"{identifier}: empty reward basis")
        reference_text = "\n".join(contract_row["reference_messages"]).lower().replace(",", "")
        for fact in contract_row["success_contract"]["required_response_facts"]:
            if fact.lower().replace(",", "") not in reference_text:
                errors.append(f"{identifier}: reference message misses `{fact}`")
        task_actions = task["evaluation_criteria"].get("actions") or []
        reference_steps = contract_row["reference_steps"]
        for action in task_actions:
            if not any(
                action["name"] == step["name"]
                and action.get("requestor", "assistant") == step.get("requestor", "assistant")
                and action.get("arguments", {}) == step.get("arguments", {})
                for step in reference_steps
            ):
                errors.append(f"{identifier}: evaluation action {action['action_id']} absent from reference steps")

    return errors, {
        "pilot_tasks": len(pilot_tasks),
        "pilot_contracts": len(pilot_contracts),
        "by_source": dict(by_source),
        "by_variant": dict(Counter(item["variant"] for item in pilot_contracts)),
        "by_level": dict(Counter(item["level"] for item in pilot_contracts)),
    }


def plain_db_snapshot(environment: Any) -> dict[str, Any]:
    return environment.tools.db.model_dump()


def changed_db_tables(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(table for table in set(before) | set(after) if before.get(table) != after.get(table))


def execute_steps(environment: Any, steps: list[dict[str, Any]], *, task_name: str) -> tuple[list[str], list[str]]:
    from tau2.data_model.message import ToolCall

    errors = []
    responses = []
    for index, step in enumerate(steps):
        response = environment.get_response(
            ToolCall(
                id=f"reference_{index}",
                name=step["name"],
                arguments=step.get("arguments") or {},
                requestor=step.get("requestor", "assistant"),
            )
        )
        try:
            decoded = json.loads(response.content)
        except json.JSONDecodeError:
            decoded = response.content
        text = decoded if isinstance(decoded, str) else json.dumps(decoded, ensure_ascii=False)
        responses.append(text)
        if response.error or text.strip().lower().startswith(("error:", "failed")):
            errors.append(f"{task_name}: step {step['name']} failed: {text[:300]}")
    return errors, responses


def set_environment_state(environment: Any, task: Any) -> None:
    initial = task.initial_state
    if initial is None:
        environment.set_state(None, None, [])
        return
    environment.set_state(
        initial.initialization_data,
        initial.initialization_actions,
        initial.message_history or [],
    )


def validate_environment(
    runtime_tasks: dict[str, dict[str, Any]],
    pilot_tasks: list[dict[str, Any]],
    pilot_contracts: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    from tau2.data_model.tasks import Task
    from tau2.domains.banking_knowledge.environment import get_environment

    errors = []
    source_reference_errors = []
    source_replays = []
    for identifier, payload in sorted(runtime_tasks.items()):
        task = Task.model_validate(payload)
        environment = get_environment(task=task, retrieval_variant="golden_retrieval")
        set_environment_state(environment, task)
        before = plain_db_snapshot(environment)
        steps = (payload.get("evaluation_criteria") or {}).get("actions") or []
        step_errors, _ = execute_steps(environment, steps, task_name=identifier)
        source_reference_errors.extend(step_errors)
        changed = changed_db_tables(before, plain_db_snapshot(environment))
        source_replays.append(
            {
                "task_id": identifier,
                "actions": len(steps),
                "changed_tables": changed,
                "reference_errors": step_errors,
            }
        )

    contract_by_id = {item["task_id"]: item for item in pilot_contracts}
    pilot_replays = []
    for payload in pilot_tasks:
        task = Task.model_validate(payload)
        item = contract_by_id[task.id]
        environment = get_environment(task=task, retrieval_variant=item["retrieval_variant"])
        set_environment_state(environment, task)
        before = plain_db_snapshot(environment)
        step_errors, responses = execute_steps(environment, item["reference_steps"], task_name=task.id)
        errors.extend(step_errors)
        changed = changed_db_tables(before, plain_db_snapshot(environment))
        expected_tables = set(item["success_contract"]["expected_changed_tables"])
        if not expected_tables <= set(changed):
            errors.append(
                f"{task.id}: expected changed tables {sorted(expected_tables)}, observed {changed}"
            )
        if item["retrieval_variant"] == "golden_retrieval":
            for document_id in task.required_documents or []:
                content = documents[document_id]["content"]
                snippet = content[:120]
                if snippet not in environment.policy:
                    errors.append(f"{task.id}: golden policy does not contain {document_id}")
        if item["variant"] == "retrieval_only":
            combined = "\n".join(responses)
            for document_id in item["success_contract"]["required_document_ids"]:
                if document_id not in combined:
                    errors.append(f"{task.id}: BM25 reference query misses {document_id}")
        pilot_replays.append(
            {
                "task_id": task.id,
                "reference_steps": len(item["reference_steps"]),
                "changed_tables": changed,
            }
        )

    return errors, {
        "schema_validated_pilots": len(pilot_tasks),
        "source_reference_replays": len(source_replays),
        "pilot_reference_replays": len(pilot_replays),
        "source_reference_error_count": len(source_reference_errors),
        "source_reference_error_tasks": sorted(
            item["task_id"] for item in source_replays if item["reference_errors"]
        ),
        "source_reference_errors": source_reference_errors,
        "pilot_validation_error_count": len(errors),
        "source_replays": source_replays,
        "pilot_replays": pilot_replays,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_command(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_runtime_tasks(args.tasks_dir)
    documents = load_documents(args.documents_dir)
    aggregate = {item["id"]: item for item in read_json(args.aggregate_tasks)}
    historical_results = load_historical_results(args.historical_runs)
    tool_types = parse_tool_types(args.tools_file)
    annotations, catalog = build_catalog(tasks, documents, aggregate, tool_types, historical_results)
    pilot_tasks, pilot_contracts = build_pilots(tasks)
    catalog_errors, catalog_summary = validate_catalog(catalog, tasks, documents)
    pilot_errors, pilot_summary = validate_pilots(pilot_tasks, pilot_contracts, documents)
    errors = catalog_errors + pilot_errors
    report = {
        "status": "pass" if not errors else "fail",
        "static": {"catalog": catalog_summary, "pilots": pilot_summary},
        "environment": None,
        "errors": errors,
    }
    if errors:
        raise ValueError("\n".join(errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "annotations.jsonl", annotations)
    write_jsonl(args.output_dir / "catalog.jsonl", catalog)
    write_json(args.output_dir / "pilot_tasks.json", pilot_tasks)
    write_jsonl(args.output_dir / "pilot_contracts.jsonl", pilot_contracts)
    write_json(args.output_dir / "validation_report.json", report)
    return report


def validate_command(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_runtime_tasks(args.tasks_dir)
    documents = load_documents(args.documents_dir)
    catalog = read_jsonl(args.output_dir / "catalog.jsonl")
    pilot_tasks = read_json(args.output_dir / "pilot_tasks.json")
    pilot_contracts = read_jsonl(args.output_dir / "pilot_contracts.jsonl")
    catalog_errors, catalog_summary = validate_catalog(catalog, tasks, documents)
    pilot_errors, pilot_summary = validate_pilots(pilot_tasks, pilot_contracts, documents)
    errors = catalog_errors + pilot_errors
    environment_summary = None
    if args.environment:
        environment_errors, environment_summary = validate_environment(
            tasks, pilot_tasks, pilot_contracts, documents
        )
        errors.extend(environment_errors)
    report = {
        "status": "pass" if not errors else "fail",
        "static": {"catalog": catalog_summary, "pilots": pilot_summary},
        "environment": environment_summary,
        "errors": errors,
    }
    write_json(args.output_dir / "validation_report.json", report)
    if errors:
        raise ValueError("\n".join(errors))
    return report


def expand_command(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_runtime_tasks(args.tasks_dir)
    documents = load_documents(args.documents_dir)
    categories = category_lookup()
    tool_types = parse_tool_types(args.tools_file)
    annotations = {
        identifier: build_annotation(task, categories[identifier], documents, tool_types)
        for identifier, task in tasks.items()
    }
    expanded_tasks, expanded_contracts = build_expanded_curriculum(tasks, annotations, documents)
    errors, summary = validate_expanded_curriculum(
        expanded_tasks,
        expanded_contracts,
        tasks,
        documents,
    )
    report = {
        "status": "pass" if not errors else "fail",
        "static": summary,
        "environment": None,
        "errors": errors,
    }
    if errors:
        raise ValueError("\n".join(errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "expanded_tasks.json", expanded_tasks)
    write_jsonl(args.output_dir / "expanded_contracts.jsonl", expanded_contracts)
    write_json(args.output_dir / "expanded_validation_report.json", report)
    return report


def validate_expanded_command(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_runtime_tasks(args.tasks_dir)
    documents = load_documents(args.documents_dir)
    expanded_tasks = read_json(args.output_dir / "expanded_tasks.json")
    expanded_contracts = read_jsonl(args.output_dir / "expanded_contracts.jsonl")
    errors, summary = validate_expanded_curriculum(
        expanded_tasks,
        expanded_contracts,
        tasks,
        documents,
    )
    environment_summary = None
    if args.environment:
        environment_errors, raw_environment = validate_environment(
            tasks,
            expanded_tasks,
            expanded_contracts,
            documents,
        )
        errors.extend(environment_errors)
        environment_summary = {
            "schema_validated_tasks": raw_environment["schema_validated_pilots"],
            "source_reference_replays": raw_environment["source_reference_replays"],
            "expanded_reference_replays": raw_environment["pilot_reference_replays"],
            "source_reference_error_count": raw_environment["source_reference_error_count"],
            "source_reference_error_tasks": raw_environment["source_reference_error_tasks"],
            "source_reference_errors": raw_environment["source_reference_errors"],
            "expanded_validation_error_count": raw_environment["pilot_validation_error_count"],
            "source_replays": raw_environment["source_replays"],
            "expanded_replays": raw_environment["pilot_replays"],
        }
    report = {
        "status": "pass" if not errors else "fail",
        "static": summary,
        "environment": environment_summary,
        "errors": errors,
    }
    write_json(args.output_dir / "expanded_validation_report.json", report)
    if errors:
        raise ValueError("\n".join(errors))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate", "expand", "validate-expanded"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
        subparser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
        subparser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if command == "build":
            subparser.add_argument("--aggregate-tasks", type=Path, default=DEFAULT_AGGREGATE_TASKS)
            subparser.add_argument("--tools-file", type=Path, default=DEFAULT_TOOLS_FILE)
            subparser.set_defaults(historical_runs=HISTORICAL_RUNS)
        elif command == "expand":
            subparser.add_argument("--tools-file", type=Path, default=DEFAULT_TOOLS_FILE)
        else:
            subparser.add_argument("--environment", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        report = build_command(args)
    elif args.command == "validate":
        report = validate_command(args)
    elif args.command == "expand":
        report = expand_command(args)
    else:
        report = validate_expanded_command(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
