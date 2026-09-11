"""Fixed curriculum and domain constants for Banking synthetic v1."""

from __future__ import annotations

from collections import OrderedDict


DATA_ORIGIN = "independent_synthetic"
TASK_ID_PREFIX = "banking_syn_v1_"
TEMPLATE_VERSION = "banking-synthetic-v1-candidate-r18"
MODEL_PATH = "/mnt/afs/models/Qwen3.8-27B"
TRAIN_TASK_COUNT = 542
DEV_TASK_COUNT = 75
CHALLENGE_TASK_COUNT = 30
FOLLOWUP_TRAIN_COUNT = 380
MIN_TRAIN_SCENARIOS = 542
MIN_GRPO_FRONTIER = 163


BUSINESS_CATEGORIES = OrderedDict(
    [
        ("信用卡选择与申请资格", "credit_card_selection"),
        ("身份验证与资料变更", "identity_profile_change"),
        ("信用卡推荐奖励", "credit_card_referral"),
        ("知识缺失与转人工", "knowledge_gap_transfer"),
        ("返现与奖励核算", "cashback_rewards"),
        ("交易争议、购买保护与补卡", "credit_dispute_protection_replacement"),
        ("信用卡保留与销户", "credit_card_retention_closure"),
        ("信用额度调整", "credit_limit_adjustment"),
        ("账户推荐、开户、销户与入金", "account_lifecycle_funding"),
        ("ATM 费用", "atm_fees"),
        ("卡片遗失或被盗", "lost_or_stolen_card"),
        ("借记卡交易争议", "debit_card_dispute"),
        ("借记卡拒付与 PIN", "debit_decline_pin"),
        ("储蓄利息", "savings_interest"),
        ("账户推荐奖励", "account_referral"),
    ]
)


CATEGORY_TOPICS = {
    "信用卡选择与申请资格": "credit-card selection and application eligibility",
    "身份验证与资料变更": "identity verification and profile updates",
    "信用卡推荐奖励": "credit-card referral rewards",
    "知识缺失与转人工": "an unsupported request that needs human assistance",
    "返现与奖励核算": "cash-back and reward calculations",
    "交易争议、购买保护与补卡": "credit-card disputes, purchase protection, and replacement",
    "信用卡保留与销户": "credit-card retention and closure",
    "信用额度调整": "credit-limit adjustments",
    "账户推荐、开户、销户与入金": "account recommendations, opening, closure, and funding",
    "ATM 费用": "ATM fees",
    "卡片遗失或被盗": "a lost or stolen card",
    "借记卡交易争议": "a debit-card transaction dispute",
    "借记卡拒付与 PIN": "debit-card declines and PIN recovery",
    "储蓄利息": "savings interest",
    "账户推荐奖励": "bank-account referral rewards",
}


PILOT_FORM_QUOTAS = OrderedDict(
    [
        ("l1_evidence", 150),
        ("l2_retrieval", 75),
        ("l3_decision", 225),
        ("l3_single_action", 300),
        ("l4_two_skill", 450),
        ("l5_medium_workflow", 225),
        ("l6_full_workflow", 75),
    ]
)


TRAIN_FORM_QUOTAS = OrderedDict(
    [
        ("l1_evidence", 54),
        ("l2_retrieval", 27),
        ("l3_decision", 81),
        ("l3_single_action", 109),
        ("l4_two_skill", 163),
        ("l5_medium_workflow", 81),
        ("l6_full_workflow", 27),
    ]
)


FRONTIER_FORM_PRIORITY = (
    "l6_full_workflow",
    "l5_medium_workflow",
    "l4_two_skill",
    "l3_single_action",
    "l2_retrieval",
    "l3_decision",
    "l1_evidence",
)


TRAIN_CATEGORY_QUOTAS = OrderedDict(
    [
        ("信用卡选择与申请资格", 38),
        ("身份验证与资料变更", 27),
        ("信用卡推荐奖励", 30),
        ("知识缺失与转人工", 27),
        ("返现与奖励核算", 43),
        ("交易争议、购买保护与补卡", 49),
        ("信用卡保留与销户", 33),
        ("信用额度调整", 30),
        ("账户推荐、开户、销户与入金", 54),
        ("ATM 费用", 30),
        ("卡片遗失或被盗", 38),
        ("借记卡交易争议", 38),
        ("借记卡拒付与 PIN", 38),
        ("储蓄利息", 35),
        ("账户推荐奖励", 32),
    ]
)


FORM_LEVEL = {
    "l1_evidence": "L1",
    "l2_retrieval": "L2",
    "l3_decision": "L3",
    "l3_single_action": "L3",
    "l4_two_skill": "L4",
    "l5_medium_workflow": "L5",
    "l6_full_workflow": "L6",
}


FORM_DOCUMENT_BOUNDS = {
    "l1_evidence": (1, 1),
    "l2_retrieval": (1, 1),
    "l3_decision": (1, 2),
    "l3_single_action": (1, 2),
    "l4_two_skill": (2, 4),
    "l5_medium_workflow": (4, 8),
    "l6_full_workflow": (6, 12),
}


FORM_ACTION_BOUNDS = {
    "l1_evidence": (0, 0),
    "l2_retrieval": (1, 1),
    "l3_decision": (0, 0),
    "l3_single_action": (1, 1),
    "l4_two_skill": (2, 4),
    "l5_medium_workflow": (5, 8),
    "l6_full_workflow": (8, 14),
}


CASE_KINDS = (
    "positive",
    "refusal",
    "user_tool",
    "dynamic_tool",
    "multi_entity",
)


FORM_CASE_KINDS = {
    "l1_evidence": ("positive",),
    "l2_retrieval": ("positive",),
    "l3_decision": ("positive",),
    "l3_single_action": ("positive", "refusal", "user_tool", "dynamic_tool"),
    "l4_two_skill": ("positive", "refusal", "user_tool", "dynamic_tool"),
    "l5_medium_workflow": ("positive", "refusal", "user_tool", "dynamic_tool"),
    "l6_full_workflow": CASE_KINDS,
}


DOCUMENT_FAMILY_PREFIXES = {
    "信用卡选择与申请资格": ("doc_credit_cards_", "doc_business_credit_cards_"),
    "身份验证与资料变更": (
        "doc_bank_accounts_",
        "doc_checking_accounts_",
        "doc_business_checking_accounts_",
    ),
    "信用卡推荐奖励": ("doc_credit_cards_", "doc_business_credit_cards_"),
    "知识缺失与转人工": (
        "doc_everyone_pay_",
        "doc_buy_now_pay_later_",
        "doc_personal_subscriptions_",
    ),
    "返现与奖励核算": ("doc_credit_cards_", "doc_business_credit_cards_"),
    "交易争议、购买保护与补卡": (
        "doc_credit_cards_",
        "doc_business_credit_cards_",
    ),
    "信用卡保留与销户": ("doc_credit_cards_", "doc_business_credit_cards_"),
    "信用额度调整": ("doc_credit_cards_", "doc_business_credit_cards_"),
    "账户推荐、开户、销户与入金": (
        "doc_bank_accounts_",
        "doc_checking_accounts_",
        "doc_savings_accounts_",
        "doc_business_checking_accounts_",
        "doc_business_savings_accounts_",
    ),
    "ATM 费用": (
        "doc_bank_accounts_",
        "doc_checking_accounts_",
        "doc_business_checking_accounts_",
    ),
    "卡片遗失或被盗": (
        "doc_credit_cards_",
        "doc_business_credit_cards_",
        "doc_checking_accounts_",
        "doc_business_checking_accounts_",
    ),
    "借记卡交易争议": (
        "doc_bank_accounts_",
        "doc_checking_accounts_",
        "doc_business_checking_accounts_",
    ),
    "借记卡拒付与 PIN": (
        "doc_checking_accounts_",
        "doc_business_checking_accounts_",
        "doc_credit_cards_",
    ),
    "储蓄利息": ("doc_savings_accounts_", "doc_business_savings_accounts_"),
    "账户推荐奖励": (
        "doc_checking_accounts_",
        "doc_savings_accounts_",
        "doc_business_checking_accounts_",
        "doc_business_savings_accounts_",
    ),
}


# These are policy anchors for state-changing/action tasks. They were selected
# from the 480-document allowlist itself; generation still has no benchmark-task
# input. Additional L4-L6 documents are drawn from the anchor's product family.
ACTION_SUPPORT_DOCUMENTS = {
    "信用卡选择与申请资格": (
        "doc_business_credit_cards_green_rewards_card_001",
    ),
    "身份验证与资料变更": (
        "doc_checking_accounts_gold_years_account_005",
    ),
    "信用卡推荐奖励": (
        "doc_business_credit_cards_green_rewards_card_010",
    ),
    "知识缺失与转人工": ("doc_everyone_pay_everyone_pay_015",),
    "返现与奖励核算": (
        "doc_credit_cards_credit_cards_(general)_021",
        "doc_business_credit_cards_business_gold_rewards_card_002",
    ),
    "交易争议、购买保护与补卡": (
        "doc_credit_cards_credit_cards_(general)_018",
        "doc_credit_cards_credit_cards_(general)_022",
    ),
    "信用卡保留与销户": (
        "doc_credit_cards_credit_card_account_logistics_001",
    ),
    "信用额度调整": ("doc_credit_cards_bronze_rewards_card_006",),
    "账户推荐、开户、销户与入金": (
        "doc_bank_accounts_bank_accounts_(general)_022",
    ),
    "ATM 费用": (
        "doc_checking_accounts_purple_account_010",
        "doc_bank_accounts_bank_accounts_(general)_019",
    ),
    "卡片遗失或被盗": (
        "doc_business_checking_accounts_sky_blue_005",
    ),
    "借记卡交易争议": (
        "doc_bank_accounts_bank_accounts_(general)_019",
    ),
    "借记卡拒付与 PIN": (
        "doc_checking_accounts_gold_years_account_005",
    ),
    "储蓄利息": (
        "doc_business_savings_accounts_platinum_reserve_account_002",
    ),
    "账户推荐奖励": ("doc_checking_accounts_evergreen_account_007",),
}


DOCUMENT_TOPIC_PATTERNS = {
    "信用卡选择与申请资格": (r"\bapply\b", r"eligib", r"card overview", r"requirements"),
    "身份验证与资料变更": (r"security", r"verif", r"account management", r"support"),
    "信用卡推荐奖励": (r"\breferral\b", r"\brefer(?:ring)?\b"),
    "知识缺失与转人工": (r"\bfaq\b", r"troubleshoot", r"support", r"help"),
    "返现与奖励核算": (
        r"cash.?back",
        r"reward points",
        r"rewards? (?:earned|paid|redeeming|maximizing)",
    ),
    "交易争议、购买保护与补卡": (
        r"dispute",
        r"purchase protection",
        r"replacement card",
        r"fraud",
    ),
    "信用卡保留与销户": (r"\bclos(?:e|ing)\b", r"annual fee", r"fee waiver", r"retention"),
    "信用额度调整": (r"credit limit",),
    "账户推荐、开户、销户与入金": (
        r"open(?:ing)? .*account",
        r"clos(?:e|ing) .*account",
        r"deposit",
        r"adding funds",
        r"moving money",
    ),
    "ATM 费用": (r"\batm\b",),
    "卡片遗失或被盗": (r"replacement card", r"fraud", r"revok", r"cancel.*card"),
    "借记卡交易争议": (r"dispute", r"statement", r"transaction history", r"fraud"),
    "借记卡拒付与 PIN": (r"declined transaction", r"transaction declined", r"security"),
    "储蓄利息": (r"interest", r"\bapy\b"),
    "账户推荐奖励": (r"\breferral\b", r"\brefer(?:ring)?\b", r"invite"),
}


DOCUMENT_CONTENT_PATTERNS = {
    "账户推荐奖励": (r"referral program",),
}


FACT_TOPIC_PATTERNS = {
    "信用卡选择与申请资格": (
        r"\bapply\b",
        r"applicat",
        r"eligib",
        r"credit score",
        r"\bfico\b",
        r"\bpaydex\b",
        r"underwriting",
    ),
    "身份验证与资料变更": (
        r"identity",
        r"verif",
        r"profile",
        r"email",
        r"phone",
        r"address",
    ),
    "信用卡推荐奖励": (
        r"referral",
        r"\brefer(?:red|ring)?\b",
        r"qualifying spend",
    ),
    "知识缺失与转人工": (
        r"support",
        r"\bhelp\b",
        r"contact us",
        r"human agent",
        r"not (?:available|supported)",
    ),
    "返现与奖励核算": (
        r"cash.?back",
        r"reward",
        r"\bearn(?:ed|ing)?\b",
        r"redeem",
    ),
    "交易争议、购买保护与补卡": (
        r"dispute",
        r"purchase protection",
        r"fraud",
        r"not delivered",
        r"replacement card",
        r"lock your card",
    ),
    "信用卡保留与销户": (
        r"\bclos(?:e|ed|ing|ure)\b",
        r"zero balance",
        r"annual fee",
        r"retention",
    ),
    "信用额度调整": (
        r"credit limit",
        r"credit line",
        r"limit increase",
        r"spending limit",
    ),
    "账户推荐、开户、销户与入金": (
        r"open(?:ing)? .*account",
        r"clos(?:e|ing) .*account",
        r"deposit",
        r"transfer",
        r"moving money",
        r"account balance",
    ),
    "ATM 费用": (r"\batm\b", r"rebate", r"withdrawal"),
    "卡片遗失或被盗": (
        r"lost",
        r"stolen",
        r"lock your card",
        r"replacement card",
        r"fraud",
    ),
    "借记卡交易争议": (r"dispute", r"transaction", r"fraud", r"charge"),
    "借记卡拒付与 PIN": (r"declin", r"\bpin\b", r"locked", r"attempt"),
    "储蓄利息": (r"interest", r"\bapy\b", r"yield"),
    "账户推荐奖励": (r"referral", r"\brefer(?:red|ring)?\b", r"\binvite\b"),
}


LOW_FORM_CATEGORIES = frozenset(
    {
        "信用卡选择与申请资格",
        "信用卡推荐奖励",
        "知识缺失与转人工",
        "返现与奖励核算",
        "交易争议、购买保护与补卡",
        "信用卡保留与销户",
        "信用额度调整",
        "账户推荐、开户、销户与入金",
        "ATM 费用",
        "储蓄利息",
        "账户推荐奖励",
    }
)


EMPTY_DB_TABLES = (
    "users",
    "accounts",
    "debit_cards",
    "referrals",
    "credit_card_applications",
    "user_discoverable_tools",
    "user_discoverable_tool_calls",
    "verification_history",
    "credit_card_transaction_history",
    "cash_back_disputes",
    "bank_account_transaction_history",
    "credit_card_accounts",
    "agent_discoverable_tools",
    "task_config",
    "human_transfer_requests",
    "transaction_disputes",
    "credit_card_orders",
    "debit_card_orders",
    "credit_card_closure_reasons",
    "credit_card_account_flags",
    "credit_limit_increase_requests",
    "payment_history",
    "debit_card_disputes",
)
