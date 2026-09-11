"""Programmatic Banking scenarios built only from synthetic entities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .constants import (
    ACTION_SUPPORT_DOCUMENTS,
    CATEGORY_TOPICS,
    CASE_KINDS,
    FORM_ACTION_BOUNDS,
    FORM_CASE_KINDS,
    FORM_DOCUMENT_BOUNDS,
    FORM_LEVEL,
    LOW_FORM_CATEGORIES,
    TEMPLATE_VERSION,
)
from .documents import (
    DocumentCatalog,
    communication_anchors,
    decisive_fact,
    indexed_body_anchor,
)
from .models import ActionNode, ScenarioSpec


@dataclass(frozen=True)
class SyntheticEntities:
    user_id: str
    name: str
    email: str
    new_email: str
    phone: str
    address: str
    date_of_birth: str
    checking_id: str
    savings_id: str
    credit_card_id: str
    debit_card_id: str
    credit_transaction_id: str
    debit_transaction_id: str


CREDIT_CARD_PRODUCTS = {
    "信用卡选择与申请资格": "Bronze Rewards Card",
    "信用卡推荐奖励": "Green Rewards Card",
    "返现与奖励核算": "Business Gold Rewards Card",
    "信用额度调整": "Bronze Rewards Card",
}


CHECKING_PRODUCTS = {
    "身份验证与资料变更": "Gold Years Account",
    "ATM 费用": "Purple Account",
    "卡片遗失或被盗": "Sky Blue Account",
    "借记卡拒付与 PIN": "Gold Years Account",
    "账户推荐奖励": "Evergreen Account",
}


REFUSAL_REQUESTS = {
    "信用卡选择与申请资格": "issue a card without an application or underwriting review",
    "身份验证与资料变更": "change the legal identity on the profile without verification",
    "信用卡推荐奖励": "guarantee a referral bonus before the referred business qualifies",
    "知识缺失与转人工": "schedule a cryptocurrency wire from a BNPL plan",
    "返现与奖励核算": "grant cash back for a returned purchase",
    "交易争议、购买保护与补卡": "erase a posted purchase without opening a dispute",
    "信用卡保留与销户": "close a credit card that still has an outstanding balance",
    "信用额度调整": "raise the limit beyond the documented product range without review",
    "账户推荐、开户、销户与入金": "transfer more money than the source account holds",
    "ATM 费用": "refund ATM fees after the monthly rebate cap is exhausted",
    "卡片遗失或被盗": "reactivate a card already confirmed stolen",
    "借记卡交易争议": "guarantee zero liability without collecting a dispute statement",
    "借记卡拒付与 PIN": "reveal the customer's existing PIN",
    "储蓄利息": "credit interest that was never earned",
    "账户推荐奖励": "pay a referral bonus before the qualifying deposit",
}


def _entities(split: str, serial: int) -> SyntheticEntities:
    token = f"{split}_{serial:06d}"
    return SyntheticEntities(
        user_id=f"syn_user_{token}",
        name=f"Synthetic Customer {serial:06d}",
        email=f"synthetic.{split}.{serial:06d}@example.test",
        new_email=f"synthetic.updated.{split}.{serial:06d}@example.test",
        phone=f"+1-555-{200 + serial % 700:03d}-{serial % 10000:04d}",
        address=f"{1000 + serial} Synthesis Avenue, Testville, OR 97000",
        date_of_birth=f"{1 + serial % 12:02d}/{1 + serial % 27:02d}/{1970 + serial % 25}",
        checking_id=f"syn_checking_{token}",
        savings_id=f"syn_savings_{token}",
        credit_card_id=f"syn_cc_{token}",
        debit_card_id=f"syn_dc_{token}",
        credit_transaction_id=f"syn_ctxn_{token}",
        debit_transaction_id=f"syn_dtxn_{token}",
    )


def _serial(entity: SyntheticEntities) -> int:
    return int(entity.user_id.rsplit("_", 1)[-1])


def _debit_last_four(entity: SyntheticEntities) -> str:
    return f"{4000 + _serial(entity) % 5000:04d}"


def _credit_last_four(entity: SyntheticEntities) -> str:
    return f"{(4821 + _serial(entity)) % 10000:04d}"


def initialization_data(
    entity: SyntheticEntities,
    category: str,
    case_kind: str,
    *,
    preverified: bool = False,
) -> dict[str, Any]:
    payment_id = f"syn_payment_{entity.user_id.rsplit('_', 1)[-1]}"
    checking_product = CHECKING_PRODUCTS.get(category, "Light Blue Account")
    savings_product = (
        "Platinum Reserve Account" if category == "储蓄利息" else "Gold Account"
    )
    savings_holdings = "100000.00" if category == "储蓄利息" else "2400.00"
    credit_card_product = CREDIT_CARD_PRODUCTS.get(category, "Gold Rewards Card")
    credit_balance = (
        "$0.00"
        if category == "信用卡保留与销户" and case_kind != "refusal"
        else "$400.00"
    )
    bank_transaction = {
        "transaction_id": entity.debit_transaction_id,
        "account_id": entity.checking_id,
        "card_id": entity.debit_card_id,
        "user_id": entity.user_id,
        "date": "11/10/2025",
        "description": "SYNTHETIC ONLINE PURCHASE",
        "amount": -64.75,
        "type": "online_purchase",
        "status": "posted",
    }
    if category == "ATM 费用":
        bank_transaction.update(
            {
                "description": "OUT-OF-NETWORK ATM FEE NOT REBATED",
                "amount": -6.0,
                "type": "atm_fee",
            }
        )
    credit_transaction = {
        "transaction_id": entity.credit_transaction_id,
        "user_id": entity.user_id,
        "credit_card_account_id": entity.credit_card_id,
        "credit_card_type": credit_card_product,
        "merchant_name": "Synthetic Market",
        "transaction_amount": "$84.25",
        "transaction_date": "11/02/2025",
        "category": "Shopping",
        "status": "COMPLETED",
        "rewards_earned": "84 points",
    }
    if category == "返现与奖励核算":
        credit_transaction.update(
            {
                "category": "Operations",
                "rewards_earned": "$0.84 cash back",
            }
        )
    data = {
        "users": {
            "data": {
                entity.user_id: {
                    "user_id": entity.user_id,
                    "name": entity.name,
                    "address": entity.address,
                    "email": entity.email,
                    "phone_number": entity.phone,
                    "date_of_birth": entity.date_of_birth,
                    "personal_credit_score": 720,
                    "business_paydex_score": 60,
                }
            }
        },
        "accounts": {
            "data": {
                entity.checking_id: {
                    "account_id": entity.checking_id,
                    "user_id": entity.user_id,
                    "account_type": "checking",
                    "account_class": checking_product,
                    "class": "checking",
                    "level": checking_product,
                    "current_holdings": "5000.00",
                    "status": "OPEN",
                    "date_opened": "01/15/2024",
                },
                entity.savings_id: {
                    "account_id": entity.savings_id,
                    "user_id": entity.user_id,
                    "account_type": "savings",
                    "account_class": savings_product,
                    "class": "savings",
                    "level": savings_product,
                    "current_holdings": savings_holdings,
                    "status": "OPEN",
                    "date_opened": "02/20/2024",
                    "apy": "5.00%" if category == "储蓄利息" else "3.50%",
                },
            }
        },
        "debit_cards": {
            "data": {
                entity.debit_card_id: {
                    "card_id": entity.debit_card_id,
                    "account_id": entity.checking_id,
                    "user_id": entity.user_id,
                    "cardholder_name": entity.name.upper(),
                    "last_4_digits": _debit_last_four(entity),
                    "expiration_date": "12/29",
                    "cvv": "731",
                    "status": "ACTIVE",
                    "issue_reason": "first_card",
                    "pin_locked": False,
                    "pin_attempts_remaining": 2,
                }
            }
        },
        "credit_card_accounts": {
            "data": {
                entity.credit_card_id: {
                    "account_id": entity.credit_card_id,
                    "user_id": entity.user_id,
                    "card_type": credit_card_product,
                    "last_4_digits": _credit_last_four(entity),
                    "status": "OPEN",
                    "account_status": "GOOD_STANDING",
                    "date_opened": "03/10/2023",
                    "credit_limit": "$8000.00",
                    "current_balance": credit_balance,
                    "reward_points": 1250,
                }
            }
        },
        "credit_card_transaction_history": {
            "data": {
                entity.credit_transaction_id: credit_transaction
            }
        },
        "bank_account_transaction_history": {
            "data": {
                entity.debit_transaction_id: bank_transaction
            }
        },
        "payment_history": {
            "data": {
                payment_id: {
                    "payment_id": payment_id,
                    "credit_card_account_id": entity.credit_card_id,
                    "payment_date": "11/01/2025",
                    "amount": "$250.00",
                    "status": "ON_TIME",
                }
            }
        },
    }
    if preverified:
        data["verification_history"] = {
            "data": {
                f"{entity.user_id}_20251114_034000_EST": {
                    "name": entity.name,
                    "user_id": entity.user_id,
                    "address": entity.address,
                    "email": entity.email,
                    "phone_number": entity.phone,
                    "date_of_birth": entity.date_of_birth,
                    "time_verified": "2025-11-14 03:40:00 EST",
                }
            }
        }
    return data


def _action(
    action_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    requestor: str = "assistant",
    compare_args: list[str] | None = None,
) -> dict[str, Any]:
    result = {
        "action_id": action_id,
        "requestor": requestor,
        "name": name,
        "arguments": arguments,
    }
    if compare_args is not None:
        result["compare_args"] = compare_args
    return result


def _env_action(name: str, arguments: dict[str, Any], requestor: str = "assistant") -> dict[str, Any]:
    return {"env_type": requestor, "func_name": name, "arguments": arguments}


def _dynamic_pair(prefix: str, tool_name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _action(f"{prefix}_unlock", "unlock_discoverable_agent_tool", {"agent_tool_name": tool_name}),
        _action(
            f"{prefix}_call",
            "call_discoverable_agent_tool",
            {
                "agent_tool_name": tool_name,
                "arguments": json.dumps(arguments, sort_keys=True),
            },
        ),
    ]


def _user_discoverable_pair(
    prefix: str, tool_name: str, arguments: dict[str, Any]
) -> list[dict[str, Any]]:
    encoded = json.dumps(arguments, sort_keys=True)
    return [
        _action(
            f"{prefix}_give",
            "give_discoverable_user_tool",
            {"discoverable_tool_name": tool_name, "arguments": encoded},
        ),
        _action(
            f"{prefix}_user",
            "call_discoverable_user_tool",
            {"discoverable_tool_name": tool_name, "arguments": encoded},
            requestor="user",
        ),
    ]


def _operation(
    category: str, entity: SyntheticEntities, prefix: str
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], str]:
    if category == "信用卡选择与申请资格":
        return (
            [
                _action(
                    f"{prefix}_apply",
                    "apply_for_credit_card",
                    {
                        "card_type": "Green Rewards Card",
                        "customer_name": entity.name,
                        "annual_income": 82000 + int(entity.user_id[-3:]),
                        "rho_bank_subscription": False,
                    },
                    requestor="user",
                )
            ],
            ("apply_for_credit_card",),
            ("credit_card_applications",),
            "review eligibility, then submit the Green Rewards Card application through the customer-side application action with a 720 personal score and 60 PAYDEX score",
        )
    if category == "身份验证与资料变更":
        return (
            [
                _action(
                    f"{prefix}_email",
                    "change_user_email",
                    {"user_id": entity.user_id, "new_email": entity.new_email},
                )
            ],
            (),
            ("users",),
            f"change the email address to {entity.new_email} after identity verification",
        )
    if category == "信用卡推荐奖励":
        return (
            _user_discoverable_pair(
                prefix,
                "get_referral_link",
                {"user_id": entity.user_id, "card_name": "Green Rewards Card"},
            ),
            ("call_discoverable_user_tool",),
            ("user_discoverable_tools", "user_discoverable_tool_calls", "referrals"),
            "have the Agent provide the customer-side referral-link action after confirming the account is in good standing, then use it for the Green Rewards Card",
        )
    if category == "知识缺失与转人工":
        return (
            [
                _action(
                    f"{prefix}_transfer",
                    "transfer_to_human_agents",
                    {
                        "summary": "The requested synthetic feature is not supported by the available policy documents.",
                        "reason": "other",
                    },
                    compare_args=[],
                )
            ],
            (),
            (),
            "decline a request to schedule a cryptocurrency wire from a BNPL plan and transfer to a human",
        )
    if category == "返现与奖励核算":
        return (
            _user_discoverable_pair(
                prefix,
                "submit_cash_back_dispute_0589",
                {
                    "user_id": entity.user_id,
                    "transaction_id": entity.credit_transaction_id,
                },
            ),
            ("call_discoverable_user_tool",),
            ("user_discoverable_tools", "user_discoverable_tool_calls", "cash_back_disputes"),
            "have the Agent provide the customer-side cash-back dispute action, then use it because $0.84 posted instead of the documented 2.5% ($2.11)",
        )
    if category == "交易争议、购买保护与补卡":
        return (
            _dynamic_pair(
                prefix,
                "file_credit_card_transaction_dispute_4829",
                {
                    "transaction_id": entity.credit_transaction_id,
                    "card_action": "keep_active",
                    "card_last_4_digits": _credit_last_four(entity),
                    "full_name": entity.name,
                    "user_id": entity.user_id,
                    "phone": entity.phone,
                    "email": entity.email,
                    "address": entity.address,
                    "contacted_merchant": True,
                    "purchase_date": "11/02/2025",
                    "issue_noticed_date": "11/12/2025",
                    "dispute_reason": "incorrect_amount",
                    "resolution_requested": "partial_refund",
                    "eligible_for_provisional_credit": True,
                    "partial_refund_amount": 36.0,
                },
            ),
            (),
            ("agent_discoverable_tools", "transaction_disputes"),
            "dispute the $36 overcharge after Synthetic Market charged $84.25 for a $48.25 purchase and could not resolve it",
        )
    if category == "信用卡保留与销户":
        return (
            _dynamic_pair(
                prefix,
                "close_credit_card_account_7834",
                {
                    "credit_card_account_id": entity.credit_card_id,
                    "user_id": entity.user_id,
                },
            ),
            (),
            ("agent_discoverable_tools", "credit_card_accounts"),
            "close the zero-balance card after confirming there are no pending disputes or replacements",
        )
    if category == "信用额度调整":
        return (
            _dynamic_pair(
                prefix,
                "approve_credit_limit_increase_5847",
                {
                    "credit_card_account_id": entity.credit_card_id,
                    "user_id": entity.user_id,
                    "new_credit_limit": 10000,
                },
            ),
            (),
            ("agent_discoverable_tools", "credit_card_accounts", "credit_limit_increase_requests"),
            "approve the Bronze Rewards Card increase from $8,000 to $10,000 after its clean payment history",
        )
    if category == "账户推荐、开户、销户与入金":
        return (
            _dynamic_pair(
                prefix,
                "transfer_funds_between_bank_accounts_7291",
                {
                    "source_account_id": entity.checking_id,
                    "destination_account_id": entity.savings_id,
                    "amount": 125.0,
                },
            ),
            (),
            ("agent_discoverable_tools", "accounts"),
            "move $125 from the open $5,000 checking account to the open savings account",
        )
    if category == "ATM 费用":
        return (
            _dynamic_pair(
                prefix,
                "apply_checking_account_credit_5829",
                {"account_id": entity.checking_id, "amount": 6.0, "credit_type": "fee_refund"},
            ),
            (),
            ("agent_discoverable_tools", "accounts", "bank_account_transaction_history"),
            "refund the eligible $6 out-of-network ATM fee that was not rebated and remains below the monthly cap",
        )
    if category == "卡片遗失或被盗":
        return (
            _dynamic_pair(prefix, "freeze_debit_card_3892", {"card_id": entity.debit_card_id}),
            (),
            ("agent_discoverable_tools", "debit_cards"),
            "temporarily freeze the reported missing debit card while the customer searches for it",
        )
    if category == "借记卡交易争议":
        return (
            _dynamic_pair(
                prefix,
                "file_debit_card_transaction_dispute_6281",
                {
                    "transaction_id": entity.debit_transaction_id,
                    "account_id": entity.checking_id,
                    "card_id": entity.debit_card_id,
                    "user_id": entity.user_id,
                    "dispute_category": "unauthorized_transaction",
                    "transaction_date": "11/10/2025",
                    "discovery_date": "11/12/2025",
                    "disputed_amount": 64.75,
                    "transaction_type": "online_purchase",
                    "card_in_possession": True,
                    "pin_compromised": "no",
                    "contacted_merchant": False,
                    "police_report_filed": False,
                    "written_statement_provided": True,
                    "provisional_credit_eligible": True,
                    "customer_max_liability_amount": 50.0,
                    "card_action": "freeze_pending_investigation",
                },
            ),
            (),
            ("agent_discoverable_tools", "debit_card_disputes"),
            "file the $64.75 unauthorized online debit-card dispute reported two days after discovery with a written statement",
        )
    if category == "借记卡拒付与 PIN":
        last_four = _debit_last_four(entity)
        return (
            _dynamic_pair(
                prefix,
                "reset_debit_card_pin_6284",
                {"card_id": entity.debit_card_id, "last_4_digits": last_four, "new_pin": "5826"},
            ),
            (),
            ("agent_discoverable_tools", "debit_cards"),
            f"reset the forgotten PIN using card digits {last_four} and the customer's new PIN 5826",
        )
    if category == "储蓄利息":
        return (
            _dynamic_pair(
                prefix,
                "apply_savings_account_credit_6831",
                {"account_id": entity.savings_id, "amount": 18.5, "credit_type": "interest_correction"},
            ),
            (),
            ("agent_discoverable_tools", "accounts", "bank_account_transaction_history"),
            "apply the verified $18.50 missing-interest correction to the 5.0% Platinum Reserve Account",
        )
    return (
        [
            _action(
                f"{prefix}_referral",
                "submit_referral",
                {"user_id": entity.user_id, "account_type": "Evergreen Account"},
                requestor="user",
            )
        ],
        ("submit_referral",),
        ("referrals",),
        "confirm the referrer's 45-day tenure, then submit the Evergreen Account referral through the customer-side referral action",
    )


def _user_case_operation(
    category: str, entity: SyntheticEntities
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], str]:
    primary = _operation(category, entity, "primary")
    if any(action["requestor"] == "user" for action in primary[0]):
        return primary
    if category in {
        "身份验证与资料变更",
        "交易争议、购买保护与补卡",
        "信用卡保留与销户",
        "信用额度调整",
    }:
        return (
            _user_discoverable_pair(
                "primary",
                "get_card_last_4_digits",
                {"credit_card_account_id": entity.credit_card_id},
            ),
            ("call_discoverable_user_tool",),
            ("user_discoverable_tools", "user_discoverable_tool_calls"),
            f"have the Agent provide the customer-side card-detail lookup, then use it to confirm the detail needed for {CATEGORY_TOPICS[category]}",
        )
    if category == "账户推荐、开户、销户与入金":
        return (
            _user_discoverable_pair(
                "primary",
                "deposit_check_3847",
                {"account_id": entity.checking_id, "check_amount": 95.0},
            ),
            ("call_discoverable_user_tool",),
            ("user_discoverable_tools", "user_discoverable_tool_calls", "accounts"),
            "have the Agent provide the customer-side mobile-check-deposit action, then use it to deposit the check into the selected account",
        )
    return (
        [
            _action(
                "primary_user_transfer",
                "request_human_agent_transfer",
                {},
                requestor="user",
            )
        ],
        ("request_human_agent_transfer",),
        ("human_transfer_requests",),
        f"record one customer-owned human-review request for {CATEGORY_TOPICS[category]} through the customer-side request action without asking the Agent to transfer",
    )


def _dynamic_case_operation(
    category: str, entity: SyntheticEntities
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], str]:
    primary = _operation(category, entity, "primary")
    if any(action["name"] == "unlock_discoverable_agent_tool" for action in primary[0]):
        return primary
    if category == "信用卡选择与申请资格":
        actions = _dynamic_pair(
            "primary",
            "get_pending_replacement_orders_5765",
            {"credit_card_account_id": entity.credit_card_id},
        )
        goal = "check the existing card relationship before making an eligibility recommendation"
        tables = ("agent_discoverable_tools",)
    elif category == "身份验证与资料变更":
        actions = _dynamic_pair(
            "primary",
            "get_payment_history_6183",
            {"credit_card_account_id": entity.credit_card_id, "months": 1},
        )
        goal = "confirm a private account-history detail before the profile update"
        tables = ("agent_discoverable_tools",)
    elif category == "信用卡推荐奖励":
        actions = _dynamic_pair(
            "primary",
            "get_pending_replacement_orders_5765",
            {"credit_card_account_id": entity.credit_card_id},
        )
        goal = "confirm the referring card account is usable before creating a referral"
        tables = ("agent_discoverable_tools",)
    elif category == "知识缺失与转人工":
        actions = _dynamic_pair("primary", "initial_transfer_to_human_agent_1822", {})
        goal = "initiate the specialized escalation path for the unsupported request"
        tables = ("agent_discoverable_tools",)
    elif category == "返现与奖励核算":
        actions = _dynamic_pair(
            "primary",
            "apply_statement_credit_8472",
                {
                    "user_id": entity.user_id,
                    "credit_card_account_id": entity.credit_card_id,
                    "amount": 1.27,
                    "reason": "error_correction",
                },
            )
        goal = "apply the verified $1.27 cash-back shortfall as a statement correction"
        tables = (
            "agent_discoverable_tools",
            "credit_card_accounts",
            "credit_card_transaction_history",
        )
    else:
        actions = _dynamic_pair(
            "primary",
            "get_bank_account_transactions_9173",
            {"account_id": entity.checking_id},
        )
        goal = "review the existing account activity before issuing an account referral"
        tables = ("agent_discoverable_tools",)
    return actions, (), tables, goal


def _secondary_operation(
    category: str, entity: SyntheticEntities
) -> tuple[list[dict[str, Any]], tuple[str, ...], str]:
    if category == "信用卡选择与申请资格":
        return (
            _dynamic_pair(
                "secondary",
                "get_payment_history_6183",
                {"credit_card_account_id": entity.credit_card_id, "months": 1},
            ),
            ("agent_discoverable_tools",),
            "review the existing card's payment history",
        )
    if category == "身份验证与资料变更":
        return (
            _dynamic_pair(
                "secondary",
                "get_all_user_accounts_by_user_id_3847",
                {"user_id": entity.user_id},
            ),
            ("agent_discoverable_tools",),
            "confirm the accounts attached to the verified profile",
        )
    if category == "信用卡推荐奖励":
        return (
            [_action("secondary_referrals", "get_referrals_by_user", {"user_id": entity.user_id})],
            (),
            "check the customer's existing referrals",
        )
    if category == "知识缺失与转人工":
        return (
            _dynamic_pair("secondary", "initial_transfer_to_human_agent_0218", {}),
            ("agent_discoverable_tools",),
            "complete the specialized human-transfer preflight",
        )
    if category == "返现与奖励核算":
        return (
            _dynamic_pair(
                "secondary",
                "update_transaction_rewards_3847",
                {
                    "transaction_id": entity.credit_transaction_id,
                    "new_rewards_earned": "211 points",
                },
            ),
            ("agent_discoverable_tools", "credit_card_transaction_history"),
            "record the corrected reward total",
        )
    if category == "交易争议、购买保护与补卡":
        return (
            _dynamic_pair(
                "secondary", "get_user_dispute_history_7291", {"user_id": entity.user_id}
            ),
            ("agent_discoverable_tools",),
            "confirm the newly filed dispute in the customer's history",
        )
    if category == "信用卡保留与销户":
        return (
            _dynamic_pair(
                "secondary",
                "get_closure_reason_history_8293",
                {"credit_card_account_id": entity.credit_card_id},
            ),
            ("agent_discoverable_tools",),
            "confirm there is no unresolved prior closure request",
        )
    if category == "信用额度调整":
        return (
            _dynamic_pair(
                "secondary",
                "get_credit_limit_increase_history_4829",
                {"credit_card_account_id": entity.credit_card_id},
            ),
            ("agent_discoverable_tools",),
            "confirm the credit-limit request history",
        )
    if category == "账户推荐、开户、销户与入金":
        return (
            _dynamic_pair(
                "secondary",
                "get_all_user_accounts_by_user_id_3847",
                {"user_id": entity.user_id},
            ),
            ("agent_discoverable_tools",),
            "confirm both linked accounts after funding",
        )
    if category == "ATM 费用":
        return (
            _dynamic_pair(
                "secondary",
                "get_bank_account_transactions_9173",
                {"account_id": entity.checking_id},
            ),
            ("agent_discoverable_tools",),
            "confirm the ATM-fee correction in account activity",
        )
    if category == "卡片遗失或被盗":
        return (
            _dynamic_pair(
                "secondary",
                "get_debit_cards_by_account_id_7823",
                {"account_id": entity.checking_id},
            ),
            ("agent_discoverable_tools",),
            "confirm the affected card under the linked account",
        )
    if category == "借记卡交易争议":
        return (
            _dynamic_pair(
                "secondary", "get_debit_dispute_status_7483", {"user_id": entity.user_id}
            ),
            ("agent_discoverable_tools",),
            "confirm the debit-card dispute status",
        )
    if category == "借记卡拒付与 PIN":
        return (
            _dynamic_pair(
                "secondary",
                "get_debit_cards_by_account_id_7823",
                {"account_id": entity.checking_id},
            ),
            ("agent_discoverable_tools",),
            "confirm the recovered card state",
        )
    if category == "储蓄利息":
        return (
            _dynamic_pair(
                "secondary",
                "get_bank_account_transactions_9173",
                {"account_id": entity.savings_id},
            ),
            ("agent_discoverable_tools",),
            "confirm the interest correction in savings-account activity",
        )
    return (
        _dynamic_pair(
            "secondary",
            "get_all_user_accounts_by_user_id_3847",
            {"user_id": entity.user_id},
        ),
        ("agent_discoverable_tools",),
        "confirm the referrer's current bank accounts",
    )


def _verification_actions(entity: SyntheticEntities, prefix: str) -> list[dict[str, Any]]:
    return [
        _action(f"{prefix}_lookup", "get_user_information_by_id", {"user_id": entity.user_id}),
        _action(f"{prefix}_time", "get_current_time", {}),
        _action(
            f"{prefix}_verify",
            "log_verification",
            {
                "name": entity.name,
                "user_id": entity.user_id,
                "address": entity.address,
                "email": entity.email,
                "phone_number": entity.phone,
                "date_of_birth": entity.date_of_birth,
                "time_verified": "2025-11-14 03:40:00 EST",
            },
        ),
    ]


def _search_action(action_id: str, query: str) -> dict[str, Any]:
    return _action(action_id, "KB_search", {"query": query}, compare_args=[])


def _action_clues(actions: list[dict[str, Any]]) -> tuple[str, ...]:
    """Expose otherwise undiscoverable synthetic action labels and case inputs."""
    clues = []
    seen = set()
    for action in actions:
        name = action["name"]
        arguments = action.get("arguments") or {}
        if name == "call_discoverable_agent_tool":
            clue = (
                "Agent must invoke its `call_discoverable_agent_tool` tool with "
                "arguments "
                + json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        elif name == "call_discoverable_user_tool":
            owner = "customer-side"
            label = arguments["discoverable_tool_name"]
            parameters = arguments.get("arguments") or "{}"
        elif action.get("requestor") == "user" and name not in {
            "call_discoverable_user_tool"
        }:
            owner = "customer-side"
            label = name
            parameters = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            continue
        if name != "call_discoverable_agent_tool":
            clue = (
                f"{owner} action label `{label}` with case parameters {parameters}"
            )
        if clue not in seen:
            clues.append(clue)
            seen.add(clue)
    return tuple(clues)


def _refusal_flow(
    form: str,
    category: str,
    entity: SyntheticEntities,
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[str, ...], str]:
    if form == "l3_single_action":
        actions, user_tools, tables, _ = _operation(
            "知识缺失与转人工", entity, "refusal"
        )
        return (
            actions,
            user_tools,
            tables,
            f"decline the unsupported request to {REFUSAL_REQUESTS[category]}, then immediately transfer to a human because the customer explicitly requests and consents to that transfer",
        )
    target = FORM_ACTION_BOUNDS[form][0]
    search_count = min(len(documents), max(1, target - 2))
    actions = [
        _search_action(f"refusal_search_{index}", indexed_body_anchor(document))
        for index, document in enumerate(documents[:search_count])
    ]
    if category in {"信用卡推荐奖励", "账户推荐奖励"}:
        category_read = _action(
            "refusal_category_read", "get_referrals_by_user", {"user_id": entity.user_id}
        )
    elif category in {
        "信用卡选择与申请资格",
        "返现与奖励核算",
        "交易争议、购买保护与补卡",
        "信用卡保留与销户",
        "信用额度调整",
        "卡片遗失或被盗",
    }:
        category_read = _action(
            "refusal_category_read",
            "get_credit_card_accounts_by_user",
            {"user_id": entity.user_id},
        )
    elif category == "知识缺失与转人工":
        category_read = _action(
            "refusal_category_read", "list_discoverable_agent_tools", {}
        )
    else:
        category_read = _action(
            "refusal_category_read",
            "get_user_information_by_id",
            {"user_id": entity.user_id},
        )
    fillers = [
        category_read,
        _action("refusal_time", "get_current_time", {}),
        _action("refusal_tools", "list_discoverable_agent_tools", {}),
    ]
    for item in fillers:
        if len(actions) >= target - 1:
            break
        actions.append(item)
    actions.append(
        _action(
            "refusal_customer_review",
            "request_human_agent_transfer",
            {},
            requestor="user",
        )
    )
    return (
        actions,
        ("request_human_agent_transfer",),
        ("human_transfer_requests",),
        f"decline the unsupported request to {REFUSAL_REQUESTS[category]}, then invite the customer to record one customer-owned human-review request without using an Agent-owned transfer action",
    )


def _flow(
    *,
    form: str,
    category: str,
    case_kind: str,
    entity: SyntheticEntities,
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...], tuple[str, ...], str]:
    if case_kind == "refusal" and form not in {"l1_evidence", "l2_retrieval", "l3_decision"}:
        actions, user_tools, tables, goal = _refusal_flow(form, category, entity, documents)
        return actions, [], user_tools, tables, goal

    primary, user_tools, tables, goal = _operation(category, entity, "primary")
    if case_kind == "user_tool":
        primary, user_tools, tables, goal = _user_case_operation(category, entity)
    elif case_kind == "dynamic_tool" and not any(
        action["name"] == "unlock_discoverable_agent_tool" for action in primary
    ):
        primary, user_tools, tables, goal = _dynamic_case_operation(category, entity)
    if form == "l3_single_action":
        initialization_actions: list[dict[str, Any]] = []
        if len(primary) == 2 and primary[0]["name"] == "unlock_discoverable_agent_tool":
            initialization_actions.append(
                _env_action(primary[0]["name"], primary[0]["arguments"])
            )
            primary = [primary[1]]
            tables = tuple(
                table for table in tables if table != "agent_discoverable_tools"
            )
        elif len(primary) == 2 and primary[0]["name"] == "give_discoverable_user_tool":
            initialization_actions.append(
                _env_action(primary[0]["name"], primary[0]["arguments"])
            )
            primary = [primary[1]]
            tables = tuple(
                table for table in tables if table != "user_discoverable_tools"
            )
            goal = goal.replace(
                "have the Agent provide the customer-side",
                "use the already available customer-side",
            ).replace(", then use it", "")
        return primary[:1], initialization_actions, user_tools, tables, goal

    if form == "l4_two_skill":
        actions = [_search_action("policy_search", indexed_body_anchor(documents[0]))]
        actions.extend(primary)
        return actions[:4], [], user_tools, tables, goal

    verification = _verification_actions(entity, "identity")
    actions = [_search_action("policy_search", indexed_body_anchor(documents[0]))]
    actions.extend(verification)
    actions.extend(primary)
    if form == "l5_medium_workflow":
        return actions[:8], [], user_tools, tuple(dict.fromkeys(("verification_history",) + tables)), goal

    actions.insert(1, _search_action("secondary_policy_search", indexed_body_anchor(documents[1])))
    secondary, secondary_tables, secondary_goal = _secondary_operation(category, entity)
    actions.extend(secondary)
    combined_tables = tuple(
        dict.fromkeys(("verification_history",) + tables + ("agent_discoverable_tools",) + secondary_tables)
    )
    return actions[:14], [], user_tools, combined_tables, f"{goal} and {secondary_goal}"


def _nodes(actions: list[dict[str, Any]], goal: str) -> tuple[ActionNode, ...]:
    nodes = []
    for index, action in enumerate(actions):
        node_id = f"node_{index + 1:02d}"
        nodes.append(
            ActionNode(
                node_id=node_id,
                subgoal=f"Step {index + 1}: use {action['name']} to {goal}.",
                action=action,
                depends_on=(nodes[-1].node_id,) if nodes else (),
            )
        )
    return tuple(nodes)


def _evidence_locator(fact: str) -> str:
    """Identify an L1 fact without copying its complete answer into the request."""

    words = fact.split()
    return " ".join(words[: min(4, max(2, len(words) - 1))])


def build_scenario(
    *,
    split: str,
    serial: int,
    category: str,
    form: str,
    catalog: DocumentCatalog,
    case_kind: str | None = None,
) -> ScenarioSpec:
    if form not in FORM_LEVEL:
        raise ValueError(f"unknown task form: {form}")
    if form in {"l1_evidence", "l2_retrieval"} and category not in LOW_FORM_CATEGORIES:
        raise ValueError(
            f"{form} is unavailable for {category}: no category-specific allowlisted document pool"
        )
    generated_case_kinds = FORM_CASE_KINDS[form]
    case_kind = case_kind or generated_case_kinds[serial % len(generated_case_kinds)]
    if case_kind not in CASE_KINDS:
        raise ValueError(f"unknown case kind: {case_kind}")
    if case_kind not in generated_case_kinds:
        raise ValueError(f"case kind {case_kind} is incompatible with {form}")
    entity = _entities(split, serial)
    category_topic = CATEGORY_TOPICS[category]
    min_docs, max_docs = FORM_DOCUMENT_BOUNDS[form]
    count = min_docs if min_docs == max_docs else min_docs + serial % (max_docs - min_docs + 1)
    support_documents = (
        ACTION_SUPPORT_DOCUMENTS[category]
        if form not in {"l1_evidence", "l2_retrieval", "l3_decision"}
        else ()
    )
    count = max(count, min(max_docs, len(support_documents)))
    document_ids = catalog.select(
        category,
        serial,
        count,
        preferred_ids=support_documents,
    )
    documents = [catalog.documents[document_id] for document_id in document_ids]
    facts = tuple(decisive_fact(document, category) for document in documents)
    titles = tuple(str(document["title"]) for document in documents)

    actions: list[dict[str, Any]] = []
    init_actions: list[dict[str, Any]] = []
    user_tools: tuple[str, ...] = ()
    changed_tables: tuple[str, ...] = ()
    goal = "answer the policy question"
    final_facts: tuple[str, ...]
    capabilities: tuple[str, ...]
    retrieval_variant = "golden_retrieval"

    if form == "l1_evidence":
        final_facts = communication_anchors(facts[0])
        capabilities = ("single_document_evidence", "fact_communication")
    elif form == "l2_retrieval":
        actions = [_search_action("retrieval_search", indexed_body_anchor(documents[0]))]
        final_facts = (document_ids[0],)
        capabilities = ("bm25_query", "document_identification")
        retrieval_variant = "bm25"
    elif form == "l3_decision":
        final_facts = ("Yes",)
        capabilities = ("policy_decision", "concise_communication")
    else:
        actions, init_actions, user_tools, changed_tables, goal = _flow(
            form=form,
            category=category,
            case_kind=case_kind,
            entity=entity,
            documents=documents,
        )
        final_facts = ()
        capabilities = tuple(
            dict.fromkeys(
                [
                    "policy_retrieval" if any(a["name"] == "KB_search" for a in actions) else "action_selection",
                    "identity_verification" if any(a["name"] == "log_verification" for a in actions) else "argument_construction",
                    "user_coordination" if any(a["requestor"] == "user" for a in actions) else "environment_update",
                ]
            )
        )
        retrieval_variant = "bm25" if any(action["name"] == "KB_search" for action in actions) else "golden_retrieval"

    action_min, action_max = FORM_ACTION_BOUNDS[form]
    if not action_min <= len(actions) <= action_max:
        raise ValueError(f"{form} generated {len(actions)} actions outside [{action_min}, {action_max}]")

    identity_known = (
        f"Customer name: {entity.name}",
        f"User ID: {entity.user_id}",
        f"Email: {entity.email}",
        f"Phone: {entity.phone}",
        f"Address: {entity.address}",
        f"Date of birth: {entity.date_of_birth}",
        f"Checking account: {entity.checking_id}",
        f"Savings account: {entity.savings_id}",
        f"Credit card account: {entity.credit_card_id}",
        f"Debit card: {entity.debit_card_id}",
    )
    action_clues = _action_clues(actions)
    preverified = (
        form in {"l3_single_action", "l4_two_skill"}
        and bool(changed_tables)
        and changed_tables != ("human_transfer_requests",)
    )
    case_preconditions = (
        (
            "Identity verification was completed and logged at 2025-11-14 03:40:00 EST before this chat; do not repeat it.",
        )
        if preverified
        else ()
    )
    intent_known = (
        (f"Requested new email: {entity.new_email}",)
        if any(action["name"] == "change_user_email" for action in actions)
        else ()
    )
    known = identity_known + intent_known + action_clues + case_preconditions
    to_collect = (
        "Ask for two identity fields before accessing private records."
        if any(action["name"] == "log_verification" for action in actions)
        else "No additional private information is required."
    ,)
    if form in {"l1_evidence", "l3_decision"}:
        title = documents[0]["title"]
        if form == "l1_evidence":
            locator = _evidence_locator(facts[0])
            request = (
                f"From the supplied material for {category_topic} [Policy title: {title}], "
                "quote verbatim the first complete policy fact that begins with "
                f"<<Evidence locator: {locator}>>. Include every value or condition in that fact. "
                "<<Response requirement: Quote the complete policy fact verbatim.>>"
            )
        else:
            request = (
                f"Decide whether the supplied material supports <<Policy claim: {facts[0]}>>. "
                f"The relevant source is [Policy title: {title}]. "
                "<<Response requirement: Begin the answer with exactly Yes or No.>>"
            )
    elif form == "l2_retrieval":
        title = documents[0]["title"]
        request = (
            f"Please search the bank knowledge base for [Policy title: {title}], then tell me "
            "the matching document ID. Do not make any account changes."
        )
    else:
        marked_clues = " ".join(
            f"<<Action clue: {clue}.>>" for clue in action_clues
        )
        marked_preconditions = " ".join(
            f"<<Case precondition: {condition}>>"
            for condition in case_preconditions
        )
        request = (
            f"I need help with {category_topic}. My goal is <<Requested outcome: {goal}>>. "
            "I know the following details, "
            f"but I will provide private fields only when asked: {'; '.join(identity_known + intent_known)}. "
            "Follow the applicable policy and do not perform unrelated operations."
        )
        if action_clues:
            request += (
                " The visible opening message already contains every non-private "
                f"case-note action detail; do not alter them: {marked_clues}"
            )
        if case_preconditions:
            request += " " + marked_preconditions

        opening = (
            f"I need help with {category_topic}. "
            f"My requested outcome is <<Requested outcome: {goal}>>."
        )
        if marked_clues:
            opening += (
                " Use these exact non-private case-note action details: "
                + marked_clues
            )
        if marked_preconditions:
            opening += " " + marked_preconditions

    if form in {"l1_evidence", "l2_retrieval", "l3_decision"}:
        opening = request

    expected_changes = tuple({"table": table, "scope": "synthetic scenario entities"} for table in changed_tables)
    scenario_id = f"banking_syn_v1_scenario_{split}_{serial:06d}"
    return ScenarioSpec(
        scenario_id=scenario_id,
        split=split,
        business_category=category,
        template_id=f"{category}:{form}:{case_kind}",
        template_version=TEMPLATE_VERSION,
        difficulty=FORM_LEVEL[form],
        task_form=form,
        case_kind=case_kind,
        atomic_capabilities=capabilities,
        initialization_data=initialization_data(
            entity, category, case_kind, preverified=preverified
        ),
        required_documents=document_ids,
        decisive_facts=facts,
        user_known_info=known,
        information_to_collect=to_collect,
        user_tools=user_tools,
        action_nodes=_nodes(actions, goal),
        expected_db_changes=expected_changes,
        final_response_facts=final_facts,
        canonical_user_request=request,
        initial_user_message=opening,
        retrieval_variant=retrieval_variant,
        initialization_actions=tuple(init_actions),
    )
