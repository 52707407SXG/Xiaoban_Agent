"""High-risk action classification for Xiaoban-Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RiskLevel = Literal["low", "medium", "high", "forbidden"]
ApprovalRequirement = Literal["none", "confirm", "owner_confirm", "deny"]


HIGH_RISK_KEYWORDS = {
    "delete": "删除数据",
    "remove": "删除数据",
    "export": "导出数据",
    "permission": "权限变更",
    "role": "权限变更",
    "mass_message": "群发消息",
    "broadcast": "群发消息",
    "deploy": "部署/重启",
    "restart": "部署/重启",
    "payment": "支付/退款/转账",
    "refund": "支付/退款/转账",
    "transfer": "支付/退款/转账",
}

FORBIDDEN_KEYWORDS = {
    "dump_source": "源码整段外发",
    "raw_source": "源码整段外发",
    "leak_secret": "密钥外泄",
}


@dataclass(frozen=True)
class ApprovalDecision:
    risk: RiskLevel
    requirement: ApprovalRequirement
    reason: str


def classify_action(action_name: str, *, side_effect: str = "read") -> ApprovalDecision:
    normalized = action_name.lower().replace("-", "_").replace(".", "_")
    for keyword, reason in FORBIDDEN_KEYWORDS.items():
        if keyword in normalized:
            return ApprovalDecision("forbidden", "deny", reason)
    for keyword, reason in HIGH_RISK_KEYWORDS.items():
        if keyword in normalized:
            return ApprovalDecision("high", "owner_confirm", reason)
    if side_effect in {"write", "external"}:
        return ApprovalDecision("medium", "confirm", "写入或外部副作用")
    return ApprovalDecision("low", "none", "只读动作")

