"""渠道无关的可信 Action Runtime（只读最小纵向闭环）。"""

from xiaoban.trusted_runtime.channels import (
    PLATFORM_CLI,
    PLATFORM_FEISHU,
    PLATFORM_WEB,
    PLATFORM_WECHAT,
    envelope_from_feishu_event,
    envelope_from_wechat_event,
    evaluate_channel_answer,
    identity_from_envelope,
)
from xiaoban.trusted_runtime.completion_guard import (
    check_completion,
    check_mystand_final_answer,
)
from xiaoban.trusted_runtime.turns import build_work_turn, classify_interaction
from xiaoban.trusted_runtime.types import (
    ActionCall,
    ActionResult,
    CommandEnvelope,
    CompletionDecision,
    EvidenceEnvelope,
    IndexReceipt,
    TrustedIdentity,
    WorkTurn,
)

__all__ = [
    "ActionCall",
    "ActionResult",
    "CommandEnvelope",
    "CompletionDecision",
    "EvidenceEnvelope",
    "IndexReceipt",
    "PLATFORM_CLI",
    "PLATFORM_FEISHU",
    "PLATFORM_WEB",
    "PLATFORM_WECHAT",
    "TrustedIdentity",
    "WorkTurn",
    "build_work_turn",
    "check_completion",
    "check_mystand_final_answer",
    "classify_interaction",
    "envelope_from_feishu_event",
    "envelope_from_wechat_event",
    "evaluate_channel_answer",
    "identity_from_envelope",
]
