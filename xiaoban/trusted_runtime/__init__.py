"""渠道无关的可信 Action Runtime（只读最小纵向闭环）。"""

from xiaoban.trusted_runtime.turns import (
    activate_turn,
    begin_action,
    begin_turn,
    current_turn,
    deactivate_turn,
    finish_action,
    gate_registry_action,
)
from xiaoban.trusted_runtime.types import (
    ACTION_OUTPUT_CONTRACTS,
    ActionCall,
    ActionResult,
    PreActionDecision,
    TrustedIdentity,
    WorkTurn,
)

__all__ = [
    "ACTION_OUTPUT_CONTRACTS",
    "ActionCall",
    "ActionResult",
    "PreActionDecision",
    "TrustedIdentity",
    "WorkTurn",
    "activate_turn",
    "begin_action",
    "begin_turn",
    "current_turn",
    "deactivate_turn",
    "finish_action",
    "gate_registry_action",
]
