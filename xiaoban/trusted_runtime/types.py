"""Trusted Action Runtime 核心类型（渠道无关）。

模型可以理解和表达，但只有程序产生的 WorkTurn / ActionCall /
ActionResult 能决定某个工具动作是否真的发生。

机制来源（固定上游审计 commit，见交接单映射矩阵）：
- OpenAI Codex 322d5b96：typed tool call 进统一 registry、唯一 call ID、
  complete/failed 与同一调用绑定；
- Claude Agent SDK f8b9ec9：PreToolUse allow/deny、PostToolUse 与
  PostToolUseFailure 分离、tool_use_id 贯穿 pre/execute/post；
- Gemini CLI 3818efbb：确定性 policy 默认拒绝、状态只来自调度事实。

第一波为只读最小纵向闭环：不新增签名 Key，SHA-256 只用于
输入/输出完整性校验，不冒充"事实真实"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ActionResult.status 全状态；失败、空、拒绝、歧义、未找到必须明确区分。
ACTION_STATUSES = (
    "success",
    "empty",
    "error",
    "denied",
    "ambiguous",
    "not_found",
    "cancelled",
)

# PreActionDecision.decision（Claude PreToolUse allow/deny 等价语义；
# clarify/confirm 预留给后续波次，本波不产生）。
PRE_ACTION_DECISIONS = ("allow", "deny", "clarify", "confirm")

# 真实状态机：只有实际进入对应代码阶段才允许发状态事件。
TURN_STATES = (
    "accepted",
    "identity_resolved",
    "validating",
    "awaiting_clarification",
    "awaiting_confirmation",
    "executing",
    "verifying",
    "succeeded",
    "failed",
    "cancelled",
    "blocked",
)

@dataclass(frozen=True)
class TrustedIdentity:
    """服务端解析的身份；消息正文、自报账号不能成为权限依据。

    scope_values 是服务端可核实的 team/company 等 DataScope 维度值；
    为空时，payload 自报的任何 team/company 字段一律 fail closed。
    """

    account_id: str
    data_scope: str
    source: str  # server_session | platform_binding | none
    scope_values: Tuple[str, ...] = ()

    @property
    def datascope_fingerprint(self) -> str:
        import hashlib

        raw = (
            f"{self.account_id}|{self.data_scope}|{self.source}"
            f"|{','.join(sorted(self.scope_values))}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ActionCall:
    call_id: str
    action_id: str
    version: str
    arguments: Dict[str, Any]
    requested_at: str


@dataclass(frozen=True)
class ActionResult:
    call_id: str
    action_id: str
    status: str
    normalized_payload: Dict[str, Any]
    error_code: str
    started_at: str
    finished_at: str
    raw_text: str = ""


@dataclass(frozen=True)
class PreActionDecision:
    """PreAction 判定；decision != "allow" 时 handler 调用数必须为 0。"""

    decision: str
    reason: str
    call: Optional[ActionCall] = None


@dataclass(frozen=True)
class ActionOutputContract:
    """动作级输出合同（动作目录的服务端侧声明）。

    ``kind`` only selects the deterministic result validator. Resource discovery
    is a normal model-selected read and is not a prerequisite for stable IDs.
    """

    action_id: str
    version: str
    kind: str


# 只适配现有只读动作，不扩大动作目录。
ACTION_OUTPUT_CONTRACTS: Dict[str, ActionOutputContract] = {
    "mystand_resource_index": ActionOutputContract(
        "mystand_resource_index",
        "v1",
        "index",
    ),
    "mystand_authorization": ActionOutputContract(
        "mystand_authorization",
        "v1",
        "read",
    ),
    "mystand_query": ActionOutputContract(
        "mystand_query",
        "v1",
        "read",
    ),
}

# 写操作不属于第一阶段只读合同：由既有写确认 + 写回执硬闸接管，
# 可信只读链对其完全旁路，不登记、不采证、不拦截。
WRITE_TOOL_NAMES = frozenset({"mystand_authorization_write"})
WRITE_OPERATIONS = frozenset({"preview_write", "commit_write"})


def is_write_action(name: str, arguments: Optional[Dict[str, Any]] = None) -> bool:
    if name in WRITE_TOOL_NAMES:
        return True
    if name == "mystand_authorization":
        return str((arguments or {}).get("operation") or "") in WRITE_OPERATIONS
    return False


@dataclass
class WorkTurn:
    turn_id: str
    request_id: str
    message_id: str
    channel: str
    identity: Optional[TrustedIdentity]
    action_calls: List[ActionCall] = field(default_factory=list)
    action_results: List[ActionResult] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    state: str = "accepted"
    terminal_reason: str = ""
    orphaned_receipts: int = 0
    rejected_cross_account: int = 0
    pre_action_denials: int = 0

    def enter(self, state: str) -> None:
        if state not in TURN_STATES:
            raise ValueError(f"unknown turn state: {state}")
        self.state = state
        self.states.append(state)
