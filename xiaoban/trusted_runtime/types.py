"""Trusted Action Runtime 核心类型（渠道无关）。

模型可以理解和表达，但只有程序产生的 WorkTurn / ActionCall /
ActionResult / EvidenceEnvelope 能决定某个动作是否真的发生、
结果是否可以引用、业务事实是否可以陈述。

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

# IndexReceipt.status
INDEX_STATUSES = (
    "found",
    "none",
    "denied",
    "unavailable",
    "no_internal_resource_needed",
)

# PreActionDecision.decision（Claude PreToolUse allow/deny 等价语义；
# clarify/confirm 预留给后续波次，本波不产生）。
PRE_ACTION_DECISIONS = ("allow", "deny", "clarify", "confirm")

# 真实状态机：只有实际进入对应代码阶段才允许发状态事件。
TURN_STATES = (
    "accepted",
    "identity_resolved",
    "indexing",
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

INTERACTION_CHAT = "CHAT"
INTERACTION_WORK = "WORK"


@dataclass(frozen=True)
class CommandEnvelope:
    """渠道适配后的统一输入；渠道只负责收消息和身份映射。"""

    request_id: str
    platform: str
    conversation_id: str
    message_id: str
    external_user_ref: str
    text: str
    received_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrustedIdentity:
    """服务端解析的身份；消息正文、自报账号不能成为权限依据。"""

    account_id: str
    data_scope: str
    source: str  # server_session | platform_binding | none

    @property
    def datascope_fingerprint(self) -> str:
        import hashlib

        raw = f"{self.account_id}|{self.data_scope}|{self.source}"
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

    - kind=index：最小资源索引读取，建立 IndexReceipt；
    - kind=scoped_read：按稳定 ID 的最小定向读取，同时建立 IndexReceipt；
    - kind=read：开放事实查询，执行前必须已有 found 的 IndexReceipt。
    allowed_fact_paths 是 Evidence 唯一允许公开的字段路径。
    """

    action_id: str
    version: str
    kind: str
    allowed_fact_paths: Tuple[str, ...]
    record_ref_paths: Tuple[str, ...]


# 第一波只适配现有只读动作，不扩大动作目录。
ACTION_OUTPUT_CONTRACTS: Dict[str, ActionOutputContract] = {
    "mystand_resource_index": ActionOutputContract(
        "mystand_resource_index",
        "v1",
        "index",
        ("items[].safeLabel",),
        ("items[].resourceUid",),
    ),
    "mystand_authorization": ActionOutputContract(
        "mystand_authorization",
        "v1",
        "scoped_read",
        ("content",),
        ("resourceUid", "authorizationId"),
    ),
    "mystand_query": ActionOutputContract(
        "mystand_query",
        "v1",
        "read",
        ("content",),
        ("resourceUid",),
    ),
}


@dataclass(frozen=True)
class IndexReceipt:
    request_id: str
    actor_fingerprint: str
    loaded_at: str
    scope_summary: str
    matched_resource_refs: List[str]
    status: str


@dataclass(frozen=True)
class EvidenceEnvelope:
    """本轮业务事实的唯一可信来源。

    allowed_facts 是动作合同允许字段路径的 JSON 投影（{path: value}），
    不是 raw text；recordRefs 来自受控执行上下文与合同声明路径。
    """

    evidence_id: str
    turn_id: str
    call_id: str
    action_id: str
    datascope_fingerprint: str
    status: str
    allowed_facts: str
    record_refs: List[str]
    input_digest: str
    output_digest: str
    verified_at: str
    verification_status: str


@dataclass(frozen=True)
class CompletionDecision:
    allowed: bool
    text: str
    reason: str


@dataclass
class WorkTurn:
    turn_id: str
    request_id: str
    message_id: str
    channel: str
    identity: Optional[TrustedIdentity]
    interaction_kind: str
    index_receipt: Optional[IndexReceipt]
    action_calls: List[ActionCall] = field(default_factory=list)
    action_results: List[ActionResult] = field(default_factory=list)
    evidence: List[EvidenceEnvelope] = field(default_factory=list)
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
