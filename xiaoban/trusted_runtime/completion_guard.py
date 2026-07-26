"""CompletionGuard：最终公开回答发送前的执行事实检查（Claude Stop 等价位置）。

边界只由服务器可信意图和真实执行生命周期构成：
- WORK 的公开业务回答由 ChannelProjection 从本轮 EvidenceEnvelope
  允许的字段路径生成，模型自然语言不能新增实体、关系或状态；
- 没有本轮 EvidenceEnvelope，WORK 只能输出固定安全失败/追问文案；
- CHAT 完全保留模型的自然表达，不扫描词语、数字、日期或历史内容；
- 是否属于 WORK 由服务端意图或真实 ActionCall 决定，不能从回答倒推；
- Guard 自身异常 fail closed。
"""

from __future__ import annotations

import json
from typing import Any, List, Mapping, Optional, Sequence

from xiaoban.trusted_runtime.turns import (
    build_work_turn,
    result_has_write_actions,
)
from xiaoban.trusted_runtime.types import (
    CompletionDecision,
    TrustedIdentity,
    WorkTurn,
    INTERACTION_CHAT,
)

# 用户可见安全文案：自然、简短，不含内部 ID、规则名或技术栈。
NO_EVIDENCE_MESSAGE = (
    "这轮我没有真正查到站内资料，所以不能给出具体的资料内容、数值或状态。"
)
EMPTY_RESULT_MESSAGE = "这轮没有找到对应的站内资料内容。"
DENIED_MESSAGE = "当前没有权限让小伴读取这份资料。"
NOT_FOUND_MESSAGE = "没有找到这份资料，或者这个站内 ID 已失效。"
AMBIGUOUS_MESSAGE = "这份资料目前无法唯一定位，请补充更完整的资料名称。"
ERROR_MESSAGE = "站内资料读取暂时没有接稳，请稍后再试。"

_STATUS_MESSAGES = {
    "empty": EMPTY_RESULT_MESSAGE,
    "denied": DENIED_MESSAGE,
    "not_found": NOT_FOUND_MESSAGE,
    "ambiguous": AMBIGUOUS_MESSAGE,
    "error": ERROR_MESSAGE,
    "cancelled": ERROR_MESSAGE,
}

def project_answer(turn: WorkTurn) -> str:
    """ChannelProjection：公开业务内容只来自 Evidence 允许字段路径。

    索引只负责资源发现（记录在 IndexReceipt），公开业务事实只从
    业务读取动作的 content 字段投影。
    """
    parts: List[str] = []
    for item in turn.evidence:
        try:
            facts = json.loads(item.allowed_facts or "{}")
        except (TypeError, ValueError):
            continue
        content = str(facts.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def _failure_message(turn: WorkTurn) -> str:
    for item in reversed(turn.action_results):
        if item.status == "denied" and item.error_code == "missing_index_receipt":
            return NO_EVIDENCE_MESSAGE
        if item.status in _STATUS_MESSAGES:
            return _STATUS_MESSAGES[item.status]
    return NO_EVIDENCE_MESSAGE


def check_completion(final_text: str, turn: WorkTurn) -> CompletionDecision:
    """对最终公开回答做确定性检查；阻断时给出安全文案与结构化原因。"""
    try:
        text = str(final_text or "")
        if turn.interaction_kind == INTERACTION_CHAT:
            return CompletionDecision(True, text, "allowed_chat")

        if turn.evidence:
            # WORK + 本轮可信证据：公开业务内容由结构化投影生成，
            # 模型文本里的新增实体/关系/状态不会进入公开回答。
            projected = project_answer(turn)
            if projected:
                return CompletionDecision(True, projected, "projected_evidence")
            return CompletionDecision(False, NO_EVIDENCE_MESSAGE, "blocked_no_evidence")

        if not turn.action_calls:
            reason = "blocked_no_action_call"
        elif not turn.action_results:
            reason = "blocked_no_action_result"
        else:
            reason = "blocked_no_evidence"
        return CompletionDecision(False, _failure_message(turn), reason)
    except Exception:
        # Guard 自身异常 fail closed。
        return CompletionDecision(False, NO_EVIDENCE_MESSAGE, "blocked_guard_error")


def _trusted_turn_binding_valid(
    turn: WorkTurn,
    *,
    channel: str,
    account_id: str,
    request_id: str,
    message_id: str,
) -> bool:
    """_trusted_turn 必须与本次服务端身份、渠道、messageId、DataScope 再绑定。"""
    identity = turn.identity
    if identity is None or not account_id or identity.account_id != account_id:
        return False
    if identity.data_scope != "mystand":
        return False
    if turn.channel != channel:
        return False
    if not request_id or turn.request_id != request_id:
        return False
    if not message_id or turn.message_id != message_id:
        return False
    return True


def check_mystand_final_answer(
    final_text: str,
    *,
    user_message: Any,
    conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
    result: Any = None,
    channel: str = "web",
    account_id: str = "",
    request_id: str = "",
    message_id: str = "",
) -> CompletionDecision:
    """构建/取用 WorkTurn 并执行 CompletionGuard（模型无关确定性路径）。

    身份只信调用方显式传入的服务端解析结果（Web 登录会话或渠道绑定），
    绝不回退读取 result 自报字段；附着的 _trusted_turn 必须与本次
    服务端身份/messageId/渠道/DataScope 再绑定，不一致立即拒绝。
    """
    if not isinstance(result, Mapping) or result.get("_mystand_request") is not True:
        return CompletionDecision(True, str(final_text or ""), "not_mystand")
    if result_has_write_actions(result):
        # 写流程由既有写确认 + 写回执硬闸（上游已先行执行）接管，
        # 此处不得再叠加一套自然语言判断。
        return CompletionDecision(True, str(final_text or ""), "write_turn_deferred")
    identity = (
        TrustedIdentity(
            account_id=account_id,
            data_scope="mystand",
            source="server_session",
        )
        if account_id
        else None
    )
    turn = result.get("_trusted_turn")
    if isinstance(turn, WorkTurn):
        if not _trusted_turn_binding_valid(
            turn,
            channel=channel,
            account_id=account_id,
            request_id=request_id,
            message_id=message_id,
        ):
            return CompletionDecision(
                False, NO_EVIDENCE_MESSAGE, "blocked_identity_rebind"
            )
    else:
        # 没有生命周期回合时，执行记录仍要逐项过同一套生命周期门禁，
        # 伪造/越权/无索引的动作在这一步被拒绝，不能洗白成证据。
        turn = build_work_turn(
            channel=channel,
            user_message=user_message,
            conversation_history=conversation_history,
            result=result,
            identity=identity,
            request_id=request_id,
            message_id=message_id,
        )
    decision = check_completion(final_text, turn)
    turn.terminal_reason = decision.reason
    turn.enter("succeeded" if decision.allowed else "blocked")
    return decision
