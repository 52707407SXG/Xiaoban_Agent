"""CompletionGuard：最终公开回答发送前的确定性检查（Claude Stop 等价位置）。

边界由程序构成，不靠提示词自律，也不靠扩充关键词/数字正则：
- WORK 的公开业务回答由 ChannelProjection 从本轮 EvidenceEnvelope
  允许的字段路径生成，模型自然语言不能新增实体、关系或状态；
- 没有本轮 EvidenceEnvelope，WORK 只能输出固定安全失败/追问文案；
- CHAT 可以自然回复，但不得伪装成 My Stand 查询结果；
- 没有真实 PostAction Verify，不得说"已核验"；Guard 自身异常 fail closed。
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Mapping, Optional, Sequence

from xiaoban.trusted_runtime.turns import build_work_turn, result_has_write_actions
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
VERIFICATION_BLOCK_MESSAGE = (
    "这轮没有完成新的真实核验，我不能说已经核验或确认。"
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

_HONESTY_RE = re.compile(
    r"(?:没查到|没有查到|没找到|没有找到|查不到|看不到|无法读取|读不到|"
    r"没有权限|无权|未授权|没有授权|没有开放|无法确认|不确定|没能|不清楚|"
    r"无法唯一定位|没有接稳|稍后再试|没有真正查到|无法判断)"
)

_VERIFICATION_CLAIM_RE = re.compile(
    r"(?:核验|核实|验证通过|已验证|复查|复核)"
)

_NEGATION_RE = re.compile(
    r"(?:没有|没能|未能|未|没|不能|无法|尚未|还没|并未|并没有|不代表|不是)"
)

# CHAT 伪装成业务结论的跳闸信号（不是事实绑定手段，只是 CHAT 边界）。
_CLAIM_VERB_RE = re.compile(
    r"(?:查到了|查到|查询到|找到了|结果显示|业主是|租户是|租客是|房东是|"
    r"状态是|记录在|登记在|名下有)"
)

_REFERENCE_ID_RE = re.compile(
    r"(?<![A-Z0-9])(?:AUTH|OUT)-[A-Z0-9][A-Z0-9-]{5,}[A-Z0-9](?![A-Z0-9])",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_DATE_RE = re.compile(r"20\d{2}\s*[年\-/.]\s*\d{1,2}(?:\s*[月\-/.]\s*\d{1,2}\s*日?)?")
_NUMBER_RE = re.compile(r"\d[\d,]{1,}(?:\.\d+)?")


def _has_positive_claim(pattern: re.Pattern, text: str) -> bool:
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 12) : match.start()]
        if not _NEGATION_RE.search(prefix):
            return True
    return False


def _answer_fact_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    tokens.extend(match.group(0) for match in _REFERENCE_ID_RE.finditer(text))
    tokens.extend(match.group(0) for match in _PHONE_RE.finditer(text))
    tokens.extend(match.group(0) for match in _DATE_RE.finditer(text))
    tokens.extend(match.group(0) for match in _NUMBER_RE.finditer(text))
    return tokens


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
        has_claim_verb = _has_positive_claim(_CLAIM_VERB_RE, text)
        has_verification_claim = _has_positive_claim(_VERIFICATION_CLAIM_RE, text)
        fact_tokens = _answer_fact_tokens(text)
        honest_admission = (
            bool(_HONESTY_RE.search(text))
            and not fact_tokens
            and not _CLAIM_VERB_RE.search(text)
            and not has_verification_claim
        )

        if turn.interaction_kind == INTERACTION_CHAT and not (
            has_claim_verb or has_verification_claim or fact_tokens
        ):
            return CompletionDecision(True, text, "allowed_chat")

        if honest_admission:
            return CompletionDecision(True, text, "allowed_honest_admission")

        if turn.evidence:
            # WORK + 本轮可信证据：公开业务内容由结构化投影生成，
            # 模型文本里的新增实体/关系/状态不会进入公开回答。
            projected = project_answer(turn)
            if projected:
                return CompletionDecision(True, projected, "projected_evidence")
            return CompletionDecision(False, NO_EVIDENCE_MESSAGE, "blocked_no_evidence")

        if has_verification_claim:
            return CompletionDecision(False, VERIFICATION_BLOCK_MESSAGE, "blocked_verification_claim")
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
        # 但失败写回合不能夹带无关的读取事实。
        if _has_positive_claim(_CLAIM_VERB_RE, str(final_text or "")):
            return CompletionDecision(
                False, NO_EVIDENCE_MESSAGE, "blocked_write_fact_leak"
            )
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
