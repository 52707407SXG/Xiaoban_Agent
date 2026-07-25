"""从真实执行记录构建 WorkTurn。

只消费程序产生的 transcript（assistant tool_calls 与 tool 回执）：
- 每个 ActionResult 严格绑定本回合匹配的 callId；
- 旧回合、伪造 callId、跨账号回执一律排除出本轮证据；
- `verifying` 状态只在 PostAction Verify 真实运行时产生。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from xiaoban.trusted_runtime.types import (
    ActionCall,
    ActionResult,
    EvidenceEnvelope,
    IndexReceipt,
    TrustedIdentity,
    WorkTurn,
    INTERACTION_CHAT,
    INTERACTION_WORK,
)

_BUSINESS_INTENT_RE = re.compile(
    r"(?:业主|房源|楼盘|客户|档案|账本|账目|欠费|提成|结算|佣金|业绩|笔记|"
    r"授权|资料|租户|租客|房东|售价|租金|月供|财务|流水|合同|钥匙|跟进|"
    r"AUTH-|OUT-|栋|单元|号楼)",
    re.IGNORECASE,
)

_BUSINESS_TOOL_NAMES = {
    "mystand_parse",
    "mystand_resource_index",
    "mystand_query",
    "mystand_authorization",
    "mystand_authorization_write",
}

_INDEX_TOOL_NAMES = {"mystand_resource_index"}

_ACCOUNT_KEYS = ("accountId", "account_id", "userId", "user_id", "ownerId", "owner_id")

_RECORD_REF_KEYS = ("resourceUid", "resourceId", "authorizationId", "sourceId")


def _visible_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, Mapping) and str(part.get("type") or "") in {
                "text",
                "input_text",
                "output_text",
            }:
                parts.append(str(part.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _current_turn_messages(result: Any) -> List[Mapping[str, Any]]:
    """只取最后一个 user 消息之后的回合；没有边界时视为无证据。"""
    if not isinstance(result, Mapping):
        return []
    raw = result.get("messages")
    if not isinstance(raw, list):
        return []
    messages = [m for m in raw if isinstance(m, Mapping)]
    last_user = -1
    for index, message in enumerate(messages):
        if str(message.get("role") or "") == "user":
            last_user = index
    if last_user < 0:
        return []
    return messages[last_user + 1 :]


def _classify_result_status(payload: Dict[str, Any], raw_text: str) -> str:
    ok = payload.get("ok")
    success = payload.get("success")
    if ok is False or success is False:
        try:
            code = int(payload.get("status") or 0)
        except (TypeError, ValueError):
            code = 0
        if code == 403:
            return "denied"
        if code == 404:
            return "not_found"
        if code == 409:
            return "ambiguous"
        return "error"
    if ok is True or success is True:
        items = payload.get("items")
        if isinstance(items, list) and not items:
            return "empty"
        if "content" in payload and not str(payload.get("content") or "").strip():
            return "empty"
        return "success"
    if '"success": true' in raw_text.lower():
        return "success"
    return "error" if len(raw_text.strip()) < 20 else "success"


def _declared_accounts(payload: Dict[str, Any]) -> List[str]:
    return [
        str(payload[key])
        for key in _ACCOUNT_KEYS
        if payload.get(key) not in (None, "")
    ]


def _record_refs(payload: Dict[str, Any]) -> List[str]:
    refs = []
    for key in _RECORD_REF_KEYS:
        value = payload.get(key)
        if value:
            refs.append(str(value))
    items = payload.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, Mapping) and item.get("resourceUid"):
                refs.append(str(item["resourceUid"]))
    return refs


def classify_interaction(
    user_message: Any,
    conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    used_business_tools: bool = False,
) -> str:
    """CHAT/WORK 分类；无法可靠区分时默认 WORK。

    分类只决定是否进入工作链，不是唯一反撒谎边界。
    """
    if used_business_tools:
        return INTERACTION_WORK
    texts = [_visible_text(user_message)]
    for message in list(conversation_history or [])[-4:]:
        if isinstance(message, Mapping) and str(message.get("role") or "") == "user":
            texts.append(_visible_text(message.get("content")))
    if any(_BUSINESS_INTENT_RE.search(text) for text in texts if text):
        return INTERACTION_WORK
    return INTERACTION_CHAT


def build_work_turn(
    *,
    channel: str,
    user_message: Any,
    conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
    result: Any = None,
    identity: Optional[TrustedIdentity] = None,
    request_id: str = "",
    message_id: str = "",
) -> WorkTurn:
    """从真实 transcript 重建本轮可信回合。"""
    turn_id = hashlib.sha256(
        f"{channel}|{request_id}|{message_id}|{_visible_text(user_message)[:200]}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    turn = WorkTurn(
        turn_id=turn_id,
        request_id=request_id,
        message_id=message_id,
        channel=channel,
        identity=identity,
        interaction_kind=INTERACTION_CHAT,
        index_receipt=None,
    )
    turn.enter("accepted")
    if identity is not None and identity.account_id:
        turn.enter("identity_resolved")

    messages = _current_turn_messages(result)

    calls: Dict[str, ActionCall] = {}
    order = 0
    for message in messages:
        if str(message.get("role") or "") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            function = function if isinstance(function, Mapping) else {}
            call_id = str(call.get("id") or "")
            name = str(function.get("name") or call.get("name") or "")
            if not call_id or not name:
                continue
            order += 1
            calls[call_id] = ActionCall(
                call_id=call_id,
                action_id=name,
                version="v1",
                arguments=_parse_json_object(
                    function.get("arguments")
                    if "arguments" in function
                    else call.get("arguments")
                ),
                requested_at=f"seq:{order}",
            )
    turn.action_calls = list(calls.values())

    used_business_tools = any(
        call.action_id in _BUSINESS_TOOL_NAMES for call in turn.action_calls
    )
    turn.interaction_kind = classify_interaction(
        user_message,
        conversation_history,
        used_business_tools=used_business_tools,
    )

    results: List[ActionResult] = []
    for message in messages:
        if str(message.get("role") or "") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        bound = calls.get(call_id)
        if bound is None:
            # 伪造或跨回合 callId：不得成为本轮证据。
            turn.orphaned_receipts += 1
            continue
        raw_text = message.get("content")
        if not isinstance(raw_text, str):
            raw_text = json.dumps(raw_text, ensure_ascii=False, default=str)
        payload = _parse_json_object(raw_text)
        results.append(
            ActionResult(
                call_id=call_id,
                action_id=bound.action_id,
                status=_classify_result_status(payload, raw_text),
                normalized_payload=payload,
                error_code=str(payload.get("code") or payload.get("error") or ""),
                started_at=bound.requested_at,
                finished_at=f"seq:{order + 1 + len(results)}",
                raw_text=raw_text,
            )
        )
    turn.action_results = results
    if turn.action_calls:
        turn.enter("executing")

    # PostAction Verify：只在真实 ActionResult 返回后执行。
    if results:
        turn.enter("verifying")
        account_id = identity.account_id if identity else ""
        for item in results:
            if item.status != "success":
                continue
            declared = _declared_accounts(item.normalized_payload)
            if declared and (
                not account_id or any(value != account_id for value in declared)
            ):
                # 跨账号 evidence 不得进入本轮。
                turn.rejected_cross_account += 1
                continue
            turn.evidence.append(
                EvidenceEnvelope(
                    evidence_id=hashlib.sha256(
                        f"{turn_id}|{item.call_id}".encode("utf-8")
                    ).hexdigest()[:16],
                    turn_id=turn_id,
                    call_id=item.call_id,
                    action_id=item.action_id,
                    datascope_fingerprint=(
                        identity.datascope_fingerprint if identity else ""
                    ),
                    status=item.status,
                    allowed_facts=item.raw_text,
                    record_refs=_record_refs(item.normalized_payload),
                    input_digest=hashlib.sha256(
                        json.dumps(
                            calls[item.call_id].arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest(),
                    output_digest=hashlib.sha256(
                        item.raw_text.encode("utf-8")
                    ).hexdigest(),
                    verified_at=item.finished_at,
                    verification_status="verified",
                )
            )

    # PreAction 最小索引回执由程序生成，模型不得伪造或补写。
    turn.enter("indexing")
    turn.index_receipt = _build_index_receipt(turn, request_id=request_id)
    return turn


def _build_index_receipt(turn: WorkTurn, *, request_id: str) -> IndexReceipt:
    fingerprint = turn.identity.datascope_fingerprint if turn.identity else ""
    index_results = [
        item for item in turn.action_results if item.action_id in _INDEX_TOOL_NAMES
    ]
    if not index_results:
        status = (
            "no_internal_resource_needed"
            if turn.interaction_kind == INTERACTION_CHAT or turn.evidence
            else "none"
        )
        return IndexReceipt(
            request_id=request_id,
            actor_fingerprint=fingerprint,
            loaded_at="",
            scope_summary="",
            matched_resource_refs=[],
            status=status,
        )
    latest = index_results[-1]
    if latest.status == "success":
        status = "found"
    elif latest.status == "empty":
        status = "none"
    elif latest.status == "denied":
        status = "denied"
    else:
        status = "unavailable"
    return IndexReceipt(
        request_id=request_id,
        actor_fingerprint=fingerprint,
        loaded_at=latest.finished_at,
        scope_summary=latest.action_id,
        matched_resource_refs=_record_refs(latest.normalized_payload),
        status=status,
    )
