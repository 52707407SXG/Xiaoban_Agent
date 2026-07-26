"""Trusted Action Runtime 真实生命周期：PreAction → Execute → PostAction。

机制映射（固定上游 commit，详见交接单）：
- Codex 322d5b96 tools/orchestrator.rs：调度前保留唯一 call ID，
  complete/failed 与同一调用绑定，无结果的调用不得视为成功；
- Claude SDK f8b9ec9 types.py：PreToolUse 在 handler 前 allow/deny，
  tool_use_id 贯穿 pre/execute/post，失败是第一等状态；
- Gemini CLI 3818efbb policy-engine.ts：确定性 allow/deny，无匹配规则、
  安全字段缺失、策略异常一律默认拒绝。

本模块不从 transcript 倒推可信结论；``build_work_turn`` 只是把既有
执行记录逐项送入同一生命周期门禁的兼容驱动（测试/夹具用），准入门禁
与生产路径完全一致。
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from xiaoban.trusted_runtime.types import (
    ACTION_OUTPUT_CONTRACTS,
    ActionCall,
    ActionResult,
    ActionOutputContract,
    EvidenceEnvelope,
    IndexReceipt,
    PreActionDecision,
    TrustedIdentity,
    WorkTurn,
    INTERACTION_CHAT,
    INTERACTION_WORK,
    is_write_action,
)

_BUSINESS_INTENT_RE = re.compile(
    r"(?:业主|房源|楼盘|客户|档案|账本|账目|欠费|提成|结算|佣金|业绩|笔记|"
    r"授权|资料|租户|租客|房东|售价|租金|月供|财务|流水|合同|钥匙|跟进|"
    r"AUTH-|OUT-|栋|单元|号楼)",
    re.IGNORECASE,
)

_ACCOUNT_KEYS = frozenset(
    {
        "accountid",
        "account_id",
        "userid",
        "user_id",
        "ownerid",
        "owner_id",
        "owneruser",
        "owner_user",
    }
)

# team/company 等 DataScope 维度：只信 identity.scope_values（服务端可核实值），
# 无法核实的一律 fail closed。
_TEAM_COMPANY_KEYS = frozenset(
    {
        "teamid",
        "team_id",
        "companyid",
        "company_id",
        "orgid",
        "org_id",
        "tenantid",
        "tenant_id",
    }
)

_MODULE_KEYS = frozenset({"moduleid", "module_id"})

# 活动可信回合：ContextVar 与 gateway.session_context 同机制，
# 每个请求执行线程各自隔离，并发同账号请求不会互相污染。
_CURRENT_TURN: "contextvars.ContextVar[Optional[WorkTurn]]" = contextvars.ContextVar(
    "xiaoban_trusted_runtime_turn", default=None
)


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
    # 连续业务追问的语境可能在 assistant 回复里（"您是想查…业主吗？"→"他是谁？"），
    # 历史扫描不限角色，漏判成 CHAT 会放行纯人名事实。
    for message in list(conversation_history or [])[-4:]:
        if isinstance(message, Mapping):
            texts.append(_visible_text(message.get("content")))
    if any(_BUSINESS_INTENT_RE.search(text) for text in texts if text):
        return INTERACTION_WORK
    return INTERACTION_CHAT


def activate_turn(turn: WorkTurn) -> "contextvars.Token":
    """在当前执行上下文中激活可信回合（registry 扼点据此找回回合）。"""
    return _CURRENT_TURN.set(turn)


def deactivate_turn(token: "contextvars.Token") -> None:
    _CURRENT_TURN.reset(token)


def current_turn() -> Optional[WorkTurn]:
    return _CURRENT_TURN.get()


def begin_turn(
    *,
    channel: str,
    user_message: Any,
    conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
    identity: Optional[TrustedIdentity] = None,
    request_id: str = "",
    message_id: str = "",
) -> WorkTurn:
    """服务端开回合：稳定 request/message ID + 服务端解析身份。"""
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
        interaction_kind=classify_interaction(user_message, conversation_history),
        index_receipt=None,
    )
    turn.enter("accepted")
    if identity is not None and identity.account_id:
        turn.enter("identity_resolved")
    return turn


def _seq(turn: WorkTurn) -> str:
    return f"seq:{len(turn.action_calls) + len(turn.action_results) + 1}"


def _record_denial(
    turn: WorkTurn, call_id: str, action_id: str, reason: str, *, status: str = "denied"
) -> PreActionDecision:
    turn.pre_action_denials += 1
    stamp = _seq(turn)
    turn.action_results.append(
        ActionResult(
            call_id=call_id,
            action_id=action_id,
            status=status,
            normalized_payload={},
            error_code=reason,
            started_at=stamp,
            finished_at=stamp,
        )
    )
    return PreActionDecision("deny", reason)


def begin_action(
    turn: WorkTurn,
    action_id: str,
    version: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    call_id: str = "",
    catalog_lookup: Optional[Callable[[str], bool]] = None,
) -> PreActionDecision:
    """PreAction：真实 handler 执行前的确定性门禁（默认拒绝）。

    allow 时一次性生成/登记 callId 并贯穿后续 finish_action；
    任何安全关键异常都拒绝，调用方必须保证 deny 时 handler 调用数为 0。
    """
    call_id = call_id or f"mystand_pre_{uuid.uuid4().hex}"
    args = dict(arguments or {})
    try:
        if is_write_action(action_id, args):
            # 写动作不属于只读合同，绝不经由只读链执行或采证。
            return _record_denial(turn, call_id, action_id, "write_isolated")
        contract = ACTION_OUTPUT_CONTRACTS.get(action_id)
        if contract is None or contract.version != version:
            # 未知动作或缺少动作级 output 合同：不执行。
            return _record_denial(turn, call_id, action_id, "unknown_action", status="error")
        if catalog_lookup is not None and not catalog_lookup(action_id):
            return _record_denial(turn, call_id, action_id, "not_in_catalog")
        if any(call.call_id == call_id for call in turn.action_calls):
            # 重复 callId 不得静默覆盖：同名调用已产生的证据一并作废。
            turn.evidence = [e for e in turn.evidence if e.call_id != call_id]
            turn.action_results = [
                ActionResult(
                    call_id=r.call_id,
                    action_id=r.action_id,
                    status="error",
                    normalized_payload={},
                    error_code="duplicate_call_id",
                    started_at=r.started_at,
                    finished_at=r.finished_at,
                )
                if r.call_id == call_id
                else r
                for r in turn.action_results
            ]
            return _record_denial(turn, call_id, action_id, "duplicate_call_id")
        # 业务动作出现即进入 WORK 链，CHAT 误分不能成为绕过口。
        turn.interaction_kind = INTERACTION_WORK
        identity = turn.identity
        if identity is None or not identity.account_id:
            return _record_denial(turn, call_id, action_id, "missing_identity")
        if not identity.datascope_fingerprint:
            return _record_denial(turn, call_id, action_id, "missing_datascope")
        if contract.kind == "read" and not (
            turn.index_receipt is not None and turn.index_receipt.status == "found"
        ):
            # My Stand WORK：开放查询前必须有本轮服务端 IndexReceipt。
            return _record_denial(turn, call_id, action_id, "missing_index_receipt")
        turn.enter("validating")
        call = ActionCall(
            call_id=call_id,
            action_id=action_id,
            version=version,
            arguments=args,
            requested_at=_seq(turn),
        )
        turn.action_calls.append(call)
        turn.enter("executing")
        return PreActionDecision("allow", "allowed", call)
    except Exception:
        return _record_denial(turn, call_id, action_id, "preaction_error")


def _scope_violations(
    node: Any,
    identity: Optional[TrustedIdentity],
    call_args: Dict[str, Any],
) -> List[str]:
    """递归复核 payload 自报的 owner/team/company/module 字段。

    owner 级字段必须等于当前服务端身份；team/company 级字段必须命中
    identity.scope_values（无服务端可核实值时一律拒绝）；module 必须
    与执行上下文调用参数一致。嵌套越权不得漏检。
    """
    violations: List[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            lowered = str(key).lower()
            if lowered in _ACCOUNT_KEYS and value not in (None, ""):
                if identity is None or str(value) != identity.account_id:
                    violations.append(str(key))
            elif lowered in _TEAM_COMPANY_KEYS and value not in (None, ""):
                allowed = set(identity.scope_values) if identity else set()
                if str(value) not in allowed:
                    violations.append(str(key))
            elif lowered in _MODULE_KEYS and value not in (None, ""):
                expected = call_args.get("module_id") or call_args.get("moduleId")
                if expected and str(value) != str(expected):
                    violations.append(str(key))
            else:
                violations.extend(_scope_violations(value, identity, call_args))
    elif isinstance(node, list):
        for item in node:
            violations.extend(_scope_violations(item, identity, call_args))
    return violations


def _classify_contract_status(
    contract: ActionOutputContract, payload: Dict[str, Any]
) -> str:
    """按动作级 output 合同分类；矛盾/未知回执一律失败关闭。"""
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
        # 矛盾回执：ok=true 同时带 error / 失败 code / 4xx-5xx（含字符串
        # 形式）/ 无法解析的 status，一律不得生成 success。
        if payload.get("error"):
            return "error"
        code_value = payload.get("code")
        if code_value not in (None, "", 0, "0", "OK", "ok"):
            return "error"
        status = payload.get("status")
        if status is not None:
            try:
                status_code = int(status)
            except (TypeError, ValueError):
                return "error"
            if status_code >= 400:
                return "error"
        if contract.kind == "index":
            items = payload.get("items")
            if not isinstance(items, list):
                return "error"
            if not items:
                return "empty"
            if not all(isinstance(item, Mapping) for item in items):
                return "error"
            return "success"
        if "content" not in payload:
            return "error"
        if not str(payload.get("content") or "").strip():
            return "empty"
        return "success"
    return "error"


def _project_allowed_facts(
    contract: ActionOutputContract, payload: Dict[str, Any]
) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    for path in contract.allowed_fact_paths:
        if path == "content":
            facts[path] = str(payload.get("content") or "")
        elif path == "items[].safeLabel":
            facts[path] = [
                str(item.get("safeLabel") or "")
                for item in payload.get("items") or []
                if isinstance(item, Mapping) and item.get("safeLabel")
            ]
    return facts


def _record_refs(contract: ActionOutputContract, payload: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for path in contract.record_ref_paths:
        if path == "items[].resourceUid":
            refs.extend(
                str(item["resourceUid"])
                for item in payload.get("items") or []
                if isinstance(item, Mapping) and item.get("resourceUid")
            )
        elif payload.get(path):
            refs.append(str(payload[path]))
    for key in ("resource_uid", "authorization_id"):
        if args.get(key):
            refs.append(str(args[key]))
    return refs


def _update_index_receipt(
    turn: WorkTurn, contract: ActionOutputContract, result: ActionResult
) -> None:
    """IndexReceipt 只来自本轮真实执行的最小索引读取，禁止反向补索引。"""
    if contract.kind != "index":
        return
    if turn.index_receipt is not None and turn.index_receipt.status == "found":
        return  # 已建立的有效回执不被后续失败冲掉
    if result.status == "success":
        status = "found"
    elif result.status == "empty":
        status = "none"
    elif result.status == "denied":
        status = "denied"
    else:
        status = "unavailable"
    turn.enter("indexing")
    turn.index_receipt = IndexReceipt(
        request_id=turn.request_id,
        actor_fingerprint=(
            turn.identity.datascope_fingerprint if turn.identity else ""
        ),
        loaded_at=result.finished_at,
        scope_summary=result.action_id,
        matched_resource_refs=_record_refs(
            contract,
            result.normalized_payload,
            next(
                (c.arguments for c in turn.action_calls if c.call_id == result.call_id),
                {},
            ),
        ),
        status=status,
    )


def finish_action(
    turn: WorkTurn,
    call_id: str,
    action_id: str,
    version: str,
    raw_content: Any,
    *,
    cancelled: bool = False,
) -> Optional[ActionResult]:
    """PostAction：严格绑定 + 合同校验 + DataScope 复核 + Evidence 构建。

    ``verifying`` 状态只在本函数真实执行时产生。重复、未知、跨回合或
    actionId/version 不一致的 callId 一律拒绝绑定。
    """
    call = next((c for c in turn.action_calls if c.call_id == call_id), None)
    if (
        call is None
        or call.action_id != action_id
        or call.version != version
        or any(r.call_id == call_id for r in turn.action_results)
    ):
        turn.orphaned_receipts += 1
        return None
    raw_text = (
        raw_content
        if isinstance(raw_content, str)
        else json.dumps(raw_content, ensure_ascii=False, default=str)
    )
    contract = ACTION_OUTPUT_CONTRACTS[action_id]
    payload = _parse_json_object(raw_text)
    if cancelled:
        status = "cancelled"
    elif not payload:
        status = "error"  # 长文本/半截回执不再洗白成 success
    else:
        status = _classify_contract_status(contract, payload)
    result = ActionResult(
        call_id=call_id,
        action_id=action_id,
        status=status,
        normalized_payload=payload if status != "error" or payload else {},
        error_code=str(payload.get("code") or payload.get("error") or ""),
        started_at=call.requested_at,
        finished_at=_seq(turn),
        raw_text=raw_text if status == "success" else "",
    )
    turn.action_results.append(result)
    _update_index_receipt(turn, contract, result)

    # PostAction Verify：只对真实返回的 ActionResult 执行。
    turn.enter("verifying")
    if status != "success" or contract.kind == "index":
        # 索引只负责资源发现与工作前置（记入 IndexReceipt），
        # 不代替业务 Evidence（计划 §4.3）。
        return result
    identity = turn.identity
    violations = _scope_violations(payload, identity, call.arguments)
    if violations:
        # 跨账号/嵌套越权/不可核实 DataScope 的 payload：拒绝成为本轮证据。
        turn.rejected_cross_account += 1
        return result
    facts = _project_allowed_facts(contract, payload)
    turn.evidence.append(
        EvidenceEnvelope(
            evidence_id=hashlib.sha256(
                f"{turn.turn_id}|{call_id}".encode("utf-8")
            ).hexdigest()[:16],
            turn_id=turn.turn_id,
            call_id=call_id,
            action_id=action_id,
            datascope_fingerprint=(
                identity.datascope_fingerprint if identity else ""
            ),
            status=status,
            allowed_facts=json.dumps(facts, ensure_ascii=False, sort_keys=True),
            record_refs=_record_refs(contract, payload, call.arguments),
            input_digest=hashlib.sha256(
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True).encode(
                    "utf-8"
                )
            ).hexdigest(),
            output_digest=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            verified_at=result.finished_at,
            verification_status="verified",
        )
    )
    return result


def gate_registry_action(
    name: str, args: Any
) -> Optional["tuple[Optional[WorkTurn], PreActionDecision]"]:
    """``ToolRegistry.dispatch`` 的 PreAction 钩子（Gemini 默认拒绝策略）。

    返回 None 表示非可信目录动作或非 My Stand 服务端会话，保持原路径；
    返回 (turn, decision) 时调用方必须按 decision 执行或拒绝。
    """
    if name not in ACTION_OUTPUT_CONTRACTS or is_write_action(
        name, args if isinstance(args, dict) else None
    ):
        return None
    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("XIAOBAN_SESSION_PLATFORM")
        user_id = get_session_env("XIAOBAN_SESSION_USER_ID")
        if platform != "api_server" or not user_id:
            # 非 My Stand 服务端会话：工具 handler 自身已有 fail-closed 门禁。
            return None
        turn = current_turn()
        if turn is None:
            return None, PreActionDecision("deny", "no_active_turn")
        decision = begin_action(turn, name, "v1", args if isinstance(args, dict) else {})
        return turn, decision
    except Exception:
        # 策略异常默认拒绝（只影响可信目录动作）。
        return None, PreActionDecision("deny", "preaction_error")


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


def result_has_write_actions(result: Any) -> bool:
    """本轮执行记录中是否含写动作（写流程由既有写回执硬闸接管）。"""
    for message in _current_turn_messages(result):
        if str(message.get("role") or "") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            function = function if isinstance(function, Mapping) else {}
            name = str(function.get("name") or call.get("name") or "")
            args = _parse_json_object(
                function.get("arguments")
                if "arguments" in function
                else call.get("arguments")
            )
            if name and is_write_action(name, args):
                return True
    return False


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
    """兼容驱动：把既有执行记录逐项送入真实生命周期门禁。

    模型/历史伪造的调用与回执会在 begin_action/finish_action 的同一
    套 PreAction、绑定与合同校验中被拒绝，不能借此洗白成证据。
    """
    turn = begin_turn(
        channel=channel,
        user_message=user_message,
        conversation_history=conversation_history,
        identity=identity,
        request_id=request_id,
        message_id=message_id,
    )
    for message in _current_turn_messages(result):
        role = str(message.get("role") or "")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                function = function if isinstance(function, Mapping) else {}
                call_id = str(call.get("id") or "")
                name = str(function.get("name") or call.get("name") or "")
                if not call_id or not name:
                    continue
                call_args = _parse_json_object(
                    function.get("arguments")
                    if "arguments" in function
                    else call.get("arguments")
                )
                if is_write_action(name, call_args):
                    # 写动作由既有写回执硬闸接管，只读链不登记不采证。
                    continue
                begin_action(
                    turn,
                    name,
                    "v1",
                    call_args,
                    call_id=call_id,
                )
        elif role == "tool":
            if is_write_action(str(message.get("name") or ""), None):
                continue
            finish_action(
                turn,
                str(message.get("tool_call_id") or ""),
                str(message.get("name") or ""),
                "v1",
                message.get("content"),
            )
    return turn
