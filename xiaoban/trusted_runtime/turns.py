"""Trusted Action Runtime 真实生命周期：PreAction → Execute → PostAction。

机制映射（固定上游 commit，详见交接单）：
- Codex 322d5b96 tools/orchestrator.rs：调度前保留唯一 call ID，
  complete/failed 与同一调用绑定，无结果的调用不得视为成功；
- Claude SDK f8b9ec9 types.py：PreToolUse 在 handler 前 allow/deny，
  tool_use_id 贯穿 pre/execute/post，失败是第一等状态；
- Gemini CLI 3818efbb policy-engine.ts：确定性 allow/deny，无匹配规则、
  安全字段缺失、策略异常一律默认拒绝。

本模块只处理当前请求的真实 ToolRegistry 生命周期，不从 transcript
倒推调用、回执或可信结论。
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional

from xiaoban.trusted_runtime.types import (
    ACTION_OUTPUT_CONTRACTS,
    ActionCall,
    ActionResult,
    ActionOutputContract,
    PreActionDecision,
    TrustedIdentity,
    WorkTurn,
    is_write_action,
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
    identity: Optional[TrustedIdentity] = None,
    request_id: str = "",
    message_id: str = "",
) -> WorkTurn:
    """Open one server-bound turn for the physical tool lifecycle."""
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
        if not turn.request_id or not turn.message_id:
            return _record_denial(turn, call_id, action_id, "missing_turn_id")
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
            # 重复 callId 不得静默覆盖既有结果。
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
        identity = turn.identity
        if identity is None or not identity.account_id:
            return _record_denial(turn, call_id, action_id, "missing_identity")
        if not identity.datascope_fingerprint:
            return _record_denial(turn, call_id, action_id, "missing_datascope")
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
            if isinstance(status, str) and status in {
                "matched",
                "success",
                "complete",
                "completed",
            }:
                pass
            else:
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
        if str(payload.get("content") or "").strip():
            return "success"
        if "content" in payload:
            return "empty"
        if (
            contract.action_id == "mystand_query"
            and payload.get("schema") == "mystand.query-result.v1"
            and isinstance(payload.get("facts"), list)
        ):
            return "success" if payload["facts"] else "empty"
        return "error"
    return "error"


def finish_action(
    turn: WorkTurn,
    call_id: str,
    action_id: str,
    version: str,
    raw_content: Any,
    *,
    cancelled: bool = False,
) -> Optional[ActionResult]:
    """Bind one physical handler result to its admitted call and DataScope."""
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
        status = "error"
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
    turn.enter("verifying")
    violations = _scope_violations(payload, turn.identity, call.arguments)
    if violations:
        turn.rejected_cross_account += 1
    return result


def gate_registry_action(
    name: str, args: Any, *, call_id: str = ""
) -> Optional["tuple[Optional[WorkTurn], PreActionDecision]"]:
    """``ToolRegistry.dispatch`` 的唯一 My Stand 物理 K3 边界。

    返回 None 表示非可信目录动作或非 My Stand 服务端会话，保持原路径；
    返回 (turn, decision) 时调用方必须按 decision 执行或拒绝。
    """
    catalog_action = name in ACTION_OUTPUT_CONTRACTS
    write_action = is_write_action(
        name,
        args if isinstance(args, dict) else None,
    )
    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("XIAOBAN_SESSION_PLATFORM")
        user_id = get_session_env("XIAOBAN_SESSION_USER_ID")
        if platform != "api_server" or not user_id:
            # 非 My Stand 服务端会话：工具 handler 自身已有 fail-closed 门禁。
            return None
        if not catalog_action:
            return None
        if write_action:
            # Existing confirmed-write runtime owns every write.
            return None
        turn = current_turn()
        if turn is None:
            return None, PreActionDecision("deny", "no_active_turn")
        if not call_id:
            from tools.approval import _approval_tool_call_id

            call_id = _approval_tool_call_id.get()
        call_id = call_id or f"mystand_pre_{uuid.uuid4().hex}"
        gated_args = dict(args) if isinstance(args, dict) else {}
        pending = next(
            (
                call
                for call in turn.action_calls
                if call.call_id == call_id
                and call.action_id == name
                and not any(result.call_id == call_id for result in turn.action_results)
            ),
            None,
        )
        decision = (
            PreActionDecision("allow", "already_allowed", pending)
            if pending is not None
            else begin_action(
                turn,
                name,
                "v1",
                gated_args,
                call_id=call_id,
            )
        )
        return turn, decision
    except Exception:
        # 策略异常默认拒绝（只影响可信目录动作）。
        if catalog_action:
            return None, PreActionDecision("deny", "preaction_error")
        return None
