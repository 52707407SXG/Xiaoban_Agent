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
import copy
import hashlib
import json
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
from xiaoban.trusted_runtime.fact_contract import (
    build_fact_query_plan,
    canonical_digest,
    evidence_requirement_digest,
    resource_read_record_refs_valid,
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


def _canonical_digest(value: Any) -> str:
    return canonical_digest(value)


def _fact_query_result_binding_valid(
    turn: WorkTurn,
    payload: Mapping[str, Any],
) -> bool:
    """Bind one internal query result to this signed plan and DataScope."""
    requirement = turn.fact_requirement
    if (
        not isinstance(requirement, Mapping)
        or not isinstance(requirement.get("query_plan"), Mapping)
    ):
        return True
    binding = requirement.get("binding")
    if not isinstance(binding, Mapping):
        return False
    expected_scope = binding.get("datascope_fingerprint")
    actual_scope = payload.get("scopeFingerprint")
    if requirement.get("fact_kind") == "collection":
        coverage = payload.get("coverage")
        actual_scope = (
            coverage.get("scopeFingerprint")
            if isinstance(coverage, Mapping)
            else None
        )
    return bool(
        payload.get("schema") == "mystand.query-result.v1"
        and payload.get("queryKind") == requirement.get("query_kind")
        and payload.get("planId") == requirement.get("plan_id")
        and payload.get("requirementDigest")
        == evidence_requirement_digest(
            requirement,
            canonical_fallback=turn.fact_requirement_digest,
        )
        and actual_scope == expected_scope
    )


def _collection_evidence_from_payload(
    turn: WorkTurn,
    call_id: str,
    payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Bind a typed My Stand collection response to this exact fact turn."""
    requirement = turn.fact_requirement
    if not isinstance(requirement, Mapping):
        return None
    server_coverage = payload.get("coverage")
    if not isinstance(server_coverage, Mapping):
        return None
    requirement_digest = evidence_requirement_digest(
        requirement,
        canonical_fallback=turn.fact_requirement_digest,
    )
    echoed_digest = payload.get("requirementDigest") or payload.get(
        "requirement_digest"
    )
    if echoed_digest != requirement_digest:
        return None
    binding = requirement.get("binding")
    if not isinstance(binding, Mapping):
        return None
    scope_fingerprint = (
        server_coverage.get("scopeFingerprint")
        or server_coverage.get("scope_fingerprint")
    )
    if scope_fingerprint != binding.get("datascope_fingerprint"):
        return None

    expected_count = server_coverage.get(
        "expectedCount",
        server_coverage.get("expected_count"),
    )
    actual_count = server_coverage.get(
        "returnedCount",
        server_coverage.get("actual_count"),
    )
    has_more = server_coverage.get(
        "hasMore",
        server_coverage.get("has_more"),
    )
    expected_digest = server_coverage.get(
        "expectedResourceRefsDigest",
        server_coverage.get("expected_digest"),
    )
    actual_digest = server_coverage.get(
        "returnedResourceRefsDigest",
        server_coverage.get("actual_digest"),
    )
    complete = server_coverage.get("complete")
    coverage_year = server_coverage.get("year")
    tie_rule = server_coverage.get("tieRule")
    raw_record_refs = payload.get("recordRefs")
    if (
        not isinstance(raw_record_refs, list)
        or any(not isinstance(ref, str) or not ref for ref in raw_record_refs)
    ):
        return None
    record_refs = sorted(set(raw_record_refs))
    if len(record_refs) != len(raw_record_refs):
        return None
    recomputed_digest = _canonical_digest(record_refs)
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or isinstance(actual_count, bool)
        or not isinstance(actual_count, int)
        or expected_count != actual_count
        or actual_count != len(record_refs)
        or has_more is not False
        or complete is not True
        or expected_digest != recomputed_digest
        or actual_digest != recomputed_digest
        or (
            str(requirement.get("time_scope") or "").isdigit()
            and coverage_year
            != int(str(requirement.get("time_scope")))
        )
        or (
            requirement.get("operation") == "rank"
            and tie_rule != "dense"
        )
    ):
        return None
    plan_id = str(requirement.get("plan_id") or "")
    echoed_plan_id = str(payload.get("planId") or payload.get("plan_id") or "")
    if plan_id and echoed_plan_id != plan_id:
        return None
    projected_facts: Dict[str, Any] = {
        "operation": requirement.get("operation"),
        "metric": requirement.get("metric"),
        "time_scope": requirement.get("time_scope"),
    }
    if requirement.get("ordinal") is not None:
        projected_facts["ordinal"] = requirement.get("ordinal")
    if isinstance(payload.get("facts"), (list, dict)):
        projected_facts["facts"] = payload.get("facts")
    if requirement.get("plan_id"):
        projected_facts["plan_id"] = requirement.get("plan_id")
    return {
        "schema": "mystand.collection-evidence.v1",
        "requirement_digest": requirement_digest,
        "binding": dict(binding),
        "status": "complete" if complete is True else "incomplete",
        "expected_count": expected_count,
        "actual_count": actual_count,
        "has_more": has_more,
        "expected_digest": expected_digest,
        "actual_digest": actual_digest,
        "source_call_ids": [call_id],
        "projected_facts": projected_facts,
        "projected_text": str(payload.get("content") or ""),
        # Kept only for the server-to-server terminal receipt projection.
        "server_coverage": dict(server_coverage),
        "record_refs": record_refs,
    }


def classify_interaction(
    user_message: Any,
    conversation_history: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    used_business_tools: bool = False,
    evidence_required: bool = False,
) -> str:
    """只按服务器执行事实分类 CHAT/WORK。

    自然语言、数字、日期和历史消息都不是可信执行信号。My Stand 后端
    签发的本轮 ``evidence_required``，或当前回合真实出现的业务工具，
    才能把回合推进到 WORK。参数保留 ``user_message`` /
    ``conversation_history`` 是为了兼容各渠道统一入口，不读取其内容。
    """
    del user_message, conversation_history
    if used_business_tools or evidence_required:
        return INTERACTION_WORK
    return INTERACTION_CHAT


def result_requires_evidence(result: Any) -> bool:
    """读取 API 运行时写入的结构化证据要求，不检查回答文本。"""
    if not isinstance(result, Mapping):
        return False
    if result.get("_mystand_evidence_required") is True:
        return True
    return any(
        result.get(key)
        for key in (
            "_mystand_required_evidence_groups",
            "_mystand_required_evidence_tools",
            "_mystand_required_evidence_tool",
        )
    )


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
    evidence_required: bool = False,
    fact_requirement: Optional[Mapping[str, Any]] = None,
) -> WorkTurn:
    """服务端开回合：稳定 request/message ID + 服务端解析身份。"""
    turn_id = hashlib.sha256(
        f"{channel}|{request_id}|{message_id}|{_visible_text(user_message)[:200]}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    bound_requirement = (
        copy.deepcopy(dict(fact_requirement))
        if isinstance(fact_requirement, Mapping)
        else None
    )
    turn = WorkTurn(
        turn_id=turn_id,
        request_id=request_id,
        message_id=message_id,
        channel=channel,
        identity=identity,
        interaction_kind=classify_interaction(
            user_message,
            conversation_history,
            evidence_required=bool(evidence_required or fact_requirement),
        ),
        index_receipt=None,
        fact_requirement=bound_requirement,
        fact_requirement_digest=(
            _canonical_digest(bound_requirement)
            if bound_requirement is not None
            else ""
        ),
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
        if action_id == "mystand_query" and (
            args.get("query_kind") is not None
            or (
                isinstance(turn.fact_requirement, Mapping)
                and turn.fact_requirement.get("fact_kind") == "collection"
            )
        ):
            signed_plan = build_fact_query_plan(turn.fact_requirement)
            if signed_plan != args:
                return _record_denial(
                    turn,
                    call_id,
                    action_id,
                    "unbound_fact_query_plan",
                )
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


def _project_allowed_facts(
    contract: ActionOutputContract, payload: Dict[str, Any]
) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    for path in contract.allowed_fact_paths:
        if path == "content":
            content = str(payload.get("content") or "")
            if content:
                facts[path] = content
        elif path == "facts":
            safe_facts: List[Dict[str, Any]] = []
            for item in payload.get("facts") or []:
                if not isinstance(item, Mapping):
                    continue
                safe_item = {
                    key: item[key]
                    for key in (
                        "kind",
                        "predicate",
                        "label",
                        "value",
                        "unit",
                        "confidence",
                    )
                    if key in item
                }
                if safe_item:
                    safe_facts.append(safe_item)
            if safe_facts:
                facts[path] = safe_facts
        elif path == "collection":
            coverage = payload.get("collection_evidence")
            if isinstance(coverage, Mapping):
                projected = coverage.get("projected_facts")
                if isinstance(projected, Mapping):
                    facts[path] = dict(projected)
        elif path == "items[].safeLabel":
            facts[path] = [
                str(item.get("safeLabel") or "")
                for item in payload.get("items") or []
                if isinstance(item, Mapping) and item.get("safeLabel")
            ]
    return facts


def serialize_allowed_facts(
    action_id: str,
    payload: Dict[str, Any],
) -> str:
    """Create the one canonical EvidenceEnvelope fact projection."""
    contract = ACTION_OUTPUT_CONTRACTS.get(action_id)
    if contract is None:
        return ""
    return json.dumps(
        _project_allowed_facts(contract, payload),
        ensure_ascii=False,
        sort_keys=True,
    )


def _record_refs(contract: ActionOutputContract, payload: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for path in contract.record_ref_paths:
        if path == "items[].resourceUid":
            refs.extend(
                str(item["resourceUid"])
                for item in payload.get("items") or []
                if isinstance(item, Mapping) and item.get("resourceUid")
            )
        elif path == "recordRefs[]":
            refs.extend(
                str(item)
                for item in payload.get("recordRefs") or []
                if isinstance(item, str) and item
            )
        elif path == "resource.resourceUid":
            resource = payload.get("resource")
            if isinstance(resource, Mapping) and resource.get("resourceUid"):
                refs.append(str(resource["resourceUid"]))
        elif payload.get(path):
            refs.append(str(payload[path]))
    for key in ("resource_uid", "authorization_id"):
        if args.get(key):
            refs.append(str(args[key]))
    return sorted(set(refs))


def _update_index_receipt(
    turn: WorkTurn, contract: ActionOutputContract, result: ActionResult
) -> None:
    """IndexReceipt 只来自本轮真实执行的最小索引读取，禁止反向补索引。"""
    if contract.kind != "index":
        return
    if turn.index_receipt is not None and turn.index_receipt.status == "found":
        return  # 已建立的有效回执不被后续失败冲掉
    raw_has_more = result.normalized_payload.get("hasMore")
    has_more = raw_has_more if isinstance(raw_has_more, bool) else None
    next_cursor = result.normalized_payload.get("nextCursor")
    signed_fact = isinstance(turn.fact_requirement, Mapping)
    legacy_pagination_undeclared = (
        not signed_fact and raw_has_more is None and next_cursor is None
    )
    terminal_page_is_consistent = (
        has_more is False and next_cursor in (None, "")
    ) or legacy_pagination_undeclared
    if result.status == "success" and terminal_page_is_consistent:
        status = "found"
    elif result.status == "empty":
        status = "none"
    elif result.status == "denied":
        status = "denied"
    else:
        status = "unavailable"
    record_refs = sorted(
        set(
            _record_refs(
                contract,
                result.normalized_payload,
                next(
                    (
                        c.arguments
                        for c in turn.action_calls
                        if c.call_id == result.call_id
                    ),
                    {},
                ),
            )
        )
    )
    turn.enter("indexing")
    turn.index_receipt = IndexReceipt(
        request_id=turn.request_id,
        actor_fingerprint=(
            turn.identity.datascope_fingerprint if turn.identity else ""
        ),
        loaded_at=result.finished_at,
        scope_summary=result.action_id,
        matched_resource_refs=record_refs,
        status=status,
        source_call_id=result.call_id,
        has_more=has_more,
        resource_count=len(record_refs),
        resource_refs_digest=_canonical_digest(record_refs),
    )


def _signed_resource_read_refs_valid(
    turn: WorkTurn,
    contract: ActionOutputContract,
    payload: Mapping[str, Any],
    arguments: Dict[str, Any],
) -> bool:
    """Bind all generic fact evidence sources to this turn's index."""
    requirement = turn.fact_requirement
    if (
        not isinstance(requirement, Mapping)
        or requirement.get("fact_kind") != "single"
        or requirement.get("query_kind") != "resource-read"
    ):
        return True
    receipt = turn.index_receipt
    if receipt is None or receipt.status != "found":
        return False
    return resource_read_record_refs_valid(
        payload.get("recordRefs"),
        _record_refs(contract, dict(payload), arguments),
        receipt.matched_resource_refs,
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
    fact_binding_invalid = bool(
        action_id == "mystand_query"
        and payload
        and not _fact_query_result_binding_valid(turn, payload)
    )
    collection_evidence = (
        _collection_evidence_from_payload(turn, call_id, payload)
        if action_id == "mystand_query"
        and payload
        and not fact_binding_invalid
        else None
    )
    collection_binding_invalid = bool(
        action_id == "mystand_query"
        and isinstance(turn.fact_requirement, Mapping)
        and turn.fact_requirement.get("fact_kind") == "collection"
        and collection_evidence is None
    )
    resource_read_binding_invalid = bool(
        action_id == "mystand_query"
        and payload
        and not _signed_resource_read_refs_valid(
            turn,
            contract,
            payload,
            call.arguments,
        )
    )
    if collection_evidence is not None:
        payload = dict(payload)
        payload["collection_evidence"] = collection_evidence
    if cancelled:
        status = "cancelled"
    elif (
        not payload
        or fact_binding_invalid
        or collection_binding_invalid
        or resource_read_binding_invalid
    ):
        status = "error"  # 长文本/半截回执不再洗白成 success
    else:
        status = _classify_contract_status(contract, payload)
        if (
            status == "empty"
            and contract.kind == "index"
            and isinstance(turn.fact_requirement, Mapping)
            and turn.fact_requirement.get("fact_kind") == "collection"
            and payload.get("schema")
            in {
                "mystand.resource-index.page.v1",
                "mystand.resource-index.complete.v1",
            }
            and payload.get("hasMore") is False
            and payload.get("nextCursor") == ""
        ):
            # A complete zero-item index is valid coverage for collection
            # facts.  It still must be followed by mystand_query so the server
            # can issue a signed zero-coverage business result.
            status = "success"
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
    # PostAction Verify：只对真实返回的 ActionResult 执行。
    turn.enter("verifying")
    identity = turn.identity
    violations = _scope_violations(payload, identity, call.arguments)
    if violations:
        turn.rejected_cross_account += 1
        return result
    _update_index_receipt(turn, contract, result)
    if status != "success" or contract.kind == "index":
        # 索引只负责资源发现与工作前置（记入 IndexReceipt），
        # 不代替业务 Evidence（计划 §4.3）。
        return result
    if collection_evidence is not None:
        turn.collection_evidence = collection_evidence
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
            allowed_facts=serialize_allowed_facts(action_id, payload),
            record_refs=_record_refs(contract, payload, call.arguments),
            input_digest=hashlib.sha256(
                json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            output_digest=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            verified_at=result.finished_at,
            verification_status="verified",
            requirement_digest=(
                collection_evidence["requirement_digest"]
                if collection_evidence is not None
                else (
                    evidence_requirement_digest(
                        turn.fact_requirement,
                        canonical_fallback=turn.fact_requirement_digest,
                    )
                    if action_id == "mystand_query"
                    and isinstance(turn.fact_requirement, Mapping)
                    else ""
                )
            ),
            coverage_digest=(
                _canonical_digest(collection_evidence)
                if collection_evidence is not None
                else ""
            ),
        )
    )
    return result


def gate_registry_action(
    name: str, args: Any, *, call_id: str = ""
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
        if not call_id:
            from tools.approval import _approval_tool_call_id

            call_id = _approval_tool_call_id.get()
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
                args if isinstance(args, dict) else {},
                call_id=call_id,
            )
        )
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
    compat_id = hashlib.sha256(
        json.dumps(_current_turn_messages(result), ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]
    request_id = request_id or f"compat-req-{compat_id}"
    message_id = message_id or f"compat-msg-{compat_id}"
    turn = begin_turn(
        channel=channel,
        user_message=user_message,
        conversation_history=conversation_history,
        identity=identity,
        request_id=request_id,
        message_id=message_id,
        evidence_required=result_requires_evidence(result),
        fact_requirement=(
            result.get("_mystand_fact_requirement")
            if isinstance(result, Mapping)
            and isinstance(result.get("_mystand_fact_requirement"), Mapping)
            else None
        ),
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
