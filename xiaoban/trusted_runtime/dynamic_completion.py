"""Dynamic-evidence-v2 completion projection and verification.

This module owns the v2-only completion contract.  The public guard keeps
protocol routing and delegates deterministic projection, binding validation,
and receipt construction here.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from xiaoban.trusted_runtime.fact_contract import canonical_digest
from xiaoban.trusted_runtime.turns import serialize_allowed_facts
from xiaoban.trusted_runtime.types import (
    ACTION_OUTPUT_CONTRACTS,
    CompletionDecision,
    MYSTAND_COMPLETION_BINDING_FIELDS,
    MYSTAND_COMPLETION_PROTOCOL_V2,
    MYSTAND_COMPLETION_VERIFICATION_SCHEMA_V2,
    WorkTurn,
)


NO_EVIDENCE_MESSAGE = (
    "这轮我没有真正查到站内资料，所以不能给出具体的资料内容、数值或状态。"
)
_MAX_COMPLETION_TEXT = 4_000
_DYNAMIC_ACTION_IDS = frozenset(ACTION_OUTPUT_CONTRACTS)
DYNAMIC_READ_NOT_DISPATCHED = "read_not_dispatched_after_index"
DYNAMIC_ACTION_NOT_DISPATCHED = "action_not_dispatched"
DYNAMIC_READ_PRECONDITION_NOT_MET = "read_precondition_not_met"
DYNAMIC_ACTION_RESULT_MISSING = "action_result_missing"
DYNAMIC_INDEX_INCOMPLETE = "index_incomplete"
_HARD_PREACTION_ERRORS = frozenset(
    {
        "duplicate_call_id",
        "missing_datascope",
        "missing_identity",
        "missing_turn_id",
        "not_in_catalog",
        "preaction_error",
        "unknown_action",
        "write_isolated",
    }
)
_TRANSIENT_TIMEOUT_CODES = frozenset(
    {
        "deadline_exceeded",
        "gateway_timeout",
        "handler_timeout",
        "provider_timeout",
        "read_timeout",
        "request_timeout",
        "timed_out",
        "timeout",
        "upstream_timeout",
    }
)
_TRANSIENT_UNAVAILABLE_CODES = frozenset(
    {
        "connection_error",
        "connection_failed",
        "econnrefused",
        "econnreset",
        "mystand_authorization_transport_failed",
        "mystand_query_transport_failed",
        "network_unavailable",
        "provider_unavailable",
        "service_unavailable",
        "upstream_unavailable",
    }
)
_TRANSIENT_RECOVERY_CODES = (
    _TRANSIENT_TIMEOUT_CODES | _TRANSIENT_UNAVAILABLE_CODES
)
_CORRECTABLE_QUERY_CODES = frozenset(
    {
        "invalid_mystand_query_arguments",
    }
)
_PRESENTATION_UNAVAILABLE_CODES = _TRANSIENT_UNAVAILABLE_CODES | frozenset(
    {
        # These codes are safe to explain as an unavailable site-data
        # connection, but they are deliberately not recoverable: retrying a
        # missing bridge/configuration cannot fix it and only spends another
        # paid call.
        "mystand_authorization_unavailable",
        "mystand_query_unavailable",
        "mystand_resource_index_transport_failed",
        "mystand_resource_index_unavailable",
    }
)
_ANSWER_DEFERRAL_RE = re.compile(
    r"(?:"
    r"(?:先不|暂不|暂时不|稍后|晚点|之后|下次).{0,12}"
    r"(?:分析|建议|判断|回答|再说)|"
    r"(?:分析|建议|判断|回答).{0,12}(?:稍后|晚点|之后|下次|再说)"
    r")"
)
def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        candidate = vars(value)
    except TypeError:
        return {}
    return candidate if isinstance(candidate, Mapping) else {}


def _canonical_digest(value: Any) -> str:
    return canonical_digest(value)


def evidence_receipt_digest(items: Sequence[Any]) -> str:
    return _canonical_digest(
        [
            {
                "evidence_id": item.evidence_id,
                "turn_id": item.turn_id,
                "call_id": item.call_id,
                "action_id": item.action_id,
                "datascope_fingerprint": item.datascope_fingerprint,
                "status": item.status,
                "allowed_facts": item.allowed_facts,
                "record_refs": item.record_refs,
                "input_digest": item.input_digest,
                "output_digest": item.output_digest,
                "requirement_digest": item.requirement_digest,
                "coverage_digest": item.coverage_digest,
                "verification_status": item.verification_status,
            }
            for item in sorted(
                items,
                key=lambda item: (
                    str(item.call_id),
                    str(item.action_id),
                    str(item.evidence_id),
                ),
            )
        ]
    )


def _completion_binding_valid(turn: WorkTurn) -> bool:
    binding = _mapping(getattr(turn, "completion_binding", None))
    identity = turn.identity
    return bool(
        turn.completion_protocol == MYSTAND_COMPLETION_PROTOCOL_V2
        and identity is not None
        and set(binding) == MYSTAND_COMPLETION_BINDING_FIELDS
        and binding.get("user_id") == identity.account_id
        and binding.get("datascope_fingerprint")
        == identity.datascope_fingerprint
        and binding.get("delivery_id") == turn.request_id
        and binding.get("message_id") == turn.message_id
        and isinstance(binding.get("session_id"), str)
        and binding.get("session_id")
        and isinstance(binding.get("attempt"), int)
        and not isinstance(binding.get("attempt"), bool)
        and binding.get("attempt") >= 1
        and re.fullmatch(
            r"[a-f0-9]{64}",
            str(binding.get("request_fingerprint") or ""),
        )
        and re.fullmatch(
            r"[a-f0-9]{64}",
            str(binding.get("invocation_fingerprint") or ""),
        )
    )


def _completion_receipt(
    turn: WorkTurn,
    *,
    completion_kind: str,
    action_count: int,
    evidence_count: int,
    output: str,
    decision: str,
) -> dict[str, Any]:
    binding = _mapping(turn.completion_binding)
    return {
        "schema": MYSTAND_COMPLETION_VERIFICATION_SCHEMA_V2,
        "completion_kind": completion_kind,
        "binding_verified": True,
        "semantic_verified": False,
        "delivery_id": str(binding.get("delivery_id") or ""),
        "request_id": turn.request_id,
        "attempt": binding.get("attempt"),
        "message_id": turn.message_id,
        "request_fingerprint": str(
            binding.get("request_fingerprint") or ""
        ),
        "invocation_fingerprint": str(
            binding.get("invocation_fingerprint") or ""
        ),
        "datascope_fingerprint": str(
            binding.get("datascope_fingerprint") or ""
        ),
        "action_count": action_count,
        "evidence_count": evidence_count,
        "output_digest": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "decision": decision,
        "verified_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
    }


def _dynamic_index_binding(
    turn: WorkTurn,
) -> tuple[str, list[Mapping[str, Any]]]:
    """Bind a model-chosen read to this turn's complete server index."""
    receipt = turn.index_receipt
    identity = turn.identity
    if (
        receipt is None
        or identity is None
        or receipt.status != "found"
        or receipt.request_id != turn.request_id
        or receipt.actor_fingerprint != identity.datascope_fingerprint
        or receipt.scope_summary != "mystand_resource_index"
        or not receipt.source_call_id
        or receipt.has_more is not False
        or receipt.resource_count <= 0
        or receipt.resource_count != len(set(receipt.matched_resource_refs))
        or receipt.resource_refs_digest
        != _canonical_digest(sorted(set(receipt.matched_resource_refs)))
    ):
        return "", []
    matching_calls = [
        call
        for call in turn.action_calls
        if call.call_id == receipt.source_call_id
        and call.action_id == "mystand_resource_index"
        and call.version == "v1"
    ]
    matching_results = [
        result
        for result in turn.action_results
        if result.call_id == receipt.source_call_id
        and result.action_id == "mystand_resource_index"
        and result.status == "success"
    ]
    if len(matching_calls) != 1 or len(matching_results) != 1:
        return "", []
    call = matching_calls[0]
    result = matching_results[0]
    if (
        result.started_at != call.requested_at
        or result.finished_at != receipt.loaded_at
        or not result.raw_text
    ):
        return "", []
    try:
        raw_payload = json.loads(result.raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "", []
    raw_items = (
        raw_payload.get("items")
        if isinstance(raw_payload, Mapping)
        else None
    )
    if (
        not isinstance(raw_payload, Mapping)
        or raw_payload.get("schema")
        not in {
            "mystand.resource-index.page.v1",
            "mystand.resource-index.complete.v1",
        }
        or raw_payload.get("ok") is not True
        or raw_payload.get("hasMore") is not False
        or raw_payload.get("nextCursor") not in (None, "")
        or not isinstance(raw_items, list)
        or not raw_items
        or _canonical_digest(raw_payload)
        != _canonical_digest(result.normalized_payload)
    ):
        return "", []
    listed_refs: list[str] = []
    projected_items: list[Mapping[str, Any]] = []
    for item in raw_items:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("resourceUid"), str)
            or not item.get("resourceUid")
            or not isinstance(item.get("safeLabel"), str)
            or not item.get("safeLabel").strip()
            or not isinstance(item.get("resourceType"), str)
            or not item.get("resourceType").strip()
            or not isinstance(item.get("canRead"), bool)
            or not isinstance(item.get("locked"), bool)
        ):
            return "", []
        listed_refs.append(item["resourceUid"])
        projected_items.append(item)
    if (
        len(listed_refs) != len(set(listed_refs))
        or sorted(listed_refs)
        != sorted(set(receipt.matched_resource_refs))
        or len(listed_refs) != receipt.resource_count
    ):
        return "", []
    digest = _canonical_digest(
        {
            "receipt": _mapping(receipt),
            "action_call": _mapping(call),
            "action_result": {
                **_mapping(result),
                "raw_text": None,
                "output_digest": hashlib.sha256(
                    result.raw_text.encode("utf-8")
                ).hexdigest(),
            },
        }
    )
    return digest, projected_items


def _hard_runtime_violation(turn: WorkTurn) -> bool:
    if turn.orphaned_receipts or turn.rejected_cross_account:
        return True
    return any(
        result.error_code in _HARD_PREACTION_ERRORS
        for result in turn.action_results
    )


def _validated_terminal_text(final_text: str) -> str:
    text = str(final_text or "")
    if not text.strip() or len(text) > _MAX_COMPLETION_TEXT or "\x00" in text:
        return ""
    return text


def _grounding_fact_texts(turn: WorkTurn) -> tuple[list[str], list[str]]:
    """Return trusted business values and field labels, never tool metadata."""
    values: list[str] = []
    labels: list[str] = []
    for evidence in turn.evidence:
        if (
            evidence.status != "success"
            or evidence.verification_status != "verified"
            or not evidence.allowed_facts
        ):
            continue
        try:
            projected = json.loads(evidence.allowed_facts)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(projected, Mapping):
            continue
        content = projected.get("content")
        if isinstance(content, str) and content.strip():
            values.append(content)
        facts = projected.get("facts")
        if isinstance(facts, Sequence) and not isinstance(
            facts,
            (str, bytes, bytearray),
        ):
            for fact in facts:
                if not isinstance(fact, Mapping):
                    continue
                label = fact.get("label")
                if isinstance(label, str) and label.strip():
                    labels.append(label)
                value = fact.get("value")
                if isinstance(value, str) and value.strip():
                    values.append(value)
                elif isinstance(value, (int, float)) and not isinstance(
                    value,
                    bool,
                ):
                    values.append(str(value))
        collection = projected.get("collection")
        if isinstance(collection, Mapping):
            for value in collection.values():
                if isinstance(value, str) and value.strip():
                    values.append(value)
                elif isinstance(value, (int, float)) and not isinstance(
                    value,
                    bool,
                ):
                    values.append(str(value))
        index_labels = projected.get("items[].safeLabel")
        if isinstance(index_labels, Sequence) and not isinstance(
            index_labels,
            (str, bytes, bytearray),
        ):
            labels.extend(
                str(item)
                for item in index_labels
                if isinstance(item, str) and item.strip()
            )
    return values, labels


def _answer_uses_bound_evidence(turn: WorkTurn, output: str) -> bool:
    """Require a content anchor, not a prescribed answer shape.

    The runtime verifies provenance and the model chooses the language.  A
    successful work reply must nevertheless contain at least one concrete
    value or phrase from the verified material, so a bare "lookup completed"
    receipt cannot masquerade as the requested answer.
    """
    values, labels = _grounding_fact_texts(turn)
    source = " ".join(values).casefold()[:200_000]
    text = str(output or "").casefold()
    if (
        (not source and not labels)
        or not text
        or _ANSWER_DEFERRAL_RE.search(text)
    ):
        return False
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._%-]{1,}", text):
        if token in source:
            return True
    for run in re.findall(r"[\u3400-\u9fff]{6,}", text):
        for width in range(min(12, len(run)), 5, -1):
            if any(
                run[start : start + width] in source
                for start in range(0, len(run) - width + 1)
            ):
                return True
    # Some typed facts intentionally use structured values (for example a
    # boolean or a small object).  Natural language will not quote their JSON
    # bytes, so bind it to the trusted field/resource label instead. Requiring
    # a non-trivial answer prevents a bare two-word label from becoming a
    # completion receipt.
    if len(text.strip()) >= 12:
        for label in labels:
            normalized = str(label).strip().casefold()
            if len(normalized) >= 2 and normalized in text:
                return True
    return False


def _failure_reason_category(
    failure_class: str,
    failures: Sequence[Any],
) -> str:
    """Reduce handler-controlled codes to a safe user-facing cause class."""
    if failure_class != "error":
        return failure_class
    error_codes = {
        str(result.error_code or "").strip().lower()
        for result in failures
        if str(result.error_code or "").strip()
    }
    if error_codes and error_codes <= _TRANSIENT_TIMEOUT_CODES:
        return "timeout"
    if error_codes and error_codes <= _PRESENTATION_UNAVAILABLE_CODES:
        return "unavailable"
    if error_codes and error_codes <= _CORRECTABLE_QUERY_CODES:
        return "invalid_arguments"
    return "execution_error"


def _validated_failure_recovery_reason(
    turn: WorkTurn,
    failures: Sequence[Any],
) -> str:
    """Bind one failed recovery to the physical call that caused it."""
    if len(failures) != 2:
        return ""
    first_result, terminal_result = failures
    calls = {
        call.call_id: call
        for call in turn.action_calls
    }
    first_call = calls.get(first_result.call_id)
    terminal_call = calls.get(terminal_result.call_id)
    if (
        first_call is None
        or terminal_call is None
        or first_result.status != "error"
        or terminal_result.status == "success"
        or first_call.action_id != terminal_call.action_id
        or first_call.version != terminal_call.version
    ):
        return ""
    positions = {
        call.call_id: index
        for index, call in enumerate(turn.action_calls)
    }
    first_position = positions.get(first_call.call_id, -1)
    terminal_position = positions.get(terminal_call.call_id, -1)
    if (
        first_position < 0
        or terminal_position <= first_position
        or sum(
            1
            for call in turn.action_calls
            if positions.get(call.call_id, -1) > first_position
        )
        != 1
    ):
        return ""
    recovery_reason = _failure_reason_category(
        "error",
        [first_result],
    )
    if recovery_reason == "invalid_arguments":
        if first_call.action_id != "mystand_query":
            return ""
        corrected_arguments = _corrected_semantic_query_arguments(
            first_call
        )
        if corrected_arguments is None or (
            _canonical_digest(terminal_call.arguments)
            != _canonical_digest(corrected_arguments)
        ):
            return ""
        return recovery_reason
    if recovery_reason in {"timeout", "unavailable"} and (
        _canonical_digest(first_call.arguments)
        == _canonical_digest(terminal_call.arguments)
    ):
        return recovery_reason
    return ""


def _failure_lifecycle_projection(
    turn: WorkTurn,
    failures: Sequence[Any],
) -> tuple[str, str, str]:
    """Project ordered first-recovery and terminal causes without flattening."""
    recovery_reason = _validated_failure_recovery_reason(turn, failures)
    considered = [failures[-1]] if recovery_reason else list(failures)
    statuses = sorted({str(result.status) for result in considered})
    failure_class = statuses[0] if len(statuses) == 1 else "mixed"
    failure_reason = _failure_reason_category(
        failure_class,
        considered,
    )
    return failure_class, failure_reason, recovery_reason


def _corrected_semantic_query_arguments(
    failed_call: Any,
) -> Optional[dict[str, Any]]:
    """Deterministically remove typed-only fields without changing target."""
    raw = _mapping(getattr(failed_call, "arguments", None))
    if raw.get("operation") != "read":
        return None
    candidate: dict[str, Any] = {"operation": "read"}
    for field in ("resource", "entities"):
        if field in raw:
            candidate[field] = json.loads(
                json.dumps(raw[field], ensure_ascii=False)
            )
    if "resource" not in candidate and "entities" not in candidate:
        return None
    if "fact_needs" in raw:
        candidate["fact_needs"] = json.loads(
            json.dumps(raw["fact_needs"], ensure_ascii=False)
        )
    else:
        # The current stage is a read-only material retrieval. These are the
        # narrow facts required to answer, not a new target selected by a model.
        candidate["fact_needs"] = [
            "document.content",
            "resource.summary",
        ]
    candidate["mode"] = raw.get("mode", "summary")
    try:
        from tools.mystand_query_tool import (
            validate_mystand_semantic_query_plan,
        )

        return validate_mystand_semantic_query_plan(candidate)
    except (TypeError, ValueError):
        return None


def _safe_recovery_tool_result(
    failure: Any,
    *,
    reason: str,
    correction: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Project one model-visible error object without replaying private text."""
    if reason == "invalid_arguments":
        status = 400
        code = "invalid_mystand_query_arguments"
        error = "查询参数里混入了当前阶段不允许的字段。"
    elif reason == "timeout":
        status = 504
        code = "read_timeout"
        error = "这次正文读取等待超时。"
    else:
        status = 503
        code = "service_unavailable"
        error = "这次正文读取服务暂时不可用。"
    return {
        "ok": False,
        "is_error": True,
        "status": status,
        "code": code,
        "error": error,
        "retryable": True,
        **(
            {"correction": dict(correction)}
            if isinstance(correction, Mapping)
            else {}
        ),
    }


def _recovery_grant(
    turn: WorkTurn,
    failed_call: Any,
    *,
    allowed_mutation: str,
) -> dict[str, Any]:
    """Issue one target-bound, request-local recovery capability."""
    arguments = _mapping(getattr(failed_call, "arguments", None))
    target = {
        key: arguments[key]
        for key in (
            "resource",
            "entities",
            "resource_uid",
            "authorization_id",
        )
        if key in arguments
    }
    grant = {
        "schema": "xiaoban.recovery-grant.v1",
        "retry_of_event_id": str(failed_call.call_id),
        "max_uses": 1,
        "tool": str(failed_call.action_id),
        "operation": str(arguments.get("operation") or ""),
        "intent_binding": str(
            turn.completion_binding.get("request_fingerprint") or ""
        ),
        "target_binding": _canonical_digest(target),
        "required_facts_binding": _canonical_digest(
            arguments.get("fact_needs") or []
        ),
        "immutable": [
            "tool",
            "operation",
            "intent_binding",
            "target_binding",
            "required_facts_binding",
        ],
        "allowed_mutation": allowed_mutation,
        "expires": "turn_end",
    }
    return {
        **grant,
        "grant_id": _canonical_digest(grant),
    }


def dynamic_transient_recovery_plan(
    turn: WorkTurn,
) -> Optional[dict[str, Any]]:
    """Allow one bounded recovery for a safe, server-classified read failure.

    This function only authenticates the current failure.  The conversation
    loop owns the one-shot budget, so a second failed physical call always
    proceeds to the normal failure finalizer. Transient failures replay the
    exact owner-bound read. A query-shape failure is rebuilt against the
    semantic-only schema instead of replaying the known-bad arguments.
    """
    if (
        turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2
        or turn.fact_requirement is not None
        or turn.evidence
        or _hard_runtime_violation(turn)
    ):
        return None
    lifecycle = _validated_failure_lifecycle(
        turn,
        include_single_preaction=True,
    )
    if lifecycle is None:
        return None
    _, failures = lifecycle
    if len(failures) != 1:
        return None
    failure = failures[0]
    failed_calls = [
        call
        for call in turn.action_calls
        if call.call_id == failure.call_id
        and call.action_id == failure.action_id
    ]
    if len(failed_calls) != 1:
        return None
    failed_call = failed_calls[0]
    error_code = str(failure.error_code or "").strip().lower()
    contract = ACTION_OUTPUT_CONTRACTS.get(failure.action_id)
    _, index_items = _dynamic_index_binding(turn)
    correctable_arguments = bool(
        failure.action_id == "mystand_query"
        and error_code in _CORRECTABLE_QUERY_CODES
    )
    if (
        failure.status != "error"
        or contract is None
        # A failed index cannot safely "change path" because no trusted scope
        # exists yet; retrying it would still need another read + finalizer and
        # can only add cost.  Recovery is therefore limited to one failed read
        # after a complete owner-bound index.
        or contract.kind != "read"
        or not index_items
        or (
            error_code not in _TRANSIENT_RECOVERY_CODES
            and not correctable_arguments
        )
    ):
        return None
    if correctable_arguments:
        corrected_arguments = _corrected_semantic_query_arguments(
            failed_call
        )
        if corrected_arguments is None:
            return None
        correction = {
            "action_id": failed_call.action_id,
            "version": failed_call.version,
            "arguments": corrected_arguments,
            "arguments_digest": _canonical_digest(corrected_arguments),
            "allowed_fields": [
                "operation",
                "resource",
                "entities",
                "fact_needs",
                "mode",
            ],
            "required_fields": ["operation", "resource", "fact_needs"],
            "locator_rule": (
                "preserve the original resource or subject entities"
            ),
            "max_calls": 1,
        }
        grant = _recovery_grant(
            turn,
            failed_call,
            allowed_mutation="schema_only",
        )
        return {
            "grant": grant,
            "reason": "invalid_arguments",
            "mode": "correct_arguments",
            "state": "正文读取参数混入了当前阶段不允许的字段",
            # A semantic recovery needs only human-readable, current-user
            # scope. Internal resource identifiers remain server-side.
            "safe_scope": [
                {
                    "safeLabel": str(item["safeLabel"]),
                    "resourceType": str(item["resourceType"]),
                    "canRead": bool(item["canRead"]),
                    "locked": bool(item["locked"]),
                }
                for item in index_items
            ],
            "failed_tool_call": {
                "call_id": failed_call.call_id,
                "action_id": failed_call.action_id,
                "version": failed_call.version,
                "arguments": dict(failed_call.arguments),
            },
            "tool_result": _safe_recovery_tool_result(
                failure,
                reason="invalid_arguments",
                correction=correction,
            ),
            "correction": correction,
        }
    reason = (
        "timeout"
        if error_code in _TRANSIENT_TIMEOUT_CODES
        else "unavailable"
    )
    safe_scope = [
        {
            "resourceUid": str(item["resourceUid"]),
            "safeLabel": str(item["safeLabel"]),
            "resourceType": str(item["resourceType"]),
            "canRead": bool(item["canRead"]),
            "locked": bool(item["locked"]),
        }
        for item in index_items
    ]
    indexed_by_ref = {
        str(item.get("resourceUid") or ""): item
        for item in index_items
        if str(item.get("resourceUid") or "")
    }
    retry_refs: list[str] = []
    for key in ("resource_uid", "authorization_id"):
        value = failed_call.arguments.get(key)
        if isinstance(value, str) and value.strip():
            retry_refs.append(value.strip())
    for key in (
        "record_refs",
        "recordRefs",
        "resource_refs",
        "resourceRefs",
    ):
        values = failed_call.arguments.get(key)
        if isinstance(values, list):
            retry_refs.extend(
                value.strip()
                for value in values
                if isinstance(value, str) and value.strip()
            )
    retry_refs = sorted(set(retry_refs))
    if (
        not retry_refs
        or _indexed_read_refs(
            payload={},
            required_refs=retry_refs,
            indexed_by_ref=indexed_by_ref,
        )
        != retry_refs
    ):
        return None
    grant = _recovery_grant(
        turn,
        failed_call,
        allowed_mutation="exact_replay",
    )
    return {
        "grant": grant,
        "reason": reason,
        "state": (
            "上一次只读处理等待超时"
            if reason == "timeout"
            else "上一次只读处理遇到暂时不可用"
        ),
        "safe_scope": safe_scope,
        "failed_tool_call": {
            "call_id": failed_call.call_id,
            "action_id": failed_call.action_id,
            "version": failed_call.version,
            "arguments": dict(failed_call.arguments),
        },
        "tool_result": _safe_recovery_tool_result(
            failure,
            reason=reason,
        ),
        "retry": {
            "action_id": failed_call.action_id,
            "version": failed_call.version,
            "arguments": dict(failed_call.arguments),
            "arguments_digest": _canonical_digest(
                failed_call.arguments
            ),
        },
    }


def dynamic_transient_recovery_tool_call_valid(
    turn: WorkTurn,
    *,
    action_id: str,
    arguments: Mapping[str, Any],
) -> bool:
    """Validate the single physical read selected for one recovery."""
    plan = dynamic_transient_recovery_plan(turn)
    if (
        isinstance(plan, Mapping)
        and plan.get("mode") == "correct_arguments"
    ):
        correction = plan.get("correction")
        if (
            not isinstance(correction, Mapping)
            or str(action_id or "")
            != str(correction.get("action_id") or "")
            or not isinstance(arguments, Mapping)
        ):
            return False
        try:
            from tools.mystand_query_tool import (
                validate_mystand_semantic_query_plan,
            )

            normalized = validate_mystand_semantic_query_plan(
                dict(arguments)
            )
        except (TypeError, ValueError):
            return False
        return bool(
            normalized
            and _canonical_digest(normalized)
            == str(correction.get("arguments_digest") or "")
            and _canonical_digest(dict(arguments))
            == str(correction.get("arguments_digest") or "")
        )
    retry = plan.get("retry") if plan else None
    return bool(
        isinstance(retry, Mapping)
        and str(action_id or "") == str(retry.get("action_id") or "")
        and isinstance(arguments, Mapping)
        and _canonical_digest(dict(arguments))
        == str(retry.get("arguments_digest") or "")
    )


def dynamic_failure_presentation(turn: WorkTurn) -> Optional[dict[str, Any]]:
    """Project one truthful failure state from the immutable action ledger."""
    failures: list[Any] = []
    recovery_reason = ""
    no_progress = _validated_no_progress_failure(turn)
    if no_progress is not None:
        failure_class = "no_progress"
        failure_reason = str(no_progress["reason"])
    else:
        lifecycle = _validated_failure_lifecycle(
            turn,
            include_single_preaction=True,
        )
        if lifecycle is None:
            return None
        _, failures = lifecycle
        (
            failure_class,
            failure_reason,
            recovery_reason,
        ) = _failure_lifecycle_projection(turn, failures)
    # These are runtime status projections, not model prompt examples.  They
    # describe generic execution stages and safe error categories shared by
    # every My Stand read target; no business module, task wording, or Chinese
    # answer pattern participates in completion.
    process_states = {
        DYNAMIC_READ_NOT_DISPATCHED: (
            "资料定位已完成，但没有继续读取完成请求所需的内容"
        ),
        DYNAMIC_ACTION_NOT_DISPATCHED: "没有发起实际处理",
        DYNAMIC_READ_PRECONDITION_NOT_MET: (
            "资料定位没有完成，因此后续读取没有发起"
        ),
        DYNAMIC_ACTION_RESULT_MISSING: (
            "处理请求已生成，但没有形成可确认结果"
        ),
        DYNAMIC_INDEX_INCOMPLETE: (
            "资料定位已发起，但返回的目录不完整"
        ),
        "empty": "读取已发起，但没有取得可回答内容",
        "not_found": "没有找到能够唯一匹配的资料",
        "denied": "没有取得完成请求所需的读取权限",
        "ambiguous": "读取目标无法唯一确认",
        "timeout": "实际处理已发起，但等待结果超时",
        "unavailable": "实际处理已发起，但读取服务暂时不可用",
        "invalid_arguments": (
            "资料定位已完成，但读取所需内容时，参数混入了"
            "当前阶段不允许的字段"
        ),
        "cancelled": "实际处理已发起，但随后被停止",
        "mixed": "实际处理已发起，但其中有步骤没有形成可靠结果",
        "execution_error": (
            "资料读取已发起，但返回了错误；"
            "当前记录没有可安全确认的更细原因"
        ),
    }
    state = process_states.get(
        failure_reason,
        process_states["execution_error"],
    )
    final_causes = {
        DYNAMIC_READ_NOT_DISPATCHED: "资料目录查询后没有继续读取正文",
        DYNAMIC_ACTION_NOT_DISPATCHED: "实际处理没有发起",
        DYNAMIC_READ_PRECONDITION_NOT_MET: "正文读取的资料定位前提没有完成",
        DYNAMIC_ACTION_RESULT_MISSING: "处理请求没有形成可确认结果",
        DYNAMIC_INDEX_INCOMPLETE: "资料目录返回不完整",
        "empty": "读取没有取得可回答内容",
        "not_found": "没有找到能够唯一匹配的资料",
        "denied": "没有取得完成任务所需的读取权限",
        "ambiguous": "读取目标无法唯一确认",
        "timeout": "等待读取结果超时",
        "unavailable": "读取服务暂时不可用",
        "invalid_arguments": "正文读取参数混入了当前阶段不允许的字段",
        "cancelled": "实际处理随后被停止",
        "mixed": "执行步骤没有形成可靠结果",
        "execution_error": (
            "资料读取返回错误，当前记录没有可安全确认的更细原因"
        ),
    }
    final_cause = final_causes.get(
        failure_reason,
        final_causes["execution_error"],
    )
    failed_attempt_count = len(failures)
    recovery_attempted = bool(recovery_reason)
    missing = (
        "完成请求所需的可靠资料内容"
        if failure_reason in {
            "empty",
            "invalid_arguments",
            "timeout",
            "unavailable",
        }
        else "完成这项任务所需的可靠结果"
    )
    if recovery_attempted:
        recovery_states = {
            "invalid_arguments": (
                "第一次读取所需内容时，参数混入了当前阶段不允许的字段；"
                "去掉这些字段后又尝试了一次"
            ),
            "timeout": (
                "第一次读取所需内容时等待超时；按原目标又尝试了一次"
            ),
            "unavailable": (
                "第一次读取所需内容时服务暂时不可用；"
                "按原目标又尝试了一次"
            ),
        }
        terminal_states = {
            "invalid_arguments": (
                "第二次读取参数仍包含当前阶段不允许的字段，"
                "所需内容没有取得"
            ),
            "timeout": "第二次等待结果仍然超时，所需内容没有取得",
            "unavailable": "第二次读取服务仍不可用，所需内容没有取得",
            "empty": "第二次读取仍没有取得可回答内容",
            "not_found": "第二次读取没有找到匹配的资料",
            "ambiguous": "第二次读取仍无法唯一确认目标",
            "denied": "第二次读取没有取得所需权限",
            "cancelled": "第二次读取随后被停止",
            "execution_error": (
                "第二次读取又返回错误，当前记录没有可安全确认的"
                "更细原因"
            ),
            "mixed": "第二次读取仍没有形成可靠结果",
        }
        state = (
            "资料目录定位已完成；"
            f"{recovery_states[recovery_reason]}；"
            f"{terminal_states.get(failure_reason, terminal_states['mixed'])}"
        )
        final_cause = terminal_states.get(
            failure_reason,
            terminal_states["mixed"],
        )
    return {
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "recovery_reason": recovery_reason,
        "state": state,
        "final_cause": final_cause,
        "failed_attempt_count": failed_attempt_count,
        "recovery_attempted": recovery_attempted,
        "missing": missing,
    }


def dynamic_turn_outcome(turn: WorkTurn) -> Optional[dict[str, Any]]:
    """Derive the terminal machine truth from calls/results, never prose."""
    presentation = dynamic_failure_presentation(turn)
    if presentation is None:
        return None
    matched = _matched_action_lifecycle(turn) or []
    failed_event_ids = [
        result.call_id
        for _call, result in matched
        if result.status != "success"
    ]
    successful_actions = [
        call.action_id
        for call, result in matched
        if result.status == "success"
    ]
    final_event_id = failed_event_ids[-1] if failed_event_ids else ""
    target_descriptors: list[dict[str, Any]] = []
    for call, _result in matched:
        arguments = _mapping(call.arguments)
        locator = {
            key: arguments[key]
            for key in (
                "resource",
                "entities",
                "resource_uid",
                "authorization_id",
            )
            if key in arguments
        }
        if locator:
            target_descriptors.append(
                {
                    "action_id": call.action_id,
                    "locator": locator,
                }
            )
    outcome = {
        "schema": "xiaoban.turn-outcome.v1",
        "turn_id": turn.turn_id,
        "terminal_status": "failed",
        "intent_binding": str(
            turn.completion_binding.get("request_fingerprint") or ""
        ),
        "target_binding": (
            _canonical_digest(target_descriptors)
            if target_descriptors
            else ""
        ),
        "attempt_event_ids": failed_event_ids,
        "attempt_count": len(failed_event_ids),
        "completed_stages": successful_actions,
        "recovery": {
            "attempted": bool(presentation["recovery_attempted"]),
            "reason": str(presentation["recovery_reason"] or ""),
        },
        "final_cause": {
            "event_id": final_event_id,
            "code": str(presentation["failure_reason"]),
            "safe_message": str(presentation["final_cause"]),
        },
        "obtained": {
            "material": False,
            "evidence_refs": [],
        },
        "missing": [str(presentation["missing"])],
        "process_summary": str(presentation["state"]),
    }
    return {
        **outcome,
        "digest": _canonical_digest(outcome),
    }


def render_dynamic_failure_report(turn: WorkTurn) -> Optional[str]:
    """Render one generic human report from TurnOutcome, without model claims."""
    outcome = dynamic_turn_outcome(turn)
    if outcome is None or outcome.get("terminal_status") != "failed":
        return None
    process = str(outcome.get("process_summary") or "").rstrip("。；; ")
    missing_items = outcome.get("missing")
    missing = (
        str(missing_items[0])
        if isinstance(missing_items, list) and missing_items
        else "完成任务所需的可靠结果"
    )
    return (
        f"{process}。由于仍缺少{missing}，"
        "小伴这次无法给出可靠答复。"
    )


def _matched_action_lifecycle(
    turn: WorkTurn,
) -> Optional[list[tuple[Any, Any]]]:
    calls: dict[str, Any] = {}
    results: dict[str, Any] = {}
    for call in turn.action_calls:
        if call.call_id in calls:
            return None
        calls[call.call_id] = call
    for result in turn.action_results:
        if result.call_id not in calls:
            continue
        if result.call_id in results:
            return None
        results[result.call_id] = result
    if not calls or set(calls) != set(results):
        return None
    matched: list[tuple[Any, Any]] = []
    for call in turn.action_calls:
        call_id = call.call_id
        result = results[call_id]
        contract = ACTION_OUTPUT_CONTRACTS.get(call.action_id)
        if (
            contract is None
            or contract.version != call.version
            or result.action_id != call.action_id
            or result.started_at != call.requested_at
        ):
            return None
        matched.append((call, result))
    return matched


def _record_refs_for_paths(
    paths: Sequence[str],
    payload: Mapping[str, Any],
) -> list[str]:
    refs: list[str] = []
    for path in paths:
        if path == "recordRefs[]":
            refs.extend(
                str(value)
                for value in payload.get("recordRefs") or []
                if isinstance(value, str) and value
            )
        elif path == "resource.resourceUid":
            resource = payload.get("resource")
            if isinstance(resource, Mapping) and resource.get("resourceUid"):
                refs.append(str(resource["resourceUid"]))
        elif payload.get(path):
            refs.append(str(payload[path]))
    return refs


def _expected_read_record_refs(
    contract: Any,
    payload: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> list[str]:
    refs = _record_refs_for_paths(contract.record_ref_paths, payload)
    for key in ("resource_uid", "authorization_id"):
        if arguments.get(key):
            refs.append(str(arguments[key]))
    return sorted(set(refs))


def _reference_payload_valid(
    contract: Any,
    payload: Mapping[str, Any],
) -> bool:
    for path in contract.record_ref_paths:
        if path == "recordRefs[]" and "recordRefs" in payload:
            values = payload.get("recordRefs")
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, str) or not value
                    for value in values
                )
                or values != sorted(set(values))
            ):
                return False
        elif path == "resource.resourceUid":
            resource = payload.get("resource")
            if (
                isinstance(resource, Mapping)
                and "resourceUid" in resource
                and (
                    not isinstance(resource.get("resourceUid"), str)
                    or not resource.get("resourceUid")
                )
            ):
                return False
        elif path in payload and (
            not isinstance(payload.get(path), str)
            or not payload.get(path)
        ):
            return False
    return True


def _argument_resource_binding_valid(
    payload: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> bool:
    bindings = (
        (
            "resource_uid",
            [
                payload.get("resourceUid"),
                (
                    payload.get("resource", {}).get("resourceUid")
                    if isinstance(payload.get("resource"), Mapping)
                    else None
                ),
            ],
        ),
        ("authorization_id", [payload.get("authorizationId")]),
    )
    for argument_key, candidates in bindings:
        expected = arguments.get(argument_key)
        observed = [
            value
            for value in candidates
            if value not in (None, "")
        ]
        if expected not in (None, "") and observed and any(
            value != expected for value in observed
        ):
            return False
    return True


def _required_index_refs(
    contract: Any,
    payload: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> list[str]:
    index_paths = tuple(
        path
        for path in contract.record_ref_paths
        if "resource" in path.lower() or path == "recordRefs[]"
    )
    refs = _record_refs_for_paths(index_paths, payload)
    if arguments.get("resource_uid"):
        refs.append(str(arguments["resource_uid"]))
    return sorted(set(refs))


def _indexed_read_refs(
    *,
    payload: Mapping[str, Any],
    required_refs: Sequence[str],
    indexed_by_ref: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if any(ref not in indexed_by_ref for ref in required_refs):
        return []
    indexed_refs = sorted(set(required_refs))
    if not indexed_refs:
        resource = payload.get("resource")
        if not isinstance(resource, Mapping):
            return []
        display_name = resource.get("display_name")
        resource_type = resource.get("type")
        if (
            not isinstance(display_name, str)
            or not display_name.strip()
            or not isinstance(resource_type, str)
            or not resource_type.strip()
        ):
            return []
        matches = [
            ref
            for ref, item in indexed_by_ref.items()
            if item.get("safeLabel") == display_name
            and item.get("resourceType") == resource_type
        ]
        if len(matches) != 1:
            return []
        indexed_refs = matches
    if any(
        indexed_by_ref[ref].get("canRead") is not True
        or indexed_by_ref[ref].get("locked") is not False
        or indexed_by_ref[ref].get("status") == "locked"
        for ref in indexed_refs
    ):
        return []
    return indexed_refs


def _validated_read_evidence(
    turn: WorkTurn,
    matched: Sequence[tuple[Any, Any]],
    index_items: Sequence[Mapping[str, Any]],
) -> Optional[tuple[list[Any], list[str]]]:
    identity = turn.identity
    if identity is None:
        return None
    indexed_by_ref = {
        str(item.get("resourceUid")): item for item in index_items
    }
    evidence_by_call: dict[str, list[Any]] = {}
    for item in turn.evidence:
        evidence_by_call.setdefault(item.call_id, []).append(item)
    verified: list[Any] = []
    public_refs: set[str] = set()
    successful_reads = [
        (call, result)
        for call, result in matched
        if ACTION_OUTPUT_CONTRACTS[call.action_id].kind == "read"
        and result.status == "success"
    ]
    if not successful_reads:
        return None
    for call, result in successful_reads:
        contract = ACTION_OUTPUT_CONTRACTS[call.action_id]
        items = evidence_by_call.get(call.call_id, [])
        if len(items) != 1 or not result.raw_text:
            return None
        item = items[0]
        try:
            payload = json.loads(result.raw_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        if (
            not _reference_payload_valid(contract, payload)
            or not _argument_resource_binding_valid(
                payload,
                call.arguments,
            )
        ):
            return None
        expected_refs = _expected_read_record_refs(
            contract,
            payload,
            call.arguments,
        )
        required_refs = _required_index_refs(
            contract,
            payload,
            call.arguments,
        )
        indexed_refs = _indexed_read_refs(
            payload=payload,
            required_refs=required_refs,
            indexed_by_ref=indexed_by_ref,
        )
        if (
            not indexed_refs
            or _canonical_digest(payload)
            != _canonical_digest(result.normalized_payload)
            or item.turn_id != turn.turn_id
            or item.call_id != call.call_id
            or item.action_id != call.action_id
            or item.evidence_id
            != hashlib.sha256(
                f"{turn.turn_id}|{call.call_id}".encode("utf-8")
            ).hexdigest()[:16]
            or item.datascope_fingerprint
            != identity.datascope_fingerprint
            or item.status != "success"
            or item.verification_status != "verified"
            or item.verified_at != result.finished_at
            or item.allowed_facts
            != serialize_allowed_facts(call.action_id, dict(payload))
            or list(item.record_refs) != expected_refs
            or item.input_digest
            != hashlib.sha256(
                json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            or item.output_digest
            != hashlib.sha256(
                result.raw_text.encode("utf-8")
            ).hexdigest()
            or item.requirement_digest
            or item.coverage_digest
        ):
            return None
        verified.append(item)
        public_refs.update(indexed_refs)
    if len(verified) != len(turn.evidence):
        return None
    return verified, sorted(public_refs)


def _validated_transient_recovery_results(
    turn: WorkTurn,
    matched: Sequence[tuple[Any, Any]],
) -> Optional[list[Any]]:
    """Bind one recoverable failure to one later successful safe read."""
    non_success = [
        (call, result)
        for call, result in matched
        if result.status != "success"
    ]
    if not non_success:
        return []
    if len(non_success) != 1:
        return None
    failed_call, failed_result = non_success[0]
    failed_contract = ACTION_OUTPUT_CONTRACTS.get(failed_call.action_id)
    error_code = str(failed_result.error_code or "").strip().lower()
    correctable_arguments = bool(
        failed_call.action_id == "mystand_query"
        and error_code in _CORRECTABLE_QUERY_CODES
    )
    if (
        failed_result.status != "error"
        or failed_contract is None
        or failed_contract.kind != "read"
        or (
            error_code not in _TRANSIENT_RECOVERY_CODES
            and not correctable_arguments
        )
    ):
        return None
    ordered_calls = {
        call.call_id: index
        for index, call in enumerate(turn.action_calls)
    }
    failed_position = ordered_calls.get(failed_call.call_id)
    failed_arguments_digest = _canonical_digest(failed_call.arguments)
    if failed_position is None:
        return None
    post_failure = [
        (call, result)
        for call, result in matched
        if (
            ordered_calls.get(call.call_id, -1) > failed_position
        )
    ]
    if len(post_failure) != 1:
        return None
    recovered_call, recovered_result = post_failure[0]
    same_action = (
        recovered_result.status == "success"
        and recovered_call.action_id == failed_call.action_id
        and recovered_call.version == failed_call.version
    )
    if correctable_arguments and same_action:
        corrected_arguments = _corrected_semantic_query_arguments(
            failed_call
        )
        recovered = bool(
            corrected_arguments is not None
            and _canonical_digest(recovered_call.arguments)
            == _canonical_digest(corrected_arguments)
        )
    else:
        recovered = bool(
            same_action
            and _canonical_digest(recovered_call.arguments)
            == failed_arguments_digest
        )
    return [failed_result] if recovered else None


def _dynamic_evidence_completion(
    turn: WorkTurn,
    final_text: str,
    *,
    user_message: Any = None,
) -> CompletionDecision:
    """Authenticate successful reads while preserving the model's answer."""
    identity = turn.identity
    output = _validated_terminal_text(final_text)
    index_receipt_digest, index_items = _dynamic_index_binding(turn)
    matched = _matched_action_lifecycle(turn)
    if (
        identity is None
        or not output
        or not _completion_binding_valid(turn)
        or not index_receipt_digest
        or _hard_runtime_violation(turn)
        or matched is None
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_evidence_binding",
        )
    validated = _validated_read_evidence(turn, matched, index_items)
    if validated is None:
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_action_binding",
        )
    verified_evidence, record_refs = validated
    transient_failures = _validated_transient_recovery_results(
        turn,
        matched,
    )
    if transient_failures is None:
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_recovery_binding",
        )
    _ = user_message  # Intent is interpreted by the model, not a keyword list.
    system_receipt = not _answer_uses_bound_evidence(turn, output)
    if system_receipt:
        output = (
            "资料已经读取成功，但最终回答没有使用本轮资料中的具体内容，"
            "无法确认它真正完成了你的要求。本次任务仍按未完成处理。"
        )
    receipt = turn.index_receipt
    verification = {
        **_completion_receipt(
            turn,
            completion_kind="evidence-bound",
            action_count=len(matched),
            evidence_count=len(verified_evidence),
            output=output,
            decision="evidence_access_verified",
        ),
        "index_count": receipt.resource_count,
        "index_resource_refs_digest": receipt.resource_refs_digest,
        "index_has_more": receipt.has_more,
        "index_receipt_digest": index_receipt_digest,
        "record_refs": record_refs,
        "record_refs_digest": _canonical_digest(record_refs),
        "evidence_digest": evidence_receipt_digest(verified_evidence),
        **(
            {
                "output_presentation": "system-receipt",
                "answer_status": "incomplete",
            }
            if system_receipt
            else {}
        ),
    }
    if transient_failures:
        verification.update(
            {
                "transient_failure_count": len(transient_failures),
                "transient_action_result_digest": _canonical_digest(
                    [
                        {
                            "call_id": result.call_id,
                            "action_id": result.action_id,
                            "status": result.status,
                            "error_code": result.error_code,
                            "payload_digest": _canonical_digest(
                                result.normalized_payload
                            ),
                            "started_at": result.started_at,
                            "finished_at": result.finished_at,
                        }
                        for result in transient_failures
                    ]
                ),
            }
        )
    return CompletionDecision(
        True,
        output,
        (
            "evidence_answer_incomplete_system_receipt"
            if system_receipt
            else "evidence_access_verified"
        ),
        verification,
    )


def _dynamic_failure_completion(
    turn: WorkTurn,
    final_text: str,
) -> CompletionDecision:
    """Bind a natural explanation to real non-success execution state."""
    output = _validated_terminal_text(final_text)
    failure_lifecycle = _validated_failure_lifecycle(
        turn,
        include_single_preaction=True,
    )
    no_progress_failure = _validated_no_progress_failure(turn)
    if (
        getattr(turn, "completion_finalization", "") != "failure"
        or not output
        or not _completion_binding_valid(turn)
        or _hard_runtime_violation(turn)
        or turn.evidence
        or (
            failure_lifecycle is None
            and no_progress_failure is None
        )
        or getattr(
            turn,
            "completion_finalization_output_digest",
            "",
        )
        != hashlib.sha256(output.encode("utf-8")).hexdigest()
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_failure_binding",
        )
    if no_progress_failure is not None:
        action_count = no_progress_failure["action_count"]
        failure_class = "no_progress"
        recovery_reason = ""
        action_result_digest = _canonical_digest(no_progress_failure)
        failed_action_count = 0
    else:
        action_count, failures = failure_lifecycle
        (
            failure_class,
            _,
            recovery_reason,
        ) = _failure_lifecycle_projection(
            turn,
            failures,
        )
        action_result_digest = _canonical_digest(
            [
                {
                    "call_id": result.call_id,
                    "action_id": result.action_id,
                    "status": result.status,
                    "error_code": result.error_code,
                    "payload_digest": _canonical_digest(
                        result.normalized_payload
                    ),
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                }
                for result in failures
            ]
        )
        failed_action_count = len(failures)
    presentation = dynamic_failure_presentation(turn)
    if (
        presentation is None
        or presentation["failure_class"] != failure_class
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_failure_binding",
        )
    failure_reason = presentation["failure_reason"]
    outcome = dynamic_turn_outcome(turn)
    runtime_output = render_dynamic_failure_report(turn)
    if outcome is None or not runtime_output:
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_failure_outcome",
        )
    # Execution facts are authenticated by TurnOutcome.  The model's Chinese
    # is never used as evidence for attempt count, cause, recovery or missing
    # material; the public failure report is rendered from those machine facts.
    output = runtime_output
    verification = {
        **_completion_receipt(
            turn,
            completion_kind="failure-bound",
            action_count=action_count,
            evidence_count=0,
            output=output,
            decision="execution_status_bound",
        ),
        "action_result_digest": action_result_digest,
        "failed_action_count": failed_action_count,
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "turn_outcome": outcome,
        "turn_outcome_digest": str(outcome["digest"]),
        **(
            {"recovery_reason": recovery_reason}
            if recovery_reason
            else {}
        ),
        "output_presentation": "system-receipt",
        "answer_status": "incomplete",
    }
    return CompletionDecision(
        True,
        output,
        "execution_status_system_receipt",
        verification,
    )


def _validated_failure_lifecycle(
    turn: WorkTurn,
    *,
    include_single_preaction: bool,
) -> Optional[tuple[int, list[Any]]]:
    """Return server-recorded failures without treating zero work as failure."""
    matched = _matched_action_lifecycle(turn)
    if turn.action_calls and matched is None:
        return None
    matched = matched or []
    _ = include_single_preaction  # retained for call-site compatibility
    matched_failures = [
        result
        for call, result in matched
        if ACTION_OUTPUT_CONTRACTS[call.action_id].kind in {"index", "read"}
        and result.status != "success"
    ]
    # A natural failure reply may describe only a handler that really ran and
    # returned a bound non-success result.  PreAction denials are protocol
    # errors/no-dispatch states, never user-facing execution evidence.
    action_order = {
        call.call_id: index
        for index, call in enumerate(turn.action_calls)
    }
    failures = sorted(
        matched_failures,
        key=lambda result: action_order.get(result.call_id, -1),
    )
    if not failures:
        return None
    return len(matched), failures


def _validated_no_progress_failure(
    turn: WorkTurn,
) -> Optional[dict[str, Any]]:
    """Authenticate a server-observed no-dispatch execution failure."""
    reason = getattr(turn, "completion_execution_failure", "")
    if (
        reason not in {
            DYNAMIC_READ_NOT_DISPATCHED,
            DYNAMIC_ACTION_NOT_DISPATCHED,
            DYNAMIC_READ_PRECONDITION_NOT_MET,
            DYNAMIC_ACTION_RESULT_MISSING,
            DYNAMIC_INDEX_INCOMPLETE,
        }
        or turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2
        or turn.fact_requirement is not None
        or turn.evidence
        or _hard_runtime_violation(turn)
    ):
        return None
    if reason == DYNAMIC_ACTION_NOT_DISPATCHED:
        if (
            turn.interaction_kind != "WORK"
            or turn.index_receipt is not None
            or turn.action_calls
            or turn.action_results
        ):
            return None
        return {
            "schema": "mystand.dynamic-execution-failure.v1",
            "reason": DYNAMIC_ACTION_NOT_DISPATCHED,
            "action_count": 0,
        }
    if reason == DYNAMIC_READ_PRECONDITION_NOT_MET:
        denials = list(turn.action_results)
        if (
            turn.interaction_kind != "WORK"
            or turn.action_calls
            or not denials
            or turn.pre_action_denials != len(denials)
            or any(
                result.status != "denied"
                or result.error_code != "missing_index_receipt"
                or ACTION_OUTPUT_CONTRACTS.get(result.action_id) is None
                or ACTION_OUTPUT_CONTRACTS[result.action_id].kind != "read"
                for result in denials
            )
        ):
            return None
        return {
            "schema": "mystand.dynamic-execution-failure.v1",
            "reason": DYNAMIC_READ_PRECONDITION_NOT_MET,
            "action_count": 0,
            "denial_digest": _canonical_digest(
                [
                    {
                        "call_id": result.call_id,
                        "action_id": result.action_id,
                        "status": result.status,
                        "error_code": result.error_code,
                    }
                    for result in denials
                ]
            ),
        }
    if reason == DYNAMIC_ACTION_RESULT_MISSING:
        calls = list(turn.action_calls)
        if not calls:
            return None
        call_ids = {call.call_id for call in calls}
        result_ids = [result.call_id for result in turn.action_results]
        if (
            len(call_ids) != len(calls)
            or len(result_ids) != len(set(result_ids))
            or any(result_id not in call_ids for result_id in result_ids)
            or call_ids == set(result_ids)
        ):
            return None
        return {
            "schema": "mystand.dynamic-execution-failure.v1",
            "reason": DYNAMIC_ACTION_RESULT_MISSING,
            "action_count": len(calls),
            "lifecycle_digest": _canonical_digest(
                {
                    "calls": [
                        {
                            "call_id": call.call_id,
                            "action_id": call.action_id,
                            "version": call.version,
                        }
                        for call in calls
                    ],
                    "results": [
                        {
                            "call_id": result.call_id,
                            "action_id": result.action_id,
                            "status": result.status,
                            "error_code": result.error_code,
                        }
                        for result in turn.action_results
                    ],
                }
            ),
        }
    if reason == DYNAMIC_INDEX_INCOMPLETE:
        matched = _matched_action_lifecycle(turn)
        receipt = turn.index_receipt
        if (
            matched is None
            or not matched
            or receipt is None
            or receipt.status != "unavailable"
            or any(
                call.action_id != "mystand_resource_index"
                or result.status != "success"
                for call, result in matched
            )
        ):
            return None
        return {
            "schema": "mystand.dynamic-execution-failure.v1",
            "reason": DYNAMIC_INDEX_INCOMPLETE,
            "action_count": len(matched),
            "index_receipt_digest": _canonical_digest(_mapping(receipt)),
        }
    index_receipt_digest, _ = _dynamic_index_binding(turn)
    matched = _matched_action_lifecycle(turn)
    if (
        not index_receipt_digest
        or matched is None
        or not matched
        or any(
            call.action_id != "mystand_resource_index"
            or result.status != "success"
            for call, result in matched
        )
        or any(
            ACTION_OUTPUT_CONTRACTS.get(call.action_id)
            and ACTION_OUTPUT_CONTRACTS[call.action_id].kind == "read"
            for call in turn.action_calls
        )
    ):
        return None
    return {
        "schema": "mystand.dynamic-execution-failure.v1",
        "reason": reason,
        "action_count": len(matched),
        "index_receipt_digest": index_receipt_digest,
    }


def mark_dynamic_read_no_progress(turn: WorkTurn) -> bool:
    """Mark a complete index lookup that never dispatched the required read."""
    previous = getattr(turn, "completion_execution_failure", "")
    turn.completion_execution_failure = DYNAMIC_READ_NOT_DISPATCHED
    if _validated_no_progress_failure(turn) is not None:
        return True
    turn.completion_execution_failure = previous
    return False


def mark_dynamic_action_no_progress(turn: WorkTurn) -> bool:
    """Mark a trusted work turn where no site action was dispatched."""
    previous = getattr(turn, "completion_execution_failure", "")
    turn.completion_execution_failure = DYNAMIC_ACTION_NOT_DISPATCHED
    if _validated_no_progress_failure(turn) is not None:
        return True
    turn.completion_execution_failure = previous
    return False


def mark_dynamic_execution_no_progress(turn: WorkTurn) -> bool:
    """Authenticate every safe unfinished lifecycle before finalization."""
    previous = getattr(turn, "completion_execution_failure", "")
    candidates = (
        DYNAMIC_READ_NOT_DISPATCHED,
        DYNAMIC_READ_PRECONDITION_NOT_MET,
        DYNAMIC_ACTION_RESULT_MISSING,
        DYNAMIC_INDEX_INCOMPLETE,
        DYNAMIC_ACTION_NOT_DISPATCHED,
    )
    for reason in candidates:
        turn.completion_execution_failure = reason
        if _validated_no_progress_failure(turn) is not None:
            return True
    turn.completion_execution_failure = previous
    return False


def dynamic_finalization_mode(
    turn: WorkTurn,
    *,
    include_single_preaction: bool = False,
) -> str:
    """Derive whether the next paid call must be the no-tool final reply."""
    if (
        turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2
        or turn.fact_requirement is not None
        or not _completion_binding_valid(turn)
        or _hard_runtime_violation(turn)
    ):
        return ""
    matched = _matched_action_lifecycle(turn)
    if turn.evidence and matched is not None:
        index_receipt_digest, index_items = _dynamic_index_binding(turn)
        if (
            index_receipt_digest
            and _validated_read_evidence(turn, matched, index_items)
            is not None
        ):
            return "evidence"
    if not turn.evidence and _validated_failure_lifecycle(
        turn,
        include_single_preaction=include_single_preaction,
    ) is not None:
        return "failure"
    if not turn.evidence and _validated_no_progress_failure(turn) is not None:
        return "failure"
    return ""


def check_dynamic_completion(
    turn: WorkTurn,
    *,
    final_text: str,
    failure_message: str,
    user_message: Any = None,
) -> Optional[CompletionDecision]:
    """Return None when legacy completion routing must continue."""
    if turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2:
        return None
    if turn.fact_requirement is not None:
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_completion_protocol_mixed",
        )
    if getattr(turn, "completion_finalization", "") == "not_executed":
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_not_executed",
        )
    if (
        getattr(turn, "completion_finalization", "") == "failure"
        and _validated_no_progress_failure(turn) is not None
    ):
        return _dynamic_failure_completion(turn, final_text)
    action_ids = {
        call.action_id for call in turn.action_calls
    } | {
        result.action_id for result in turn.action_results
    }
    if not action_ids.intersection(_DYNAMIC_ACTION_IDS):
        return None
    if turn.evidence:
        return _dynamic_evidence_completion(
            turn,
            final_text,
            user_message=user_message,
        )
    if getattr(turn, "completion_finalization", "") == "failure":
        return _dynamic_failure_completion(turn, final_text)
    if not turn.action_calls:
        reason = "blocked_no_action_call"
    elif not turn.action_results:
        reason = "blocked_no_action_result"
    else:
        reason = "blocked_no_evidence"
    return CompletionDecision(False, failure_message, reason)


def validate_dynamic_result_protocol(
    result: Mapping[str, Any],
    signed_fact_requirement: Any,
) -> tuple[bool, Optional[CompletionDecision]]:
    """Validate top-level result routing before a trusted turn is accepted."""
    completion_protocol = str(
        result.get("_mystand_completion_protocol") or ""
    )
    dynamic_completion = (
        completion_protocol == MYSTAND_COMPLETION_PROTOCOL_V2
    )
    if completion_protocol and not dynamic_completion:
        return dynamic_completion, CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_completion_protocol",
        )
    if dynamic_completion and isinstance(signed_fact_requirement, Mapping):
        return dynamic_completion, CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_completion_protocol_mixed",
        )
    return dynamic_completion, None


def dynamic_result_turn_binding_valid(
    result: Mapping[str, Any],
    turn: WorkTurn,
) -> bool:
    return bool(
        turn.completion_protocol
        == str(result.get("_mystand_completion_protocol") or "")
        and _mapping(turn.completion_binding)
        == _mapping(result.get("_mystand_completion_binding"))
        and _completion_binding_valid(turn)
    )
