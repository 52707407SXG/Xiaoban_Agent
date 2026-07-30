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
import hashlib
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional, Sequence

from xiaoban.trusted_runtime.dynamic_completion import (
    NO_EVIDENCE_MESSAGE,
    check_dynamic_completion,
    dynamic_result_turn_binding_valid,
    evidence_receipt_digest as _evidence_receipt_digest,
    validate_dynamic_result_protocol,
)
from xiaoban.trusted_runtime.turns import (
    build_work_turn,
    result_has_write_actions,
    serialize_allowed_facts,
)
from xiaoban.trusted_runtime.fact_contract import (
    SIGNED_FACT_INDEX_MAX_ITEMS,
    SIGNED_FACT_INDEX_MAX_PAGES,
    SIGNED_FACT_INDEX_PAGE_LIMIT,
    canonical_digest,
    evidence_requirement_digest,
    resource_read_record_refs_valid,
)
from xiaoban.trusted_runtime.types import (
    CompletionDecision,
    TrustedIdentity,
    WorkTurn,
    INTERACTION_CHAT,
)

# 用户可见安全文案：自然、简短，不含内部 ID、规则名或技术栈。
INCOMPLETE_COLLECTION_MESSAGE = (
    "这轮没有取得完整且可验证的站内证据，不能给出这项业务事实。"
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


def _blocked_collection(reason: str) -> CompletionDecision:
    return CompletionDecision(False, INCOMPLETE_COLLECTION_MESSAGE, reason)


def _receipt_coverage(
    requirement: Mapping[str, Any],
    requirement_binding: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project only the completeness fields needed by My Stand's ledger."""
    raw = coverage.get("server_coverage")
    raw = raw if isinstance(raw, Mapping) else {}
    projected: Dict[str, Any] = {
        "expectedCount": raw.get(
            "expectedCount",
            coverage.get("expected_count"),
        ),
        "returnedCount": raw.get(
            "returnedCount",
            coverage.get("actual_count"),
        ),
        "hasMore": raw.get("hasMore", coverage.get("has_more")),
        "expectedResourceRefsDigest": raw.get(
            "expectedResourceRefsDigest",
            coverage.get("expected_digest"),
        ),
        "returnedResourceRefsDigest": raw.get(
            "returnedResourceRefsDigest",
            coverage.get("actual_digest"),
        ),
        "scopeFingerprint": raw.get(
            "scopeFingerprint",
            requirement_binding.get("datascope_fingerprint"),
        ),
        "complete": raw.get(
            "complete",
            coverage.get("status") == "complete",
        ),
    }
    if isinstance(raw.get("year"), int) and not isinstance(raw.get("year"), bool):
        projected["year"] = raw["year"]
    tie_rule = raw.get("tieRule")
    if isinstance(tie_rule, str) and tie_rule:
        projected["tieRule"] = tie_rule
    return projected


def _coverage_from_result(payload: Any) -> Mapping[str, Any]:
    normalized = _mapping(payload)
    nested = normalized.get("collection_evidence")
    if not isinstance(nested, Mapping):
        nested = normalized.get("collectionEvidence")
    if isinstance(nested, Mapping):
        return nested
    if normalized.get("schema") == "mystand.collection-evidence.v1":
        return normalized
    return {}


def _verified_index_receipt_digest(
    turn: WorkTurn,
    requirement: Mapping[str, Any],
) -> str:
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
        or receipt.resource_count != len(set(receipt.matched_resource_refs))
        or receipt.resource_refs_digest
        != _canonical_digest(sorted(set(receipt.matched_resource_refs)))
        or requirement.get("index_count") != receipt.resource_count
        or requirement.get("index_has_more") is not receipt.has_more
        or requirement.get("index_resource_refs_digest")
        != receipt.resource_refs_digest
    ):
        return ""
    matching_calls = [
        call
        for call in turn.action_calls
        if call.call_id == receipt.source_call_id
        and call.action_id == "mystand_resource_index"
    ]
    matching_results = [
        result
        for result in turn.action_results
        if result.call_id == receipt.source_call_id
        and result.action_id == "mystand_resource_index"
        and result.status == "success"
    ]
    if len(matching_calls) != 1 or len(matching_results) != 1:
        return ""
    call = matching_calls[0]
    result = matching_results[0]
    expected_arguments = {
        "operation": "list_resources",
        "module_id": str(requirement.get("module_id") or ""),
        "status": "all",
        "limit": SIGNED_FACT_INDEX_PAGE_LIMIT,
    }
    if (
        call.version != "v1"
        or call.arguments != expected_arguments
        or result.started_at != call.requested_at
        or result.finished_at != receipt.loaded_at
    ):
        return ""
    try:
        raw_payload = json.loads(result.raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if (
        not isinstance(raw_payload, Mapping)
        or _canonical_digest(raw_payload)
        != _canonical_digest(result.normalized_payload)
    ):
        return ""
    raw_items = raw_payload.get("items")
    raw_schema = raw_payload.get("schema")
    if (
        raw_payload.get("ok") is not True
        or raw_schema not in {
            "mystand.resource-index.page.v1",
            "mystand.resource-index.complete.v1",
        }
        or not isinstance(raw_items, list)
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("resourceUid"), str)
            or not item.get("resourceUid")
            for item in raw_items
        )
        or (
            not raw_items
            and requirement.get("fact_kind") != "collection"
        )
        or (
            raw_schema == "mystand.resource-index.complete.v1"
            and (
                isinstance(raw_payload.get("pageCount"), bool)
                or not isinstance(raw_payload.get("pageCount"), int)
                or raw_payload.get("pageCount") < 1
                or raw_payload.get("pageCount")
                > SIGNED_FACT_INDEX_MAX_PAGES
                or len(raw_items) > SIGNED_FACT_INDEX_MAX_ITEMS
            )
        )
        or (
            raw_schema == "mystand.resource-index.page.v1"
            and len(raw_items) > SIGNED_FACT_INDEX_PAGE_LIMIT
        )
    ):
        return ""
    listed_resource_refs = [
        str(item.get("resourceUid"))
        for item in raw_items
        if isinstance(item, Mapping) and item.get("resourceUid")
    ]
    raw_resource_refs = sorted(set(listed_resource_refs))
    if (
        len(raw_resource_refs) != len(listed_resource_refs)
        or raw_resource_refs != sorted(set(receipt.matched_resource_refs))
        or raw_payload.get("hasMore") is not receipt.has_more
        or raw_payload.get("nextCursor") != ""
    ):
        return ""
    return _canonical_digest(
        {
            "receipt": _mapping(receipt),
            "action_call": {
                "call_id": call.call_id,
                "action_id": call.action_id,
                "version": call.version,
                "arguments": call.arguments,
                "requested_at": call.requested_at,
            },
            "action_result": {
                "call_id": result.call_id,
                "action_id": result.action_id,
                "status": result.status,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "output_digest": hashlib.sha256(
                    result.raw_text.encode("utf-8")
                ).hexdigest(),
            },
        }
    )


def _collection_completion(turn: WorkTurn) -> CompletionDecision:
    """Validate one complete collection proof and project only server facts."""
    requirement = _mapping(getattr(turn, "fact_requirement", None))
    coverage = _mapping(getattr(turn, "collection_evidence", None))
    if not turn.action_calls:
        return _blocked_collection("blocked_collection_no_action")
    if (
        turn.state == "cancelled"
        or coverage.get("status") in {"cancelled", "canceled"}
        or any(
            result.status == "cancelled"
            for result in turn.action_results
        )
    ):
        return _blocked_collection("blocked_collection_cancelled")
    if not coverage:
        return _blocked_collection("blocked_collection_incomplete")
    if coverage.get("schema") != "mystand.collection-evidence.v1":
        return _blocked_collection("blocked_collection_schema")

    signed_index_receipt_digest = ""
    if turn.fact_requirement_digest:
        if turn.fact_requirement_digest != _canonical_digest(requirement):
            return _blocked_collection("blocked_collection_binding")
        if (
            len(turn.action_calls) != 2
            or len(turn.action_results) != 2
            or len(turn.evidence) != 1
        ):
            return _blocked_collection("blocked_collection_action_count")
        signed_index_receipt_digest = _verified_index_receipt_digest(
            turn,
            requirement,
        )
        if not signed_index_receipt_digest:
            return _blocked_collection("blocked_collection_index_receipt")

    requirement_digest = evidence_requirement_digest(
        requirement,
        canonical_fallback=turn.fact_requirement_digest,
    )
    if coverage.get("requirement_digest") != requirement_digest:
        return _blocked_collection("blocked_collection_digest")
    requirement_binding = _mapping(requirement.get("binding"))
    coverage_binding = _mapping(coverage.get("binding"))
    if not requirement_binding or dict(coverage_binding) != dict(requirement_binding):
        return _blocked_collection("blocked_collection_binding")
    identity = turn.identity
    if (
        identity is None
        or requirement_binding.get("user_id") != identity.account_id
        or requirement_binding.get("datascope_fingerprint")
        != identity.datascope_fingerprint
        or requirement_binding.get("delivery_id") != turn.request_id
        or requirement_binding.get("message_id") != turn.message_id
    ):
        return _blocked_collection("blocked_collection_binding")

    source_call_ids = coverage.get("source_call_ids")
    if (
        not isinstance(source_call_ids, list)
        or not source_call_ids
        or len(source_call_ids) != len(set(source_call_ids))
        or any(not isinstance(call_id, str) or not call_id for call_id in source_call_ids)
    ):
        return _blocked_collection("blocked_collection_action_binding")
    if turn.fact_requirement_digest and (
        len(source_call_ids) != 1
        or turn.index_receipt is None
        or {
            call.call_id
            for call in turn.action_calls
        }
        != {
            turn.index_receipt.source_call_id,
            source_call_ids[0],
        }
    ):
        return _blocked_collection("blocked_collection_action_binding")

    record_refs: List[str] = []
    receipt_evidence: List[Any] = []
    for call_id in source_call_ids:
        calls = [
            call
            for call in turn.action_calls
            if call.call_id == call_id and call.action_id == "mystand_query"
        ]
        results = [
            result
            for result in turn.action_results
            if result.call_id == call_id and result.action_id == "mystand_query"
        ]
        evidence_items = [
            item
            for item in turn.evidence
            if item.call_id == call_id and item.action_id == "mystand_query"
        ]
        if len(calls) != 1 or len(results) != 1:
            return _blocked_collection("blocked_collection_action_binding")
        if results[0].status == "cancelled":
            return _blocked_collection("blocked_collection_cancelled")
        if results[0].status != "success":
            return _blocked_collection("blocked_collection_incomplete")
        result_coverage = _coverage_from_result(results[0].normalized_payload)
        if not result_coverage or _canonical_digest(result_coverage) != _canonical_digest(
            coverage
        ):
            return _blocked_collection("blocked_collection_digest")
        if not evidence_items:
            return _blocked_collection("blocked_collection_no_evidence")
        for item in evidence_items:
            if (
                item.turn_id != turn.turn_id
                or item.datascope_fingerprint != identity.datascope_fingerprint
                or item.status != "success"
                or item.verification_status != "verified"
            ):
                return _blocked_collection("blocked_collection_binding")
            if turn.fact_requirement_digest:
                if (
                    item.requirement_digest != requirement_digest
                    or item.coverage_digest != _canonical_digest(coverage)
                    or item.input_digest
                    != hashlib.sha256(
                        json.dumps(
                            calls[0].arguments,
                            ensure_ascii=False,
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest()
                    or item.output_digest
                    != hashlib.sha256(
                        results[0].raw_text.encode("utf-8")
                    ).hexdigest()
                ):
                    return _blocked_collection("blocked_collection_digest")
            elif (
                item.input_digest != requirement_digest
                or item.output_digest != _canonical_digest(coverage)
            ):
                # Compatibility for the committed low-level RED fixture.  A
                # server-signed turn always takes the stricter branch above.
                return _blocked_collection("blocked_collection_digest")
            if not isinstance(item.record_refs, list):
                return _blocked_collection("blocked_collection_digest")
            record_refs.extend(str(ref) for ref in item.record_refs)
            receipt_evidence.append(item)

    if coverage.get("status") != "complete":
        return _blocked_collection("blocked_collection_incomplete")
    expected_count = coverage.get("expected_count")
    actual_count = coverage.get("actual_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
        or isinstance(actual_count, bool)
        or not isinstance(actual_count, int)
        or actual_count < 0
        or expected_count != actual_count
        or actual_count != len(record_refs)
    ):
        return _blocked_collection("blocked_collection_incomplete")
    if coverage.get("has_more") is not False:
        return _blocked_collection("blocked_collection_has_more")
    expected_digest = coverage.get("expected_digest")
    actual_digest = coverage.get("actual_digest")
    if (
        not isinstance(expected_digest, str)
        or not isinstance(actual_digest, str)
        or len(expected_digest) != 64
        or len(actual_digest) != 64
        or expected_digest != actual_digest
        or actual_digest != _canonical_digest(record_refs)
    ):
        return _blocked_collection("blocked_collection_digest")

    projected_facts = _mapping(coverage.get("projected_facts"))
    projected_text = coverage.get("projected_text")
    if (
        not projected_facts
        or not isinstance(projected_text, str)
        or not projected_text.strip()
        or projected_facts.get("operation") != requirement.get("operation")
        or (
            requirement.get("ordinal") is not None
            and projected_facts.get("ordinal") != requirement.get("ordinal")
        )
        or (
            requirement.get("metric")
            and projected_facts.get("metric") != requirement.get("metric")
        )
        or (
            requirement.get("time_scope")
            and projected_facts.get("time_scope") != requirement.get("time_scope")
        )
    ):
        return _blocked_collection("blocked_collection_projection")

    receipt_coverage = _receipt_coverage(
        requirement,
        requirement_binding,
        coverage,
    )
    final_projection = projected_text.strip()
    successful_action_count = len(
        {
            (result.call_id, result.action_id)
            for result in turn.action_results
            if result.status == "success"
        }
    )
    verification = {
        "schema": "mystand.xiaoban-fact-verification.v1",
        "verified": True,
        "request_id": turn.request_id,
        "delivery_id": str(requirement_binding.get("delivery_id") or ""),
        "attempt": requirement_binding.get("attempt"),
        "message_id": turn.message_id,
        "request_fingerprint": str(
            requirement_binding.get("request_fingerprint") or ""
        ),
        "plan_id": str(
            requirement.get("plan_id")
            or projected_facts.get("plan_id")
            or f"fact-{requirement_digest[:24]}"
        ),
        "requirement_digest": requirement_digest,
        "coverage": dict(receipt_coverage),
        "coverage_digest": _canonical_digest(receipt_coverage),
        "action_count": successful_action_count,
        "evidence_count": len(receipt_evidence),
        "evidence_digest": _evidence_receipt_digest(receipt_evidence),
        "output_digest": hashlib.sha256(
            final_projection.encode("utf-8")
        ).hexdigest(),
        "decision": "projected_complete_collection",
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "datascope_fingerprint": identity.datascope_fingerprint,
    }
    if signed_index_receipt_digest:
        verification["index_receipt_digest"] = signed_index_receipt_digest
        verification["index_count"] = requirement.get("index_count")
        verification["index_resource_refs_digest"] = requirement.get(
            "index_resource_refs_digest"
        )
        verification["index_has_more"] = requirement.get("index_has_more")
    index_evidence_digest = requirement.get("index_evidence_digest")
    if isinstance(index_evidence_digest, str) and index_evidence_digest:
        verification["index_evidence_digest"] = index_evidence_digest
    return CompletionDecision(
        True,
        final_projection,
        "projected_complete_collection",
        verification,
    )


def _generic_fact_completion(turn: WorkTurn) -> CompletionDecision:
    """Validate a signed single-resource fact turn and issue its terminal receipt."""
    requirement = _mapping(getattr(turn, "fact_requirement", None))
    binding = _mapping(requirement.get("binding"))
    identity = turn.identity
    if (
        identity is None
        or not binding
        or binding.get("user_id") != identity.account_id
        or binding.get("datascope_fingerprint") != identity.datascope_fingerprint
        or binding.get("delivery_id") != turn.request_id
        or binding.get("message_id") != turn.message_id
    ):
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_binding",
        )
    if (
        not turn.fact_requirement_digest
        or turn.fact_requirement_digest != _canonical_digest(requirement)
    ):
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_binding",
        )
    index_receipt_digest = _verified_index_receipt_digest(
        turn,
        requirement,
    )
    if not index_receipt_digest:
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_index_receipt",
        )

    successful_results = {
        (result.call_id, result.action_id)
        for result in turn.action_results
        if result.status == "success"
    }
    successful_calls = [
        call
        for call in turn.action_calls
        if (call.call_id, call.action_id) in successful_results
    ]
    if (
        len(turn.action_calls) != 2
        or len(turn.action_results) != 2
        or len(turn.evidence) != 1
        or len(successful_calls) != 2
    ):
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_action_count",
        )
    if not any(call.action_id == "mystand_resource_index" for call in successful_calls):
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_index_action",
        )
    if {
        call.action_id
        for call in successful_calls
    } != {"mystand_resource_index", "mystand_query"}:
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_read_action",
        )
    verified_evidence = [
        item
        for item in turn.evidence
        if item.action_id == "mystand_query"
        and item.status == "success"
        and item.verification_status == "verified"
        and item.turn_id == turn.turn_id
        and item.datascope_fingerprint == identity.datascope_fingerprint
        and (item.call_id, item.action_id) in successful_results
    ]
    if len(verified_evidence) != 1:
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_evidence",
        )
    query_evidence = verified_evidence[0]
    query_calls = [
        call
        for call in successful_calls
        if call.action_id == "mystand_query"
        and call.call_id == query_evidence.call_id
    ]
    query_results = [
        result
        for result in turn.action_results
        if result.action_id == "mystand_query"
        and result.call_id == query_evidence.call_id
        and result.status == "success"
    ]
    requirement_digest = evidence_requirement_digest(
        requirement,
        canonical_fallback=turn.fact_requirement_digest,
    )
    if (
        len(query_calls) != 1
        or len(query_results) != 1
        or query_evidence.requirement_digest != requirement_digest
        or query_evidence.coverage_digest
        or not query_evidence.record_refs
        or query_evidence.input_digest
        != hashlib.sha256(
            json.dumps(
                query_calls[0].arguments,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        or query_evidence.output_digest
        != hashlib.sha256(
            query_results[0].raw_text.encode("utf-8")
        ).hexdigest()
    ):
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_evidence_binding",
        )
    normalized_result = _mapping(query_results[0].normalized_payload)
    try:
        raw_result = json.loads(query_results[0].raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        raw_result = None
    if (
        not isinstance(raw_result, dict)
        or _canonical_digest(raw_result)
        != _canonical_digest(normalized_result)
        or query_evidence.allowed_facts
        != serialize_allowed_facts("mystand_query", raw_result)
        or normalized_result.get("queryKind")
        != requirement.get("query_kind")
        or normalized_result.get("planId") != requirement.get("plan_id")
        or normalized_result.get("requirementDigest")
        != requirement_digest
        or normalized_result.get("scopeFingerprint")
        != binding.get("datascope_fingerprint")
        or turn.index_receipt is None
        or not resource_read_record_refs_valid(
            normalized_result.get("recordRefs"),
            query_evidence.record_refs,
            turn.index_receipt.matched_resource_refs,
        )
    ):
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_result_binding",
        )
    projected = project_answer(turn)
    if not projected:
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_projection",
        )

    evidence_digest = _evidence_receipt_digest(verified_evidence)
    verification = {
        "schema": "mystand.xiaoban-fact-verification.v1",
        "verified": True,
        "request_id": turn.request_id,
        "delivery_id": str(binding.get("delivery_id") or ""),
        "attempt": binding.get("attempt"),
        "message_id": turn.message_id,
        "request_fingerprint": str(binding.get("request_fingerprint") or ""),
        "plan_id": str(
            requirement.get("plan_id")
            or f"fact-{requirement_digest[:24]}"
        ),
        "requirement_digest": requirement_digest,
        "action_count": len(successful_calls),
        "evidence_count": len(verified_evidence),
        "index_receipt_digest": index_receipt_digest,
        "index_count": requirement.get("index_count"),
        "index_resource_refs_digest": requirement.get(
            "index_resource_refs_digest"
        ),
        "index_has_more": requirement.get("index_has_more"),
        "evidence_digest": evidence_digest,
        "output_digest": hashlib.sha256(
            projected.encode("utf-8")
        ).hexdigest(),
        "datascope_fingerprint": identity.datascope_fingerprint,
        "decision": "projected_evidence",
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    index_evidence_digest = requirement.get("index_evidence_digest")
    if isinstance(index_evidence_digest, str) and index_evidence_digest:
        verification["index_evidence_digest"] = index_evidence_digest
    return CompletionDecision(
        True,
        projected,
        "projected_evidence",
        verification,
    )


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
            continue
        structured_facts = facts.get("facts")
        if isinstance(structured_facts, list):
            rendered: List[str] = []
            for fact in structured_facts:
                if not isinstance(fact, Mapping) or "value" not in fact:
                    continue
                label = str(
                    fact.get("label")
                    or fact.get("predicate")
                    or fact.get("kind")
                    or "资料"
                ).strip()
                value = fact.get("value")
                if isinstance(value, str):
                    value_text = value.strip()
                else:
                    value_text = json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                if value_text:
                    rendered.append(f"{label}：{value_text}")
            if rendered:
                parts.append("\n".join(rendered))
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
        dynamic_decision = check_dynamic_completion(
            turn,
            final_text=text,
            failure_message=_failure_message(turn),
        )
        if dynamic_decision is not None:
            return dynamic_decision
        if turn.interaction_kind == INTERACTION_CHAT:
            return CompletionDecision(True, text, "allowed_chat")

        requirement = _mapping(getattr(turn, "fact_requirement", None))
        if requirement.get("fact_kind") == "collection":
            return _collection_completion(turn)
        if requirement.get("fact_kind") == "single":
            return _generic_fact_completion(turn)

        if turn.evidence:
            # WORK + 本轮可信证据：公开业务内容由结构化投影生成，
            # 模型文本里的新增实体/关系/状态不会进入公开回答。
            projected = project_answer(turn)
            if projected:
                return CompletionDecision(
                    True,
                    projected,
                    "projected_evidence",
                )
            return CompletionDecision(
                False,
                NO_EVIDENCE_MESSAGE,
                "blocked_no_evidence",
            )

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
    signed_fact_requirement = result.get("_mystand_fact_requirement")
    dynamic_completion, protocol_rejection = validate_dynamic_result_protocol(
        result,
        signed_fact_requirement,
    )
    if protocol_rejection is not None:
        return protocol_rejection
    if (
        isinstance(signed_fact_requirement, Mapping)
        and not isinstance(result.get("_trusted_turn"), WorkTurn)
    ):
        return CompletionDecision(
            False,
            INCOMPLETE_COLLECTION_MESSAGE,
            "blocked_fact_missing_trusted_turn",
        )
    if dynamic_completion and not isinstance(
        result.get("_trusted_turn"),
        WorkTurn,
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_completion_missing_trusted_turn",
        )
    if result_has_write_actions(result):
        if isinstance(signed_fact_requirement, Mapping):
            return CompletionDecision(
                False,
                INCOMPLETE_COLLECTION_MESSAGE,
                "blocked_fact_write_mixed",
            )
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
        if dynamic_completion and not dynamic_result_turn_binding_valid(
            result,
            turn,
        ):
            return CompletionDecision(
                False,
                NO_EVIDENCE_MESSAGE,
                "blocked_completion_binding_rebind",
            )
        if isinstance(signed_fact_requirement, Mapping) and (
            not turn.fact_requirement_digest
            or turn.fact_requirement_digest
            != _canonical_digest(signed_fact_requirement)
            or _mapping(turn.fact_requirement) != signed_fact_requirement
        ):
            return CompletionDecision(
                False,
                INCOMPLETE_COLLECTION_MESSAGE,
                "blocked_fact_requirement_rebind",
            )
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
