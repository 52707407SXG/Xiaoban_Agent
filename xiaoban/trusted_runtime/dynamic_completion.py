"""Dynamic-evidence-v2 completion projection and verification.

This module owns the v2-only completion contract.  The public guard keeps
protocol routing and delegates deterministic projection, binding validation,
and receipt construction here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from xiaoban.trusted_runtime.fact_contract import canonical_digest
from xiaoban.trusted_runtime.turns import serialize_allowed_facts
from xiaoban.trusted_runtime.types import (
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
_DYNAMIC_ACTION_IDS = {
    "mystand_resource_index",
    "mystand_query",
    "mystand_authorization",
}
_DYNAMIC_PROJECTION_ACTION_IDS = {
    "mystand_resource_index",
    "mystand_query",
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


def _normalized_fact_value(value: Any) -> str:
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value).strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return json.dumps(value, ensure_ascii=False)
    return ""


def _project_dynamic_facts(facts: Any) -> str:
    if not isinstance(facts, list) or not facts:
        return ""
    normalized: list[tuple[str, str, str]] = []
    values_by_kind: dict[str, str] = {}
    for item in facts:
        if not isinstance(item, Mapping):
            return ""
        kind = str(item.get("kind") or item.get("predicate") or "").strip()
        label = str(
            item.get("label")
            or item.get("predicate")
            or item.get("kind")
            or ""
        ).strip()
        raw_value = item.get("value")
        if kind == "property.parking":
            if (
                not isinstance(raw_value, Mapping)
                or set(raw_value) != {"available"}
                or not isinstance(raw_value.get("available"), bool)
            ):
                return ""
            value = "有" if raw_value["available"] else "没有"
        else:
            value = _normalized_fact_value(raw_value)
        if (
            not kind
            or not label
            or not value
            or len(kind) > 120
            or len(label) > 160
            or len(value) > 2_000
            or any(mark in label for mark in ("\r", "\n", "\x00"))
            or "\x00" in value
        ):
            return ""
        prior_value = values_by_kind.get(kind)
        if prior_value is not None:
            if prior_value != value:
                return ""
            continue
        values_by_kind[kind] = value
        normalized.append((kind, label, value))
    if len(normalized) == 1 and normalized[0][0] == "property.parking":
        projected = normalized[0][2]
    else:
        projected = "\n".join(
            f"{label}：{value}"
            for _kind, label, value in normalized
        )
    if not projected or len(projected) > _MAX_COMPLETION_TEXT:
        return ""
    return projected


def project_dynamic_answer(turn: WorkTurn) -> Optional[str]:
    """Return None when the turn is outside v2 projection routing."""
    action_ids = {
        call.action_id for call in turn.action_calls
    } | {
        result.action_id for result in turn.action_results
    }
    if not (
        turn.completion_protocol == MYSTAND_COMPLETION_PROTOCOL_V2
        and action_ids.intersection(_DYNAMIC_PROJECTION_ACTION_IDS)
        and len(turn.evidence) == 1
        and turn.evidence[0].action_id == "mystand_query"
    ):
        return None
    try:
        allowed = json.loads(turn.evidence[0].allowed_facts or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    return _project_dynamic_facts(allowed.get("facts"))


def _dynamic_evidence_completion(turn: WorkTurn) -> CompletionDecision:
    """Seal facts only after the model actually chose and completed a read."""
    identity = turn.identity
    index_receipt_digest, index_items = _dynamic_index_binding(turn)
    if (
        identity is None
        or not _completion_binding_valid(turn)
        or not index_receipt_digest
        or turn.pre_action_denials
        or turn.orphaned_receipts
        or turn.rejected_cross_account
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_evidence_binding",
        )
    calls = {call.call_id: call for call in turn.action_calls}
    results = {result.call_id: result for result in turn.action_results}
    if (
        len(calls) != len(turn.action_calls)
        or len(results) != len(turn.action_results)
        or set(calls) != set(results)
        or len(calls) != 2
        or any(result.status != "success" for result in results.values())
        or {call.action_id for call in calls.values()}
        != {"mystand_resource_index", "mystand_query"}
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_action_binding",
        )
    query_calls = [
        call
        for call in calls.values()
        if call.action_id == "mystand_query"
    ]
    query_results = [
        result
        for result in results.values()
        if result.action_id == "mystand_query"
    ]
    verified_evidence = [
        item
        for item in turn.evidence
        if item.action_id == "mystand_query"
        and item.status == "success"
        and item.verification_status == "verified"
    ]
    if (
        len(query_calls) != 1
        or len(query_results) != 1
        or len(turn.evidence) != 1
        or len(verified_evidence) != 1
        or verified_evidence[0].call_id != query_calls[0].call_id
        or query_results[0].call_id != query_calls[0].call_id
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_evidence_binding",
        )
    call = query_calls[0]
    result = query_results[0]
    item = verified_evidence[0]
    try:
        raw_result = json.loads(result.raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        raw_result = None
    if (
        not isinstance(raw_result, Mapping)
        or _canonical_digest(raw_result)
        != _canonical_digest(result.normalized_payload)
        or raw_result.get("schema") != "mystand.query-result.v1"
        or raw_result.get("ok") is not True
        or raw_result.get("status") != "matched"
        or raw_result.get("missing_facts") != []
        or not isinstance(raw_result.get("facts"), list)
        or not raw_result["facts"]
        or call.arguments.get("operation") != "read"
        or "query_kind" in call.arguments
        or item.turn_id != turn.turn_id
        or item.action_id != "mystand_query"
        or item.datascope_fingerprint != identity.datascope_fingerprint
        or item.allowed_facts
        != serialize_allowed_facts("mystand_query", dict(raw_result))
        or item.input_digest
        != hashlib.sha256(
            json.dumps(
                call.arguments,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        or item.output_digest
        != hashlib.sha256(result.raw_text.encode("utf-8")).hexdigest()
        or item.requirement_digest
        or item.coverage_digest
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_query_binding",
        )
    indexed_by_ref = {
        str(index_item.get("resourceUid")): index_item
        for index_item in index_items
    }
    explicit_refs = list(item.record_refs)
    raw_record_refs = raw_result.get("recordRefs")
    if (
        explicit_refs != sorted(set(explicit_refs))
        or (
            raw_record_refs is not None
            and (
                not isinstance(raw_record_refs, list)
                or not raw_record_refs
                or any(
                    not isinstance(ref, str) or not ref
                    for ref in raw_record_refs
                )
                or raw_record_refs != sorted(set(raw_record_refs))
            )
        )
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_record_refs",
        )
    if explicit_refs:
        resource = raw_result.get("resource")
        selected_ref = (
            str(resource.get("resourceUid") or "")
            if isinstance(resource, Mapping)
            else ""
        )
        if (
            not selected_ref
            or selected_ref not in explicit_refs
            or any(ref not in indexed_by_ref for ref in explicit_refs)
            or (
                isinstance(raw_record_refs, list)
                and (
                    selected_ref not in raw_record_refs
                    or explicit_refs
                    != sorted(set(raw_record_refs + [selected_ref]))
                )
            )
        ):
            return CompletionDecision(
                False,
                NO_EVIDENCE_MESSAGE,
                "blocked_dynamic_record_refs",
            )
        selected_item = indexed_by_ref[selected_ref]
    else:
        resource = raw_result.get("resource")
        if (
            not isinstance(resource, Mapping)
            or not isinstance(resource.get("display_name"), str)
            or not resource.get("display_name").strip()
            or not isinstance(resource.get("type"), str)
            or not resource.get("type").strip()
        ):
            return CompletionDecision(
                False,
                NO_EVIDENCE_MESSAGE,
                "blocked_dynamic_resource_join",
            )
        matches = [
            index_item
            for index_item in index_items
            if index_item.get("safeLabel") == resource["display_name"]
            and index_item.get("resourceType") == resource["type"]
        ]
        if len(matches) != 1:
            return CompletionDecision(
                False,
                NO_EVIDENCE_MESSAGE,
                "blocked_dynamic_resource_join",
            )
        selected_item = matches[0]
        explicit_refs = [str(selected_item["resourceUid"])]
    if (
        selected_item.get("canRead") is not True
        or selected_item.get("locked") is not False
        or selected_item.get("status") == "locked"
        or any(
            indexed_by_ref[ref].get("canRead") is not True
            or indexed_by_ref[ref].get("locked") is not False
            or indexed_by_ref[ref].get("status") == "locked"
            for ref in explicit_refs
        )
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_resource_access",
        )
    projected = project_dynamic_answer(turn)
    if not projected:
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_projection",
        )
    receipt = turn.index_receipt
    verification = {
        **_completion_receipt(
            turn,
            completion_kind="evidence-bound",
            action_count=2,
            evidence_count=1,
            output=projected,
            decision="projected_evidence",
        ),
        "index_count": receipt.resource_count,
        "index_resource_refs_digest": receipt.resource_refs_digest,
        "index_has_more": receipt.has_more,
        "index_receipt_digest": index_receipt_digest,
        "record_refs": explicit_refs,
        "record_refs_digest": _canonical_digest(explicit_refs),
        "evidence_digest": evidence_receipt_digest(verified_evidence),
    }
    return CompletionDecision(
        True,
        projected,
        "projected_evidence",
        verification,
    )


def check_dynamic_completion(
    turn: WorkTurn,
    *,
    failure_message: str,
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
    action_ids = {
        call.action_id for call in turn.action_calls
    } | {
        result.action_id for result in turn.action_results
    }
    if not action_ids.intersection(_DYNAMIC_ACTION_IDS):
        return None
    if turn.evidence:
        return _dynamic_evidence_completion(turn)
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
