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
    for call_id in sorted(calls):
        call = calls[call_id]
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


def _dynamic_evidence_completion(
    turn: WorkTurn,
    final_text: str,
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
    receipt = turn.index_receipt
    verification = {
        **_completion_receipt(
            turn,
            completion_kind="evidence-bound",
            action_count=1 + len(verified_evidence),
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
    }
    return CompletionDecision(
        True,
        output,
        "evidence_access_verified",
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
    if (
        getattr(turn, "completion_finalization", "") != "failure"
        or not output
        or not _completion_binding_valid(turn)
        or _hard_runtime_violation(turn)
        or turn.evidence
        or failure_lifecycle is None
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
    action_count, failures = failure_lifecycle
    failure_statuses = sorted({result.status for result in failures})
    failure_class = (
        failure_statuses[0]
        if len(failure_statuses) == 1
        else "mixed"
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
        "failed_action_count": len(failures),
        "failure_class": failure_class,
    }
    return CompletionDecision(
        True,
        output,
        "execution_status_bound",
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
    failures = sorted(
        matched_failures,
        key=lambda result: (
            str(result.call_id),
            str(result.action_id),
            str(result.error_code),
        ),
    )
    if not failures:
        return None
    return len(matched), failures


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
    return ""


def check_dynamic_completion(
    turn: WorkTurn,
    *,
    final_text: str,
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
    if getattr(turn, "completion_finalization", "") == "not_executed":
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_not_executed",
        )
    action_ids = {
        call.action_id for call in turn.action_calls
    } | {
        result.action_id for result in turn.action_results
    }
    if not action_ids.intersection(_DYNAMIC_ACTION_IDS):
        return None
    if turn.evidence:
        return _dynamic_evidence_completion(turn, final_text)
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
