"""Shared deterministic helpers for signed My Stand fact requirements."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Dict, Mapping, Optional, Sequence

SIGNED_FACT_INDEX_PAGE_LIMIT = 100
SIGNED_FACT_INDEX_MAX_PAGES = 200
SIGNED_FACT_INDEX_MAX_ITEMS = (
    SIGNED_FACT_INDEX_PAGE_LIMIT * SIGNED_FACT_INDEX_MAX_PAGES
)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def normalized_fact_query_text(value: Any) -> str:
    """Mirror My Stand's trusted fact compiler text normalization."""
    return (
        unicodedata.normalize("NFKC", str(value or "").replace("\x00", ""))
        .strip()[:6_000]
    )


def evidence_requirement_digest(
    requirement: Mapping[str, Any],
    *,
    canonical_fallback: str = "",
) -> str:
    """Use My Stand's signed business digest, falling back for v1 RED clients."""
    declared = requirement.get("requirement_digest")
    if isinstance(declared, str) and len(declared) == 64:
        return declared
    return canonical_fallback or canonical_digest(requirement)


def resource_read_record_refs_valid(
    raw_record_refs: Any,
    evidence_record_refs: Any,
    matched_index_refs: Sequence[str],
) -> bool:
    """Bind every resource-read evidence source to this turn's index.

    Collection queries intentionally do not use this shape: their record refs
    describe the complete result set.  A generic resource-read may legitimately
    join a root resource with linked profile/property material, but every
    reported source must be unique and present in the signed IndexReceipt.
    """
    if (
        not isinstance(raw_record_refs, list)
        or not raw_record_refs
        or any(
            not isinstance(ref, str) or not ref
            for ref in raw_record_refs
        )
        or len(set(raw_record_refs)) != len(raw_record_refs)
        or not isinstance(evidence_record_refs, list)
        or evidence_record_refs != sorted(raw_record_refs)
    ):
        return False
    return set(raw_record_refs).issubset(set(matched_index_refs))


def build_fact_query_plan(
    requirement: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the signed typed plan or the v1 rank compatibility plan."""
    if not isinstance(requirement, Mapping):
        return None
    plan = requirement.get("query_plan")
    if isinstance(plan, Mapping):
        return json.loads(json.dumps(plan, ensure_ascii=False))
    if (
        requirement.get("fact_kind") == "collection"
        and requirement.get("operation") == "rank"
        and requirement.get("metric") == "settled_performance"
        and str(requirement.get("time_scope") or "").isdigit()
        and isinstance(requirement.get("ordinal"), int)
    ):
        return {
            "operation": "read",
            "query_kind": "rank",
            "module_id": str(requirement.get("module_id") or ""),
            "fact_paths": ["finance.performance.rank"],
            "query_args": {
                "year": int(str(requirement["time_scope"])),
                "rank": int(requirement["ordinal"]),
            },
            "coverage_required": True,
        }
    return None
