"""Process-local capability for one verified My Stand egress projection.

Provider and tool payloads are ordinary JSON-compatible values.  A boolean
marker and a matching text digest therefore cannot prove that CompletionGuard
ran.  Only this module can attach the process-local capability used to reuse a
projection.  Durable replay receives a fresh capability only after the
encrypted outcome has been recovered and validated by the durable store.
"""

from __future__ import annotations

import hashlib
from typing import Any


class _MyStandEgressSeal:
    __slots__ = ()

    def __copy__(self) -> "_MyStandEgressSeal":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_MyStandEgressSeal":
        return self


_MYSTAND_EGRESS_SEAL = _MyStandEgressSeal()
_MYSTAND_EGRESS_SEAL_FIELD = "_mystand_egress_seal"


def discard_untrusted_mystand_egress_projection(result: Any) -> None:
    """Remove provider-forgeable projection metadata before verification."""

    if not isinstance(result, dict) or is_mystand_egress_sealed(result):
        return
    for field in (
        "_mystand_egress_finalized",
        "_mystand_egress_output_digest",
        "_mystand_egress_seal",
        "_mystand_completion_allowed",
        "_mystand_trusted_verification",
    ):
        result.pop(field, None)


def seal_mystand_egress_projection(result: Any) -> None:
    """Attach the capability after validating the visible text digest."""

    if not isinstance(result, dict):
        raise RuntimeError("My Stand egress result is unavailable")
    final_text = result.get("final_response")
    output_digest = result.get("_mystand_egress_output_digest")
    if (
        result.get("_mystand_egress_finalized") is not True
        or not isinstance(final_text, str)
        or not isinstance(output_digest, str)
        or hashlib.sha256(final_text.encode("utf-8")).hexdigest()
        != output_digest
    ):
        raise RuntimeError("My Stand finalized egress digest mismatch")
    result[_MYSTAND_EGRESS_SEAL_FIELD] = _MYSTAND_EGRESS_SEAL


def is_mystand_egress_sealed(result: Any) -> bool:
    """Return true only for a digest-bound projection sealed in this process."""

    if (
        not isinstance(result, dict)
        or result.get("_mystand_egress_finalized") is not True
        or result.get(_MYSTAND_EGRESS_SEAL_FIELD) is not _MYSTAND_EGRESS_SEAL
    ):
        return False
    final_text = result.get("final_response")
    output_digest = result.get("_mystand_egress_output_digest")
    return bool(
        isinstance(final_text, str)
        and isinstance(output_digest, str)
        and hashlib.sha256(final_text.encode("utf-8")).hexdigest()
        == output_digest
    )
