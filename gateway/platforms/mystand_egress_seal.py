"""Process-local capability for one verified My Stand egress projection.

Provider and tool payloads are ordinary JSON-compatible values.  A boolean
marker and a matching text digest therefore cannot prove that an earlier stage
ran.  Only this module can attach the process-local capability used to reuse a
projection.  Durable replay receives a fresh capability only after the
encrypted outcome has been recovered and validated by the durable store.
"""

from __future__ import annotations

import hashlib
from typing import Any


class _MyStandEgressSeal(str):
    """Immutable process-local capability bound to one output digest."""

    __slots__ = ()

    def __new__(cls, output_digest: str) -> "_MyStandEgressSeal":
        return str.__new__(cls, output_digest)

    @property
    def output_digest(self) -> str:
        return str(self)

    def __copy__(self) -> "_MyStandEgressSeal":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_MyStandEgressSeal":
        return self


_MYSTAND_EGRESS_SEAL_FIELD = "_mystand_egress_seal"


def discard_untrusted_mystand_egress_projection(result: Any) -> None:
    """Remove provider-forgeable projection metadata before verification."""

    if not isinstance(result, dict) or is_mystand_egress_sealed(result):
        return
    for field in (
        "_mystand_egress_finalized",
        "_mystand_egress_output_digest",
        "_mystand_egress_seal",
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
    # Bind the verified digest inside the process-local capability.  Mutating
    # both the text and provider-shaped digest metadata after sealing must not
    # manufacture a second valid projection.
    result[_MYSTAND_EGRESS_SEAL_FIELD] = _MyStandEgressSeal(output_digest)


def is_mystand_egress_sealed(result: Any) -> bool:
    """Return true only for a digest-bound projection sealed in this process."""

    if (
        not isinstance(result, dict)
        or result.get("_mystand_egress_finalized") is not True
    ):
        return False
    seal = result.get(_MYSTAND_EGRESS_SEAL_FIELD)
    if not isinstance(seal, _MyStandEgressSeal):
        return False
    final_text = result.get("final_response")
    output_digest = result.get("_mystand_egress_output_digest")
    return bool(
        isinstance(final_text, str)
        and isinstance(output_digest, str)
        and seal.output_digest == output_digest
        and hashlib.sha256(final_text.encode("utf-8")).hexdigest()
        == seal.output_digest
    )
