"""Cross-service trusted-runtime contract is exact and fail closed."""

from __future__ import annotations

import pytest

from xiaoban.trusted_runtime.protocol_contract import (
    TRUSTED_RUNTIME_CONTRACT,
    TRUSTED_RUNTIME_CONTRACT_DIGEST,
    TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER,
    TRUSTED_RUNTIME_CONTRACT_REVISION,
    TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER,
    TrustedRuntimeContractError,
    validate_trusted_runtime_contract_headers,
)


def _valid_headers() -> dict[str, str]:
    return {
        TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER:
            TRUSTED_RUNTIME_CONTRACT_REVISION,
        TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER:
            TRUSTED_RUNTIME_CONTRACT_DIGEST,
    }


def test_trusted_runtime_contract_protects_non_negotiable_invariants():
    compatibility = TRUSTED_RUNTIME_CONTRACT["compatibility"]
    assert compatibility == {
        "mode": "exact-match",
        "silentFallbackAllowed": False,
        "providerCanAuthorizeCompletion": False,
        "completionGuardRequired": True,
        "identitySource": "server-only",
        "dataScopeSource": "server-only",
    }
    assert TRUSTED_RUNTIME_CONTRACT["write"]["receiptVersion"] == (
        "authorization-write-receipt-v2"
    )
    assert TRUSTED_RUNTIME_CONTRACT["write"]["successRequiresOk"] is True
    assert (
        TRUSTED_RUNTIME_CONTRACT["write"]["successRequiresVerified"]
        is True
    )
    with pytest.raises(TypeError):
        TRUSTED_RUNTIME_CONTRACT["compatibility"]["mode"] = "fallback"
    with pytest.raises(TypeError):
        TRUSTED_RUNTIME_CONTRACT["billing"]["trueMoa"]["slots"][0][
            "provider"
        ] = "untrusted-provider"


@pytest.mark.parametrize(
    "mutate",
    (
        lambda headers: headers.pop(
            TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER
        ),
        lambda headers: headers.__setitem__(
            TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER,
            "future-uncoordinated-revision",
        ),
        lambda headers: headers.__setitem__(
            TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER,
            "0" * 64,
        ),
    ),
)
def test_trusted_runtime_contract_rejects_missing_or_drifted_peer(mutate):
    headers = _valid_headers()
    mutate(headers)
    with pytest.raises(TrustedRuntimeContractError):
        validate_trusted_runtime_contract_headers(headers)


def test_trusted_runtime_contract_accepts_exact_peer():
    validate_trusted_runtime_contract_headers(_valid_headers())
