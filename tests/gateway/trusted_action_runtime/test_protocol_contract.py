"""Cross-service trusted-runtime contract is exact and fail closed."""

from __future__ import annotations

import json

import pytest

from scripts.check_xiaoban_trusted_runtime_contract import (
    assert_unique_contract_revision,
)
from xiaoban.trusted_runtime.protocol_contract import (
    MYSTAND_BUSINESS_TOOL_MODE_DISABLED_VALUE,
    MYSTAND_BUSINESS_TOOL_MODE_ENABLED_VALUE,
    MYSTAND_BUSINESS_TOOL_MODE_HEADER,
    TRUSTED_RUNTIME_CONTRACT,
    TRUSTED_RUNTIME_CONTRACT_DIGEST,
    TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER,
    TRUSTED_RUNTIME_CONTRACT_REVISION,
    TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER,
    TrustedRuntimeContractError,
    validate_trusted_runtime_approved_policy,
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
    completion = TRUSTED_RUNTIME_CONTRACT["completion"]
    assert MYSTAND_BUSINESS_TOOL_MODE_HEADER == (
        "x-xiaoban-business-tool-mode"
    )
    assert MYSTAND_BUSINESS_TOOL_MODE_ENABLED_VALUE == "enabled"
    assert MYSTAND_BUSINESS_TOOL_MODE_DISABLED_VALUE == "disabled"
    assert completion["businessToolModeRequired"] is True
    assert completion["businessToolModeSource"] == "server-trusted-intent"
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


def test_trusted_runtime_contract_requires_independent_approved_policy():
    validate_trusted_runtime_approved_policy(
        {
            "XIAOBAN_TRUSTED_RUNTIME_APPROVED_REVISION":
                TRUSTED_RUNTIME_CONTRACT_REVISION,
            "XIAOBAN_TRUSTED_RUNTIME_APPROVED_DIGEST":
                TRUSTED_RUNTIME_CONTRACT_DIGEST,
        },
        required=True,
    )
    with pytest.raises(RuntimeError, match="approved ops policy"):
        validate_trusted_runtime_approved_policy({}, required=True)
    with pytest.raises(RuntimeError, match="approved ops policy"):
        validate_trusted_runtime_approved_policy(
            {
                "XIAOBAN_TRUSTED_RUNTIME_APPROVED_REVISION":
                    TRUSTED_RUNTIME_CONTRACT_REVISION,
                "XIAOBAN_TRUSTED_RUNTIME_APPROVED_DIGEST": "0" * 64,
            },
            required=True,
        )


def test_trusted_runtime_contract_cannot_change_under_same_revision():
    current = json.dumps(
        {
            "revision": TRUSTED_RUNTIME_CONTRACT_REVISION,
            "billing": {"model": "deepseek-v4-pro"},
        },
        sort_keys=True,
    ).encode()
    downgraded_same_revision = json.dumps(
        {
            "revision": TRUSTED_RUNTIME_CONTRACT_REVISION,
            "billing": {"model": "weaker-fallback"},
        },
        sort_keys=True,
    ).encode()
    with pytest.raises(SystemExit, match="without a new revision"):
        assert_unique_contract_revision(
            downgraded_same_revision,
            [current],
        )
    bumped = json.dumps(
        {
            "revision": f"{TRUSTED_RUNTIME_CONTRACT_REVISION}-next",
            "billing": {"model": "reviewed-future-model"},
        },
        sort_keys=True,
    ).encode()
    assert_unique_contract_revision(bumped, [current])
