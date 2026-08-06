"""Authoritative My Stand/Xiaoban cross-service protocol contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


TRUSTED_RUNTIME_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "xiaoban-trusted-runtime-contract.v1.json"
)
_CONTRACT_BYTES = TRUSTED_RUNTIME_CONTRACT_PATH.read_bytes()
_CONTRACT_RAW = json.loads(_CONTRACT_BYTES)
if (
    not isinstance(_CONTRACT_RAW, dict)
    or _CONTRACT_RAW.get("contract")
    != "mystand.xiaoban-trusted-runtime-contract.v1"
    or _CONTRACT_RAW.get("compatibility", {}).get("mode")
    != "exact-match"
    or _CONTRACT_RAW.get("compatibility", {}).get(
        "silentFallbackAllowed"
    )
    is not False
):
    raise RuntimeError("invalid Xiaoban trusted runtime contract")


def _freeze_contract(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {
                str(key): _freeze_contract(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_contract(item) for item in value)
    return value


TRUSTED_RUNTIME_CONTRACT: Mapping[str, Any] = _freeze_contract(
    _CONTRACT_RAW
)
TRUSTED_RUNTIME_CONTRACT_REVISION = str(_CONTRACT_RAW["revision"])
TRUSTED_RUNTIME_CONTRACT_DIGEST = hashlib.sha256(
    _CONTRACT_BYTES
).hexdigest()
TRUSTED_RUNTIME_APPROVED_REVISION_ENV = (
    "XIAOBAN_TRUSTED_RUNTIME_APPROVED_REVISION"
)
TRUSTED_RUNTIME_APPROVED_DIGEST_ENV = (
    "XIAOBAN_TRUSTED_RUNTIME_APPROVED_DIGEST"
)
TRUSTED_RUNTIME_POLICY_REQUIRED_ENV = (
    "XIAOBAN_TRUSTED_RUNTIME_POLICY_REQUIRED"
)
TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER = str(
    _CONTRACT_RAW["transport"]["revisionHeader"]
)
TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER = str(
    _CONTRACT_RAW["transport"]["digestHeader"]
)


def validate_trusted_runtime_approved_policy(
    env: Mapping[str, Any] | None = None,
    *,
    required: bool | None = None,
) -> None:
    """Bind repository code to the independently approved ops baseline."""

    source = os.environ if env is None else env
    approved_revision = str(
        source.get(TRUSTED_RUNTIME_APPROVED_REVISION_ENV, "")
    ).strip()
    approved_digest = str(
        source.get(TRUSTED_RUNTIME_APPROVED_DIGEST_ENV, "")
    ).strip().lower()
    policy_required = (
        str(source.get(TRUSTED_RUNTIME_POLICY_REQUIRED_ENV, "")).strip()
        == "1"
        if required is None
        else bool(required)
    )
    if not policy_required and not approved_revision and not approved_digest:
        return
    if (
        approved_revision != TRUSTED_RUNTIME_CONTRACT_REVISION
        or approved_digest != TRUSTED_RUNTIME_CONTRACT_DIGEST
    ):
        raise RuntimeError(
            "Xiaoban trusted runtime does not match the approved ops policy"
        )


validate_trusted_runtime_approved_policy()

_USAGE = _CONTRACT_RAW["usage"]
MYSTAND_AGENT_CALL_USAGE_SCHEMA = str(_USAGE["agentCallSchema"])
MYSTAND_TRUE_MOA_USAGE_SCHEMA = str(_USAGE["trueMoaSchema"])
MYSTAND_USAGE_MAX_BYTES = int(_USAGE["maxBytes"])
MYSTAND_NORMAL_CALL_LIMIT = int(_USAGE["normalCallLimit"])
MYSTAND_TRUE_MOA_TOTAL_CALL_LIMIT = int(
    _USAGE["trueMoaTotalCallLimit"]
)
MYSTAND_TRUE_MOA_FINAL_CALL_LIMIT = int(
    _USAGE["trueMoaFinalCallLimit"]
)

_NORMAL_BILLING = _CONTRACT_RAW["billing"]["normal"]
_TRUE_MOA_BILLING = _CONTRACT_RAW["billing"]["trueMoa"]
MYSTAND_NORMAL_BILLING_POLICY_REVISION = str(
    _NORMAL_BILLING["policyRevision"]
)
MYSTAND_NORMAL_PROVIDER = str(_NORMAL_BILLING["provider"])
MYSTAND_NORMAL_MODEL = str(_NORMAL_BILLING["model"])
MYSTAND_NORMAL_ROLE = str(_NORMAL_BILLING["role"])
MYSTAND_NORMAL_INPUT_MAX_BYTES = (
    int(_NORMAL_BILLING["inputMaxBytes"])
    if _NORMAL_BILLING.get("inputMaxBytes") is not None
    else None
)
MYSTAND_NORMAL_RESERVATION_INPUT_MAX_TOKENS = int(
    _NORMAL_BILLING["reservationInputMaxTokens"]
)
MYSTAND_NORMAL_OUTPUT_MAX_TOKENS = int(
    _NORMAL_BILLING["outputMaxTokens"]
)
MYSTAND_TRUE_MOA_PRESET_ID = str(_TRUE_MOA_BILLING["presetId"])
MYSTAND_TRUE_MOA_PRESET_REVISION = str(
    _TRUE_MOA_BILLING["presetRevision"]
)
MYSTAND_TRUE_MOA_MODE = str(_TRUE_MOA_BILLING["mode"])
MYSTAND_TRUE_MOA_AUTHORIZATION_SOURCE = str(
    _TRUE_MOA_BILLING["authorizationSource"]
)
MYSTAND_TRUE_MOA_ADVISOR_INPUT_MAX_BYTES = int(
    _TRUE_MOA_BILLING["advisorInputMaxBytes"]
)
MYSTAND_TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS = int(
    _TRUE_MOA_BILLING["advisorOutputMaxTokens"]
)
MYSTAND_TRUE_MOA_FINAL_INPUT_MAX_BYTES = int(
    _TRUE_MOA_BILLING["finalInputMaxBytes"]
)
MYSTAND_TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS = int(
    _TRUE_MOA_BILLING["finalOutputMaxTokens"]
)
MYSTAND_TRUE_MOA_SLOTS = tuple(
    MappingProxyType(dict(slot))
    for slot in _TRUE_MOA_BILLING["slots"]
)

_DURABILITY = _CONTRACT_RAW["durability"]
MYSTAND_COMPLETED_OUTCOME_SCHEMA = str(
    _DURABILITY["completedOutcomeSchema"]
)
MYSTAND_OUTCOME_BINDING_SCHEMA = str(
    _DURABILITY["outcomeBindingSchema"]
)
MYSTAND_OUTCOME_AAD_SCHEMA = str(_DURABILITY["outcomeAadSchema"])
MYSTAND_WRITE_RECEIPT_VERSION = str(
    _CONTRACT_RAW["write"]["receiptVersion"]
)


class TrustedRuntimeContractError(ValueError):
    """A My Stand caller did not prove an exact cross-service contract."""

    code = "xiaoban_trusted_runtime_contract_mismatch"


def _header_value(headers: Mapping[str, Any], name: str) -> str:
    direct = headers.get(name)
    if direct is not None:
        return str(direct).strip()
    expected = name.lower()
    for key, value in headers.items():
        if str(key).lower() == expected:
            return str(value).strip()
    return ""


def validate_trusted_runtime_contract_headers(
    headers: Mapping[str, Any],
) -> None:
    """Fail closed before dispatch when API and Agent contracts differ."""

    observed_revision = _header_value(
        headers,
        TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER,
    )
    observed_digest = _header_value(
        headers,
        TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER,
    ).lower()
    if (
        observed_revision != TRUSTED_RUNTIME_CONTRACT_REVISION
        or observed_digest != TRUSTED_RUNTIME_CONTRACT_DIGEST
    ):
        raise TrustedRuntimeContractError(
            "My Stand and Xiaoban trusted runtime contracts do not match"
        )
