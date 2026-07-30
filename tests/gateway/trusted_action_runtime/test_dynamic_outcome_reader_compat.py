"""Reader-first compatibility for newly emitted dynamic terminal outcomes."""

from __future__ import annotations

import hashlib

import pytest

from xiaoban.trusted_runtime.true_moa_durable_shared import (
    _project_trusted_verification,
)
from xiaoban.trusted_runtime.true_moa_durable import (
    TrueMoAOutcomeBindingError,
)
from xiaoban.trusted_runtime.fact_contract import canonical_digest


OUTPUT = "可信终态"
OUTPUT_DIGEST = hashlib.sha256(OUTPUT.encode("utf-8")).hexdigest()
BINDING = {
    "completionProtocol": "dynamic-evidence-v2",
    "deliveryId": "delivery-reader-compat",
    "messageId": "message-reader-compat",
    "attempt": 1,
    "requestFingerprint": "a" * 64,
    "invocationFingerprint": "b" * 64,
    "datascopeFingerprint": "c" * 16,
}


def _common(*, completion_kind: str, action_count: int, evidence_count: int):
    return {
        "schema": "mystand.xiaoban-completion-verification.v2",
        "completion_kind": completion_kind,
        "binding_verified": True,
        "semantic_verified": False,
        "delivery_id": BINDING["deliveryId"],
        "request_id": BINDING["deliveryId"],
        "attempt": BINDING["attempt"],
        "message_id": BINDING["messageId"],
        "request_fingerprint": BINDING["requestFingerprint"],
        "invocation_fingerprint": BINDING["invocationFingerprint"],
        "datascope_fingerprint": BINDING["datascopeFingerprint"],
        "action_count": action_count,
        "evidence_count": evidence_count,
        "output_digest": OUTPUT_DIGEST,
        "decision": (
            "execution_status_bound"
            if completion_kind == "failure-bound"
            else "evidence_access_verified"
        ),
        "verified_at": "2026-07-30T12:00:00Z",
    }


@pytest.mark.parametrize(
    ("failure_class", "action_count", "failed_action_count"),
    [
        ("no_progress", 0, 0),
        ("cancelled", 1, 1),
    ],
)
def test_reader_accepts_new_failure_classes(
    failure_class: str,
    action_count: int,
    failed_action_count: int,
):
    verification = {
        **_common(
            completion_kind="failure-bound",
            action_count=action_count,
            evidence_count=0,
        ),
        "action_result_digest": "d" * 64,
        "failed_action_count": failed_action_count,
        "failure_class": failure_class,
    }

    assert _project_trusted_verification(
        verification,
        output_digest=OUTPUT_DIGEST,
        binding=BINDING,
    ) == verification


def test_reader_accepts_one_bound_transient_failure_before_success():
    verification = {
        **_common(
            completion_kind="evidence-bound",
            action_count=3,
            evidence_count=1,
        ),
        "index_count": 1,
        "index_has_more": False,
        "index_receipt_digest": "e" * 64,
        "index_resource_refs_digest": "f" * 64,
        "record_refs": ["resource-reader-compat"],
        "record_refs_digest": canonical_digest(
            ["resource-reader-compat"]
        ),
        "evidence_digest": "2" * 64,
        "transient_failure_count": 1,
        "transient_action_result_digest": "3" * 64,
    }

    assert _project_trusted_verification(
        verification,
        output_digest=OUTPUT_DIGEST,
        binding=BINDING,
    ) == verification

    incomplete = dict(verification)
    incomplete.pop("transient_action_result_digest")
    with pytest.raises(TrueMoAOutcomeBindingError):
        _project_trusted_verification(
            incomplete,
            output_digest=OUTPUT_DIGEST,
            binding=BINDING,
        )
