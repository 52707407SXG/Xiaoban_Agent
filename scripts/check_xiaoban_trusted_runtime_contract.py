#!/usr/bin/env python3
"""Fail closed on local or cross-service trusted-runtime contract drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
EGRESS_SEAL_MODULE = Path(
    "gateway/platforms/mystand_egress_seal.py"
)
EGRESS_SEAL_CALLERS = {
    EGRESS_SEAL_MODULE,
    Path("gateway/platforms/api_server.py"),
    Path("gateway/platforms/true_moa_idempotency.py"),
}

from xiaoban.trusted_runtime.agent_call_usage_codec import (
    AGENT_CALL_LIMIT,
    AGENT_CALL_USAGE_MAX_BYTES,
    AGENT_CALL_USAGE_SCHEMA,
)
from xiaoban.trusted_runtime.paid_call_policy import (
    SIGNED_MYSTAND_AGENT_POLICY,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION,
)
from xiaoban.trusted_runtime.protocol_contract import (
    TRUSTED_RUNTIME_CONTRACT,
    TRUSTED_RUNTIME_CONTRACT_DIGEST,
    TRUSTED_RUNTIME_CONTRACT_PATH,
    validate_trusted_runtime_approved_policy,
)
from xiaoban.trusted_runtime.true_moa_contracts import (
    TRUE_MOA_ADVISOR_INPUT_MAX_BYTES,
    TRUE_MOA_ADVISOR_RESERVATION_INPUT_MAX_TOKENS,
    TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS,
    TRUE_MOA_ALL_SLOTS,
    TRUE_MOA_FINAL_CALL_LIMIT,
    TRUE_MOA_FINAL_INPUT_MAX_BYTES,
    TRUE_MOA_FINAL_RESERVATION_INPUT_MAX_TOKENS,
    TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS,
    TRUE_MOA_MODE,
    TRUE_MOA_PRESET_ID,
    TRUE_MOA_PRESET_REVISION,
    TRUE_MOA_TOTAL_CALL_LIMIT,
    TRUE_MOA_USAGE_SCHEMA,
)
from xiaoban.trusted_runtime.true_moa_durable_shared import (
    TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
    TRUE_MOA_OUTCOME_BINDING_SCHEMA,
)
def _expect(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise SystemExit(
            f"trusted runtime contract drift: {label}: "
            f"{actual!r} != {expected!r}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_egress_seal_callers() -> None:
    for root_name in ("agent", "gateway", "tools", "xiaoban"):
        for path in (REPO_ROOT / root_name).rglob("*.py"):
            relative = path.relative_to(REPO_ROOT)
            source = path.read_text(encoding="utf-8")
            if (
                "seal_mystand_egress_projection" in source
                and relative not in EGRESS_SEAL_CALLERS
            ):
                raise SystemExit(
                    "unauthorized My Stand egress seal caller: "
                    f"{relative}"
                )
            if (
                "_MYSTAND_EGRESS_SEAL" in source
                and relative != EGRESS_SEAL_MODULE
            ):
                raise SystemExit(
                    "egress seal capability escaped its module: "
                    f"{relative}"
                )


def assert_unique_contract_revision(
    current_bytes: bytes,
    historical_blobs: list[bytes],
) -> None:
    """One revision may never identify two different contract byte streams."""

    try:
        current_revision = str(json.loads(current_bytes)["revision"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit("invalid current trusted runtime contract") from exc
    for shown in historical_blobs:
        try:
            historical_revision = str(
                json.loads(shown)["revision"]
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SystemExit(
                "invalid trusted runtime contract in Git history"
            ) from exc
        if (
            historical_revision == current_revision
            and shown != current_bytes
        ):
            raise SystemExit(
                "trusted runtime contract changed without a new revision"
            )


def _check_revision_history() -> None:
    if not (REPO_ROOT / ".git").exists():
        build_sha_path = REPO_ROOT / ".xiaoban_build_sha"
        build_sha = (
            build_sha_path.read_text(encoding="utf-8").strip()
            if build_sha_path.is_file()
            else ""
        )
        if re.fullmatch(r"[0-9a-f]{40}", build_sha) is None:
            raise SystemExit(
                "immutable Xiaoban release build SHA is missing or invalid"
            )
        return
    relative = TRUSTED_RUNTIME_CONTRACT_PATH.relative_to(REPO_ROOT)
    history = subprocess.run(
        [
            "git",
            "log",
            "--all",
            "--format=%H",
            "--",
            str(relative),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    historical_blobs = [
        subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        for commit in history
    ]
    assert_unique_contract_revision(
        TRUSTED_RUNTIME_CONTRACT_PATH.read_bytes(),
        historical_blobs,
    )


def _peer_file(peer_root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = peer_root / candidate
        if path.is_file():
            return path
    raise SystemExit(
        "trusted runtime peer contract is missing: "
        + ", ".join(str(peer_root / item) for item in candidates)
    )


def check_local() -> None:
    _check_egress_seal_callers()
    _check_revision_history()
    contract = TRUSTED_RUNTIME_CONTRACT
    usage = contract["usage"]
    normal = contract["billing"]["normal"]
    true_moa = contract["billing"]["trueMoa"]
    durability = contract["durability"]

    _expect(
        "contract digest",
        _sha256(TRUSTED_RUNTIME_CONTRACT_PATH),
        TRUSTED_RUNTIME_CONTRACT_DIGEST,
    )
    api_server_source = (
        REPO_ROOT / "gateway/platforms/api_server.py"
    ).read_text(encoding="utf-8")
    if any(
        legacy_receiver in api_server_source
        for legacy_receiver in (
            "X-Xiaoban-Fact-Requirement",
            "X-Xiaoban-Fact-Signature",
            "X-Xiaoban-Completion-Protocol",
            "dynamic-evidence-v2",
            "_MYSTAND_BUSINESS_TOOL_MODE_HEADER",
            "_MYSTAND_EVIDENCE_REQUIRED_HEADER",
        )
    ):
        raise SystemExit(
            "legacy G1-G4 compatibility input is still active in API runtime"
        )
    _expect("agent usage schema", AGENT_CALL_USAGE_SCHEMA, usage["agentCallSchema"])
    _expect("agent call limit", AGENT_CALL_LIMIT, usage["normalCallLimit"])
    _expect("usage max bytes", AGENT_CALL_USAGE_MAX_BYTES, usage["maxBytes"])
    _expect("normal policy revision", SIGNED_MYSTAND_AGENT_POLICY_REVISION, normal["policyRevision"])
    _expect("normal provider", SIGNED_MYSTAND_AGENT_POLICY.provider, normal["provider"])
    _expect("normal model", SIGNED_MYSTAND_AGENT_POLICY.model, normal["model"])
    _expect("normal role", SIGNED_MYSTAND_AGENT_POLICY.role, normal["role"])
    _expect("normal input cap", SIGNED_MYSTAND_AGENT_POLICY.input_max_bytes, normal["inputMaxBytes"])
    _expect("normal output cap", SIGNED_MYSTAND_AGENT_POLICY.output_max_tokens, normal["outputMaxTokens"])
    _expect("true MoA mode", TRUE_MOA_MODE, true_moa["mode"])
    _expect("true MoA preset", TRUE_MOA_PRESET_ID, true_moa["presetId"])
    _expect("true MoA revision", TRUE_MOA_PRESET_REVISION, true_moa["presetRevision"])
    _expect("true MoA usage schema", TRUE_MOA_USAGE_SCHEMA, usage["trueMoaSchema"])
    _expect("true MoA total calls", TRUE_MOA_TOTAL_CALL_LIMIT, usage["trueMoaTotalCallLimit"])
    _expect("true MoA final calls", TRUE_MOA_FINAL_CALL_LIMIT, usage["trueMoaFinalCallLimit"])
    _expect("advisor input cap", TRUE_MOA_ADVISOR_INPUT_MAX_BYTES, true_moa["advisorInputMaxBytes"])
    _expect(
        "advisor input reservation",
        TRUE_MOA_ADVISOR_RESERVATION_INPUT_MAX_TOKENS,
        true_moa["advisorReservationInputMaxTokens"],
    )
    _expect("advisor output cap", TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS, true_moa["advisorOutputMaxTokens"])
    _expect("final input cap", TRUE_MOA_FINAL_INPUT_MAX_BYTES, true_moa["finalInputMaxBytes"])
    _expect(
        "final input reservation",
        TRUE_MOA_FINAL_RESERVATION_INPUT_MAX_TOKENS,
        true_moa["finalReservationInputMaxTokens"],
    )
    _expect("final output cap", TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS, true_moa["finalOutputMaxTokens"])
    _expect(
        "true MoA slots",
        [
            {
                "slotId": slot.slot_id,
                "provider": slot.provider,
                "model": slot.model,
                "role": slot.role,
                "maxCalls": (
                    TRUE_MOA_FINAL_CALL_LIMIT
                    if slot.role == "final_executor"
                    else 1
                ),
            }
            for slot in TRUE_MOA_ALL_SLOTS
        ],
        list(true_moa["slots"]),
    )
    _expect("outcome schema", TRUE_MOA_COMPLETED_OUTCOME_SCHEMA, durability["completedOutcomeSchema"])
    _expect("outcome binding", TRUE_MOA_OUTCOME_BINDING_SCHEMA, durability["outcomeBindingSchema"])

    for name, relative in (
        ("resourceIndex", "contracts/mystand-resource-index-tool.v1.json"),
        ("query", "contracts/mystand-query-tool.v1.json"),
    ):
        _expect(
            f"{name} schema hash",
            _sha256(REPO_ROOT / relative),
            contract["tools"][name]["sha256"],
        )
    _expect(
        "true MoA parity fixture hash",
        _sha256(
            REPO_ROOT / "contracts/true-moa-usage-parity.v1.json"
        ),
        contract["parityFixtures"]["trueMoaUsageSha256"],
    )


def check_peer(peer_root: Path) -> None:
    peer_root = peer_root.resolve()
    peer_contract = _peer_file(
        peer_root,
        (
            "contracts/xiaoban-trusted-runtime-contract.v1.json",
            "packages/xiaoban-trusted-runtime-contract/schemas/xiaoban-trusted-runtime-contract.v1.json",
        ),
    )
    _expect(
        "cross-service manifest bytes",
        peer_contract.read_bytes(),
        TRUSTED_RUNTIME_CONTRACT_PATH.read_bytes(),
    )
    peer_resource = _peer_file(
        peer_root,
        (
            "contracts/mystand-resource-index-tool.v1.json",
            "packages/resource-index-contract/schemas/mystand-resource-index-tool.v1.json",
        ),
    )
    peer_query = _peer_file(
        peer_root,
        (
            "contracts/mystand-query-tool.v1.json",
            "packages/xiaoban-trusted-runtime-contract/schemas/mystand-query-tool.v1.json",
        ),
    )
    peer_parity = _peer_file(
        peer_root,
        (
            "contracts/true-moa-usage-parity.v1.json",
            "packages/xiaoban-trusted-runtime-contract/fixtures/true-moa-usage-parity.v1.json",
        ),
    )
    _expect(
        "cross-service resource-index schema",
        _sha256(peer_resource),
        TRUSTED_RUNTIME_CONTRACT["tools"]["resourceIndex"]["sha256"],
    )
    _expect(
        "cross-service query schema",
        _sha256(peer_query),
        TRUSTED_RUNTIME_CONTRACT["tools"]["query"]["sha256"],
    )
    _expect(
        "cross-service true MoA parity fixture",
        peer_parity.read_bytes(),
        (
            REPO_ROOT / "contracts/true-moa-usage-parity.v1.json"
        ).read_bytes(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-root", type=Path)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    check_local()
    if args.release:
        if args.peer_root is None:
            raise SystemExit(
                "trusted runtime release check requires --peer-root"
            )
        validate_trusted_runtime_approved_policy(required=True)
    if args.peer_root is not None:
        check_peer(args.peer_root)
    print("ok Xiaoban trusted runtime contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
