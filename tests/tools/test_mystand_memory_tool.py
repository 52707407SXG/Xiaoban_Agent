from __future__ import annotations

import json
import os

from gateway.session_context import clear_session_vars, set_session_vars
from plugins.memory.holographic.scope import open_scoped_memory_store
from tools.mystand_memory_tool import MYSTAND_MEMORY_SCHEMA, mystand_memory_handler
from xiaoban.mystand_account_memory import list_account_documents


def _configure(monkeypatch, tmp_path):
    secret_file = tmp_path / "memory-scope.key"
    secret_file.write_text("test-secret-that-is-long-enough-for-scope", encoding="utf-8")
    os.chmod(secret_file, 0o600)
    monkeypatch.setenv("XIAOBAN_HOME", str(tmp_path))
    monkeypatch.setenv("XIAOBAN_MYSTAND_MEMORY_SCOPE_SECRET_FILE", str(secret_file))
    monkeypatch.setenv("MYSTAND_XIAOBAN_OWNER_USER_ID", "52707407")
    return secret_file.read_text(encoding="utf-8")


def test_tool_is_bound_to_current_account(monkeypatch, tmp_path):
    secret = _configure(monkeypatch, tmp_path)
    alice_tokens = set_session_vars(
        platform="api_server",
        source="mystand",
        user_id="broker-a",
        memory_site_id="mystand-test",
        memory_tier="notebook",
    )
    try:
        written = json.loads(mystand_memory_handler({
            "action": "upsert",
            "target": "notebook",
            "summary": "偏好先看城南片区",
        }))
    finally:
        clear_session_vars(alice_tokens)

    bob_tokens = set_session_vars(
        platform="api_server",
        source="mystand",
        user_id="broker-b",
        memory_site_id="mystand-test",
        memory_tier="notebook",
    )
    try:
        bob = json.loads(mystand_memory_handler({"action": "skip"}))
    finally:
        clear_session_vars(bob_tokens)

    alice_store = open_scoped_memory_store(
        secret=secret,
        site_id="mystand-test",
        user_id="broker-a",
        xiaoban_home=tmp_path,
    )
    try:
        alice = list_account_documents(alice_store, tier="notebook")
    finally:
        alice_store.close()

    assert written["ok"] is True
    assert "城南片区" in alice[0]["content"]
    assert bob["documents"] == []


def test_forged_owner_tier_is_rejected(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    tokens = set_session_vars(
        platform="api_server",
        source="mystand",
        user_id="broker-a",
        memory_site_id="mystand-test",
        memory_tier="owner",
    )
    try:
        result = json.loads(mystand_memory_handler({"action": "skip"}))
    finally:
        clear_session_vars(tokens)

    assert result["ok"] is False
    assert result["code"] == "invalid_memory_scope"


def test_tool_contract_uses_only_structured_harness_decisions():
    action = MYSTAND_MEMORY_SCHEMA["parameters"]["properties"]["action"]

    assert action["enum"] == ["skip", "upsert", "correct", "forget"]
    assert "summary" in MYSTAND_MEMORY_SCHEMA["parameters"]["properties"]
    assert "resourceRefs" in MYSTAND_MEMORY_SCHEMA["parameters"]["properties"]
    assert "entries" not in MYSTAND_MEMORY_SCHEMA["parameters"]["properties"]


def test_tool_accepts_only_current_turn_authorized_resource_refs(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    allowed_reference = "AUTH-AAAAAAAA-BBBBBBBB-CCCCCCCC-DDDDDDDD"
    tokens = set_session_vars(
        platform="api_server",
        source="mystand",
        user_id="broker-a",
        memory_site_id="mystand-test",
        memory_tier="notebook",
        memory_resource_refs=((allowed_reference, "business-archive"),),
    )
    try:
        accepted = json.loads(mystand_memory_handler({
            "action": "upsert",
            "target": "notebook",
            "summary": "后续按已授权资料跟进",
            "resourceRefs": [{
                "referenceId": allowed_reference,
                "sourceType": "business-archive",
            }],
        }))
        forged = json.loads(mystand_memory_handler({
            "action": "upsert",
            "target": "notebook",
            "summary": "伪造引用不得保存",
            "resourceRefs": [{
                "referenceId": "AUTH-11111111-22222222-33333333-44444444",
                "sourceType": "business-archive",
            }],
        }))
    finally:
        clear_session_vars(tokens)

    assert accepted["ok"] is True
    assert accepted["recorded"] is True
    assert forged["ok"] is False
    assert forged["code"] == "invalid_memory_operation"
