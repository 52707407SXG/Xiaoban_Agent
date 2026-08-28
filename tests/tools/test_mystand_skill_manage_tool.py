import json
from pathlib import Path
from unittest.mock import patch

from tools import mystand_skill_manage_tool as bridge
from tools import skill_manager_tool


SKILL = """---
name: test-workflow
description: A bounded test workflow.
metadata:
  mystand:
    tools: []
---

# Test

Run the bounded workflow.
"""


def _session_value(key, default=""):
    return {
        "XIAOBAN_SESSION_PLATFORM": "api_server",
        "XIAOBAN_SESSION_USER_ID": "52707407",
    }.get(key, default)


def _broker_session_value(key, default=""):
    return {
        "XIAOBAN_SESSION_PLATFORM": "api_server",
        "XIAOBAN_SESSION_USER_ID": "ZYJ099",
    }.get(key, default)


def test_non_owner_cannot_manage_shared_skill(tmp_path):
    skills_root = tmp_path / "skills"
    with patch.object(bridge, "get_session_env", side_effect=_broker_session_value), \
         patch.object(skill_manager_tool, "SKILLS_DIR", skills_root), \
         patch.object(bridge, "request_gateway_action_approval") as approval:
        result = json.loads(bridge.mystand_skill_manage_handler({
            "action": "create",
            "name": "test-workflow",
            "content": SKILL,
        }))

    assert result["success"] is False
    assert result["code"] == "mystand_skill_owner_required"
    approval.assert_not_called()
    assert not skills_root.exists()


def test_mystand_skill_create_is_approval_bound_and_category_scoped(tmp_path):
    skills_root = tmp_path / "skills"
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(skill_manager_tool, "SKILLS_DIR", skills_root), \
         patch.object(bridge, "request_gateway_action_approval", return_value={"approved": True}), \
         patch.object(bridge, "_scan_and_rollback", return_value=""):
        result = json.loads(bridge.mystand_skill_manage_handler({
            "action": "create",
            "name": "test-workflow",
            "content": SKILL,
        }))

    assert result["success"] is True
    assert result["path"] == "mystand/test-workflow/SKILL.md"
    assert (skills_root / "mystand" / "test-workflow" / "SKILL.md").read_text() == SKILL


def test_mystand_skill_denied_approval_writes_nothing(tmp_path):
    skills_root = tmp_path / "skills"
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(skill_manager_tool, "SKILLS_DIR", skills_root), \
         patch.object(bridge, "request_gateway_action_approval", return_value={"approved": False, "message": "denied"}):
        result = json.loads(bridge.mystand_skill_manage_handler({
            "action": "create",
            "name": "test-workflow",
            "content": SKILL,
        }))

    assert result["success"] is False
    assert result["code"] == "mystand_skill_approval_denied"
    assert not skills_root.exists()


def test_mystand_skill_rejects_missing_tool_contract_before_approval(tmp_path):
    skills_root = tmp_path / "skills"
    invalid = SKILL.replace("metadata:\n  mystand:\n    tools: []\n", "")
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(skill_manager_tool, "SKILLS_DIR", skills_root), \
         patch.object(bridge, "request_gateway_action_approval") as approval:
        result = json.loads(bridge.mystand_skill_manage_handler({
            "action": "create",
            "name": "test-workflow",
            "content": invalid,
        }))

    assert result["code"] == "mystand_skill_contract_invalid"
    approval.assert_not_called()
    assert not skills_root.exists()


def test_mystand_skill_rejects_unregistered_declared_tool(tmp_path):
    skills_root = tmp_path / "skills"
    invalid = SKILL.replace("tools: []", "tools: [mystand_missing_tool]")
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(skill_manager_tool, "SKILLS_DIR", skills_root), \
         patch.object(bridge.registry, "get_entry", return_value=None), \
         patch.object(bridge, "request_gateway_action_approval") as approval:
        result = json.loads(bridge.mystand_skill_manage_handler({
            "action": "create",
            "name": "test-workflow",
            "content": invalid,
        }))

    assert result["code"] == "mystand_skill_contract_invalid"
    assert "尚未注册" in result["error"]
    approval.assert_not_called()
    assert not skills_root.exists()


def test_mystand_skill_returns_verified_tool_contract(tmp_path):
    skills_root = tmp_path / "skills"
    content = SKILL.replace("tools: []", "tools: [mystand_test_tool]")
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(skill_manager_tool, "SKILLS_DIR", skills_root), \
         patch.object(bridge.registry, "get_entry", return_value=object()), \
         patch.object(bridge, "_owner_available_tool_names", return_value={"mystand_test_tool"}), \
         patch.object(bridge, "request_gateway_action_approval", return_value={"approved": True}), \
         patch.object(bridge, "_scan_and_rollback", return_value=""):
        result = json.loads(bridge.mystand_skill_manage_handler({
            "action": "create",
            "name": "test-workflow",
            "content": content,
        }))

    assert result["success"] is True
    assert result["tools"] == ["mystand_test_tool"]
    assert result["toolRegistrationVerified"] is True


def test_mystand_skill_cannot_edit_skill_outside_mystand_category(tmp_path):
    skills_root = tmp_path / "skills"
    outside = skills_root / "research" / "test-workflow"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text(SKILL)
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(skill_manager_tool, "SKILLS_DIR", skills_root), \
         patch.object(skill_manager_tool, "_find_skill", return_value={"path": outside}):
        result = json.loads(bridge.mystand_skill_manage_handler({
            "action": "edit",
            "name": "test-workflow",
            "content": SKILL,
        }))
    assert result["code"] == "mystand_skill_scope_denied"


def test_mystand_skill_schema_has_no_delete_or_supporting_file_actions():
    action = bridge.MYSTAND_SKILL_MANAGE_SCHEMA["parameters"]["properties"]["action"]
    assert action["enum"] == ["create", "edit", "patch"]
    properties = bridge.MYSTAND_SKILL_MANAGE_SCHEMA["parameters"]["properties"]
    assert "file_path" not in properties
    assert "file_content" not in properties
