"""Restricted, approval-bound Skill authoring for the My Stand owner chat."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml

from gateway.session_context import get_session_env
from tools.approval import request_gateway_action_approval
from tools.registry import registry
from xiaoban.mystand_owner import is_configured_mystand_owner


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_ACTIONS = {"create", "edit", "patch"}
_CATEGORY = "mystand"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _error(message: str, code: str, status: int = 400) -> str:
    return _json({"success": False, "status": status, "code": code, "error": message})


def _frontmatter(content: object) -> dict:
    text = str(content or "")
    if not text.startswith("---\n"):
        return {}
    closing = text.find("\n---\n", 4)
    if closing < 0:
        return {}
    try:
        value = yaml.safe_load(text[4:closing])
    except yaml.YAMLError:
        return {}
    return value if isinstance(value, dict) else {}


def _frontmatter_name(content: object) -> str:
    return str(_frontmatter(content).get("name") or "").strip()


def _owner_available_tool_names() -> set[str]:
    """Resolve the real My Stand owner tool policy without duplicating it."""
    from gateway.platforms.api_server import APIServerAdapter
    from toolsets import resolve_multiple_toolsets

    toolsets = APIServerAdapter._toolsets_for_request_policy("mystand-owner")
    return set(resolve_multiple_toolsets(toolsets))


def _validate_mystand_contract(content: object, expected_name: str) -> tuple[list[str], str]:
    """Validate the small My Stand-specific contract layered on SKILL.md."""
    frontmatter = _frontmatter(content)
    if str(frontmatter.get("name") or "").strip() != expected_name:
        return [], "SKILL.md frontmatter 的 name 必须与目标 Skill 名称完全一致。"

    metadata = frontmatter.get("metadata")
    mystand = metadata.get("mystand") if isinstance(metadata, dict) else None
    if not isinstance(mystand, dict):
        return [], "SKILL.md 必须声明 metadata.mystand.tools，用来核对它依赖的 Tool。"

    tools = mystand.get("tools")
    if not isinstance(tools, list):
        return [], "metadata.mystand.tools 必须是 Tool 名称数组；没有依赖时使用空数组。"

    normalized: list[str] = []
    for value in tools:
        name = str(value or "").strip()
        if not _TOOL_NAME_RE.fullmatch(name):
            return [], f"metadata.mystand.tools 包含无效 Tool 名称：{name or '<empty>'}。"
        if name in normalized:
            return [], f"metadata.mystand.tools 重复声明了 Tool：{name}。"
        normalized.append(name)

    missing = [name for name in normalized if registry.get_entry(name) is None]
    if missing:
        return [], f"Skill 依赖的 Tool 尚未注册：{', '.join(missing)}。"

    if not normalized:
        return [], ""

    available = _owner_available_tool_names()
    unavailable = [name for name in normalized if name not in available]
    if unavailable:
        return [], f"Skill 依赖的 Tool 未进入 My Stand 管理员工具集：{', '.join(unavailable)}。"
    return normalized, ""


def _mystand_skill_root(skill_manager) -> Path:
    return (skill_manager.SKILLS_DIR / _CATEGORY).resolve()


def _owned_existing(skill_manager, name: str):
    existing = skill_manager._find_skill(name)
    if not existing:
        return None
    try:
        path = Path(existing["path"]).resolve()
    except (KeyError, OSError, RuntimeError):
        return False
    return existing if path.parent == _mystand_skill_root(skill_manager) else False


def _scan_and_rollback(
    *,
    skill_manager,
    skill_dir: Path,
    action: str,
    original_content: str | None,
) -> str:
    skill_error = ""
    try:
        content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        _, skill_error = _validate_mystand_contract(content, skill_dir.name)
        if not skill_error:
            from tools.skills_guard import format_scan_report, scan_skill, should_allow_install

            scan = scan_skill(skill_dir, source="agent-created")
            allowed, reason = should_allow_install(scan)
            if allowed is not True:
                skill_error = f"安全扫描拒绝了这个 Skill（{reason}）：\n{format_scan_report(scan)}"
    except Exception:
        skill_error = "Skill 合同或安全扫描没有成功完成。"
    if not skill_error:
        return ""
    try:
        if action == "create":
            if skill_dir.parent == _mystand_skill_root(skill_manager):
                shutil.rmtree(skill_dir)
        elif original_content is not None:
            skill_manager._atomic_write_text(skill_dir / "SKILL.md", original_content)
    finally:
        try:
            from agent.prompt_builder import clear_skills_system_prompt_cache

            clear_skills_system_prompt_cache(clear_snapshot=True)
        except Exception:
            pass
    return skill_error


def mystand_skill_manage_handler(args, **_kwargs) -> str:
    if not isinstance(args, dict):
        return _error("Skill 参数必须是对象。", "invalid_mystand_skill_arguments")
    if get_session_env("XIAOBAN_SESSION_PLATFORM", "").strip().lower() != "api_server":
        return _error("该工具只允许 My Stand 已登录网页会话使用。", "mystand_session_required", 403)
    session_user_id = get_session_env("XIAOBAN_SESSION_USER_ID", "").strip()
    if not session_user_id:
        return _error("当前 My Stand 登录身份无效。", "mystand_session_required", 403)
    if not is_configured_mystand_owner(session_user_id):
        return _error(
            "只有当前配置的 My Stand owner 账号可以创建、修改或安排共享 Skill。",
            "mystand_skill_owner_required",
            403,
        )
    action = str(args.get("action") or "").strip().lower()
    name = str(args.get("name") or "").strip()
    if action not in _ALLOWED_ACTIONS:
        return _error("My Stand 只允许创建、完整更新或定点修订 Skill。", "mystand_skill_action_denied", 403)
    if not _NAME_RE.fullmatch(name):
        return _error("Skill 名称只能使用小写字母、数字、短横线或下划线，最长 64 个字符。", "invalid_mystand_skill_name")

    from tools import skill_manager_tool as skill_manager

    existing = _owned_existing(skill_manager, name)
    if action == "create" and existing is not None:
        return _error("同名 Skill 已存在，不能覆盖创建。", "mystand_skill_already_exists", 409)
    if action != "create" and existing is None:
        return _error("没有找到这个 My Stand 自建 Skill。", "mystand_skill_not_found", 404)
    if existing is False:
        return _error("只能修改 mystand 分类下由 My Stand 创建的 Skill。", "mystand_skill_scope_denied", 403)

    content = args.get("content")
    old_string = args.get("old_string")
    new_string = args.get("new_string")
    if action in {"create", "edit"}:
        if not isinstance(content, str) or not content.strip():
            return _error("创建或完整更新 Skill 时必须提供完整 SKILL.md。", "mystand_skill_content_required")
        if _frontmatter_name(content) != name:
            return _error("SKILL.md frontmatter 的 name 必须与目标 Skill 名称完全一致。", "mystand_skill_name_mismatch")
        _, contract_error = _validate_mystand_contract(content, name)
        if contract_error:
            return _error(contract_error, "mystand_skill_contract_invalid")
    if action == "patch":
        if not isinstance(old_string, str) or not old_string:
            return _error("定点修订必须提供 old_string。", "mystand_skill_patch_source_required")
        if not isinstance(new_string, str):
            return _error("定点修订必须提供 new_string。", "mystand_skill_patch_target_required")

    approval = request_gateway_action_approval(
        pattern_key=f"mystand-skill:{action}:{name}",
        description=f"持久化{ {'create': '创建', 'edit': '更新', 'patch': '修订'}[action] } My Stand Skill：{name}",
        command=f"My Stand Skill {action}: {name}",
        surface="mystand-skill",
    )
    if not approval.get("approved"):
        return _error(str(approval.get("message") or "Skill 操作未获确认。"), "mystand_skill_approval_denied", 403)

    skill_dir = _mystand_skill_root(skill_manager) / name
    original_content = None
    if action != "create":
        try:
            original_content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        except OSError:
            return _error("读取现有 Skill 失败，操作没有执行。", "mystand_skill_read_failed", 500)
    bypass_token = skill_manager._skill_gate_bypass.set(True)
    try:
        raw = skill_manager.skill_manage(
            action=action,
            name=name,
            content=content if action in {"create", "edit"} else None,
            category=_CATEGORY if action == "create" else None,
            old_string=old_string if action == "patch" else None,
            new_string=new_string if action == "patch" else None,
            replace_all=bool(args.get("replace_all", False)),
        )
    finally:
        skill_manager._skill_gate_bypass.reset(bypass_token)
    try:
        result = json.loads(raw)
    except (TypeError, ValueError):
        return _error("Skill 管理器返回异常。", "mystand_skill_invalid_result", 502)
    if not isinstance(result, dict) or result.get("success") is not True:
        message = str(result.get("error") or "Skill 操作失败。") if isinstance(result, dict) else "Skill 操作失败。"
        message = message.replace(str(skill_manager.SKILLS_DIR), "skills")
        return _error(message[:1000], "mystand_skill_write_failed")
    scan_error = _scan_and_rollback(
        skill_manager=skill_manager,
        skill_dir=skill_dir,
        action=action,
        original_content=original_content,
    )
    if scan_error:
        return _error(scan_error[:2000], "mystand_skill_security_blocked", 403)
    persisted_content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    declared_tools, _ = _validate_mystand_contract(persisted_content, name)
    return _json({
        "success": True,
        "action": action,
        "name": name,
        "category": _CATEGORY,
        "path": f"{_CATEGORY}/{name}/SKILL.md",
        "tools": declared_tools,
        "toolRegistrationVerified": True,
        "message": f"My Stand Skill『{name}』已{ {'create': '创建', 'edit': '更新', 'patch': '修订'}[action] }。",
    })


MYSTAND_SKILL_MANAGE_SCHEMA = {
    "name": "mystand_skill_manage",
    "description": (
        "仅当前 My Stand 登录账号与服务端配置的 owner 身份一致，且用户明确要求制作、学习或更新可复用工作流程时，创建或修订全站共享 Skill。"
        "只能操作 mystand 分类下的 SKILL.md，不支持删除、脚本、附件或其他目录；每次持久化都需要用户在网页确认。"
        "SKILL.md 必须在 metadata.mystand.tools 中声明依赖的 Tool；写入前会自动核对 ToolRegistry 和管理员工具集。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "edit", "patch"]},
            "name": {"type": "string", "description": "小写 Skill 名称。"},
            "content": {"type": "string", "description": "create/edit 时的完整 SKILL.md；frontmatter 必须包含 metadata: {mystand: {tools: [实际依赖的 Tool 名称]}}，无依赖时写空数组。"},
            "old_string": {"type": "string", "description": "patch 时要替换的原文。"},
            "new_string": {"type": "string", "description": "patch 时的替换文本，可为空字符串。"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["action", "name"],
        "additionalProperties": False,
    },
}


registry.register(
    name="mystand_skill_manage",
    toolset="mystand_skill_manage",
    schema=MYSTAND_SKILL_MANAGE_SCHEMA,
    handler=mystand_skill_manage_handler,
    requires_env=[],
    is_async=False,
    description=MYSTAND_SKILL_MANAGE_SCHEMA["description"],
    emoji="📚",
)
