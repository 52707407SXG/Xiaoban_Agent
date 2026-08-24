"""Owner-scoped, approval-bound Cron management for the My Stand chat."""

from __future__ import annotations

import json
import re

from gateway.session_context import get_session_env
from tools.approval import request_gateway_action_approval
from tools.registry import registry


_ACTIONS = {"create", "list", "update", "pause", "resume", "remove", "run"}
_MUTATIONS = _ACTIONS - {"list"}
_SAFE_CRON_TOOLSETS = ["web", "skills_readonly"]
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:/.-]{0,127}$")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _error(message: str, code: str, status: int = 400) -> str:
    return _json({"success": False, "status": status, "code": code, "error": message})


def _session() -> tuple[str, str]:
    return (
        get_session_env("XIAOBAN_SESSION_PLATFORM", "").strip().lower(),
        get_session_env("XIAOBAN_SESSION_USER_ID", "").strip(),
    )


def _owned_jobs(user_id: str, *, include_disabled: bool = True) -> list[dict]:
    from cron.jobs import list_jobs

    result = []
    for job in list_jobs(include_disabled=include_disabled):
        origin = job.get("origin") if isinstance(job.get("origin"), dict) else {}
        if (
            str(origin.get("platform") or "").strip().lower() == "api_server"
            and str(origin.get("user_id") or "").strip() == user_id
        ):
            result.append(job)
    return result


def _resolve_owned_job(reference: str, user_id: str):
    clean_ref = str(reference or "").strip()
    if not clean_ref:
        return None, "missing"
    jobs = _owned_jobs(user_id)
    exact_id = [job for job in jobs if str(job.get("id") or "") == clean_ref]
    if len(exact_id) == 1:
        return exact_id[0], ""
    exact_name = [job for job in jobs if str(job.get("name") or "") == clean_ref]
    if len(exact_name) == 1:
        return exact_name[0], ""
    if len(exact_name) > 1:
        return None, "ambiguous"
    return None, "not_found"


def _public_job(job: dict) -> dict:
    repeat = job.get("repeat") if isinstance(job.get("repeat"), dict) else {}
    prompt = str(job.get("prompt") or "")
    return {
        "job_id": str(job.get("id") or ""),
        "name": str(job.get("name") or ""),
        "prompt_preview": prompt[:100] + ("..." if len(prompt) > 100 else ""),
        "schedule": str(job.get("schedule_display") or ""),
        "skills": list(job.get("skills") or []),
        "repeat": {
            "times": repeat.get("times"),
            "completed": int(repeat.get("completed") or 0),
        },
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "last_delivery_error": job.get("last_delivery_error"),
        "enabled": bool(job.get("enabled", True)),
        "state": str(job.get("state") or "scheduled"),
        "delivery": "My Stand 当前对话",
    }


def _skills(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("skills 必须是不超过 8 项的数组")
    result: list[str] = []
    for item in value:
        name = str(item or "").strip()
        if not _SKILL_NAME_RE.fullmatch(name):
            raise ValueError("skills 包含无效名称")
        if name not in result:
            result.append(name)
    return result


def mystand_cron_handler(args, **kwargs) -> str:
    if not isinstance(args, dict):
        return _error("Cron 参数必须是对象。", "invalid_mystand_cron_arguments")
    platform, user_id = _session()
    if platform != "api_server" or not user_id:
        return _error("该工具只允许 My Stand 已登录网页会话使用。", "mystand_session_required", 403)
    action = str(args.get("action") or "").strip().lower()
    if action not in _ACTIONS:
        return _error("不支持这个 Cron 操作。", "invalid_mystand_cron_action")
    try:
        skills = _skills(args.get("skills"))
    except ValueError as exc:
        return _error(str(exc), "invalid_mystand_cron_skills")

    if action == "list":
        jobs = [_public_job(job) for job in _owned_jobs(user_id)]
        return _json({"success": True, "count": len(jobs), "jobs": jobs})

    job = None
    if action != "create":
        job, resolve_error = _resolve_owned_job(args.get("job_id"), user_id)
        if resolve_error == "missing":
            return _error("这个操作必须提供 job_id。", "mystand_cron_job_required")
        if resolve_error == "ambiguous":
            return _error("找到多个同名定时任务，请先列出任务并使用准确 job_id。", "mystand_cron_job_ambiguous", 409)
        if job is None:
            return _error("没有找到当前账号创建的这个 My Stand 定时任务。", "mystand_cron_job_not_found", 404)

    prompt = args.get("prompt")
    schedule = args.get("schedule")
    if prompt is not None:
        prompt = str(prompt).strip()
        if not prompt or len(prompt) > 8_000:
            return _error("Cron prompt 必须是 1-8000 个字符。", "invalid_mystand_cron_prompt")
    if schedule is not None:
        schedule = str(schedule).strip()
        if not schedule or len(schedule) > 200:
            return _error("Cron schedule 无效。", "invalid_mystand_cron_schedule")
    repeat = args.get("repeat")
    if repeat is not None and (isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0 or repeat > 10_000):
        return _error("repeat 必须是 0-10000 之间的整数。", "invalid_mystand_cron_repeat")
    if action == "create" and prompt is None:
        return _error("创建 Cron 必须提供完整自包含 prompt。", "mystand_cron_prompt_required")
    if action == "create" and schedule is None:
        return _error("创建 Cron 必须提供 schedule。", "mystand_cron_schedule_required")
    if schedule is not None:
        try:
            from cron.jobs import parse_schedule

            parse_schedule(schedule)
        except (TypeError, ValueError) as exc:
            return _error(str(exc)[:1000], "invalid_mystand_cron_schedule")
    if action == "update" and not any(
        key in args for key in ("prompt", "schedule", "name", "repeat", "skills")
    ):
        return _error("更新 Cron 时至少提供一个要修改的字段。", "mystand_cron_update_required")

    name = str(args.get("name") or (job or {}).get("name") or "定时任务").strip()[:120]
    approval = request_gateway_action_approval(
        pattern_key=f"mystand-cron:{action}",
        description=f"{ {'create': '创建', 'update': '更新', 'pause': '暂停', 'resume': '恢复', 'remove': '删除', 'run': '立即执行'}[action] } My Stand 定时任务：{name}",
        command=f"My Stand Cron {action}: {name}",
        surface="mystand-cron",
    )
    if not approval.get("approved"):
        return _error(str(approval.get("message") or "Cron 操作未获确认。"), "mystand_cron_approval_denied", 403)

    from tools.cronjob_tools import cronjob

    raw = cronjob(
        action=action,
        job_id=str(job.get("id")) if job else None,
        prompt=prompt,
        schedule=schedule,
        name=str(args.get("name") or "").strip()[:120] or None,
        repeat=repeat,
        deliver="origin" if action in {"create", "update"} else None,
        include_disabled=True,
        skills=skills,
        enabled_toolsets=_SAFE_CRON_TOOLSETS if action in {"create", "update"} else None,
        reason=str(args.get("reason") or "").strip()[:300] or None,
        task_id=kwargs.get("task_id"),
    )
    try:
        result = json.loads(raw)
    except (TypeError, ValueError):
        return _error("Cron 管理器返回异常。", "mystand_cron_invalid_result", 502)
    if not isinstance(result, dict) or result.get("success") is not True:
        return _error(
            str(result.get("error") or "Cron 操作失败。")[:1000] if isinstance(result, dict) else "Cron 操作失败。",
            "mystand_cron_operation_failed",
        )
    if action == "remove":
        return _json({"success": True, "action": action, "message": f"定时任务『{name}』已删除。"})
    from cron.jobs import get_job

    current_id = str(result.get("job_id") or result.get("job", {}).get("job_id") or (job or {}).get("id") or "")
    current = get_job(current_id) if current_id else None
    return _json({
        "success": True,
        "action": action,
        "job": _public_job(current) if current else None,
        "message": str(result.get("message") or f"定时任务『{name}』操作完成。"),
    })


MYSTAND_CRON_SCHEMA = {
    "name": "mystand_cron",
    "description": (
        "在 My Stand 当前对话中创建、查看、更新、暂停、恢复、删除或立即运行定时任务。"
        "任务固定回传当前 My Stand 对话，只能使用公开网页检索和只读 Skill，不能运行脚本、终端、代码或读取 My Stand 私有业务资料。"
        "除查看外的操作都需要用户在网页确认。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "update", "pause", "resume", "remove", "run"]},
            "job_id": {"type": "string", "description": "非 create/list 操作使用的准确任务 ID；不要猜，先 list。"},
            "prompt": {"type": "string", "description": "create 时的完整自包含任务，update 时可选。"},
            "schedule": {"type": "string", "description": "如 30m、every 2h、0 7 * * * 或 ISO 时间。"},
            "name": {"type": "string"},
            "repeat": {"type": "integer", "minimum": 0, "maximum": 10000},
            "skills": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
            "reason": {"type": "string", "description": "暂停原因。"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


registry.register(
    name="mystand_cron",
    toolset="mystand_cron",
    schema=MYSTAND_CRON_SCHEMA,
    handler=mystand_cron_handler,
    requires_env=[],
    is_async=False,
    description=MYSTAND_CRON_SCHEMA["description"],
    emoji="⏰",
)
