import json
from unittest.mock import patch

from tools import mystand_cron_tool as bridge


def _session_value(key, default=""):
    return {
        "XIAOBAN_SESSION_PLATFORM": "api_server",
        "XIAOBAN_SESSION_USER_ID": "owner-user-001",
    }.get(key, default)


def _job(job_id="job-own", user_id="owner-user-001"):
    return {
        "id": job_id,
        "name": "早报",
        "prompt": "查公开天气并给出早报",
        "schedule_display": "0 7 * * *",
        "skills": [],
        "repeat": {"times": None, "completed": 0},
        "origin": {"platform": "api_server", "chat_id": "web:test", "user_id": user_id},
        "enabled": True,
        "state": "scheduled",
    }


def test_mystand_cron_list_is_filtered_to_current_owner():
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch("cron.jobs.list_jobs", return_value=[_job(), _job("job-other", "OTHER")]):
        result = json.loads(bridge.mystand_cron_handler({"action": "list"}))
    assert result["count"] == 1
    assert result["jobs"][0]["job_id"] == "job-own"


def test_mystand_cron_create_forces_origin_delivery_and_safe_toolsets():
    created = _job()
    raw_result = json.dumps({"success": True, "job_id": "job-own", "message": "created"})
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(bridge, "request_gateway_action_approval", return_value={"approved": True}), \
         patch("tools.cronjob_tools.cronjob", return_value=raw_result) as cronjob, \
         patch("cron.jobs.get_job", return_value=created):
        result = json.loads(bridge.mystand_cron_handler({
            "action": "create",
            "name": "早报",
            "prompt": "查公开天气并给出早报",
            "schedule": "0 7 * * *",
        }))

    assert result["success"] is True
    kwargs = cronjob.call_args.kwargs
    assert kwargs["deliver"] == "origin"
    assert kwargs["enabled_toolsets"] == ["web", "skills_readonly"]
    assert "script" not in kwargs
    assert "workdir" not in kwargs
    assert "model" not in kwargs


def test_mystand_cron_cannot_mutate_another_users_job():
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch("cron.jobs.list_jobs", return_value=[_job("job-other", "OTHER")]), \
         patch("tools.cronjob_tools.cronjob") as cronjob:
        result = json.loads(bridge.mystand_cron_handler({
            "action": "remove",
            "job_id": "job-other",
        }))
    assert result["code"] == "mystand_cron_job_not_found"
    cronjob.assert_not_called()


def test_mystand_cron_validates_before_requesting_approval():
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(bridge, "request_gateway_action_approval") as approval:
        result = json.loads(bridge.mystand_cron_handler({
            "action": "create",
            "name": "坏任务",
            "prompt": "查公开天气",
            "schedule": "不是时间表达式",
        }))

    assert result["code"] == "invalid_mystand_cron_schedule"
    approval.assert_not_called()


def test_mystand_cron_schema_excludes_raw_execution_and_routing_controls():
    properties = bridge.MYSTAND_CRON_SCHEMA["parameters"]["properties"]
    for forbidden in (
        "deliver", "model", "provider", "base_url", "script", "no_agent",
        "context_from", "enabled_toolsets", "workdir",
    ):
        assert forbidden not in properties
