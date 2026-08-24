from unittest.mock import patch

from tools import approval as approval


def test_mystand_action_approval_never_reuses_or_persists_session_grants():
    session_key = "api_server:test-session"
    with patch.object(approval, "get_current_session_key", return_value=session_key), \
         patch.object(approval, "_is_gateway_approval_context", return_value=True), \
         patch.dict(approval._gateway_notify_cbs, {session_key: lambda _data: None}, clear=True), \
         patch.object(approval, "_await_gateway_decision", return_value={"resolved": True, "choice": "session"}), \
         patch.object(approval, "is_approved") as cached_approval, \
         patch.object(approval, "approve_session") as persist_approval:
        result = approval.request_gateway_action_approval(
            pattern_key="mystand-skill:create:test",
            description="创建测试 Skill",
        )

    assert result["approved"] is True
    cached_approval.assert_not_called()
    persist_approval.assert_not_called()
