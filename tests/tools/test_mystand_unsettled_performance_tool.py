import json
from unittest.mock import patch

from tools import mystand_unsettled_performance_tool as bridge


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.payload


def _session_value(key, default=""):
    return {
        "XIAOBAN_SESSION_PLATFORM": "api_server",
        "XIAOBAN_SESSION_USER_ID": "52707407",
    }.get(key, default)


def test_exact_unsettled_tool_calls_only_dedicated_endpoint():
    payload = {
        "ok": True,
        "counts": {"totalUnpaidRecordCount": 6, "readyToPayRecordCount": 4},
        "records": [{"id": "one", "readyToPay": True}],
    }
    with patch.object(bridge, "get_session_env", side_effect=_session_value), \
         patch.object(bridge, "_internal_token", return_value="token"), \
         patch.object(bridge, "_api_base_url", return_value="http://127.0.0.1:18083"), \
         patch.object(bridge, "mark_mystand_private_query_turn") as mark_private, \
         patch.object(bridge.urllib.request, "urlopen", return_value=_Response(payload)) as urlopen:
        result = json.loads(bridge.mystand_unsettled_performance_handler({}))

    assert result["ok"] is True
    assert json.loads(result["content"])["counts"] == payload["counts"]
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:18083/api/xiaoban/internal/finance/unsettled-ready"
    assert json.loads(request.data)["userId"] == "52707407"
    mark_private.assert_called_once_with()


def test_exact_unsettled_tool_rejects_filters_and_non_mystand_sessions():
    assert json.loads(bridge.mystand_unsettled_performance_handler({"year": 2026}))["code"] == "invalid_unsettled_performance_arguments"
    with patch.object(bridge, "get_session_env", return_value=""):
        result = json.loads(bridge.mystand_unsettled_performance_handler({}))
    assert result["status"] == 403
