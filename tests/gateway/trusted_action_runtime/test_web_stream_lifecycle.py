"""波次 2 RED：My Stand 流式（web）trusted 生命周期接缝断言（执行单 §B / G1-G5）。

- R1（G2）/R2（G1）/R3（G3）/R5b（G5）/R6（G4）在 base 149ccb7 上必须失败；
- R4/R5a 是既有行为表征（Guard 替换文案、普通聊天无 verification 事件），必须现在通过。

全部 fixture 均为脱敏虚构值；不打真实网络/模型；断言只落在 HTTP 层、
SSE 帧和 begin_turn 调用边界上。
"""

import asyncio
import hashlib
import json
import threading
import time
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from xiaoban.trusted_runtime.paid_call_policy import (
    SIGNED_MYSTAND_AGENT_POLICY_REVISION,
    SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
)
from xiaoban.trusted_runtime.protocol_contract import (
    MYSTAND_EVIDENCE_REQUIRED_HEADER,
    TRUSTED_RUNTIME_CONTRACT_DIGEST,
    TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER,
    TRUSTED_RUNTIME_CONTRACT_REVISION,
    TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER,
)

_USAGE = {
    "input_tokens": 1,
    "output_tokens": 1,
    "total_tokens": 2,
    "agent_calls": {
        "schema": "mystand.agent-call-usage.v1",
        "executionId": "b" * 32,
        "status": "completed",
        "calls": [
            {
                "callId": f"{'b' * 32}:call:000001",
                "ordinal": 1,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "role": "agent",
                "startedAtMs": 1,
                "endedAtMs": 2,
                "status": "completed",
                "inputTokens": 1,
                "outputTokens": 1,
                "totalTokens": 2,
                "cachedInputTokens": 0,
                "usageStatus": "reported",
            }
        ],
    },
}

# 脱敏虚构 delivery 身份：每个用例独立的 delivery-id，避免共享 _idem_cache 串扰。
DELIVERY_R1 = "xbd_" + "01" * 20
DELIVERY_R2 = "xbd_" + "02" * 20
DELIVERY_R3 = "xbd_" + "03" * 20
DELIVERY_R4 = "xbd_" + "04" * 20
DELIVERY_R5A = "xbd_" + "05" * 20
DELIVERY_R5B = "xbd_" + "06" * 20
DELIVERY_R6A = "xbd_" + "07" * 20
DELIVERY_R6B = "xbd_" + "08" * 20
DELIVERY_R3_USER = "xbd_" + "09" * 20
DELIVERY_R3_SITE = "xbd_" + "0a" * 20
DELIVERY_R2_ATTEMPT = "xbd_" + "0b" * 20
DELIVERY_R5_DENIED = "xbd_" + "0c" * 20
DELIVERY_DYNAMIC_INVALID = "xbd_" + "0e" * 20
DELIVERY_DYNAMIC_CONFLICT = "xbd_" + "0f" * 20
DELIVERY_DYNAMIC_WORK = "xbd_" + "10" * 20
DELIVERY_DYNAMIC_CHAT = "xbd_" + "11" * 20
DELIVERY_DYNAMIC_DIAGNOSTIC = "xbd_" + "12" * 20
DELIVERY_DYNAMIC_TOOL_MODE_CONFLICT = "xbd_" + "13" * 20
FINGERPRINT_A = hashlib.sha256(b"wave2-stream-fingerprint-a").hexdigest()
FINGERPRINT_B = hashlib.sha256(b"wave2-stream-fingerprint-b").hexdigest()


def _make_adapter() -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={"key": "sk-secret"})
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/chat/completions/stop", adapter._handle_stop_idempotent_chat_completion)
    return app


def _mystand_stream_headers(
    delivery_id: str,
    *,
    user: str = "alice",
    site: str = "mystand-test-site",
    attempt: str = "1",
    fingerprint: str = FINGERPRINT_A,
) -> dict[str, str]:
    """My Stand 流式全套可信 header（按 G1 合同全部合法），脱敏虚构。"""
    return {
        "Authorization": "Bearer sk-secret",
        "X-Xiaoban-Site-Id": site,
        "X-Xiaoban-User-Id": user,
        "X-Xiaoban-Toolset-Policy": "mystand-broker-basic",
        "X-Xiaoban-Memory-Mode": "disabled",
        "X-Xiaoban-Session-Key": f"session-{user}",
        "X-Xiaoban-Session-Id": f"session-{user}",
        "X-Xiaoban-Message-Id": f"message-{delivery_id}",
        "X-Xiaoban-Delivery-Id": delivery_id,
        "X-Xiaoban-Attempt": attempt,
        "X-Xiaoban-Request-Fingerprint": fingerprint,
        SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER: (
            SIGNED_MYSTAND_AGENT_POLICY_REVISION
        ),
        TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER: (
            TRUSTED_RUNTIME_CONTRACT_REVISION
        ),
        TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER: (
            TRUSTED_RUNTIME_CONTRACT_DIGEST
        ),
    }


def _stream_body(message: str) -> dict:
    return {"model": "test", "messages": [{"role": "user", "content": message}], "stream": True}


def _dynamic_stream_headers(
    delivery_id: str,
    *,
    evidence_required: str = "0",
    business_tool_mode: str = "enabled",
) -> dict[str, str]:
    headers = _mystand_stream_headers(delivery_id)
    headers.update({
        "X-Xiaoban-Completion-Protocol": "dynamic-evidence-v2",
        MYSTAND_EVIDENCE_REQUIRED_HEADER: evidence_required,
        "X-Xiaoban-Business-Tool-Mode": business_tool_mode,
        "X-Xiaoban-Delivery-Attempt": "1",
        "X-Xiaoban-Invocation-Fingerprint": FINGERPRINT_B,
    })
    return headers


def _mock_agent(final_response: str) -> MagicMock:
    agent = MagicMock()
    agent.provider = "deepseek"
    agent.model = "deepseek-v4-pro"
    agent.max_iterations = 8
    agent.run_conversation.return_value = {"final_response": final_response, "messages": []}
    agent.session_prompt_tokens = 1
    agent.session_completion_tokens = 1
    agent.session_total_tokens = 2
    agent.valid_tool_names = []
    return agent


def _visible_sse_text(body: str) -> str:
    parts = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[len("data: "):])
        for choice in payload.get("choices", []):
            parts.append(choice.get("delta", {}).get("content", ""))
    return "".join(parts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "empty",
        "word",
        "number",
        "missing_tool_mode",
        "invalid_tool_mode",
        "without_protocol",
    ],
)
async def test_dynamic_evidence_header_rejects_before_agent_dispatch(
    mutation,
):
    adapter = _make_adapter()
    headers = _dynamic_stream_headers(DELIVERY_DYNAMIC_INVALID)
    if mutation == "missing":
        headers.pop(MYSTAND_EVIDENCE_REQUIRED_HEADER)
    elif mutation == "empty":
        headers[MYSTAND_EVIDENCE_REQUIRED_HEADER] = ""
    elif mutation == "word":
        headers[MYSTAND_EVIDENCE_REQUIRED_HEADER] = "true"
    elif mutation == "number":
        headers[MYSTAND_EVIDENCE_REQUIRED_HEADER] = "2"
    elif mutation == "missing_tool_mode":
        headers.pop("X-Xiaoban-Business-Tool-Mode")
    elif mutation == "invalid_tool_mode":
        headers["X-Xiaoban-Business-Tool-Mode"] = "diagnostic"
    else:
        headers.pop("X-Xiaoban-Completion-Protocol")
    mock_run = AsyncMock(
        return_value=({"final_response": "不得执行", "messages": []}, _USAGE)
    )
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/chat/completions",
                headers=headers,
                json=_stream_body("请处理这件事"),
            )
            body = await response.text()

    assert response.status == 400, body
    assert json.loads(body)["error"]["code"] == (
        "invalid_completion_protocol"
    )
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_dynamic_work_bit_is_idempotency_bound():
    adapter = _make_adapter()
    mock_run = AsyncMock(
        return_value=({"final_response": "中性回复", "messages": []}, _USAGE)
    )
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/v1/chat/completions",
                headers=_dynamic_stream_headers(
                    DELIVERY_DYNAMIC_CONFLICT,
                    evidence_required="0",
                ),
                json=_stream_body("请处理这件事"),
            )
            await first.read()
            second = await cli.post(
                "/v1/chat/completions",
                headers=_dynamic_stream_headers(
                    DELIVERY_DYNAMIC_CONFLICT,
                    evidence_required="1",
                ),
                json=_stream_body("请处理这件事"),
            )
            second_body = await second.text()

    assert first.status == 200
    assert second.status == 409, second_body
    assert mock_run.call_count == 1


@pytest.mark.asyncio
async def test_dynamic_business_tool_mode_is_idempotency_bound():
    adapter = _make_adapter()
    mock_run = AsyncMock(
        return_value=({"final_response": "中性回复", "messages": []}, _USAGE)
    )
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/v1/chat/completions",
                headers=_dynamic_stream_headers(
                    DELIVERY_DYNAMIC_TOOL_MODE_CONFLICT,
                    business_tool_mode="enabled",
                ),
                json=_stream_body("请解释上一轮"),
            )
            await first.read()
            second = await cli.post(
                "/v1/chat/completions",
                headers=_dynamic_stream_headers(
                    DELIVERY_DYNAMIC_TOOL_MODE_CONFLICT,
                    business_tool_mode="disabled",
                ),
                json=_stream_body("请解释上一轮"),
            )
            second_body = await second.text()

    assert first.status == 200
    assert second.status == 409, second_body
    assert mock_run.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_normal_diagnostic_tool_mode_reaches_stream_and_nonstream_runner(
    monkeypatch,
    tmp_path,
    stream,
):
    from gateway.platforms import api_server

    adapter = _make_adapter()
    delivery_id = "xbd_" + ("14" if stream else "15") * 20
    headers = _dynamic_stream_headers(
        delivery_id,
        evidence_required="0",
        business_tool_mode="disabled",
    )
    if not stream:
        headers["Idempotency-Key"] = delivery_id
    mock_run = AsyncMock(
        return_value=(
            {"final_response": "上一轮没有形成可用结果。", "messages": []},
            _USAGE,
        ),
    )
    cache = api_server._IdempotencyCache(
        durable_path=str(tmp_path / f"normal-diagnostic-{stream}.sqlite"),
    )
    monkeypatch.setattr(api_server, "_idem_cache", cache)
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "test",
                    "messages": [{
                        "role": "user",
                        "content": (
                            "刚才查 AUTH-ABCDEFG 为什么失败？"
                            "不要索引，直接读库"
                        ),
                    }],
                    "stream": stream,
                },
            )
            await response.read()
    cache._durable.close()

    assert response.status == 200
    mock_run.assert_awaited_once()
    runner_kwargs = mock_run.await_args.kwargs
    assert runner_kwargs["business_tools_disabled"] is True
    assert runner_kwargs["dynamic_evidence_required"] is False
    assert runner_kwargs["true_moa_snapshot"] is None


@pytest.mark.asyncio
async def test_dynamic_work_and_chat_use_only_server_bit_for_zero_call_egress():
    adapter = _make_adapter()
    answer = "同一条没有工具调用的模型正文"
    mock_agent = _mock_agent(answer)
    interaction_kinds: list[str] = []
    business_tool_modes: list[bool] = []
    from xiaoban.trusted_runtime.turns import begin_turn as real_begin_turn

    def _recording_begin_turn(*args, **kwargs):
        turn = real_begin_turn(*args, **kwargs)
        interaction_kinds.append(turn.interaction_kind)
        business_tool_modes.append(turn.business_tools_disabled)
        return turn

    app = _create_app(adapter)
    with (
        patch.object(adapter, "_create_agent", return_value=mock_agent),
        patch(
            "xiaoban.trusted_runtime.turns.begin_turn",
            new=_recording_begin_turn,
        ),
    ):
        async with TestClient(TestServer(app)) as cli:
            work = await cli.post(
                "/v1/chat/completions",
                headers=_dynamic_stream_headers(
                    DELIVERY_DYNAMIC_WORK,
                    evidence_required="1",
                ),
                json=_stream_body("请处理这件事"),
            )
            work_body = await work.text()
            chat = await cli.post(
                "/v1/chat/completions",
                headers=_dynamic_stream_headers(
                    DELIVERY_DYNAMIC_CHAT,
                    evidence_required="0",
                ),
                json=_stream_body("请处理这件事"),
            )
            chat_body = await chat.text()
            diagnostic = await cli.post(
                "/v1/chat/completions",
                headers=_dynamic_stream_headers(
                    DELIVERY_DYNAMIC_DIAGNOSTIC,
                    evidence_required="0",
                    business_tool_mode="disabled",
                ),
                json=_stream_body("请解释上一轮为什么失败"),
            )
            diagnostic_body = await diagnostic.text()

    assert work.status == chat.status == diagnostic.status == 200
    assert interaction_kinds == ["WORK", "CHAT", "CHAT"]
    assert business_tool_modes == [False, False, True]
    assert _visible_sse_text(work_body) == ""
    assert "这轮我没有真正查到站内资料" not in work_body
    assert "event: xiaoban.error" in work_body
    assert _visible_sse_text(chat_body) == answer
    assert _visible_sse_text(diagnostic_body) == answer


# R1（§B R1 / G2）：trusted turn request_id 绑定 Delivery-Id 原值


@pytest.mark.asyncio
async def test_r1_stream_trusted_turn_request_id_binds_delivery_id():
    """执行单 §B R1（G2）：begin_turn 的 request_id 必须 == X-Xiaoban-Delivery-Id。

    base 上 request_id 是 ``mystand-req-<uuid4>`` 随机值，本断言失败。
    """
    adapter = _make_adapter()
    captured: list[dict] = []
    from xiaoban.trusted_runtime.turns import begin_turn as real_begin_turn

    def _recording_begin_turn(*args, **kwargs):
        captured.append(kwargs)
        return real_begin_turn(*args, **kwargs)

    mock_agent = _mock_agent("你好，我是站小伴。")
    app = _create_app(adapter)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(adapter, "_create_agent", return_value=mock_agent)
        )
        # _run_agent 当前在函数内 import begin_turn；同时防御 GREEN 改成模块级 import。
        stack.enter_context(
            patch(
                "xiaoban.trusted_runtime.turns.begin_turn",
                new=_recording_begin_turn,
            )
        )
        stack.enter_context(
            patch(
                "gateway.platforms.api_server.begin_turn",
                new=_recording_begin_turn,
                create=True,
            )
        )
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(DELIVERY_R1),
                json=_stream_body("你好"),
            )
            await resp.read()

    assert resp.status == 200
    assert captured, "mystand 流式回合必须创建 trusted turn"
    assert any(
        kwargs.get("request_id") == DELIVERY_R1 for kwargs in captured
    ), "trusted turn request_id 必须等于 X-Xiaoban-Delivery-Id 原值，禁止随机生成"


# R2（§B R2 / G1）：流式 delivery 身份校验


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["missing_delivery_id", "malformed_fingerprint", "missing_attempt"],
)
async def test_r2_stream_rejects_invalid_delivery_identity(mutation):
    """执行单 §B R2（G1）：缺 Delivery-Id / 畸形 fingerprint / 缺 Attempt
    必须 400 + invalid_delivery_identity 且 Agent 零调用。base 不校验，失败。"""
    adapter = _make_adapter()
    headers = _mystand_stream_headers(DELIVERY_R2)
    if mutation == "missing_delivery_id":
        headers.pop("X-Xiaoban-Delivery-Id")
    elif mutation == "malformed_fingerprint":
        headers["X-Xiaoban-Request-Fingerprint"] = "not-a-sha256-fingerprint"
    else:
        headers.pop("X-Xiaoban-Attempt")

    mock_run = AsyncMock(return_value=({"final_response": "ok", "messages": []}, _USAGE))
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=headers,
                json=_stream_body("你好"),
            )
            body = await resp.text()

    assert resp.status == 400, body
    data = json.loads(body)
    assert data["error"]["code"] == "invalid_delivery_identity"
    mock_run.assert_not_called()


# R3（§B R3 / G3）：流式幂等登记与 fingerprint 冲突


@pytest.mark.asyncio
async def test_r3_stream_replay_with_changed_fingerprint_conflicts():
    """执行单 §B R3（G3）：同 delivery-id+attempt 换 fingerprint 必须 409
    且 Agent 调用数仍为 1。base 流式不入 _idem_cache，第二发照常执行，失败。"""
    adapter = _make_adapter()
    mock_run = AsyncMock(
        return_value=({"final_response": "你好。", "messages": []}, _USAGE)
    )
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(
                    DELIVERY_R3, fingerprint=FINGERPRINT_A
                ),
                json=_stream_body("你好"),
            )
            await first.read()
            second = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(
                    DELIVERY_R3, fingerprint=FINGERPRINT_B
                ),
                json=_stream_body("你好"),
            )
            second_body = await second.text()

    assert first.status == 200
    assert second.status == 409, second_body
    assert mock_run.call_count == 1, "同 key 换 fingerprint 的重放不得再次执行 Agent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delivery_id", "changed_headers"),
    [
        (DELIVERY_R3_USER, {"user": "bob"}),
        (DELIVERY_R3_SITE, {"site": "mystand-other-site"}),
    ],
)
async def test_r3_stream_replay_with_changed_server_identity_conflicts(
    delivery_id,
    changed_headers,
):
    """同 delivery+attempt 换 owner/site 必须在第二次 Agent 执行前 409。

    scoped key 本身包含 owner/site，不能因此把相同 delivery 当成新请求。
    """
    adapter = _make_adapter()
    mock_run = AsyncMock(
        return_value=({"final_response": "你好。", "messages": []}, _USAGE)
    )
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(delivery_id),
                json=_stream_body("你好"),
            )
            await first.read()
            second = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(delivery_id, **changed_headers),
                json=_stream_body("你好"),
            )
            second_body = await second.text()

    assert first.status == 200
    assert second.status == 409, second_body
    assert mock_run.call_count == 1, "跨身份重放不得触发第二次 Agent"


@pytest.mark.asyncio
async def test_r2_stream_rejects_conflicting_attempt_headers():
    """两个服务端 attempt 头同时存在时必须完全一致，否则 Agent 零调用。"""
    adapter = _make_adapter()
    headers = _mystand_stream_headers(DELIVERY_R2_ATTEMPT, attempt="1")
    headers["X-Xiaoban-Delivery-Attempt"] = "2"
    mock_run = AsyncMock(
        return_value=({"final_response": "你好。", "messages": []}, _USAGE)
    )
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=headers,
                json=_stream_body("你好"),
            )
            body = await resp.text()

    assert resp.status == 400, body
    assert json.loads(body)["error"]["code"] == "invalid_delivery_identity"
    mock_run.assert_not_called()


# R4（§B R4，表征）：Guard 拒绝的业务回答在 SSE 里只有替换文案


@pytest.mark.asyncio
async def test_r4_guarded_stream_replaces_unverified_business_claim():
    """执行单 §B R4（表征，base 必须通过）：未验证写入成功_claim 的原始 delta
    与最终文本都不得出站，SSE 可见文本只有 Guard 替换文案。"""
    adapter = _make_adapter()
    claim = "滨江一号3栋802的钥匙状态已改为已交接。"

    async def _mock_run_agent(**kwargs):
        callback = kwargs.get("stream_delta_callback")
        if callback:
            callback(claim)
        return (
            {
                "_mystand_request": True,
                "_mystand_evidence_required": True,
                "final_response": claim,
                "messages": [
                    {"role": "user", "content": "确认"},
                    {"role": "assistant", "content": claim},
                ],
            },
            dict(_USAGE),
        )

    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(DELIVERY_R4),
                json=_stream_body("确认"),
            )
            body = await resp.text()

    assert resp.status == 200
    visible = _visible_sse_text(body)
    assert "已交接" not in visible
    assert "滨江一号" not in visible
    # 既有 Guard 替换文案（blocked_no_action_call 路径），表征锁定。
    assert "这轮我没有真正查到站内资料" in visible


# R5（§B R5 / G5）：xiaoban.trusted.verification 尾包事件


@pytest.mark.asyncio
async def test_r5a_plain_chat_stream_has_no_trusted_verification_event():
    """执行单 §B R5a（表征，base 必须通过）：无 action_results 的普通聊天
    流式响应不得出现 xiaoban.trusted.verification 事件。"""
    adapter = _make_adapter()
    mock_agent = _mock_agent("你好，我是站小伴。")
    app = _create_app(adapter)
    with patch.object(adapter, "_create_agent", return_value=mock_agent):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(DELIVERY_R5A),
                json=_stream_body("你好"),
            )
            body = await resp.text()

    assert resp.status == 200
    assert "xiaoban.trusted.verification" not in body


@pytest.mark.asyncio
async def test_plain_chat_forwards_real_deltas_without_guard_rewrite_or_duplicate():
    adapter = _make_adapter()
    answer_parts = ["你好，", "我是站小伴。"]

    async def _mock_run_agent(**kwargs):
        callback = kwargs.get("stream_delta_callback")
        for part in answer_parts:
            callback(part)
        return (
            {
                "_mystand_request": True,
                "final_response": "".join(answer_parts),
                "messages": [],
            },
            dict(_USAGE),
        )

    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers("xbd_" + "0d" * 20),
                json=_stream_body("你好，介绍一下你自己"),
            )
            body = await resp.text()

    assert resp.status == 200
    assert _visible_sse_text(body) == "".join(answer_parts)
    assert "没有真正查到站内资料" not in body
    assert "xiaoban.trusted.verification" not in body


@pytest.mark.asyncio
async def test_r5b_work_turn_stream_emits_trusted_verification_event(monkeypatch):
    """执行单 §B R5b（G5）：真实 trusted 证据的工作回合（索引 + 定向读取，
    action_results=2）必须在内容 chunk 之后、finish chunk 之前发出
    xiaoban.trusted.verification，request_id == delivery-id。base 无此事件，失败。"""
    monkeypatch.setattr(
        "tools.mystand_authorization_tool.mystand_authorization_tool_handler",
        lambda args: '{"ok":true,"content":"地址3401号"}',
    )
    monkeypatch.setattr(
        "tools.mystand_resource_index_tool.mystand_resource_index_tool_handler",
        lambda args: '{"ok":true,"items":[{"resourceUid":"res-demo-1","safeLabel":"档案"}]}',
    )
    adapter = _make_adapter()
    mock_agent = _mock_agent("地址3401号。")
    mock_agent.valid_tool_names = [
        "mystand_authorization",
        "mystand_resource_index",
        "mystand_query",
    ]
    app = _create_app(adapter)
    with patch.object(adapter, "_create_agent", return_value=mock_agent):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(DELIVERY_R5B),
                json=_stream_body("读取 AUTH-ABC12345"),
            )
            body = await resp.text()

    assert resp.status == 200
    frames = [frame for frame in body.split("\n\n") if frame.strip()]
    verification = [
        frame
        for frame in frames
        if frame.startswith("event: xiaoban.trusted.verification")
    ]
    assert len(verification) == 1, "工作回合必须恰好发出一次 trusted verification 事件"
    data_line = next(
        line for line in verification[0].splitlines() if line.startswith("data: ")
    )
    payload = json.loads(data_line[len("data: "):])
    assert payload["verified"] is True
    assert payload["action_count"] == 1
    assert payload["request_id"] == DELIVERY_R5B
    # 位置合同：内容 chunk 之后、finish chunk 之前。
    event_at = body.index("event: xiaoban.trusted.verification")
    content_at = body.index('"delta": {"content"')
    finish_at = body.index('"finish_reason": "stop"')
    assert content_at < event_at < finish_at


@pytest.mark.asyncio
async def test_r5c_denied_action_does_not_emit_verified_event(monkeypatch):
    """权限拒绝虽有 ActionResult，但没有 verified Evidence，不得标记 verified=true。"""
    monkeypatch.setattr(
        "tools.mystand_authorization_tool.mystand_authorization_tool_handler",
        lambda args: '{"ok":false,"status":403,"code":"read_not_authorized"}',
    )
    monkeypatch.setattr(
        "tools.mystand_resource_index_tool.mystand_resource_index_tool_handler",
        lambda args: '{"ok":true,"items":[{"resourceUid":"res-demo-denied","safeLabel":"档案"}]}',
    )
    adapter = _make_adapter()
    mock_agent = _mock_agent("当前没有权限读取这份资料。")
    mock_agent.valid_tool_names = [
        "mystand_authorization",
        "mystand_resource_index",
        "mystand_query",
    ]
    app = _create_app(adapter)
    with patch.object(adapter, "_create_agent", return_value=mock_agent):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(DELIVERY_R5_DENIED),
                json=_stream_body("读取 AUTH-ABC12345"),
            )
            body = await resp.text()

    assert resp.status == 200
    assert "xiaoban.trusted.verification" not in body


# R6（§B R6 / G4）：stop 与流式执行的竞态收口


@pytest.mark.asyncio
async def test_r6a_stop_interrupts_running_stream_agent():
    """执行单 §B R6a（G4）：运行中的流式请求被 /stop（idempotency_key=
    delivery-id + X-Xiaoban-Attempt）命中：202 + agent.interrupt 被调 +
    SSE 全文不含业务内容 token。base 流式不入册、stop 碰不到 Agent，失败。"""
    adapter = _make_adapter()
    agent_started = threading.Event()
    interrupted = threading.Event()
    give_up = threading.Event()
    business_claim = "游某今年结算业绩是 32105.68 元。"
    mock_agent = _mock_agent("")
    mock_agent.interrupt.side_effect = lambda *a, **k: interrupted.set()

    def _run_conversation(**_kwargs):
        agent_started.set()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if interrupted.is_set():
                return {
                    "final_response": "",
                    "completed": False,
                    "failed": True,
                    "interrupted": True,
                    "error": "completion stopped",
                    "messages": [],
                }
            if give_up.is_set():
                # base 路径：stop 打不到 Agent，回合带着业务文本跑完。
                return {"final_response": business_claim, "messages": []}
            time.sleep(0.01)
        return {"final_response": business_claim, "messages": []}

    mock_agent.run_conversation.side_effect = _run_conversation
    app = _create_app(adapter)
    with patch.object(adapter, "_create_agent", return_value=mock_agent):
        async with TestClient(TestServer(app)) as cli:
            stream_task = asyncio.create_task(
                cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(DELIVERY_R6A),
                    json=_stream_body("查一下游某今年的结算业绩"),
                )
            )
            for _ in range(300):
                if agent_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert agent_started.is_set(), "流式 Agent 未进入执行"

            stop = await cli.post(
                "/v1/chat/completions/stop",
                headers=_mystand_stream_headers(DELIVERY_R6A),
                json={"idempotency_key": DELIVERY_R6A},
            )
            assert stop.status == 202
            for _ in range(300):
                if interrupted.is_set():
                    break
                await asyncio.sleep(0.01)
            give_up.set()  # base 上 stop 打不到 Agent，手动收口避免悬挂
            resp = await stream_task
            body = await resp.text()

    mock_agent.interrupt.assert_called_once()
    assert "32105.68" not in body, "stop 之后 SSE 不得再输出业务内容 chunk"


@pytest.mark.asyncio
async def test_r6b_stream_after_stop_tombstone_conflicts():
    """执行单 §B R6b（G4）：先 /stop 一个未见过的 delivery key（202 墓碑），
    再发同 key 流式请求：必须 409 且 Agent 零调用。base 流式不查墓碑，失败。"""
    adapter = _make_adapter()
    mock_run = AsyncMock(
        return_value=({"final_response": "你好。", "messages": []}, _USAGE)
    )
    app = _create_app(adapter)
    with patch.object(adapter, "_run_agent", new=mock_run):
        async with TestClient(TestServer(app)) as cli:
            stop = await cli.post(
                "/v1/chat/completions/stop",
                headers=_mystand_stream_headers(DELIVERY_R6B),
                json={"idempotency_key": DELIVERY_R6B},
            )
            assert stop.status == 202
            resp = await cli.post(
                "/v1/chat/completions",
                headers=_mystand_stream_headers(DELIVERY_R6B),
                json=_stream_body("你好"),
            )
            body = await resp.text()

    assert resp.status == 409, body
    mock_run.assert_not_called()
