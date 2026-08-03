"""My Stand 流式（web）请求身份、幂等、自然回复与停止生命周期断言。

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
DELIVERY_R6A = "xbd_" + "07" * 20
DELIVERY_R6B = "xbd_" + "08" * 20
DELIVERY_R3_USER = "xbd_" + "09" * 20
DELIVERY_R3_SITE = "xbd_" + "0a" * 20
DELIVERY_R2_ATTEMPT = "xbd_" + "0b" * 20
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


@pytest.mark.asyncio
async def test_plain_chat_forwards_real_deltas_once():
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
