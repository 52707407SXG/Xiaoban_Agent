"""HTTP delivery proof for signed fact projection and terminal receipts."""

from __future__ import annotations

import json
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _URL_EVIDENCE_FAILURE,
    _idem_cache,
    cors_middleware,
    security_headers_middleware,
)
from tests.gateway.trusted_action_runtime.test_fact_requirement_real_chain_p0 import (
    DELIVERY_ID,
    IDENTITY,
    INDEX_REFS,
    MESSAGE_ID,
    REQUEST_FINGERPRINT,
    SESSION_ID,
    TEST_SIGNING_KEY,
    _base_requirement,
    _begin,
    _finish_index,
    _index_payload,
    _signed_headers,
)
from tools import (
    mystand_authorization_tool,
    mystand_query_tool,
    mystand_resource_index_tool,
)
from xiaoban.trusted_runtime.fact_contract import (
    canonical_digest,
    normalized_fact_query_text,
)
from xiaoban.trusted_runtime.turns import begin_action, finish_action


MODEL_WRONG_TEXT = "模型伪造：2026年业绩排名第4的是合成经纪人甲。"
SERVER_TEXT = "2026年业绩排名第4的是合成经纪人丁。"
USAGE = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


@pytest.fixture(autouse=True)
def _isolate_idempotency_cache():
    assert not _idem_cache._inflight
    _idem_cache._store.clear()
    _idem_cache._agent_refs.clear()
    _idem_cache._stopped.clear()
    yield
    assert not _idem_cache._inflight
    _idem_cache._store.clear()
    _idem_cache._agent_refs.clear()
    _idem_cache._stopped.clear()


def _complete_rank_turn():
    plan = {
        "operation": "read",
        "query_kind": "rank",
        "module_id": "finance-ledger",
        "fact_paths": ["finance.performance.rank"],
        "query_args": {"year": 2026, "rank": 4},
        "coverage_required": True,
    }
    requirement = {
        **_base_requirement(plan, fact_kind="collection", operation="rank"),
        "metric": "settled_performance",
        "ordinal": 4,
    }
    turn = _begin(requirement)
    _finish_index(turn)
    assert begin_action(
        turn,
        "mystand_query",
        "v1",
        plan,
        call_id="call-query",
    ).decision == "allow"
    record_refs = [
        f"finance-performance:2026:broker-{index:02d}"
        for index in range(1, 20)
    ]
    refs_digest = canonical_digest(record_refs)
    result = finish_action(
        turn,
        "call-query",
        "mystand_query",
        "v1",
        {
            "schema": "mystand.query-result.v1",
            "ok": True,
            "status": "matched",
            "queryKind": "rank",
            "planId": requirement["plan_id"],
            "requirementDigest": requirement["requirement_digest"],
            "content": SERVER_TEXT,
            "facts": [
                {
                    "kind": "finance.performance.rank",
                    "value": {
                        "year": 2026,
                        "rank": 4,
                        "brokers": ["合成经纪人丁"],
                    },
                }
            ],
            "missing_facts": [],
            "recordRefs": record_refs,
            "coverage": {
                "expectedCount": 19,
                "returnedCount": 19,
                "hasMore": False,
                "expectedResourceRefsDigest": refs_digest,
                "returnedResourceRefsDigest": refs_digest,
                "year": 2026,
                "scopeFingerprint": IDENTITY.datascope_fingerprint,
                "tieRule": "dense",
                "complete": True,
            },
        },
    )
    assert result is not None and result.status == "success"
    return requirement, turn


def _headers(requirement: dict) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {TEST_SIGNING_KEY}",
        "X-Xiaoban-Site-Id": "synthetic-mystand-site",
        "X-Xiaoban-User-Id": IDENTITY.account_id,
        "X-Xiaoban-Toolset-Policy": "mystand-broker-basic",
        "X-Xiaoban-Memory-Mode": "disabled",
        "X-Xiaoban-Async-Delivery": "disabled",
        "X-Xiaoban-Session-Key": SESSION_ID,
        "X-Xiaoban-Session-Id": SESSION_ID,
        "X-Xiaoban-Message-Id": MESSAGE_ID,
        "X-Xiaoban-Delivery-Id": DELIVERY_ID,
        "X-Xiaoban-Attempt": "1",
        "X-Xiaoban-Request-Fingerprint": REQUEST_FINGERPRINT,
        **_signed_headers(requirement),
    }


def _app(adapter: APIServerAdapter) -> web.Application:
    middlewares = [
        item
        for item in (cors_middleware, security_headers_middleware)
        if item is not None
    ]
    app = web.Application(middlewares=middlewares)
    app["api_server_adapter"] = adapter
    app.router.add_post(
        "/v1/chat/completions",
        adapter._handle_chat_completions,
    )
    return app


def _agent_result(requirement: dict, turn) -> dict:
    return {
        "final_response": MODEL_WRONG_TEXT,
        "messages": [],
        "_mystand_request": True,
        "_mystand_user_id": IDENTITY.account_id,
        "_mystand_request_id": DELIVERY_ID,
        "_mystand_message_id": MESSAGE_ID,
        "_mystand_evidence_required": True,
        "_mystand_fact_requirement": requirement,
        "_trusted_turn": turn,
    }


def _assert_full_receipt(receipt: dict, requirement: dict) -> None:
    common = {
        "schema",
        "verified",
        "request_id",
        "delivery_id",
        "attempt",
        "message_id",
        "request_fingerprint",
        "plan_id",
        "requirement_digest",
        "action_count",
        "evidence_count",
        "evidence_digest",
        "output_digest",
        "verified_at",
        "datascope_fingerprint",
        "decision",
        "coverage",
        "coverage_digest",
        "index_receipt_digest",
        "index_count",
        "index_resource_refs_digest",
        "index_has_more",
    }
    assert common <= set(receipt)
    assert receipt["schema"] == "mystand.xiaoban-fact-verification.v1"
    assert receipt["verified"] is True
    assert receipt["request_id"] == receipt["delivery_id"] == DELIVERY_ID
    assert receipt["message_id"] == MESSAGE_ID
    assert receipt["request_fingerprint"] == REQUEST_FINGERPRINT
    assert receipt["plan_id"] == requirement["plan_id"]
    assert receipt["requirement_digest"] == requirement["requirement_digest"]
    assert receipt["action_count"] == 2
    assert receipt["evidence_count"] == 1
    assert receipt["output_digest"] == hashlib.sha256(
        SERVER_TEXT.encode("utf-8")
    ).hexdigest()
    assert receipt["index_count"] == len(INDEX_REFS)
    assert receipt["index_has_more"] is False
    assert receipt["decision"] == "projected_complete_collection"


def _visible_sse_text(body: str) -> str:
    parts = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line.removeprefix("data: "))
        for choice in payload.get("choices", []):
            parts.append(choice.get("delta", {}).get("content", ""))
    return "".join(parts)


@pytest.mark.asyncio
async def test_signed_fact_sse_projects_server_text_and_emits_full_receipt():
    requirement, turn = _complete_rank_turn()
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": TEST_SIGNING_KEY})
    )
    run_agent = AsyncMock(
        return_value=(_agent_result(requirement, turn), dict(USAGE))
    )
    with patch.object(adapter, "_run_agent", new=run_agent):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(requirement),
                json={
                    "model": "synthetic",
                    "messages": [
                        {"role": "user", "content": "今年业绩第四名是谁？"}
                    ],
                    "stream": True,
                },
            )
            body = await response.text()

    assert response.status == 200
    assert _visible_sse_text(body) == SERVER_TEXT
    assert "合成经纪人甲" not in _visible_sse_text(body)
    frames = [
        frame
        for frame in body.split("\n\n")
        if frame.startswith("event: xiaoban.trusted.verification")
    ]
    assert len(frames) == 1
    receipt = json.loads(
        next(
            line.removeprefix("data: ")
            for line in frames[0].splitlines()
            if line.startswith("data: ")
        )
    )
    _assert_full_receipt(receipt, requirement)
    assert body.index('"delta": {"content"') < body.index(
        "event: xiaoban.trusted.verification"
    ) < body.index('"finish_reason": "stop"')


@pytest.mark.asyncio
async def test_signed_fact_nonstream_projects_server_text_and_nests_full_receipt():
    requirement, turn = _complete_rank_turn()
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": TEST_SIGNING_KEY})
    )
    run_agent = AsyncMock(
        return_value=(_agent_result(requirement, turn), dict(USAGE))
    )
    with patch.object(adapter, "_run_agent", new=run_agent):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(requirement),
                json={
                    "model": "synthetic",
                    "messages": [
                        {"role": "user", "content": "今年业绩第四名是谁？"}
                    ],
                    "stream": False,
                },
            )
            payload = await response.json()

    assert response.status == 200
    assert payload["choices"][0]["message"]["content"] == SERVER_TEXT
    assert "合成经纪人甲" not in json.dumps(payload, ensure_ascii=False)
    receipt = payload["xiaoban"]["trusted_verification"]
    _assert_full_receipt(receipt, requirement)


@pytest.mark.asyncio
async def test_signed_fact_drops_receipt_when_later_egress_guard_changes_text():
    requirement, turn = _complete_rank_turn()
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": TEST_SIGNING_KEY})
    )
    run_agent = AsyncMock(
        return_value=(_agent_result(requirement, turn), dict(USAGE))
    )
    with patch.object(adapter, "_run_agent", new=run_agent):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(requirement),
                json={
                    "model": "synthetic",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "今年业绩第四名是谁？再分析 "
                                "https://example.com/report"
                            ),
                        }
                    ],
                    "stream": False,
                },
            )
            payload = await response.json()

    assert response.status == 200
    assert (
        payload["choices"][0]["message"]["content"]
        == _URL_EVIDENCE_FAILURE
    )
    assert "trusted_verification" not in payload.get("xiaoban", {})


@pytest.mark.asyncio
async def test_signed_fact_rejects_async_session_delivery_before_agent():
    requirement, _turn = _complete_rank_turn()
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": TEST_SIGNING_KEY})
    )
    run_agent = AsyncMock()
    headers = {
        **_headers(requirement),
        "X-Xiaoban-Async-Delivery": "session-events",
    }
    with patch.object(adapter, "_run_agent", new=run_agent):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={
                    "model": "synthetic",
                    "messages": [
                        {"role": "user", "content": "今年业绩第四名是谁？"}
                    ],
                    "stream": True,
                },
            )
            payload = await response.json()

    assert response.status == 409
    assert payload["error"]["code"] == "fact_async_delivery_unsupported"
    run_agent.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (True, False))
async def test_signed_rank_full_http_chain_runs_real_handlers_and_guard(
    monkeypatch,
    stream: bool,
):
    requirement, _turn = _complete_rank_turn()
    record_refs = [
        f"finance-performance:2026:broker-{index:02d}"
        for index in range(1, 20)
    ]
    refs_digest = canonical_digest(record_refs)
    query_calls = []
    monkeypatch.setattr(
        mystand_resource_index_tool,
        "_post_internal",
        lambda payload, user_id: json.dumps(
            _index_payload(),
            ensure_ascii=False,
        ),
    )

    def _query_transport(payload, session):
        query_calls.append((dict(payload), dict(session)))
        return json.dumps(
            {
                "schema": "mystand.query-result.v1",
                "ok": True,
                "status": "matched",
                "queryKind": "rank",
                "planId": requirement["plan_id"],
                "requirementDigest": requirement["requirement_digest"],
                "content": SERVER_TEXT,
                "facts": [
                    {
                        "kind": "finance.performance.rank",
                        "value": {
                            "year": 2026,
                            "rank": 4,
                            "brokers": ["合成经纪人丁"],
                        },
                    }
                ],
                "missing_facts": [],
                "recordRefs": record_refs,
                "coverage": {
                    "expectedCount": 19,
                    "returnedCount": 19,
                    "hasMore": False,
                    "expectedResourceRefsDigest": refs_digest,
                    "returnedResourceRefsDigest": refs_digest,
                    "year": 2026,
                    "scopeFingerprint": IDENTITY.datascope_fingerprint,
                    "tieRule": "dense",
                    "complete": True,
                },
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(
        mystand_query_tool,
        "_post_internal",
        _query_transport,
    )
    persisted = []
    fake_agent = MagicMock()
    fake_agent.valid_tool_names = {
        "mystand_resource_index",
        "mystand_query",
    }
    fake_agent.tools = [
        {"function": {"name": "mystand_resource_index"}},
        {"function": {"name": "mystand_query"}},
    ]
    fake_agent.ephemeral_system_prompt = ""
    fake_agent.session_prompt_tokens = 1
    fake_agent.session_completion_tokens = 1
    fake_agent.session_total_tokens = 2
    fake_agent._persist_session.side_effect = (
        lambda messages, history=None: persisted.append(
            json.loads(json.dumps(messages, ensure_ascii=False))
        )
    )

    def _model_run(**_kwargs):
        assert fake_agent.valid_tool_names == set()
        assert fake_agent.tools == []
        transcript = [
            {"role": "user", "content": "今年业绩第四名是谁？"},
            {"role": "assistant", "content": MODEL_WRONG_TEXT},
        ]
        fake_agent._persist_session(transcript, [])
        return {
            "final_response": MODEL_WRONG_TEXT,
            "messages": transcript,
        }

    fake_agent.run_conversation.side_effect = _model_run
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": TEST_SIGNING_KEY})
    )
    with patch.object(adapter, "_create_agent", return_value=fake_agent):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(requirement),
                json={
                    "model": "synthetic",
                    "messages": [
                        {"role": "user", "content": "今年业绩第四名是谁？"}
                    ],
                    "stream": stream,
                },
            )
            body = await response.text()

    assert response.status == 200
    visible = (
        _visible_sse_text(body)
        if stream
        else json.loads(body)["choices"][0]["message"]["content"]
    )
    assert visible == SERVER_TEXT
    assert "合成经纪人甲" not in visible
    assert len(query_calls) == 1
    sent, _session = query_calls[0]
    assert sent["query_kind"] == "rank"
    assert sent["query_args"] == {"year": 2026, "rank": 4}
    assert sent["requirement_digest"] == requirement["requirement_digest"]
    assert sent["scope_fingerprint"] == IDENTITY.datascope_fingerprint
    assert persisted
    assert all(
        "合成经纪人甲"
        not in json.dumps(snapshot, ensure_ascii=False)
        for snapshot in persisted
    )
    assert any(
        SERVER_TEXT in json.dumps(snapshot, ensure_ascii=False)
        for snapshot in persisted
    )
    if stream:
        frame = next(
            frame
            for frame in body.split("\n\n")
            if frame.startswith("event: xiaoban.trusted.verification")
        )
        receipt = json.loads(
            next(
                line.removeprefix("data: ")
                for line in frame.splitlines()
                if line.startswith("data: ")
            )
        )
    else:
        receipt = json.loads(body)["xiaoban"]["trusted_verification"]
    assert receipt["output_digest"] == hashlib.sha256(
        visible.encode("utf-8")
    ).hexdigest()


@pytest.mark.asyncio
async def test_signed_generic_full_http_chain_uses_only_trusted_query_text(
    monkeypatch,
):
    question = "　请读取 AUTH-ABC12345 的面积。　"
    normalized_question = normalized_fact_query_text(question)
    plan = {
        "operation": "read",
        "query_kind": "resource-read",
        "module_id": "property-maintenance",
        "fact_paths": ["content"],
        "query_args": {
            "semanticQueryDigest": hashlib.sha256(
                normalized_question.encode("utf-8")
            ).hexdigest(),
            "stableReference": "AUTH-ABC12345",
        },
        "coverage_required": False,
    }
    requirement = _base_requirement(
        plan,
        fact_kind="single",
        operation="read",
    )
    monkeypatch.setattr(
        mystand_resource_index_tool,
        "_post_internal",
        lambda payload, user_id: json.dumps(
            _index_payload(),
            ensure_ascii=False,
        ),
    )
    sent_payloads = []

    def _generic_transport(payload, session):
        sent_payloads.append(dict(payload))
        return json.dumps(
            {
                "schema": "mystand.query-result.v1",
                "ok": True,
                "status": "matched",
                "queryKind": "resource-read",
                "planId": requirement["plan_id"],
                "requirementDigest": requirement["requirement_digest"],
                "scopeFingerprint": IDENTITY.datascope_fingerprint,
                "content": "面积：100平方米",
                "recordRefs": ["resource-generic-01"],
                "facts": [
                    {
                        "kind": "property.area",
                        "label": "面积",
                        "value": "100平方米",
                    }
                ],
                "missing_facts": [],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(
        mystand_query_tool,
        "_post_internal",
        _generic_transport,
    )
    blocked_authorization = MagicMock(
        side_effect=AssertionError(
            "signed stableReference must use typed mystand_query"
        )
    )
    monkeypatch.setattr(
        mystand_authorization_tool,
        "mystand_authorization_tool_handler",
        blocked_authorization,
    )
    fake_agent = MagicMock()
    fake_agent.valid_tool_names = {
        "mystand_resource_index",
        "mystand_query",
    }
    fake_agent.tools = [
        {"function": {"name": "mystand_resource_index"}},
        {"function": {"name": "mystand_query"}},
    ]
    fake_agent.ephemeral_system_prompt = ""
    fake_agent.session_prompt_tokens = 1
    fake_agent.session_completion_tokens = 1
    fake_agent.session_total_tokens = 2
    fake_agent.run_conversation.return_value = {
        "final_response": "模型改查另一份资料并编造面积：88平方米。",
        "messages": [],
    }
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": TEST_SIGNING_KEY})
    )
    with patch.object(adapter, "_create_agent", return_value=fake_agent):
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=_headers(requirement),
                json={
                    "model": "synthetic",
                    "messages": [{"role": "user", "content": question}],
                    "stream": False,
                },
            )
            payload = await response.json()

    assert response.status == 200
    assert payload["choices"][0]["message"]["content"] == "面积：100平方米"
    receipt = payload["xiaoban"]["trusted_verification"]
    assert receipt["action_count"] == 2
    assert receipt["evidence_count"] == 1
    assert receipt["output_digest"] == hashlib.sha256(
        payload["choices"][0]["message"]["content"].encode("utf-8")
    ).hexdigest()
    assert len(sent_payloads) == 1
    sent = sent_payloads[0]
    assert sent["queryText"] == normalized_question
    assert sent["query_args"]["semanticQueryDigest"] == hashlib.sha256(
        normalized_question.encode("utf-8")
    ).hexdigest()
    assert sent["query_args"]["stableReference"] == "AUTH-ABC12345"
    blocked_authorization.assert_not_called()
    assert not {
        "resource",
        "entities",
        "fact_needs",
        "mode",
    }.intersection(sent)
