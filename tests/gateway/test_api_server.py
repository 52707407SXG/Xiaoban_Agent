"""
Tests for the OpenAI-compatible API server gateway adapter.

Tests cover:
- Chat Completions endpoint (request parsing, response format)
- Responses API endpoint (request parsing, response format)
- previous_response_id chaining (store/retrieve)
- Auth (valid key, invalid key, no key configured)
- /v1/models endpoint
- /health endpoint
- System prompt extraction
- Error handling (invalid JSON, missing fields)
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    CompletionStoppedError,
    IdempotencyConflictError,
    ResponseStore,
    _IdempotencyCache,
    _build_mystand_stream_replay_envelope,
    _build_api_temporal_context,
    _decode_mystand_stream_replay_envelope,
    _derive_chat_session_id,
    _mystand_tool_result_failed,
    _merge_temporal_context,
    _progress_sensitive_values,
    _progress_tool_batch_context,
    _public_progress_summary,
    _todo_result_progress_projection,
    _trim_chat_history_for_context,
    check_api_server_requirements,
    cors_middleware,
    security_headers_middleware,
)
from xiaoban.trusted_runtime.protocol_contract import (
    TRUSTED_RUNTIME_CONTRACT_DIGEST,
    TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER,
    TRUSTED_RUNTIME_CONTRACT_REVISION,
    TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER,
)


# ---------------------------------------------------------------------------
# check_api_server_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:
    def test_returns_true_when_aiohttp_available(self):
        assert check_api_server_requirements() is True

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_api_server_requirements() is False


class TestMystandToolResultFailure:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            ({"ok": False}, True),
            ({"status": 404}, True),
            ({"status": "503"}, True),
            ({"success": False}, True),
            ({"failed": True}, True),
            ({"error": "remote unavailable"}, True),
            ({"ok": True, "failed": False}, False),
            ('{"ok":true,"failed":false}', False),
            ({"data": {"failed": False}}, False),
            ('{"data":{"failed":false}}', False),
            (
                "[Tool execution cancelled — mystand_query was skipped due to user interrupt]",
                True,
            ),
            ({"data": {"items": [1]}}, False),
        ],
        ids=[
            "ok-false-without-error",
            "http-4xx-status",
            "http-5xx-string-status",
            "success-false",
            "failed-true",
            "non-empty-error",
            "ok-true-failed-false",
            "json-ok-true-failed-false",
            "nested-failed-false",
            "json-nested-failed-false",
            "explicit-tool-cancelled",
            "normal-result",
        ],
    )
    def test_structured_result_fields_take_priority(self, result, expected):
        assert _mystand_tool_result_failed("mystand_query", result) is expected


class TestProgressSummaryDlp:
    def test_todo_numeric_ids_do_not_suppress_natural_commentary(self):
        tool_calls = [{
            "id": "call-todo-plan",
            "type": "function",
            "function": {
                "name": "todo",
                "arguments": json.dumps({
                    "todos": [
                        {
                            "id": "1",
                            "content": "查授权列表，找到2026年7月结算相关的财务档案",
                            "status": "in_progress",
                        },
                        {
                            "id": "2",
                            "content": "逐个拉取档案确认状态（经纪人确认、店长确认、是否结单）",
                            "status": "pending",
                        },
                        {
                            "id": "3",
                            "content": "统计未确认人数和名单",
                            "status": "pending",
                        },
                    ],
                }, ensure_ascii=False),
            },
        }]

        bindings, protected, complete = _progress_tool_batch_context(tool_calls)

        assert bindings == {"call-todo-plan": "todo"}
        assert complete is True
        summary = _public_progress_summary(
            "我会先定位2026年7月结算档案。随后逐项核对确认状态，并汇总未确认名单。",
            protected,
        )
        assert summary
        assert "汇总未确认名单" in summary

    @pytest.mark.parametrize(
        ("content", "commentary"),
        [
            (
                "查2026年7月结算卡还有谁没点",
                "我理解你要确认谁没点。我会核对完整名单和人数。",
            ),
            (
                "核对2026年7月结算卡点击情况",
                "我先查清7月结算卡没点的人员。我会核对完整覆盖后汇总。",
            ),
        ],
    )
    def test_todo_generic_settlement_wording_keeps_real_commentary(
        self,
        content,
        commentary,
    ):
        _, protected, complete = _progress_tool_batch_context([{
            "id": "call-todo-plan",
            "type": "function",
            "function": {
                "name": "todo",
                "arguments": json.dumps({
                    "todos": [{
                        "id": "1",
                        "content": content,
                        "status": "in_progress",
                    }],
                }, ensure_ascii=False),
            },
        }])

        assert complete is True
        assert _public_progress_summary(commentary, protected) == commentary

    def test_todo_single_character_content_remains_fail_closed(self):
        _, _, complete = _progress_tool_batch_context([{
            "id": "call-todo-plan",
            "type": "function",
            "function": {
                "name": "todo",
                "arguments": json.dumps({
                    "todos": [
                        {"id": "1", "content": "王", "status": "in_progress"},
                    ],
                }, ensure_ascii=False),
            },
        }])

        assert complete is False

    @pytest.mark.parametrize(
        "arguments",
        [
            {"todos": ["甲某"]},
            {"todos": [{
                "id": "1",
                "content": "核对结算卡",
                "status": "in_progress",
                "x": "甲某",
            }]},
            {"todos": "甲某"},
            {"todos": [{
                "id": "1",
                "content": "核对😀结算卡",
                "status": "in_progress",
            }]},
        ],
    )
    def test_invalid_or_unparsed_todo_shape_remains_fail_closed(
        self,
        arguments,
    ):
        _, _, complete = _progress_tool_batch_context([{
            "id": "call-todo-plan",
            "type": "function",
            "function": {
                "name": "todo",
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }])

        assert complete is False

    @pytest.mark.parametrize(
        ("content", "commentary", "private_fragment"),
        [
            ("核对𠀀野结算卡", "正在核对𠀀野", "𠀀野"),
            ("核对やまだ结算卡", "正在核对やまだ", "やまだ"),
            ("核对김민수结算卡", "正在核对김민수", "김민수"),
        ],
    )
    def test_unicode_todo_entities_remain_protected(
        self,
        content,
        commentary,
        private_fragment,
    ):
        _, protected, complete = _progress_tool_batch_context([{
            "id": "call-todo-plan",
            "type": "function",
            "function": {
                "name": "todo",
                "arguments": json.dumps({
                    "todos": [{
                        "id": "1",
                        "content": content,
                        "status": "in_progress",
                    }],
                }, ensure_ascii=False),
            },
        }])

        assert complete is True
        summary = _public_progress_summary(commentary, protected)
        assert private_fragment not in summary

    @pytest.mark.parametrize(
        ("content", "commentary", "private_fragments"),
        [
            ("核对甲某结算卡", "我先核对甲某", ("甲某",)),
            (
                "逐个拉取档案确认状态——已查甲某，继续查乙某/丙某",
                "已查甲某，继续查乙某和丙某",
                ("甲某", "乙某", "丙某"),
            ),
            (
                "处理东方花园3栋2单元结算卡",
                "正在处理东方花园3栋",
                ("东方花园", "3栋"),
            ),
            (
                "核对Broker-X7结算卡",
                "正在核对Broker-X7",
                ("Broker-X7",),
            ),
            (
                "核对东方结算花园档案",
                "正在核对东方结算花园",
                ("东方", "花园"),
            ),
        ],
    )
    def test_todo_content_fragments_remain_protected(
        self,
        content,
        commentary,
        private_fragments,
    ):
        _, protected, complete = _progress_tool_batch_context([{
            "id": "call-todo-plan",
            "type": "function",
            "function": {
                "name": "todo",
                "arguments": json.dumps({
                    "todos": [{
                        "id": "1",
                        "content": content,
                        "status": "in_progress",
                    }],
                }, ensure_ascii=False),
            },
        }])

        assert complete is True
        summary = _public_progress_summary(commentary, protected)
        assert all(fragment not in summary for fragment in private_fragments)

    def test_todo_items_survive_strict_stream_replay(self):
        todo_items = [
            {"id": "1", "content": "核对确认状态", "status": "in_progress"},
            {"id": "2", "content": "汇总名单", "status": "pending"},
        ]
        envelope = _build_mystand_stream_replay_envelope(
            [("__tool_progress__", {
                "tool": "todo",
                "status": "completed",
                "todoItems": todo_items,
            })],
            {"final_response": "已完成", "completed": True, "failed": False},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

        assert envelope is not None
        decoded = _decode_mystand_stream_replay_envelope(envelope)
        assert decoded is not None
        assert decoded[0][0][1]["todoItems"] == todo_items

    def test_todo_projection_keeps_items_and_unique_active_id(self):
        items, active_id = _todo_result_progress_projection({
            "todos": [
                {"id": "1", "content": "读取授权", "status": "completed"},
                {"id": "2", "content": "核对确认状态", "status": "in_progress"},
                {"id": "3", "content": "汇总名单", "status": "pending"},
            ],
        })

        assert [item["id"] for item in items] == ["1", "2", "3"]
        assert [item["status"] for item in items] == [
            "completed", "in_progress", "pending",
        ]
        assert active_id == "2"

    @pytest.mark.parametrize(
        "unsafe_case",
        [
            "long-string",
            "deep-container",
            "wide-container",
            "malformed-json",
            "non-finite-number",
            "single-chinese-character",
            "single-digit",
        ],
    )
    def test_uncovered_argument_shapes_are_explicitly_incomplete(
        self,
        unsafe_case,
    ):
        if unsafe_case == "long-string":
            value = {"financeBody": "长" * 513}
        elif unsafe_case == "deep-container":
            nested = "深层客户资料"
            for index in range(6):
                nested = {f"level{index}": nested}
            value = {"payload": nested}
        elif unsafe_case == "wide-container":
            value = {"items": [f"资料-{index:03d}" for index in range(129)]}
        elif unsafe_case == "malformed-json":
            value = "{not-json}"
        elif unsafe_case == "non-finite-number":
            value = {"amount": float("nan")}
        elif unsafe_case == "single-chinese-character":
            value = {"customerName": "王"}
        else:
            value = {"roomNumber": 8}

        _, complete = _progress_sensitive_values(
            value,
            protect_all_strings=True,
            limit=512,
        )

        assert complete is False

    def test_numeric_and_executor_error_reason_are_exactly_protected(self):
        numeric_values, numeric_complete = _progress_sensitive_values(
            {"amount": 6350},
            protect_all_strings=True,
            limit=512,
        )
        error_values, error_complete = _progress_sensitive_values(
            "Error executing tool 'mystand_query': 松鹤居",
            protect_all_strings=True,
            limit=512,
        )
        wrapped_values = []
        for wrapped_result in (
            {"error": "Context engine tool 'lcm_grep' failed: 蓝湾苑"},
            {"error": "Memory tool 'hindsight_search' failed: 蓝湾苑"},
        ):
            values, complete = _progress_sensitive_values(
                wrapped_result,
                protect_all_strings=True,
                limit=512,
            )
            assert complete is True
            wrapped_values.extend(values)

        assert numeric_complete is True
        assert error_complete is True
        assert _public_progress_summary(
            "正在核对 6350",
            numeric_values,
        ) == "正在核对 相关资料"
        assert "松鹤居" not in _public_progress_summary(
            "我已经找到松鹤居，接着核对登记状态。",
            error_values,
        )
        assert "蓝湾苑" not in _public_progress_summary(
            "我已经找到蓝湾苑，接着核对登记状态。",
            wrapped_values,
        )

    def test_overlong_commentary_is_not_silently_truncated(self):
        assert _public_progress_summary("甲" * 2_001) == ""

    def test_derived_entity_and_identifier_fragments_suppress_summary(self):
        protected, complete = _progress_sensitive_values(
            {
                "company": "松鹤居房地产经纪有限公司",
                "phone": "13800001234",
                "account": "6222020200000123",
                "address": "城南一号2栋10楼",
            },
            protect_all_strings=True,
            limit=512,
        )

        assert complete is True
        for derived_summary in (
            "我已找到松鹤居地产，接着核对登记状态。",
            "我先核对松鹤对应资料。",
            "我先核对联系电话尾号1234。",
            "我先核对收款账号尾号0123。",
            "我先核对城南一号的登记状态。",
        ):
            assert _public_progress_summary(derived_summary, protected) == ""
        assert _public_progress_summary(
            "我先核对相关登记资料。",
            protected,
        ) == "我先核对相关登记资料。"

    def test_exact_argument_is_redacted_before_fragment_check(self):
        protected, complete = _progress_sensitive_values(
            {"query": "chain-1"},
            protect_all_strings=True,
            limit=512,
        )
        prior_values, prior_complete = _progress_sensitive_values(
            {"evidence": "ev_8ab5abcd97c9d0717c6285b80979"},
            protect_all_strings=True,
            limit=512,
        )

        assert complete is True
        assert prior_complete is True
        assert _public_progress_summary(
            "第一步完成，现在用 chain-1 继续核对。",
            protected,
        ) == "第一步完成，现在用 相关资料 继续核对。"
        assert _public_progress_summary(
            "第一步完成，ev_8ab5abcd97c9d0717c6285b80979。"
            "现在用 chain-1 继续核对。",
            [*protected, *prior_values],
        ) == "第一步完成，相关资料。现在用 相关资料 继续核对。"
        assert _public_progress_summary(
            "第一步完成，现在核对 chain。",
            protected,
        ) == ""

    @pytest.mark.parametrize(
        "private_summary",
        [
            "我先核对。<think>PRIVATE_REASONING",
            "PRIVATE_REASONING</ANALYSIS>我再继续。",
            "我先核对。<REASONING_SCRATCHPAD>PRIVATE_REASONING",
            "我先核对。<reasoning-scratchpad>PRIVATE_REASONING",
        ],
    )
    def test_unclosed_or_orphan_private_markers_suppress_summary(
        self,
        private_summary,
    ):
        assert _public_progress_summary(private_summary) == ""

    def test_closed_private_block_is_removed_but_safe_context_remains(self):
        assert _public_progress_summary(
            "我先核对。<ReAsOnInG_ScRaTcHpAd>PRIVATE_REASONING"
            "</ReAsOnInG_ScRaTcHpAd>接着整理公开结果。"
        ) == "我先核对。接着整理公开结果。"

    @pytest.mark.parametrize(
        ("query", "summary"),
        [
            ("核对佣金", "我先核对相关佣金规则。"),
            ("读取授权", "我先读取可用授权状态。"),
            ("查询状态", "我先查询当前处理状态。"),
            ("查询当前登记状态", "我先查询当前状态。"),
        ],
    )
    def test_generic_query_fragments_do_not_suppress_progress(
        self,
        query,
        summary,
    ):
        protected, complete = _progress_sensitive_values(
            {"query": query},
            protect_all_strings=True,
            limit=512,
        )

        assert complete is True
        assert _public_progress_summary(summary, protected) == ""

    @pytest.mark.parametrize(
        ("query", "derived_summary"),
        [
            (
                "查找松鹤居房地产经纪有限公司",
                "我先核对松鹤居的资料。",
            ),
            ("查询城南一号2栋10楼", "我先核对城南一号的登记。"),
            ("查找客户张小明", "我先核对张小明的资料。"),
            ("检索蓝湾苑客户资料", "我先核对蓝湾苑的登记。"),
            ("查找阿黎电话", "我先核对阿黎的电话。"),
            ("查询世纪大道100号", "我先核对100号的登记。"),
            ("查找客户CUST-ABC12345", "我先核对ABC12345。"),
        ],
    )
    def test_query_entity_fragments_suppress_progress(
        self,
        query,
        derived_summary,
    ):
        protected, complete = _progress_sensitive_values(
            {"query": query},
            protect_all_strings=True,
            limit=512,
        )

        assert complete is True
        assert _public_progress_summary(derived_summary, protected) == ""

    @pytest.mark.parametrize(
        ("key", "value", "derived_summary"),
        [
            ("account", 6222020200000123, "我先核对账号尾号0123。"),
            ("phone", 13800001234, "我先核对电话尾号1234。"),
            ("customerId", 123456789.0, "我先核对客户编号尾号6789。"),
        ],
    )
    def test_numeric_identifier_leaves_protect_tail_fragments(
        self,
        key,
        value,
        derived_summary,
    ):
        protected, complete = _progress_sensitive_values(
            {key: value},
            protect_all_strings=True,
            limit=512,
        )

        assert complete is True
        assert _public_progress_summary(derived_summary, protected) == ""

    @pytest.mark.parametrize(
        ("key", "value", "derived_summary"),
        [
            ("query", "AUTH-7F93A1B2", "我先核对标识 7F93A1B2。"),
            ("resourceId", "OUT-customer", "我先核对 customer 资源。"),
            ("query", "gang@example.com", "我先核对 gang 的邮箱登记。"),
        ],
    )
    def test_ascii_identifier_fragments_suppress_progress(
        self,
        key,
        value,
        derived_summary,
    ):
        protected, complete = _progress_sensitive_values(
            {key: value},
            protect_all_strings=True,
            limit=512,
        )

        assert complete is True
        assert _public_progress_summary(derived_summary, protected) == ""

    @pytest.mark.parametrize(
        "mutation",
        ["duplicate-id", "padded-id", "padded-name", "invalid-name"],
    )
    def test_batch_binding_rejects_ambiguous_ids_and_names(self, mutation):
        first = {
            "id": "call-safe-one",
            "type": "function",
            "function": {
                "name": "mystand_query",
                "arguments": '{"query":"阿黎"}',
            },
        }
        calls = [first]
        if mutation == "duplicate-id":
            calls.append({
                "id": "call-safe-one",
                "type": "function",
                "function": {
                    "name": "mystand_query",
                    "arguments": '{"query":"蓝湾苑"}',
                },
            })
        elif mutation == "padded-id":
            first["id"] = " call-safe-one"
        elif mutation == "padded-name":
            first["function"]["name"] = "mystand_query "
        else:
            first["function"]["name"] = "mystand/query"

        bindings, protected, complete = _progress_tool_batch_context(calls)

        assert bindings == {}
        assert protected == []
        assert complete is False


class TestChatHistoryContextBudget:
    def test_trim_chat_history_keeps_recent_messages_within_budget(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_CHAT_HISTORY_MAX_MESSAGES", "3")
        monkeypatch.setenv("API_SERVER_CHAT_HISTORY_CHAR_BUDGET", "4000")
        history = [{"role": "user", "content": f"message {i}"} for i in range(8)]

        trimmed = _trim_chat_history_for_context(history)

        assert trimmed == history[-3:]

    def test_trim_chat_history_truncates_single_oversized_latest_message(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_CHAT_HISTORY_MAX_MESSAGES", "10")
        monkeypatch.setenv("API_SERVER_CHAT_HISTORY_CHAR_BUDGET", "4000")
        latest = "x" * 5000

        trimmed = _trim_chat_history_for_context([
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": latest},
        ])

        assert len(trimmed) == 1
        assert trimmed[0]["role"] == "assistant"
        assert "前文已按上下文预算截断" in trimmed[0]["content"]
        assert len(trimmed[0]["content"]) <= 4000


class TestAPIServerTemporalContext:
    def test_temporal_context_defaults_to_beijing_window(self):
        context = _build_api_temporal_context(
            now_utc=datetime(2026, 6, 24, 7, 25, 0, tzinfo=timezone.utc),
        )

        assert "IANA时区：Asia/Shanghai" in context
        assert "当前用户本地时间：2026-06-24 15:25:00 CST+0800" in context
        assert "今天窗口：2026-06-24 00:00:00 CST+0800 至 2026-06-25 00:00:00 CST+0800" in context
        assert "今晚/今夜窗口：2026-06-24 18:00:00 CST+0800 至 2026-06-25 12:00:00 CST+0800" in context
        assert "ET/PT/CT/MT/BST/UTC/local time/venue local time" in context
        assert "不要假设 date-filtered schedule page" in context

    def test_merge_temporal_context_preserves_upstream_system_prompt_and_header_timezone(self):
        headers = {
            "X-Xiaoban-User-Timezone": "America/New_York",
            "X-Xiaoban-User-Locale": "en-US",
        }
        merged = _merge_temporal_context(
            "upstream system prompt",
            headers=headers,
            now_utc=datetime(2026, 6, 24, 7, 25, 0, tzinfo=timezone.utc),
        )

        assert "IANA时区：America/New_York" in merged
        assert "locale=en-US" in merged
        assert merged.index("【Xiaoban deterministic temporal context】") < merged.index("upstream system prompt")


# ---------------------------------------------------------------------------
# ResponseStore
# ---------------------------------------------------------------------------


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.get("resp_1") == {"output": "hello"}

    def test_get_missing_returns_none(self):
        store = ResponseStore(max_size=10)
        assert store.get("resp_missing") is None

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Adding a 4th should evict resp_1
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert len(store) == 3

    def test_access_refreshes_lru(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Access resp_1 to move it to end
        store.get("resp_1")
        # Now resp_2 is the oldest — adding a 4th should evict resp_2
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_2") is None
        assert store.get("resp_1") is not None

    def test_update_existing_key(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "v1"})
        store.put("resp_1", {"output": "v2"})
        assert store.get("resp_1") == {"output": "v2"}
        assert len(store) == 1

    def test_delete_existing(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.delete("resp_1") is True
        assert store.get("resp_1") is None
        assert len(store) == 0

    def test_delete_missing(self):
        store = ResponseStore(max_size=10)
        assert store.delete("resp_missing") is False


class TestAPIServerSessionEvents:
    @pytest.mark.asyncio
    async def test_send_queues_assistant_message_event(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))

        result = await adapter.send("web:company:user:chat", "授权成功。")

        assert result.success is True
        events = adapter._session_event_snapshot("web:company:user:chat", since=0)
        assert len(events) == 1
        assert events[0]["event"] == "assistant.message"
        assert events[0]["message"]["role"] == "assistant"
        assert events[0]["message"]["content"] == "授权成功。"
        assert events[0]["seq"] > 0

    def test_session_events_requested_requires_explicit_header(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
        request = MagicMock()
        request.headers = {}
        assert adapter._session_events_requested(request) is False

        request.headers = {"X-Xiaoban-Async-Delivery": "session-events"}
        assert adapter._session_events_requested(request) is True

    def test_session_event_snapshot_filters_by_sequence(self):
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
        first = adapter._enqueue_session_event("s1", "assistant.message", {"message": {"content": "one"}})
        second = adapter._enqueue_session_event("s1", "assistant.message", {"message": {"content": "two"}})

        events = adapter._session_event_snapshot("s1", since=int(first["seq"]))

        assert [event["id"] for event in events] == [second["id"]]

    def test_delete_clears_conversation_mapping(self):
        """Deleting a response also removes conversation mappings that reference it."""
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        store.set_conversation("chat-a", "resp_1")
        assert store.get_conversation("chat-a") == "resp_1"
        store.delete("resp_1")
        assert store.get_conversation("chat-a") is None

    def test_eviction_clears_conversation_mapping(self):
        """LRU eviction also removes conversation mappings for evicted responses."""
        store = ResponseStore(max_size=2)
        store.put("resp_1", {"output": "one"})
        store.set_conversation("chat-a", "resp_1")
        store.put("resp_2", {"output": "two"})
        store.set_conversation("chat-b", "resp_2")
        # Adding a 3rd should evict resp_1 and its conversation mapping
        store.put("resp_3", {"output": "three"})
        assert store.get("resp_1") is None
        assert store.get_conversation("chat-a") is None
        # resp_2 mapping should still be intact
        assert store.get_conversation("chat-b") == "resp_2"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are platform-specific")
    def test_file_store_created_owner_only_under_permissive_umask(self, tmp_path):
        """response_store.db must be 0o600 on creation even under umask 022."""
        db_path = tmp_path / "response_store.db"
        store = None
        old_umask = os.umask(0o022)
        try:
            store = ResponseStore(max_size=10, db_path=str(db_path))
            store.put(
                "resp_secret",
                {
                    "response": {"id": "resp_secret"},
                    "conversation_history": [{"role": "tool", "content": "dummy-marker"}],
                },
            )
        finally:
            os.umask(old_umask)
            if store is not None:
                store.close()

        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        # WAL/SHM sidecars are owner-only too when present. WAL mode may be
        # unavailable on some filesystems (NFS/SMB) — only assert when the
        # sidecar files actually exist.
        for sidecar in (
            db_path.with_name(db_path.name + "-wal"),
            db_path.with_name(db_path.name + "-shm"),
        ):
            if sidecar.exists():
                assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# _IdempotencyCache
# ---------------------------------------------------------------------------


class TestIdempotencyCache:
    @pytest.mark.asyncio
    async def test_concurrent_same_key_and_fingerprint_runs_once(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return ("response", {"total_tokens": 1})

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))

        await started.wait()
        assert calls == 1

        gate.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result == second_result == ("response", {"total_tokens": 1})

    @pytest.mark.asyncio
    async def test_different_fingerprint_is_rejected_without_second_run(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return calls

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        await started.wait()
        with pytest.raises(IdempotencyConflictError):
            await cache.get_or_set("idem-key", "fp-2", compute)
        assert calls == 1

        gate.set()
        assert await first == 1

    @pytest.mark.asyncio
    async def test_stop_interrupts_only_the_matching_inflight_key(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        agent = MagicMock()
        agent_ref = [agent, False]

        async def compute():
            await gate.wait()
            return "done"

        task = asyncio.create_task(cache.get_or_set("scoped-key", "fp-1", compute, agent_ref=agent_ref))
        await asyncio.sleep(0)
        assert cache.stop("other-key") is True
        agent.interrupt.assert_not_called()
        assert cache.stop("scoped-key") is True
        assert agent_ref[1] is True
        agent.interrupt.assert_called_once_with("Stop requested via My Stand delivery")
        gate.set()
        stopped = await task
        assert stopped["final_response"] == ""
        assert stopped["messages"] == []
        assert stopped["interrupted"] is True
        state, cached = cache.result_state("scoped-key")
        assert state == "stopped"
        assert cached == stopped

    def test_stop_closes_chat_control_bridge_before_cancel_and_interrupt(
        self,
        monkeypatch,
    ):
        from gateway.platforms import true_moa_stop_projection

        order = []
        bridge = MagicMock()
        bridge.close.side_effect = lambda: order.append("bridge.close")
        controller = MagicMock()
        controller.cancel.side_effect = lambda: order.append("controller.cancel") or True
        agent = MagicMock()
        monkeypatch.setattr(
            true_moa_stop_projection,
            "_interrupt_agent_async",
            lambda current, reason: order.append("agent.interrupt"),
        )

        assert true_moa_stop_projection._cancel_chat_agent_ref(
            [agent, False, controller, bridge],
            "stop fence won",
        ) is True

        assert order == ["bridge.close", "controller.cancel", "agent.interrupt"]

    @pytest.mark.asyncio
    async def test_cancelled_waiter_does_not_drop_shared_inflight_task(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return "response"

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))

        await started.wait()
        assert calls == 1

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        await asyncio.sleep(0)
        assert calls == 1

        gate.set()
        assert await second == "response"


# ---------------------------------------------------------------------------
# Adapter initialization
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_default_config(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 8642
        assert adapter._api_key == ""
        assert adapter.platform == Platform.API_SERVER

    def test_custom_config_from_extra(self):
        config = PlatformConfig(
            enabled=True,
            extra={
                "host": "0.0.0.0",
                "port": 9999,
                "key": "sk-test",
                "cors_origins": ["http://localhost:3000"],
            },
        )
        adapter = APIServerAdapter(config)
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 9999
        assert adapter._api_key == "sk-test"
        assert adapter._cors_origins == ("http://localhost:3000",)

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_HOST", "10.0.0.1")
        monkeypatch.setenv("API_SERVER_PORT", "7777")
        monkeypatch.setenv("API_SERVER_KEY", "sk-env")
        monkeypatch.setenv("API_SERVER_CORS_ORIGINS", "http://localhost:3000, http://127.0.0.1:3000")
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "10.0.0.1"
        assert adapter._port == 7777
        assert adapter._api_key == "sk-env"
        assert adapter._cors_origins == (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        )

    def test_invalid_port_from_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_PORT", "not-a-port")
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._port == 8642

    def test_create_agent_forwards_config_reasoning_effort(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai-codex",
                "base_url": "https://example.test/v1",
                "api_mode": "codex_responses",
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5.5")
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {"agent": {"reasoning_effort": "xhigh"}},
        )
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda: {"enabled": True, "effort": "xhigh"}),
        )
        monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
        monkeypatch.setattr("xiaoban_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["reasoning_config"] == {"enabled": True, "effort": "xhigh"}

    def test_create_agent_refreshes_max_iterations_from_runtime_config(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai",
                "base_url": "https://example.test/v1",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5")
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"agent": {"max_turns": 200}})
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda: {}),
        )
        monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
        monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 200)
        monkeypatch.setattr("xiaoban_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["max_iterations"] == 200


# ---------------------------------------------------------------------------
# Auth checking
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_key_configured_allows_all(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        assert adapter._check_auth(mock_request) is None

    def test_valid_key_passes(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer sk-test123"}
        assert adapter._check_auth(mock_request) is None

    def test_invalid_key_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer wrong-key"}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401

    def test_missing_auth_header_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401

    def test_malformed_auth_header_returns_401(self):
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}
        result = adapter._check_auth(mock_request)
        assert result is not None
        assert result.status == 401


# ---------------------------------------------------------------------------
# Concurrency cap (gateway.api_server.max_concurrent_runs) — #7483
# ---------------------------------------------------------------------------


class TestConcurrencyCap:
    def test_resolve_defaults_to_10_when_unset(self):
        with patch("xiaoban_cli.config.load_config", return_value={}):
            assert APIServerAdapter._resolve_max_concurrent_runs() == 10

    def test_resolve_reads_config_value(self):
        cfg = {"gateway": {"api_server": {"max_concurrent_runs": 3}}}
        with patch("xiaoban_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_concurrent_runs() == 3

    def test_resolve_clamps_negative_to_zero(self):
        cfg = {"gateway": {"api_server": {"max_concurrent_runs": -5}}}
        with patch("xiaoban_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_concurrent_runs() == 0

    def test_resolve_malformed_falls_back_to_default(self):
        cfg = {"gateway": {"api_server": {"max_concurrent_runs": "not-an-int"}}}
        with patch("xiaoban_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_concurrent_runs() == 10

    def test_under_cap_returns_none(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 5
        adapter._inflight_agent_runs = 2
        assert adapter._concurrency_limited_response() is None

    def test_at_cap_returns_429_with_retry_after(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 3
        adapter._inflight_agent_runs = 3
        resp = adapter._concurrency_limited_response()
        assert resp is not None
        assert resp.status == 429
        assert resp.headers.get("Retry-After")

    def test_cap_counts_both_buckets(self):
        # /v1/runs (tracked by _run_streams) + chat/responses (inflight)
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 4
        adapter._inflight_agent_runs = 2
        adapter._run_streams = {"r1": object(), "r2": object()}
        resp = adapter._concurrency_limited_response()
        assert resp is not None
        assert resp.status == 429

    def test_zero_disables_cap(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 0
        adapter._inflight_agent_runs = 9999
        assert adapter._concurrency_limited_response() is None


# ---------------------------------------------------------------------------
# Helpers for HTTP tests
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", cors_origins=None) -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app from the adapter (without starting the full server)."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    app.router.add_get("/v1/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/v1/skills", adapter._handle_skills)
    app.router.add_get("/v1/toolsets", adapter._handle_toolsets)
    app.router.add_get("/v1/mystand/memory", adapter._handle_mystand_memory)
    app.router.add_post("/v1/mystand/memory", adapter._handle_mystand_memory)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/chat/completions/stop", adapter._handle_stop_idempotent_chat_completion)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_delete("/v1/responses/{response_id}", adapter._handle_delete_response)
    return app


def _mystand_idempotent_headers(
    key: str,
    *,
    user: str = "alice",
    attempt: str = "1",
    request_fingerprint: str | None = None,
) -> dict[str, str]:
    return {
        "Authorization": "Bearer sk-secret",
        "Idempotency-Key": key,
        "X-Xiaoban-Site-Id": "mystand-test-site",
        "X-Xiaoban-User-Id": user,
        "X-Xiaoban-Toolset-Policy": "mystand-broker-basic",
        "X-Xiaoban-Memory-Mode": "disabled",
        "X-Xiaoban-Session-Key": f"session-{user}",
        "X-Xiaoban-Session-Id": f"session-{user}",
        "X-Xiaoban-Message-Id": f"message-{key}",
        "X-Xiaoban-Attempt": attempt,
        "X-Xiaoban-Request-Fingerprint": request_fingerprint or hashlib.sha256(
            f"request:{key}".encode("utf-8")
        ).hexdigest(),
        TRUSTED_RUNTIME_CONTRACT_REVISION_HEADER: (
            TRUSTED_RUNTIME_CONTRACT_REVISION
        ),
        TRUSTED_RUNTIME_CONTRACT_DIGEST_HEADER: TRUSTED_RUNTIME_CONTRACT_DIGEST,
    }


def _mystand_stream_headers(delivery_id: str) -> dict[str, str]:
    headers = _mystand_idempotent_headers(delivery_id)
    headers.pop("Idempotency-Key")
    headers["X-Xiaoban-Delivery-Id"] = delivery_id
    headers["X-Xiaoban-Delivery-Attempt"] = headers["X-Xiaoban-Attempt"]
    return headers


def _xiaoban_progress_payloads(body: str) -> list[dict]:
    payloads = []
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "event: xiaoban.tool.progress":
            continue
        for follow in lines[index + 1:index + 4]:
            if follow.startswith("data: "):
                payloads.append(json.loads(follow[len("data: "):]))
                break
    return payloads


def _chat_stream_public_projection(body: str) -> dict:
    content = []
    events = []
    finishes = []
    current_event = None
    for line in body.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):]
            continue
        if not line:
            current_event = None
            continue
        if not line.startswith("data: "):
            continue
        raw = line[len("data: "):]
        if raw == "[DONE]":
            continue
        payload = json.loads(raw)
        if current_event:
            if current_event != "xiaoban.status":
                events.append((current_event, payload))
            continue
        if payload.get("object") != "chat.completion.chunk":
            continue
        for choice in payload.get("choices", []):
            value = choice.get("delta", {}).get("content")
            if isinstance(value, str):
                content.append(value)
            if choice.get("finish_reason") is not None:
                finishes.append({
                    "finish_reason": choice.get("finish_reason"),
                    "usage": payload.get("usage"),
                })
    return {
        "content": "".join(content),
        "progress": _xiaoban_progress_payloads(body),
        "events": events,
        "finishes": finishes,
        "done": body.count("data: [DONE]"),
    }


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# Adapter internals
# ---------------------------------------------------------------------------


class TestAgentExecution:
    @pytest.mark.asyncio
    async def test_run_agent_uses_session_id_as_task_id(self, adapter):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent.session_prompt_tokens = 1
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 3

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            result, usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-123",
            )

        # _run_agent annotates result with the effective agent.session_id
        # when it's a real string, so the response-header writer can track
        # compression-triggered session rotations (#16938). The mock agent
        # here doesn't set an explicit session_id string so the guard skips
        # the annotation — header will fall back to the provided session_id.
        assert result["final_response"] == "ok"
        assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        mock_agent.run_conversation.assert_called_once_with(
            user_message="hello",
            conversation_history=[],
            task_id="session-123",
        )

    @pytest.mark.asyncio
    async def test_mystand_multimodal_text_binds_as_trusted_user_message(
        self,
        auth_adapter,
    ):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {
            "final_response": "ok",
            "messages": [],
        }
        mock_agent.session_prompt_tokens = 1
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 3
        user_message = [
            {"type": "text", "text": "把城南一号2栋10楼的特征卡改一下"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            },
        ]

        with (
            patch.object(auth_adapter, "_create_agent", return_value=mock_agent),
            patch.object(
                auth_adapter,
                "_bind_api_server_session",
                return_value=[],
            ) as bind_session,
        ):
            await auth_adapter._run_agent(
                user_message=user_message,
                conversation_history=[],
                session_id="session-multimodal-trusted-text",
                request_headers=_mystand_idempotent_headers(
                    "multimodal-trusted-text"
                ),
            )

        assert bind_session.call_args.kwargs["user_message"] == (
            "把城南一号2栋10楼的特征卡改一下"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("agent_result", "expected_error_code"),
        [
            ({"final_response": "", "failed": True}, "agent_error"),
            ({"final_response": "partial", "partial": True}, "output_truncated"),
            ({"final_response": "", "completed": False}, "agent_error"),
        ],
    )
    async def test_mystand_structured_non_success_emits_failed_metadata(
        self,
        auth_adapter,
        agent_result,
        expected_error_code,
    ):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = dict(agent_result)
        mock_agent.session_prompt_tokens = 1
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 3
        metadata_trace = MagicMock()
        metadata_trace.elapsed_ms.return_value = 7

        with (
            patch.object(auth_adapter, "_create_agent", return_value=mock_agent),
            patch(
                "xiaoban.observability.mystand_metadata.MystandMetadataTrace",
                return_value=metadata_trace,
            ),
        ):
            await auth_adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-structured-failure",
                request_headers=_mystand_idempotent_headers("metadata-structured-failure"),
            )

        emitted = metadata_trace.safe_emit.call_args_list
        assert [call.args[0] for call in emitted] == ["request_started", "request_failed"]
        assert emitted[-1].kwargs["status"] == "failed"
        assert emitted[-1].kwargs["error_code"] == expected_error_code

    @pytest.mark.asyncio
    async def test_completion_stopped_error_emits_stop_metadata_once(self, auth_adapter):
        mock_agent = MagicMock()
        metadata_trace = MagicMock()
        metadata_trace.elapsed_ms.return_value = 9
        agent_ref = [None, True]

        with (
            patch.object(auth_adapter, "_create_agent", return_value=mock_agent),
            patch(
                "xiaoban.observability.mystand_metadata.MystandMetadataTrace",
                return_value=metadata_trace,
            ),
        ):
            with pytest.raises(CompletionStoppedError):
                await auth_adapter._run_agent(
                    user_message="hello",
                    conversation_history=[],
                    session_id="session-stopped-before-execution",
                    request_headers=_mystand_idempotent_headers("metadata-stopped"),
                    agent_ref=agent_ref,
                )

        mock_agent.interrupt.assert_called_once_with("Stop requested via My Stand delivery")
        emitted = metadata_trace.safe_emit.call_args_list
        assert [call.args[0] for call in emitted] == ["request_started", "request_failed"]
        assert emitted[-1].kwargs["error_code"] == "completion_stopped"


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_security_headers_present(self, adapter):
        """Responses should include basic security headers."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("Content-Security-Policy") == "default-src 'none'; frame-ancestors 'none'"
            assert resp.headers.get("Permissions-Policy") == "camera=(), microphone=(), geolocation=()"
            assert resp.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert resp.headers.get("X-XSS-Protection") == "0"
            assert resp.headers.get("Referrer-Policy") == "no-referrer"

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "xiaoban-agent"

    @pytest.mark.asyncio
    async def test_health_reports_version(self, adapter):
        """GET /health must expose a non-empty version so orchestrators (e.g.
        AgentOS) can read the gateway version without scraping. Regression
        guard for the missing-version gap."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert "version" in data
            assert isinstance(data["version"], str)
            assert data["version"] != ""

    @pytest.mark.asyncio
    async def test_v1_health_alias_returns_ok(self, adapter):
        """GET /v1/health should return the same response as /health."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["platform"] == "xiaoban-agent"
            assert data.get("version")


# ---------------------------------------------------------------------------
# /health/detailed endpoint
# ---------------------------------------------------------------------------


class TestHealthDetailedEndpoint:
    @pytest.mark.asyncio
    async def test_health_detailed_returns_ok(self, adapter):
        """GET /health/detailed returns status, platform, and runtime fields."""
        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value={
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "active_agents": 2,
            "exit_reason": None,
            "updated_at": "2026-04-14T00:00:00Z",
        }):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["platform"] == "xiaoban-agent"
                assert data["gateway_state"] == "running"
                assert data["platforms"] == {"telegram": {"state": "connected"}}
                assert data["active_agents"] == 2
                # Derived busy/drainable: this endpoint is served BY the live
                # gateway, so running + 2 agents ⇒ busy and drainable.
                assert data["gateway_busy"] is True
                assert data["gateway_drainable"] is True
                assert isinstance(data["pid"], int)
                assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_health_detailed_no_runtime_status(self, adapter):
        """When gateway_state.json is missing, fields are None."""
        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value=None):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["gateway_state"] is None
                assert data["platforms"] == {}
                # No runtime file ⇒ state None ⇒ not busy, not drainable.
                assert data["gateway_busy"] is False
                assert data["gateway_drainable"] is False

    @pytest.mark.asyncio
    async def test_health_detailed_does_not_require_auth(self, auth_adapter):
        """Health detailed endpoint should be accessible without auth, like /health."""
        app = _create_app(auth_adapter)
        with patch("gateway.status.read_runtime_status", return_value=None):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200


# ---------------------------------------------------------------------------
# /v1/models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_returns_xiaoban_agent(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "xiaoban-agent"
            assert data["data"][0]["owned_by"] == "xiaoban"

    @pytest.mark.asyncio
    async def test_models_returns_profile_name(self):
        """When running under a named profile, /v1/models advertises the profile name."""
        with patch("gateway.platforms.api_server.APIServerAdapter._resolve_model_name", return_value="lucas"):
            adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["data"][0]["id"] == "lucas"
            assert data["data"][0]["root"] == "lucas"

    @pytest.mark.asyncio
    async def test_models_returns_explicit_model_name(self):
        """Explicit model_name in config overrides profile name."""
        extra = {"model_name": "my-custom-agent"}
        config = PlatformConfig(enabled=True, extra=extra)
        adapter = APIServerAdapter(config)
        assert adapter._model_name == "my-custom-agent"

    def test_resolve_model_name_explicit(self):
        assert APIServerAdapter._resolve_model_name("my-bot") == "my-bot"

    def test_resolve_model_name_default_profile(self):
        """Default profile falls back to 'xiaoban-agent'."""
        with patch("xiaoban_cli.profiles.get_active_profile_name", return_value="default"):
            assert APIServerAdapter._resolve_model_name("") == "xiaoban-agent"

    def test_resolve_model_name_named_profile(self):
        """Named profile uses the profile name as model name."""
        with patch("xiaoban_cli.profiles.get_active_profile_name", return_value="lucas"):
            assert APIServerAdapter._resolve_model_name("") == "lucas"

    @pytest.mark.asyncio
    async def test_models_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_models_with_valid_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(
                "/v1/models",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200


# ---------------------------------------------------------------------------
# /v1/capabilities endpoint
# ---------------------------------------------------------------------------


class TestCapabilitiesEndpoint:
    @pytest.mark.asyncio
    async def test_capabilities_advertises_plugin_safe_contract(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "xiaoban.api_server.capabilities"
            assert data["platform"] == "xiaoban-agent"
            assert data["model"] == "xiaoban-agent"
            assert data["auth"]["type"] == "bearer"
            assert data["auth"]["required"] is False
            assert data["runtime"]["mode"] == "server_agent"
            assert data["runtime"]["tool_execution"] == "server"
            assert data["runtime"]["split_runtime"] is False
            assert "API-server host" in data["runtime"]["description"]
            assert data["features"]["chat_completions"] is True
            assert data["features"]["run_status"] is True
            assert data["features"]["run_events_sse"] is True
            assert data["features"]["session_continuity_header"] == "X-Xiaoban-Session-Id"
            assert data["endpoints"]["run_status"]["path"] == "/v1/runs/{run_id}"
            assert data["endpoints"]["skills"] == {"method": "GET", "path": "/v1/skills"}
            assert data["endpoints"]["toolsets"] == {"method": "GET", "path": "/v1/toolsets"}

    @pytest.mark.asyncio
    async def test_capabilities_requires_auth_when_key_configured(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 401

            authed = await cli.get(
                "/v1/capabilities",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert authed.status == 200
            data = await authed.json()
            assert data["auth"]["required"] is True


# ---------------------------------------------------------------------------
# /v1/skills and /v1/toolsets endpoints
# ---------------------------------------------------------------------------


class TestSkillsEndpoint:
    @pytest.mark.asyncio
    async def test_skills_returns_list_envelope(self, adapter):
        fake_skills = [
            {"name": "github", "description": "GitHub workflow skill", "category": "github"},
            {"name": "ascii-art", "description": "ASCII art generation", "category": "creative"},
        ]
        with patch(
            "tools.skills_tool._find_all_skills",
            return_value=list(fake_skills),
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/skills")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                names = sorted(s["name"] for s in data["data"])
                assert names == ["ascii-art", "github"]
                for entry in data["data"]:
                    assert set(entry.keys()) >= {"name", "description", "category"}

    @pytest.mark.asyncio
    async def test_skills_handles_enumeration_failure(self, adapter):
        with patch(
            "tools.skills_tool._find_all_skills",
            side_effect=RuntimeError("boom"),
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/skills")
                assert resp.status == 500
                data = await resp.json()
                assert "error" in data

    @pytest.mark.asyncio
    async def test_skills_requires_auth_when_key_configured(self, auth_adapter):
        with patch("tools.skills_tool._find_all_skills", return_value=[]):
            app = _create_app(auth_adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/skills")
                assert resp.status == 401

                authed = await cli.get(
                    "/v1/skills",
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert authed.status == 200


class TestToolsetsEndpoint:
    @pytest.mark.asyncio
    async def test_toolsets_returns_resolved_tools(self, adapter):
        fake_toolsets = [
            ("default", "Default Tools", "Core tools"),
            ("web", "Web Tools", "Search and extract"),
        ]
        with patch(
            "xiaoban_cli.tools_config._get_effective_configurable_toolsets",
            return_value=fake_toolsets,
        ), patch(
            "xiaoban_cli.tools_config._get_platform_tools",
            return_value={"default"},
        ), patch(
            "xiaoban_cli.tools_config._toolset_has_keys",
            return_value=True,
        ), patch(
            "toolsets.resolve_toolset",
            side_effect=lambda name: {
                "default": ["terminal", "read_file"],
                "web": ["web_search"],
            }[name],
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/toolsets")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                assert data["platform"] == "api_server"
                by_name = {ts["name"]: ts for ts in data["data"]}
                assert by_name["default"]["enabled"] is True
                assert by_name["default"]["tools"] == ["read_file", "terminal"]
                assert by_name["web"]["enabled"] is False
                assert by_name["web"]["tools"] == ["web_search"]
                assert by_name["default"]["configured"] is True

    @pytest.mark.asyncio
    async def test_toolsets_handles_resolution_failure_per_toolset(self, adapter):
        """If one toolset fails to resolve, others still appear with empty tools."""
        fake_toolsets = [
            ("broken", "Broken", "fails"),
            ("ok", "OK", "works"),
        ]

        def _resolve(name):
            if name == "broken":
                raise RuntimeError("nope")
            return ["some_tool"]

        with patch(
            "xiaoban_cli.tools_config._get_effective_configurable_toolsets",
            return_value=fake_toolsets,
        ), patch(
            "xiaoban_cli.tools_config._get_platform_tools",
            return_value=set(),
        ), patch(
            "xiaoban_cli.tools_config._toolset_has_keys",
            return_value=False,
        ), patch(
            "toolsets.resolve_toolset",
            side_effect=_resolve,
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/toolsets")
                assert resp.status == 200
                data = await resp.json()
                by_name = {ts["name"]: ts for ts in data["data"]}
                assert by_name["broken"]["tools"] == []
                assert by_name["ok"]["tools"] == ["some_tool"]

    @pytest.mark.asyncio
    async def test_toolsets_requires_auth_when_key_configured(self, auth_adapter):
        with patch(
            "xiaoban_cli.tools_config._get_effective_configurable_toolsets",
            return_value=[],
        ), patch(
            "xiaoban_cli.tools_config._get_platform_tools",
            return_value=set(),
        ):
            app = _create_app(auth_adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/toolsets")
                assert resp.status == 401

                authed = await cli.get(
                    "/v1/toolsets",
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert authed.status == 200


# ---------------------------------------------------------------------------
# /v1/chat/completions endpoint
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    def test_mystand_idempotency_key_is_account_scoped(self, auth_adapter):
        first = auth_adapter._scoped_idempotency_key(_mystand_idempotent_headers("delivery-1", user="alice"), "delivery-1")
        same = auth_adapter._scoped_idempotency_key(_mystand_idempotent_headers("delivery-1", user="alice"), "delivery-1")
        other = auth_adapter._scoped_idempotency_key(_mystand_idempotent_headers("delivery-1", user="bob"), "delivery-1")
        next_attempt = auth_adapter._scoped_idempotency_key(_mystand_idempotent_headers("delivery-1", user="alice", attempt="2"), "delivery-1")

        assert first == same
        assert first != other
        assert first != next_attempt
        assert "alice" not in first
        assert "mystand-test-site" not in first

    @pytest.mark.asyncio
    async def test_mystand_request_fails_closed_without_api_authentication(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_idempotent_headers("missing-server-auth"),
                    json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                )
                data = await response.json()

        assert response.status == 503
        assert data["error"]["code"] == "mystand_auth_unavailable"
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_with_idempotency_key_fails_closed(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_idempotent_headers("stream-delivery"),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                response_data = await resp.json()

        assert resp.status == 409
        assert response_data["error"]["code"] == "idempotency_stream_unsupported"
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_mystand_delivery_uses_stable_ledger_fingerprint(self, auth_adapter):
        app = _create_app(auth_adapter)
        gate = asyncio.Event()
        started = asyncio.Event()
        key = f"delivery-{uuid.uuid4().hex}"
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
        }

        async def _mock_run_agent(**_kwargs):
            started.set()
            await gate.wait()
            return (
                {"final_response": "done", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run:
                first = asyncio.create_task(cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_idempotent_headers(key),
                    json=body,
                ))
                await started.wait()
                second = asyncio.create_task(cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_idempotent_headers(key),
                    json={
                        **body,
                        "messages": [
                            {"role": "system", "content": "server time: later"},
                            {"role": "user", "content": "hello"},
                        ],
                    },
                ))
                await asyncio.sleep(0)
                assert mock_run.call_count == 1
                gate.set()
                first_resp, second_resp = await asyncio.gather(first, second)
                assert first_resp.status == second_resp.status == 200
                await first_resp.read()
                await second_resp.read()

                conflict = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_idempotent_headers(
                        key,
                        request_fingerprint=hashlib.sha256(b"changed-ledger-request").hexdigest(),
                    ),
                    json={**body, "messages": [{"role": "user", "content": "changed"}]},
                )
                conflict_data = await conflict.json()
                changed_context_headers = {
                    **_mystand_idempotent_headers(key),
                    "X-Xiaoban-User-Timezone": "America/New_York",
                }
                context_conflict = await cli.post(
                    "/v1/chat/completions",
                    headers=changed_context_headers,
                    json=body,
                )
                context_conflict_data = await context_conflict.json()

        assert conflict.status == 409
        assert conflict_data["error"]["code"] == "idempotency_conflict"
        assert context_conflict.status == 409
        assert context_conflict_data["error"]["code"] == "idempotency_conflict"
        assert mock_run.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_fingerprint", [None, "not-a-sha256", "f" * 63])
    async def test_mystand_idempotency_requires_valid_ledger_fingerprint(
        self,
        auth_adapter,
        request_fingerprint,
    ):
        key = f"delivery-invalid-fingerprint-{uuid.uuid4().hex}"
        headers = _mystand_idempotent_headers(key)
        if request_fingerprint is None:
            headers.pop("X-Xiaoban-Request-Fingerprint")
        else:
            headers["X-Xiaoban-Request-Fingerprint"] = request_fingerprint

        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                )
                data = await response.json()

        assert response.status == 400
        assert data["error"]["code"] == "invalid_idempotency_scope"
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_mystand_runs_fail_closed_before_agent_creation(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create_agent:
                response = await cli.post(
                    "/v1/runs",
                    headers=_mystand_idempotent_headers("runs-not-supported"),
                    json={"input": "private request"},
                )
                data = await response.json()

        assert response.status == 409
        assert data["error"]["code"] == "mystand_runs_unsupported"
        mock_create_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_mystand_memory_http_is_account_scoped_and_cannot_add(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        from plugins.memory.holographic.scope import open_scoped_memory_store

        secret = "stable-memory-scope-secret-for-tests-20260721"
        monkeypatch.setenv("XIAOBAN_MYSTAND_MEMORY_SCOPE_SECRET", secret)
        monkeypatch.setenv("XIAOBAN_HOME", str(tmp_path))
        store = open_scoped_memory_store(
            secret=secret,
            site_id="mystand-test-site",
            user_id="alice",
            xiaoban_home=tmp_path,
        )
        try:
            fact_id = store.add_fact("alice private preference", category="user_pref")
        finally:
            store.close()

        alice_headers = {
            **_mystand_idempotent_headers("memory-alice", user="alice"),
            "X-Xiaoban-Memory-Mode": "user",
        }
        bob_headers = {
            **_mystand_idempotent_headers("memory-bob", user="bob"),
            "X-Xiaoban-Memory-Mode": "user",
        }
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            alice = await cli.get("/v1/mystand/memory", headers=alice_headers)
            alice_data = await alice.json()
            bob = await cli.get("/v1/mystand/memory", headers=bob_headers)
            bob_data = await bob.json()
            forged_update = await cli.post(
                "/v1/mystand/memory",
                headers=bob_headers,
                json={"action": "update", "factId": fact_id, "content": "forged"},
            )
            active_add = await cli.post(
                "/v1/mystand/memory",
                headers=alice_headers,
                json={"action": "add", "content": "must not be accepted"},
            )
            active_add_data = await active_add.json()

        assert alice.status == 200
        assert [fact["content"] for fact in alice_data["facts"]] == ["alice private preference"]
        assert bob.status == 200
        assert bob_data["facts"] == []
        assert forged_update.status == 404
        assert active_add.status == 400
        assert active_add_data["error"] == "invalid_action"

    @pytest.mark.asyncio
    async def test_mystand_user_memory_fails_closed_without_stable_secret(
        self,
        auth_adapter,
        monkeypatch,
    ):
        monkeypatch.delenv("XIAOBAN_MYSTAND_MEMORY_SCOPE_SECRET", raising=False)
        headers = {
            **_mystand_idempotent_headers("memory-no-secret"),
            "X-Xiaoban-Memory-Mode": "user",
        }
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            response = await cli.get("/v1/mystand/memory", headers=headers)
            response_data = await response.json()

        assert response.status == 400
        assert response_data["error"] == "invalid_memory_scope"

    @pytest.mark.asyncio
    async def test_stop_interrupts_matching_mystand_delivery(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "stop-running.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        app = _create_app(auth_adapter)
        gate = asyncio.Event()
        started = asyncio.Event()
        agent = MagicMock()
        key = f"delivery-stop-{uuid.uuid4().hex}"
        headers = _mystand_idempotent_headers(key)

        async def _mock_run_agent(**kwargs):
            kwargs["agent_ref"][0] = agent
            started.set()
            await gate.wait()
            return (
                {
                    "final_response": "agent ignored interrupt and returned success",
                    "messages": [],
                    "api_calls": 1,
                    "completed": True,
                    "failed": False,
                    "interrupted": False,
                },
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run:
                completion = asyncio.create_task(cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                ))
                await started.wait()
                stopped = await cli.post(
                    "/v1/chat/completions/stop",
                    headers=headers,
                    json={"idempotency_key": key},
                )
                assert stopped.status == 202
                agent.interrupt.assert_called_once_with("Stop requested via My Stand delivery")
                gate.set()
                completion_response = await completion
                completion_data = await completion_response.json()
                replay = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                )
                replay_data = await replay.json()

        assert completion_response.status == 409
        assert completion_data["error"]["code"] == "completion_stopped"
        assert replay.status == 409
        assert replay_data["error"]["code"] == "completion_stopped"
        assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_stop_before_completion_request_prevents_agent_start(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "stop-before-start.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        app = _create_app(auth_adapter)
        key = f"delivery-stop-before-start-{uuid.uuid4().hex}"
        headers = _mystand_idempotent_headers(key)

        async with TestClient(TestServer(app)) as cli:
            stopped = await cli.post(
                "/v1/chat/completions/stop",
                headers=headers,
                json={"idempotency_key": key},
            )
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                completion = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={"model": "test", "messages": [{"role": "user", "content": "hello"}]},
                )
                completion_data = await completion.json()

        assert stopped.status == 202
        assert completion.status == 409
        assert completion_data["error"]["code"] == "completion_stopped"
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_mystand_responses_idempotency_fails_closed_for_every_account(self, auth_adapter):
        app = _create_app(auth_adapter)
        body = {"model": "test", "input": "same private question"}
        key = f"responses-{uuid.uuid4().hex}"

        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                first = await cli.post(
                    "/v1/responses",
                    headers=_mystand_idempotent_headers(key, user="alice"),
                    json=body,
                )
                first_data = await first.json()
                second = await cli.post(
                    "/v1/responses",
                    headers=_mystand_idempotent_headers(key, user="bob"),
                    json=body,
                )
                second_data = await second.json()

        assert first.status == second.status == 409
        assert first_data["error"]["code"] == "mystand_responses_idempotency_unsupported"
        assert second_data["error"]["code"] == "mystand_responses_idempotency_unsupported"
        mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert "Invalid JSON" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "messages" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_empty_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test", "messages": []})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_stream_true_returns_sse(self, adapter):
        """stream=true returns SSE format with the full response."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                # Simulate streaming: invoke stream_delta_callback with tokens
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello!")
                    cb(None)  # End signal
                return (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                assert resp.headers.get("X-Accel-Buffering") == "no"
                body = await resp.text()
                assert "data: " in body
                assert "[DONE]" in body
                assert "Hello!" in body
                assert "event: xiaoban.status" in body
                assert "小伴正在处理中....." in body

    @pytest.mark.asyncio
    async def test_stream_string_false_returns_json_completion(self, adapter):
        """Quoted false must not route chat completions into SSE mode."""
        mock_result = {
            "final_response": "Hello! How can I help you today?",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": "false",
                    },
                )

            assert resp.status == 200
            assert "text/event-stream" not in resp.headers.get("Content-Type", "")
            data = await resp.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["content"] == mock_result["final_response"]

    @pytest.mark.asyncio
    async def test_wechat_article_answer_with_tool_evidence_passes(self, adapter):
        mock_result = {
            "final_response": "这篇文章主要讲 Hermes 的 MoA 混合 Agent 模式。",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "mystand_parse", "arguments": "{}"}},
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"success": true, "title": "Hermes MoA", "text": "正文内容..."}',
                },
            ],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [
                            {
                                "role": "user",
                                "content": "总结这篇公众号：https://mp.weixin.qq.com/s/pbHlRqN_w1RLXnC_IgC8Ag",
                            }
                        ],
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["message"]["content"] == mock_result["final_response"]

    @pytest.mark.asyncio
    async def test_stream_task_done_callback_enqueues_eos_for_chat_completions(self, adapter):
        """Regression guard for #24451: completion callback must signal SSE EOS."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            class _FakeTask:
                def __init__(self):
                    self.callbacks = []

                def add_done_callback(self, cb):
                    self.callbacks.append(cb)

            fake_task = _FakeTask()

            def _fake_ensure_future(coro):
                # We short-circuit task scheduling in this unit test.
                coro.close()
                return fake_task

            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new=AsyncMock(
                        return_value=(
                            {"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        )
                    ),
                ),
                patch("gateway.platforms.api_server.asyncio.ensure_future", side_effect=_fake_ensure_future),
                patch.object(adapter, "_write_sse_chat_completion", new_callable=AsyncMock) as mock_write_sse,
            ):
                mock_write_sse.return_value = web.Response(status=200, text="ok")
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200

            assert len(fake_task.callbacks) == 1
            stream_q = mock_write_sse.call_args.args[4]
            assert stream_q.empty()
            fake_task.callbacks[0](fake_task)
            assert stream_q.get_nowait() is None

    @pytest.mark.asyncio
    async def test_stream_sends_keepalive_during_quiet_tool_gap(self, adapter):
        """Idle SSE streams should send keepalive comments while tools run silently."""
        import asyncio
        import gateway.platforms.api_server as api_server_mod

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Working")
                    await asyncio.sleep(0.65)
                    cb("...done")
                return (
                    {"final_response": "Working...done", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with (
                patch.object(api_server_mod, "CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS", 0.01),
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent),
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "do the thing"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert ": keepalive" in body
                assert "Working" in body
                assert "...done" in body
                assert "[DONE]" in body

    @pytest.mark.asyncio
    async def test_stream_survives_tool_call_none_sentinel(self, adapter):
        """stream_delta_callback(None) mid-stream (tool calls) must NOT kill the SSE stream.

        The agent fires stream_delta_callback(None) to tell the CLI display to
        close its response box before executing tool calls.  The API server's
        _on_delta must filter this out so the SSE response stays open and the
        final answer (streamed after tool execution) reaches the client.
        """
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    # Simulate: agent streams partial text, then fires None
                    # (tool call box-close signal), then streams the final answer
                    cb("Thinking")
                    cb(None)          # mid-stream None from tool calls
                    await asyncio.sleep(0.05)  # simulate tool execution delay
                    cb(" about it...")
                    cb(None)          # another None (possible second tool round)
                    await asyncio.sleep(0.05)
                    cb(" The answer is 42.")
                return (
                    {"final_response": "Thinking about it... The answer is 42.", "messages": [], "api_calls": 3},
                    {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "What is the answer?"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # The final answer text must appear in the SSE stream
                assert "The answer is 42." in body
                # All partial text must be present too
                assert "Thinking" in body
                assert " about it..." in body

    @pytest.mark.asyncio
    async def test_stream_includes_tool_progress(self, adapter):
        """tool_start_callback fires → progress appears as custom SSE event, not in delta.content."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                # Simulate the structured tool start the gateway now consumes.
                if ts_cb:
                    ts_cb("call_terminal_1", "terminal", {"command": "ls -la"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Here are the files.")
                return (
                    {"final_response": "Here are the files.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list files"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # Tool progress must appear as a custom SSE event, not in
                # delta.content — prevents model from learning to imitate
                # markers instead of calling tools (#6972).
                assert "event: xiaoban.tool.progress" in body
                assert '"tool": "terminal"' in body
                # ``label`` is now derived by ``build_tool_preview`` from the
                # tool args rather than passed by the caller, so we assert
                # only that *some* label exists rather than a literal value.
                assert '"label":' in body
                # The progress marker must NOT appear inside any
                # chat.completion.chunk delta.content field.
                import json as _json
                for line in body.splitlines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            chunk = _json.loads(line[len("data: "):])
                        except _json.JSONDecodeError:
                            continue
                        if chunk.get("object") == "chat.completion.chunk":
                            for choice in chunk.get("choices", []):
                                content = choice.get("delta", {}).get("content", "")
                                # Tool emoji markers must never leak into content
                                assert "ls -la" not in content or content == "Here are the files."
                # Final content must also be present
                assert "Here are the files." in body

    @pytest.mark.asyncio
    async def test_stream_tool_progress_skips_internal_events(self, adapter):
        """Internal tool calls (name starting with ``_``) are not streamed."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                if ts_cb:
                    ts_cb("call_internal_1", "_thinking", {"text": "some internal state"})
                    ts_cb("call_search_1", "web_search", {"query": "Python docs"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Found it.")
                return (
                    {"final_response": "Found it.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "search"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                # Internal _thinking event should NOT appear anywhere
                assert "some internal state" not in body
                assert "call_internal_1" not in body
                # Real tool progress should appear as custom SSE event
                assert "event: xiaoban.tool.progress" in body
                assert '"tool": "web_search"' in body
                # Label is derived from the args dict by build_tool_preview;
                # asserting on the structural fact (label exists, call id
                # is correlated) rather than a literal preview string keeps
                # the test robust against preview-formatter tweaks.
                assert '"label":' in body
                assert '"toolCallId": "call_search_1"' in body

    @pytest.mark.asyncio
    async def test_stream_emits_tool_lifecycle_with_call_id(self, adapter):
        """Regression for #16588.

        ``/v1/chat/completions`` streaming previously emitted only a
        ``tool.started``-style ``xiaoban.tool.progress`` event; clients
        rendering tool lifecycle UI had no way to mark a tool as finished
        because no matching ``status: completed`` event was emitted, and
        no ``toolCallId`` was carried for correlation.

        The fix adds ``tool_start_callback`` / ``tool_complete_callback``
        to the chat completions agent invocation and writes both halves
        of the lifecycle pair on the same ``event: xiaoban.tool.progress``
        SSE line, with stable ``toolCallId`` and ``status``.
        """
        import asyncio
        import json as _json

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # The structured callbacks own the chat-completions SSE
                # channel now; ``tool_progress_callback`` is intentionally
                # not wired so each tool start emits exactly one event.
                if ts_cb:
                    ts_cb("call_terminal_1", "terminal", {"command": "ls -la"})
                if tc_cb:
                    tc_cb("call_terminal_1", "terminal", {"command": "ls -la"}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("done.")
                return (
                    {"final_response": "done.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            # Walk the SSE body and collect *(status, toolCallId)* pairs
            # per event so the assertions verify per-event correlation —
            # an event missing ``toolCallId`` would not pass even if a
            # different event happens to carry the right id.
            pairs: list[tuple[str | None, str | None]] = []
            lines = body.splitlines()
            for i, line in enumerate(lines):
                if line.strip() != "event: xiaoban.tool.progress":
                    continue
                for follow in lines[i + 1: i + 4]:
                    if follow.startswith("data: "):
                        try:
                            payload = _json.loads(follow[len("data: "):])
                        except _json.JSONDecodeError:
                            break
                        pairs.append((payload.get("status"), payload.get("toolCallId")))
                        break

            # Each tool start must emit exactly one event (no duplicate
            # legacy + new emit), and each lifecycle pair must carry the
            # same toolCallId on every event — not just somewhere in the
            # aggregate.
            assert len(pairs) == 2, f"expected 2 events (running+completed), got {pairs}"
            assert pairs[0] == ("running", "call_terminal_1"), pairs
            assert pairs[1] == ("completed", "call_terminal_1"), pairs

    @pytest.mark.asyncio
    async def test_stream_tool_lifecycle_reports_failed_result(self, adapter):
        """A real failed tool result must not be rendered as completed."""
        import asyncio
        import json as _json

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                if ts_cb:
                    ts_cb("call_query_failed", "mystand_query", {"query": "missing"})
                if tc_cb:
                    tc_cb(
                        "call_query_failed",
                        "mystand_query",
                        {"query": "missing"},
                        {"ok": False, "error": "not found"},
                    )
                if cb:
                    await asyncio.sleep(0.05)
                    cb("没有查到结果。")
                return (
                    {"final_response": "没有查到结果。", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "查询不存在的资料"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            pairs: list[tuple[str | None, str | None]] = []
            lines = body.splitlines()
            for i, line in enumerate(lines):
                if line.strip() != "event: xiaoban.tool.progress":
                    continue
                for follow in lines[i + 1: i + 4]:
                    if follow.startswith("data: "):
                        payload = _json.loads(follow[len("data: "):])
                        pairs.append((payload.get("status"), payload.get("toolCallId")))
                        break

            assert pairs == [
                ("running", "call_query_failed"),
                ("failed", "call_query_failed"),
            ], pairs

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outcome", "expected_status"),
        [
            ("success", "completed"),
            ("empty", "completed"),
            ("not_found", "failed"),
            ("denied", "failed"),
            ("failed", "failed"),
            ("unknown", "failed"),
            ("cancelled", "stopped"),
        ],
    )
    async def test_stream_tool_terminal_uses_safe_canonical_metadata(
        self,
        adapter,
        outcome,
        expected_status,
    ):
        """Terminal SSE is a bounded projection of the canonical sidecar."""
        app = _create_app(adapter)
        call_id = f"call_canonical_{outcome}"
        raw_secret = f"PRIVATE_RAW_RESULT_{outcome}"
        nested_secret = f"PRIVATE_NESTED_METADATA_{outcome}"

        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                start_cb = kwargs.get("tool_start_callback")
                complete_cb = kwargs.get("tool_complete_callback")
                if start_cb:
                    start_cb(call_id, "mystand_query", {"query": "safe-label"})
                if complete_cb:
                    complete_cb(
                        call_id,
                        "mystand_query",
                        {"privateArg": "MUST_NOT_REACH_TERMINAL_SSE"},
                        raw_secret,
                        {
                            "schema": "xiaoban.tool-result.v1",
                            "requestId": "delivery-e1a",
                            "turnId": "turn-e1a",
                            "callId": call_id,
                            "toolName": "mystand_query",
                            "dispatchState": "dispatched",
                            "outcome": outcome,
                            "retrySafe": False,
                            "recordRefs": [nested_secret],
                            "continuation": {"private": nested_secret},
                        },
                    )
                return (
                    {"final_response": "done", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        terminal_payloads = []
        lines = body.splitlines()
        for index, line in enumerate(lines):
            if line.strip() != "event: xiaoban.tool.progress":
                continue
            for follow in lines[index + 1:index + 4]:
                if follow.startswith("data: "):
                    payload = json.loads(follow[len("data: "):])
                    if payload.get("status") != "running":
                        terminal_payloads.append(payload)
                    break

        assert terminal_payloads == [
            {
                "tool": "mystand_query",
                "toolCallId": call_id,
                "status": expected_status,
                "schema": "xiaoban.tool-result.v1",
                "requestId": "delivery-e1a",
                "turnId": "turn-e1a",
                "dispatchState": "dispatched",
                "outcome": outcome,
                "retrySafe": False,
            }
        ]
        assert raw_secret not in body
        assert nested_secret not in body
        assert "MUST_NOT_REACH_TERMINAL_SSE" not in json.dumps(
            terminal_payloads,
            ensure_ascii=False,
        )

    @pytest.mark.asyncio
    async def test_mystand_tool_sse_payloads_bind_to_accepted_turn(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        """Every real My Stand tool frame carries the accepted turn binding."""
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "tool-binding.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_id = "call-bound-tool"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started",
                delivery_id,
                turn_id,
                None,
            )
            kwargs["tool_start_callback"](
                call_id,
                "mystand_query",
                {"query": "safe"},
            )
            kwargs["tool_complete_callback"](
                call_id,
                "mystand_query",
                {"query": "safe"},
                {"ok": True},
                {
                    "schema": "xiaoban.tool-result.v1",
                    "requestId": delivery_id,
                    "turnId": turn_id,
                    "callId": call_id,
                    "toolName": "mystand_query",
                    "dispatchState": "dispatched",
                    "outcome": "success",
                    "retrySafe": False,
                },
            )
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": "done",
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(
                    auth_adapter,
                    "_run_agent",
                    side_effect=_mock_run_agent,
                ),
                patch(
                    "agent.display.get_tool_emoji",
                    return_value="🔎",
                ),
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        tool_payloads = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("toolCallId") == call_id
        ]
        assert tool_payloads == [
            {
                "tool": "mystand_query",
                "emoji": "🔎",
                "label": "mystand_query",
                "toolCallId": call_id,
                "status": "running",
                "progressSchema": "xiaoban.progress.v2",
                "requestId": delivery_id,
                "turnId": turn_id,
            },
            {
                "tool": "mystand_query",
                "toolCallId": call_id,
                "status": "completed",
                "schema": "xiaoban.tool-result.v1",
                "progressSchema": "xiaoban.progress.v2",
                "requestId": delivery_id,
                "turnId": turn_id,
                "dispatchState": "dispatched",
                "outcome": "success",
                "retrySafe": False,
            },
        ]

    @pytest.mark.asyncio
    async def test_mystand_commentary_binds_to_real_tools_without_becoming_reply(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        """R2-B keeps one natural summary per real call and out of final text."""
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "r2b-commentary.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_ids = ["call-r2b-one", "call-r2b-two"]
        protected_name = "阿黎"
        protected_phone = "13800001234"
        protected_finance = "R2B_FINANCE_BODY_CANARY_781"
        protected_amount = 6350
        commentary = (
            f"我先核对{protected_name}的结算资料 {protected_phone} "
            f"{protected_finance}，金额 {protected_amount}。"
        )
        final_summary = (
            "这轮两项核对已经结束。第一项资料读取成功；第二项因当前"
            "授权范围不足而未能读取，我没有继续越权，也没有把它说成"
            "不存在。若要完成剩余部分，请先补齐第二项授权，我可以从"
            "这里继续。"
        )
        app = _create_app(auth_adapter)
        arguments_by_call = {
            call_ids[0]: {"query": protected_name},
            call_ids[1]: {
                "phone": protected_phone,
                "financeBody": protected_finance,
                "amount": protected_amount,
            },
        }
        tool_calls = [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "mystand_query",
                    "arguments": json.dumps(
                        arguments_by_call[call_id],
                        ensure_ascii=False,
                    ),
                },
            }
            for call_id in call_ids
        ]

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            kwargs["stream_delta_callback"](commentary)
            kwargs["interim_assistant_callback"](
                commentary,
                already_streamed=True,
                tool_calls=tool_calls,
                source="provider",
                provider_sequence=1,
                provider_event_at=1786214400.0,
            )
            kwargs["stream_delta_callback"](None)
            for call_id in call_ids:
                kwargs["tool_start_callback"](
                    call_id,
                    "mystand_query",
                    arguments_by_call[call_id],
                )
            for call_id in call_ids:
                outcome = (
                    "success" if call_id == call_ids[0] else "denied"
                )
                kwargs["tool_complete_callback"](
                    call_id,
                    "mystand_query",
                    arguments_by_call[call_id],
                    (
                        {"ok": True}
                        if outcome == "success"
                        else {"ok": False, "status": 403}
                    ),
                    {
                        "schema": "xiaoban.tool-result.v1",
                        "requestId": delivery_id,
                        "turnId": turn_id,
                        "callId": call_id,
                        "toolName": "mystand_query",
                        "dispatchState": "dispatched",
                        "outcome": outcome,
                        "retrySafe": False,
                    },
                )
            kwargs["stream_delta_callback"](final_summary)
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": final_summary,
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        running = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("toolCallId") in call_ids
            and payload.get("status") == "running"
        ]
        assert [payload["toolCallId"] for payload in running] == call_ids
        assert all(
            payload.get("progressSchema") == "xiaoban.progress.v2"
            and payload.get("requestId") == delivery_id
            and payload.get("turnId") == turn_id
            for payload in running
        )
        commentary_events = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("type") == "assistant.commentary"
        ]
        assert [event["status"] for event in commentary_events] == [
            "completed",
        ]
        assert commentary_events[-1]["source"] == "provider"
        assert commentary_events[-1]["providerSequence"] == 1
        assert commentary_events[-1]["relatedCallIds"] == ",".join(call_ids)
        assert commentary_events[-1].get("toolCallId") is None
        assert commentary_events[-1].get("callId") is None
        terminals = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("toolCallId") in call_ids
            and payload.get("status") in {"completed", "failed"}
        ]
        assert [
            (
                payload["toolCallId"],
                payload["status"],
                payload["outcome"],
            )
            for payload in terminals
        ] == [
            (call_ids[0], "completed", "success"),
            (call_ids[1], "failed", "denied"),
        ]
        decoded_frames = []
        final_chunks = []
        for line in body.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                frame = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            decoded_frames.append(frame)
            if frame.get("object") == "chat.completion.chunk":
                final_chunks.extend(
                    choice.get("delta", {}).get("content", "")
                    for choice in frame.get("choices", [])
                )
        decoded_wire = json.dumps(decoded_frames, ensure_ascii=False)
        for canary in (
            protected_name,
            protected_phone,
            protected_finance,
            str(protected_amount),
        ):
            assert canary not in decoded_wire
        assert commentary not in decoded_wire
        assert "".join(final_chunks) == final_summary
        assert "系统" not in final_summary

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("query", "commentary", "canary", "expected_summary"),
        [
            ("查找阿黎电话", "我先核对阿黎的电话。", "阿黎", None),
            ("查询世纪大道100号", "我先核对100号的登记。", "100号", None),
            (
                "查找客户CUST-ABC12345",
                "我先核对ABC12345。",
                "ABC12345",
                None,
            ),
            ("核对佣金", "我先核对相关佣金规则。", "佣金", None),
            ("读取授权", "我先读取可用授权状态。", "授权", None),
            ("查询状态", "我先查询当前处理状态。", "状态", None),
            (
                "市场资料",
                "我先整理公开市场资料。",
                "市场资料",
                "我先整理公开相关资料。",
            ),
        ],
    )
    async def test_mystand_parameter_overlap_never_uses_fixed_tool_summary(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        query,
        commentary,
        canary,
        expected_summary,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / f"fallback-{uuid.uuid4().hex}.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_id = "call-fixed-summary"
        app = _create_app(auth_adapter)
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "mystand_query",
                "arguments": json.dumps(
                    {"query": query},
                    ensure_ascii=False,
                ),
            },
        }

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            kwargs["interim_assistant_callback"](
                commentary,
                tool_calls=[tool_call],
                source="provider",
                provider_sequence=1,
                provider_event_at=1786214400.0,
            )
            kwargs["stream_delta_callback"](None)
            kwargs["tool_start_callback"](
                call_id,
                "mystand_query",
                {"query": query},
            )
            kwargs["tool_complete_callback"](
                call_id,
                "mystand_query",
                {"query": query},
                {"ok": True},
                {
                    "schema": "xiaoban.tool-result.v1",
                    "requestId": delivery_id,
                    "turnId": turn_id,
                    "callId": call_id,
                    "toolName": "mystand_query",
                    "dispatchState": "dispatched",
                    "outcome": "success",
                    "retrySafe": False,
                },
            )
            kwargs["stream_delta_callback"]("安全最终答复。")
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": "安全最终答复。",
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        running = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("toolCallId") == call_id
            and payload.get("status") == "running"
        ]
        assert len(running) == 1
        assert running[0].get("summary") is None
        commentary_events = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("type") == "assistant.commentary"
        ]
        if expected_summary is None:
            assert commentary_events == []
        else:
            assert len(commentary_events) == 1
            assert commentary_events[0].get("summary") == expected_summary
            assert commentary_events[0].get("callId") is None
        assert commentary not in body
        assert canary not in json.dumps(
            commentary_events,
            ensure_ascii=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "unsafe_case",
        [
            "long-finance-body",
            "deep-arguments",
            "wide-arguments",
            "overlong-commentary",
            "single-character",
            "single-digit",
            "non-finite-number",
        ],
    )
    async def test_mystand_unsafe_progress_input_suppresses_entire_summary(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        unsafe_case,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / f"unsafe-progress-{unsafe_case}.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_id = "call-unsafe-progress"
        if unsafe_case == "long-finance-body":
            marker = "R2B_LONG_FINANCE_PREFIX_CANARY_781"
            arguments = {"financeBody": marker + "长" * 513}
            commentary = f"我先核对{arguments['financeBody']}。"
        elif unsafe_case == "deep-arguments":
            marker = "R2B_DEEP_ARGUMENT_CANARY_781"
            nested = marker
            for index in range(6):
                nested = {f"level{index}": nested}
            arguments = {"payload": nested}
            commentary = f"我先核对{marker}。"
        elif unsafe_case == "wide-arguments":
            marker = "R2B_WIDE_ARGUMENT_CANARY_781"
            items = [f"资料-{index:03d}" for index in range(128)]
            items.append(marker)
            arguments = {"items": items}
            commentary = f"我先核对{marker}。"
        elif unsafe_case == "overlong-commentary":
            marker = "R2B_BOUNDARY_CANARY_781"
            arguments = {"financeBody": marker}
            commentary = "甲" * 1_990 + marker
        elif unsafe_case == "single-character":
            marker = "客户王"
            arguments = {"customerName": "王"}
            commentary = f"我先核对{marker}。"
        elif unsafe_case == "single-digit":
            marker = "房号8号"
            arguments = {"roomNumber": 8}
            commentary = f"我先核对{marker}。"
        else:
            marker = "R2B_NONFINITE_AMOUNT_CANARY_781"
            arguments = {"amount": float("nan")}
            commentary = f"我先核对{marker}。"
        app = _create_app(auth_adapter)
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "mystand_query",
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            kwargs["stream_delta_callback"](commentary)
            kwargs["interim_assistant_callback"](
                commentary,
                already_streamed=True,
                tool_calls=[tool_call],
            )
            kwargs["stream_delta_callback"](None)
            kwargs["tool_start_callback"](
                call_id, "mystand_query", arguments,
            )
            kwargs["tool_complete_callback"](
                call_id,
                "mystand_query",
                arguments,
                {"ok": True},
                {
                    "schema": "xiaoban.tool-result.v1",
                    "requestId": delivery_id,
                    "turnId": turn_id,
                    "callId": call_id,
                    "toolName": "mystand_query",
                    "dispatchState": "dispatched",
                    "outcome": "success",
                    "retrySafe": False,
                },
            )
            kwargs["stream_delta_callback"]("安全最终答复。")
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": "安全最终答复。",
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        running = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("toolCallId") == call_id
            and payload.get("status") == "running"
        ]
        assert len(running) == 1
        assert running[0].get("summary") is None
        assert marker not in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "history_case",
        ["short-string", "numeric", "executor-error", "overflow"],
    )
    async def test_mystand_later_commentary_protects_complete_prior_results(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        history_case,
    ):
        """Historical values never enter later public commentary."""
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / f"r2b-prior-{history_case}.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        first_call_id = "call-prior-one"
        second_call_id = "call-prior-two"
        if history_case == "numeric":
            first_result = {"amount": 8421}
            prior_canary = "8421"
        elif history_case == "executor-error":
            first_result = "Error executing tool 'mystand_query': 松鹤居"
            prior_canary = "松鹤居"
        elif history_case == "overflow":
            values = {"estate": "松鹤居"}
            values.update({
                f"field-{index:03d}": f"历史资料-{index:03d}"
                for index in range(127)
            })
            first_result = {"values": values}
            prior_canary = "松鹤居"
        else:
            first_result = {"estate": "松鹤居"}
            prior_canary = "松鹤居"
        second_commentary = (
            f"我已经找到{prior_canary}，接着核对登记状态。"
        )
        app = _create_app(auth_adapter)

        def _tool_call(call_id, arguments):
            return {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "mystand_query",
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }

        def _terminal_metadata(call_id):
            return {
                "schema": "xiaoban.tool-result.v1",
                "requestId": delivery_id,
                "turnId": turn_id,
                "callId": call_id,
                "toolName": "mystand_query",
                "dispatchState": "dispatched",
                "outcome": "success",
                "retrySafe": False,
            }

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            first_args = {"query": "基础资料"}
            kwargs["interim_assistant_callback"](
                "我先核对基础资料。",
                tool_calls=[_tool_call(first_call_id, first_args)],
                source="provider",
                provider_sequence=1,
                provider_event_at=1786214400.0,
            )
            kwargs["stream_delta_callback"](None)
            kwargs["tool_start_callback"](
                first_call_id, "mystand_query", first_args,
            )
            kwargs["tool_complete_callback"](
                first_call_id,
                "mystand_query",
                first_args,
                first_result,
                _terminal_metadata(first_call_id),
            )

            second_args = {"query": "蓝湾苑"}
            kwargs["interim_assistant_callback"](
                second_commentary,
                tool_calls=[_tool_call(second_call_id, second_args)],
                source="provider",
                provider_sequence=2,
                provider_event_at=1786214401.0,
            )
            kwargs["stream_delta_callback"](None)
            kwargs["tool_start_callback"](
                second_call_id, "mystand_query", second_args,
            )
            kwargs["tool_complete_callback"](
                second_call_id,
                "mystand_query",
                second_args,
                {"ok": True},
                _terminal_metadata(second_call_id),
            )
            kwargs["stream_delta_callback"]("安全最终答复。")
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": "安全最终答复。",
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        second_running = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("toolCallId") == second_call_id
            and payload.get("status") == "running"
        ]
        assert len(second_running) == 1
        second_commentary_events = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("type") == "assistant.commentary"
            and payload.get("providerSequence") == 2
        ]
        if history_case == "overflow":
            assert second_commentary_events == []
        else:
            assert len(second_commentary_events) == 1
            assert second_commentary_events[0].get("summary") == (
                "我已经找到相关资料，接着核对登记状态。"
            )
            assert second_commentary_events[0].get("callId") is None
        assert prior_canary not in json.dumps(
            second_commentary_events,
            ensure_ascii=False,
        )

    @pytest.mark.asyncio
    async def test_mystand_turn_failure_projects_only_allowlisted_details(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        """R2-B exposes typed location/reason without raw failure strings."""
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "r2b-failure.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        private_failure = "PRIVATE_FAILURE_CANARY /root/customer.txt"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("failed")
            return (
                {
                    "final_response": None,
                    "completed": False,
                    "failed": True,
                    "partial": False,
                    "interrupted": False,
                    "error": private_failure,
                    "failure": {
                        "schema": "xiaoban.agent-failure.v1",
                        "kind": "fatal",
                        "code": "provider_call_failed",
                        "phase": "provider_call",
                        "reason": private_failure,
                        "retryable": True,
                    },
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 0,
                    "total_tokens": 1,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        failed_turns = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("type") == "turn.failed"
        ]
        assert failed_turns == [{
            "progressSchema": "xiaoban.progress.v2",
            "type": "turn.failed",
            "requestId": delivery_id,
            "turnId": turn_id,
            "status": "failed",
            "phase": "provider_call",
            "errorCategory": "provider_call_failed",
            "retrySafe": True,
            "summary": "模型服务调用失败，未取得可用响应。",
        }]
        assert private_failure not in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("invalid_case", "raw_result"),
        [
            ("requestId", {"ok": True}),
            (
                "turnId",
                {"ok": False, "error": "PRIVATE_RAW_FAILURE"},
            ),
            ("not-dispatched-success", {"ok": True}),
            (
                "not-dispatched-failed",
                {"ok": False, "error": "PRIVATE_RAW_FAILURE"},
            ),
        ],
        ids=[
            "cross-request",
            "cross-turn",
            "invalid-not-dispatched-success",
            "invalid-not-dispatched-failed",
        ],
    )
    async def test_mystand_tool_terminal_rejects_cross_binding_metadata(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        invalid_case,
        raw_result,
    ):
        """Invalid canonical metadata fails closed without trusting raw status."""
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / f"tool-invalid-{invalid_case}.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_id = f"call-invalid-{invalid_case}"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started",
                delivery_id,
                turn_id,
                None,
            )
            kwargs["tool_start_callback"](
                call_id,
                "mystand_query",
                {"query": "safe"},
            )
            metadata = {
                "schema": "xiaoban.tool-result.v1",
                "requestId": delivery_id,
                "turnId": turn_id,
                "callId": call_id,
                "toolName": "mystand_query",
                "dispatchState": "dispatched",
                "outcome": "success",
                "retrySafe": False,
                "recordRefs": ["PRIVATE_METADATA_MUST_NOT_LEAK"],
            }
            if invalid_case in {"requestId", "turnId"}:
                metadata[invalid_case] = (
                    "xbd_" + "f" * 40
                    if invalid_case == "requestId"
                    else "f" * 16
                )
            else:
                metadata["dispatchState"] = "not_dispatched"
                metadata["outcome"] = invalid_case.rsplit("-", 1)[-1]
            kwargs["tool_complete_callback"](
                call_id,
                "mystand_query",
                {"private": "PRIVATE_ARGS_MUST_NOT_LEAK"},
                raw_result,
                metadata,
            )
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": "done",
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(
                    auth_adapter,
                    "_run_agent",
                    side_effect=_mock_run_agent,
                ),
                patch(
                    "agent.display.get_tool_emoji",
                    return_value="🔎",
                ),
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        terminal_payloads = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("toolCallId") == call_id
            and payload.get("status") != "running"
        ]
        assert terminal_payloads == [
            {
                "tool": "mystand_query",
                "toolCallId": call_id,
                "status": "failed",
                "schema": "xiaoban.tool-result.v1",
                "requestId": delivery_id,
                "turnId": turn_id,
                "dispatchState": "dispatched",
                "outcome": "unknown",
                "retrySafe": False,
                "progressSchema": "xiaoban.progress.v2",
            }
        ]
        assert "PRIVATE_" not in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_call_id", "tool_name"),
        [
            (" call-padded", "mystand_query"),
            ("call/invalid", "mystand_query"),
            ("call-valid", " mystand_query"),
            ("call-valid", "mystand/query"),
        ],
        ids=["padded-id", "invalid-id", "padded-name", "invalid-name"],
    )
    async def test_mystand_invalid_tool_start_fails_turn_without_fake_lifecycle(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        tool_call_id,
        tool_name,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(
                tmp_path / f"invalid-start-{uuid.uuid4().hex}.sqlite"
            ),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        final_canary = "PRIVATE_INVALID_START_FINAL_MUST_NOT_LEAK"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            kwargs["tool_start_callback"](
                tool_call_id,
                tool_name,
                {"query": "阿黎"},
            )
            kwargs["stream_delta_callback"](final_canary)
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": final_canary,
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        payloads = _xiaoban_progress_payloads(body)
        assert not [payload for payload in payloads if "toolCallId" in payload]
        assert [
            (payload.get("type"), payload.get("status"))
            for payload in payloads
            if str(payload.get("type") or "").startswith("turn.")
        ] == [
            ("turn.started", "running"),
            ("turn.failed", "failed"),
        ]
        assert final_canary not in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "integrity_case",
        [
            "orphan-complete",
            "complete-name-mismatch",
            "padded-complete-id",
            "padded-complete-name",
            "duplicate-same-turn",
            "duplicate-different-turn",
        ],
    )
    async def test_mystand_duplicate_or_orphan_lifecycle_fails_closed(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        integrity_case,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / f"integrity-{integrity_case}.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_id = "call-integrity"
        final_canary = "PRIVATE_LIFECYCLE_INTEGRITY_FINAL_CANARY_781"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            progress = kwargs["tool_progress_callback"]
            progress("turn.started", delivery_id, turn_id, None)
            if integrity_case.startswith("duplicate-"):
                duplicate_turn_id = (
                    turn_id
                    if integrity_case == "duplicate-same-turn"
                    else uuid.uuid4().hex[:16]
                )
                progress(
                    "turn.started",
                    delivery_id,
                    duplicate_turn_id,
                    None,
                )
            else:
                if integrity_case != "orphan-complete":
                    kwargs["tool_start_callback"](
                        call_id,
                        "mystand_query",
                        {"query": "阿黎"},
                    )
                complete_id = (
                    f" {call_id}"
                    if integrity_case == "padded-complete-id"
                    else call_id
                )
                complete_name = {
                    "complete-name-mismatch": "mystand_authorization",
                    "padded-complete-name": " mystand_query",
                }.get(integrity_case, "mystand_query")
                kwargs["tool_complete_callback"](
                    complete_id,
                    complete_name,
                    {"query": "阿黎"},
                    {"ok": True},
                    {
                        "schema": "xiaoban.tool-result.v1",
                        "requestId": delivery_id,
                        "turnId": turn_id,
                        "callId": complete_id,
                        "toolName": complete_name,
                        "dispatchState": "dispatched",
                        "outcome": "success",
                        "retrySafe": False,
                    },
                )
            kwargs["stream_delta_callback"](final_canary)
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": final_canary,
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        payloads = _xiaoban_progress_payloads(body)
        assert not [
            payload
            for payload in payloads
            if payload.get("status") == "completed"
        ]
        assert [
            (payload.get("type"), payload.get("status"))
            for payload in payloads
            if str(payload.get("type") or "").startswith("turn.")
        ] == [
            ("turn.started", "running"),
            ("turn.failed", "failed"),
        ]
        assert final_canary not in body

    @pytest.mark.asyncio
    async def test_mystand_tool_callbacks_without_accepted_turn_emit_nothing(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        """A My Stand tool cannot invent its request/turn binding."""
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "tool-no-turn.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_id = "call-without-turn"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            kwargs["tool_start_callback"](
                call_id,
                "mystand_query",
                {"query": "safe"},
            )
            kwargs["tool_complete_callback"](
                call_id,
                "mystand_query",
                {},
                {"ok": True},
                {
                    "schema": "xiaoban.tool-result.v1",
                    "requestId": delivery_id,
                    "turnId": turn_id,
                    "callId": call_id,
                    "toolName": "mystand_query",
                    "dispatchState": "dispatched",
                    "outcome": "success",
                    "retrySafe": False,
                },
            )
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": "done",
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        assert not [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if payload.get("toolCallId") == call_id
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("final_result", "ledger_status", "expected_type", "expected_status"),
        [
            (
                {
                    "final_response": "done",
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                "completed",
                "turn.completed",
                "completed",
            ),
            (
                {
                    "final_response": "",
                    "completed": False,
                    "failed": True,
                    "partial": False,
                    "interrupted": False,
                    "error": "provider call durable settlement failed",
                    "messages": [],
                },
                "failed",
                "turn.failed",
                "failed",
            ),
            (
                {
                    "final_response": "PRIVATE_PARTIAL_FINAL_MUST_NOT_LEAK",
                    "completed": False,
                    "failed": False,
                    "partial": True,
                    "interrupted": False,
                    "error": "response incomplete",
                    "messages": [],
                },
                "failed",
                "turn.failed",
                "failed",
            ),
            (
                {
                    "final_response": "",
                    "completed": False,
                    "failed": True,
                    "partial": False,
                    "interrupted": True,
                    "error": "completion stopped",
                    "messages": [],
                },
                "cancelled",
                "turn.stopped",
                "stopped",
            ),
        ],
        ids=[
            "tool-less-success",
            "settlement-reversed-failure",
            "partial",
            "stopped",
        ],
    )
    async def test_mystand_stream_turn_lifecycle_uses_real_ids_and_settled_result(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        final_result,
        ledger_status,
        expected_type,
        expected_status,
    ):
        """Tool-less turns close from the settled result, never from tools."""
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / f"turn-{ledger_status}.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            progress = kwargs.get("tool_progress_callback")
            assert progress is not None
            progress("turn.started", delivery_id, turn_id, None)
            stream_canary = (
                "done"
                if final_result.get("completed") is True
                else f"PRIVATE_{ledger_status.upper()}_STREAM_MUST_NOT_LEAK"
            )
            kwargs["stream_delta_callback"](stream_canary)
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status(ledger_status)
            usage = {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "agent_calls": ledger.to_dict(),
            }
            return dict(final_result), usage

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        assert response.status == 200
        turn_payloads = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if str(payload.get("type") or "").startswith("turn.")
        ]
        assert turn_payloads == [
            {
                "progressSchema": "xiaoban.progress.v2",
                "type": "turn.started",
                "requestId": delivery_id,
                "turnId": turn_id,
                "status": "running",
            },
            {
                "progressSchema": "xiaoban.progress.v2",
                "type": expected_type,
                "requestId": delivery_id,
                "turnId": turn_id,
                "status": expected_status,
            },
        ]
        assert body.index(f'"type": "{expected_type}"') < body.index(
            "data: [DONE]"
        )
        if final_result.get("completed") is not True:
            assert "PRIVATE_" not in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reported_request_id", "reported_turn_id"),
        [
            ("xbd_" + "f" * 40, "1" * 16),
            (None, "not-a-real-turn"),
        ],
        ids=["cross-request", "invalid-turn-id"],
    )
    async def test_mystand_stream_drops_unbound_turn_lifecycle(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        reported_request_id,
        reported_turn_id,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "turn-unbound.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        app = _create_app(auth_adapter)

        private_final = "PRIVATE_INVALID_TURN_FINAL_MUST_NOT_LEAK"

        async def _mock_run_agent(**kwargs):
            progress = kwargs.get("tool_progress_callback")
            assert progress is not None
            progress(
                "turn.started",
                reported_request_id or delivery_id,
                reported_turn_id,
                None,
            )
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": private_final,
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        assert response.status == 200
        assert not [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if str(payload.get("type") or "").startswith("turn.")
        ]
        assert private_final not in body
        assert "event: xiaoban.error" in body
        assert '"finish_reason": "error"' in body

    @pytest.mark.asyncio
    async def test_mystand_stream_pre_turn_failure_does_not_invent_turn(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "turn-preflight-failure.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**_kwargs):
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("failed")
            return (
                {
                    "final_response": "",
                    "completed": False,
                    "failed": True,
                    "partial": False,
                    "interrupted": False,
                    "error": "agent preflight failed",
                    "messages": [],
                },
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        assert response.status == 200
        assert not [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if str(payload.get("type") or "").startswith("turn.")
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("start_turn", "expected_terminal"),
        [
            (True, "turn.failed"),
            (False, None),
        ],
        ids=["post-start-exception", "pre-turn-exception"],
    )
    async def test_mystand_stream_exception_only_closes_a_started_turn(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        caplog,
        start_turn,
        expected_terminal,
    ):
        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / f"turn-exception-{start_turn}.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        private_error = "PRIVATE_PROVIDER_EXCEPTION_MUST_NOT_LEAK"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            if start_turn:
                kwargs["tool_progress_callback"](
                    "turn.started",
                    delivery_id,
                    turn_id,
                    None,
                )
            raise RuntimeError(private_error)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        turn_payloads = [
            payload
            for payload in _xiaoban_progress_payloads(body)
            if str(payload.get("type") or "").startswith("turn.")
        ]
        if expected_terminal is None:
            assert turn_payloads == []
        else:
            assert turn_payloads == [
                {
                    "progressSchema": "xiaoban.progress.v2",
                    "type": "turn.started",
                    "requestId": delivery_id,
                    "turnId": turn_id,
                    "status": "running",
                },
                {
                    "progressSchema": "xiaoban.progress.v2",
                    "type": expected_terminal,
                    "requestId": delivery_id,
                    "turnId": turn_id,
                    "status": "failed",
                },
            ]
        assert private_error not in body
        assert private_error not in caplog.text

    @pytest.mark.asyncio
    async def test_post_start_cancelled_error_queues_stopped_before_eos(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "turn-cancelled-error.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        captured_items = []
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started",
                delivery_id,
                turn_id,
                None,
            )
            raise asyncio.CancelledError()

        async def _capture_cancelled_stream(*args, **_kwargs):
            stream_q = args[4]
            agent_task = args[5]
            with pytest.raises(asyncio.CancelledError):
                await agent_task
            await asyncio.sleep(0)
            while not stream_q.empty():
                captured_items.append(stream_q.get_nowait())
            return web.Response(status=200, text="cancelled")

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(
                    auth_adapter,
                    "_run_agent",
                    side_effect=_mock_run_agent,
                ),
                patch.object(
                    auth_adapter,
                    "_write_sse_chat_completion",
                    side_effect=_capture_cancelled_stream,
                ),
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )

        progress = [
            item[1]
            for item in captured_items
            if isinstance(item, tuple)
            and len(item) == 2
            and item[0] == "__tool_progress__"
        ]
        assert progress == [
            {
                "progressSchema": "xiaoban.progress.v2",
                "type": "turn.started",
                "requestId": delivery_id,
                "turnId": turn_id,
                "status": "running",
            },
            {
                "progressSchema": "xiaoban.progress.v2",
                "type": "turn.stopped",
                "requestId": delivery_id,
                "turnId": turn_id,
                "status": "stopped",
            },
        ]
        assert captured_items[-1] is None

    @pytest.mark.asyncio
    async def test_turn_terminal_queues_after_open_tool_close_and_before_eos(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "turn-queue-order.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_id = "call-open-at-turn-end"
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            kwargs["tool_progress_callback"](
                "turn.started",
                delivery_id,
                turn_id,
                None,
            )
            kwargs["tool_start_callback"](
                call_id,
                "mystand_query",
                {"query": "safe"},
            )
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("failed")
            return (
                {
                    "final_response": "",
                    "completed": False,
                    "failed": True,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 0,
                    "total_tokens": 1,
                    "agent_calls": ledger.to_dict(),
                },
            )

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(
                    auth_adapter,
                    "_run_agent",
                    side_effect=_mock_run_agent,
                ),
                patch(
                    "agent.display.get_tool_emoji",
                    return_value="🔎",
                ),
            ):
                response = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(delivery_id),
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                body = await response.text()

        payloads = _xiaoban_progress_payloads(body)
        assert payloads == [
            {
                "progressSchema": "xiaoban.progress.v2",
                "type": "turn.started",
                "requestId": delivery_id,
                "turnId": turn_id,
                "status": "running",
            },
            {
                "tool": "mystand_query",
                "emoji": "🔎",
                "label": "mystand_query",
                "toolCallId": call_id,
                "status": "running",
                "progressSchema": "xiaoban.progress.v2",
                "requestId": delivery_id,
                "turnId": turn_id,
            },
            {
                "tool": "mystand_query",
                "toolCallId": call_id,
                "status": "failed",
                "schema": "xiaoban.tool-result.v1",
                "requestId": delivery_id,
                "turnId": turn_id,
                "dispatchState": "dispatched",
                "outcome": "unknown",
                "retrySafe": False,
                "progressSchema": "xiaoban.progress.v2",
            },
            {
                "progressSchema": "xiaoban.progress.v2",
                "type": "turn.failed",
                "requestId": delivery_id,
                "turnId": turn_id,
                "status": "failed",
            },
        ]
        assert body.index('"type": "turn.failed"') < body.index("data: [DONE]")

    @pytest.mark.asyncio
    async def test_mystand_same_process_stream_replays_public_envelope_once(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "stream-replay.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        call_id = "call-stream-replay"
        private_argument = "PRIVATE_REPLAY_ARGUMENT_MUST_NOT_CACHE"
        private_result = "PRIVATE_REPLAY_RESULT_MUST_NOT_CACHE"
        final_text = "这是可重放的最终正文。"
        counters = {"run": 0, "tool": 0}
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            counters["run"] += 1
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            tool_calls = [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "mystand_query",
                    "arguments": json.dumps(
                        {"query": private_argument},
                        ensure_ascii=False,
                    ),
                },
            }]
            kwargs["interim_assistant_callback"](
                "我先核对公开登记状态。",
                tool_calls=tool_calls,
            )
            kwargs["stream_delta_callback"](None)
            counters["tool"] += 1
            kwargs["tool_start_callback"](
                call_id,
                "mystand_query",
                {"query": private_argument},
            )
            kwargs["tool_complete_callback"](
                call_id,
                "mystand_query",
                {"query": private_argument},
                {"rawError": private_result},
                {
                    "schema": "xiaoban.tool-result.v1",
                    "requestId": delivery_id,
                    "turnId": turn_id,
                    "callId": call_id,
                    "toolName": "mystand_query",
                    "dispatchState": "dispatched",
                    "outcome": "success",
                    "retrySafe": False,
                },
            )
            kwargs["stream_delta_callback"](final_text)
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": final_text,
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 7,
                    "output_tokens": 5,
                    "total_tokens": 12,
                    "agent_calls": ledger.to_dict(),
                },
            )

        def _public_projection(body):
            json_frames = []
            named_events = []
            current_event = None
            for line in body.splitlines():
                if line.startswith("event: "):
                    current_event = line[len("event: "):]
                    continue
                if not line.startswith("data: "):
                    if not line:
                        current_event = None
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    continue
                payload = json.loads(data)
                if current_event:
                    if current_event != "xiaoban.status":
                        named_events.append((current_event, payload))
                elif payload.get("object") == "chat.completion.chunk":
                    json_frames.append(payload)
            contents = [
                choice.get("delta", {}).get("content", "")
                for frame in json_frames
                for choice in frame.get("choices", [])
                if choice.get("delta", {}).get("content") is not None
            ]
            finishes = [
                {
                    "finish_reason": choice.get("finish_reason"),
                    "usage": frame.get("usage"),
                }
                for frame in json_frames
                for choice in frame.get("choices", [])
                if choice.get("finish_reason") is not None
            ]
            return {
                "content": contents,
                "events": named_events,
                "finishes": finishes,
                "done": body.count("data: [DONE]"),
            }

        headers = _mystand_stream_headers(delivery_id)
        request_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "run once"}],
            "stream": True,
        }
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                first = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                first_body = await first.text()
                second = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                second_body = await second.text()

        assert first.status == second.status == 200
        assert counters == {"run": 1, "tool": 1}
        assert _public_projection(first_body) == _public_projection(second_body)
        assert "".join(_public_projection(first_body)["content"]) == final_text
        assert _public_projection(first_body)["finishes"] == [{
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 5,
                "total_tokens": 12,
            },
        }]
        first_progress = _xiaoban_progress_payloads(first_body)
        assert first_progress == _xiaoban_progress_payloads(second_body)
        assert first_progress[-1]["type"] == "turn.completed"
        scoped_key = auth_adapter._scoped_idempotency_key(
            headers,
            delivery_id,
        )
        fingerprint = auth_adapter._chat_idempotency_fingerprint(
            request_body,
            headers,
        )
        envelope = durable_cache.load_stream_replay(scoped_key, fingerprint)
        envelope_wire = json.dumps(envelope, ensure_ascii=False)
        assert envelope is not None
        assert private_argument not in first_body + second_body + envelope_wire
        assert private_result not in first_body + second_body + envelope_wire
        assert '"arguments"' not in envelope_wire
        assert '"result"' not in json.dumps(
            envelope.get("items"),
            ensure_ascii=False,
        )

    @pytest.mark.asyncio
    async def test_mystand_concurrent_stream_attaches_inflight_at_run_limit(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        auth_adapter._max_concurrent_runs = 1
        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "stream-replay-inflight.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        private_argument = "PRIVATE_INFLIGHT_ARGUMENT_MUST_NOT_REPLAY"
        private_result = "PRIVATE_INFLIGHT_RESULT_MUST_NOT_REPLAY"
        run_started = asyncio.Event()
        release_run = asyncio.Event()
        counters = {"run": 0, "tool": 0}
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            counters["run"] += 1
            auth_adapter._inflight_agent_runs = 1
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            run_started.set()
            await release_run.wait()
            kwargs["interim_assistant_callback"](
                "我先核对公开资料。",
                tool_calls=[{
                    "id": "call-inflight",
                    "type": "function",
                    "function": {
                        "name": "mystand_query",
                        "arguments": json.dumps({
                            "query": private_argument,
                        }),
                    },
                }],
            )
            kwargs["stream_delta_callback"](None)
            counters["tool"] += 1
            kwargs["tool_start_callback"](
                "call-inflight",
                "mystand_query",
                {"query": private_argument},
            )
            kwargs["tool_complete_callback"](
                "call-inflight",
                "mystand_query",
                {"query": private_argument},
                {"rawError": private_result},
                {
                    "schema": "xiaoban.tool-result.v1",
                    "requestId": delivery_id,
                    "turnId": turn_id,
                    "callId": "call-inflight",
                    "toolName": "mystand_query",
                    "dispatchState": "dispatched",
                    "outcome": "success",
                    "retrySafe": False,
                },
            )
            kwargs["stream_delta_callback"]("并发重放正文。")
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            auth_adapter._inflight_agent_runs = 0
            return (
                {
                    "final_response": "并发重放正文。",
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "agent_calls": ledger.to_dict(),
                },
            )

        headers = _mystand_stream_headers(delivery_id)
        request_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "attach inflight"}],
            "stream": True,
        }
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                first_task = asyncio.create_task(cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                ))
                await run_started.wait()
                second_task = asyncio.create_task(cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                ))
                await asyncio.sleep(0.05)
                release_run.set()
                first, second = await asyncio.gather(
                    first_task,
                    second_task,
                )
                first_body, second_body = await asyncio.gather(
                    first.text(),
                    second.text(),
                )

        assert first.status == second.status == 200
        assert counters == {"run": 1, "tool": 1}
        assert _chat_stream_public_projection(
            first_body
        ) == _chat_stream_public_projection(second_body)
        assert private_argument not in first_body + second_body
        assert private_result not in first_body + second_body

    @pytest.mark.asyncio
    async def test_mystand_stream_replay_coalesces_many_final_chunks(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "stream-replay-many-chunks.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        final_text = "片" * 4_096
        run_count = 0
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            nonlocal run_count
            run_count += 1
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            for chunk in final_text:
                kwargs["stream_delta_callback"](chunk)
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": final_text,
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 4_096,
                    "total_tokens": 4_097,
                    "agent_calls": ledger.to_dict(),
                },
            )

        headers = _mystand_stream_headers(delivery_id)
        request_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "many chunks"}],
            "stream": True,
        }
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                first = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                first_body = await first.text()
                second = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                second_body = await second.text()

        first_projection = _chat_stream_public_projection(first_body)
        second_projection = _chat_stream_public_projection(second_body)
        assert run_count == 1
        assert first_projection["content"] == final_text
        assert second_projection["content"] == final_text
        assert first_projection["progress"] == second_projection["progress"]
        assert first_projection["finishes"] == second_projection["finishes"]
        assert first_projection["progress"][-1]["type"] == "turn.completed"

    @pytest.mark.asyncio
    async def test_mystand_stream_replay_survives_sixty_five_distinct_runs(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        """A reusable response and its public replay envelope evict together."""
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / "stream-replay-capacity.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        deliveries = [
            "xbd_" + uuid.uuid4().hex + f"{index:08x}"
            for index in range(65)
        ]
        expected = {
            delivery_id: {
                "turn": hashlib.sha256(
                    delivery_id.encode("utf-8")
                ).hexdigest()[:16],
                "text": f"容量回放正文-{index:02d}。",
            }
            for index, delivery_id in enumerate(deliveries)
        }
        run_count = 0
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            nonlocal run_count
            run_count += 1
            delivery_id = kwargs["request_headers"].get(
                "X-Xiaoban-Delivery-Id"
            )
            case = expected[delivery_id]
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, case["turn"], None,
            )
            kwargs["stream_delta_callback"](case["text"])
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": case["text"],
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                    "agent_calls": ledger.to_dict(),
                },
            )

        request_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "capacity replay"}],
            "stream": True,
        }
        first_body = ""
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                for index, delivery_id in enumerate(deliveries):
                    response = await cli.post(
                        "/v1/chat/completions",
                        headers=_mystand_stream_headers(delivery_id),
                        json=request_body,
                    )
                    assert response.status == 200
                    body = await response.text()
                    if index == 0:
                        first_body = body
                replay = await cli.post(
                    "/v1/chat/completions",
                    headers=_mystand_stream_headers(deliveries[0]),
                    json=request_body,
                )
                replay_body = await replay.text()

        first_projection = _chat_stream_public_projection(first_body)
        replay_projection = _chat_stream_public_projection(replay_body)
        assert replay.status == 200
        assert run_count == 65
        assert first_projection == replay_projection
        assert first_projection["content"] == expected[deliveries[0]]["text"]
        assert first_projection["progress"][-1]["type"] == "turn.completed"
        assert first_projection["finishes"] == [{
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }]
        assert first_projection["done"] == 1

    @pytest.mark.asyncio
    async def test_mystand_restart_without_local_envelope_fails_closed(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_path = tmp_path / "stream-replay-restart.sqlite"
        first_cache = _IdempotencyCache(
            durable_path=str(durable_path),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            first_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        final_text = "仅首进程可见的完整正文。"
        run_count = 0
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            nonlocal run_count
            run_count += 1
            kwargs["tool_progress_callback"](
                "turn.started", delivery_id, turn_id, None,
            )
            kwargs["stream_delta_callback"](final_text)
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": final_text,
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        headers = _mystand_stream_headers(delivery_id)
        request_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "restart replay"}],
            "stream": True,
        }
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                first = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                first_body = await first.text()
                first_cache._durable.close()
                restarted_cache = _IdempotencyCache(
                    durable_path=str(durable_path),
                    outcome_keys={"test-v1": b"\x31" * 32},
                )
                monkeypatch.setattr(
                    "gateway.platforms.api_server._idem_cache",
                    restarted_cache,
                )
                second = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                second_body = await second.text()

        first_projection = _chat_stream_public_projection(first_body)
        second_projection = _chat_stream_public_projection(second_body)
        assert run_count == 1
        assert first_projection["content"] == final_text
        assert second_projection["content"] == ""
        assert '"finish_reason": "stop"' in first_body
        assert '"finish_reason": "error"' in second_body
        assert "event: xiaoban.error" in second_body
        restarted_cache._durable.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_lifecycle", ["open-tool", "missing-turn"])
    async def test_mystand_stream_never_replays_success_without_closed_turn(
        self,
        auth_adapter,
        monkeypatch,
        tmp_path,
        invalid_lifecycle,
    ):
        from xiaoban.trusted_runtime.agent_call_usage import AgentCallUsageLedger

        durable_cache = _IdempotencyCache(
            durable_path=str(tmp_path / f"replay-{invalid_lifecycle}.sqlite"),
            outcome_keys={"test-v1": b"\x31" * 32},
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server._idem_cache",
            durable_cache,
        )
        delivery_id = "xbd_" + uuid.uuid4().hex + "12345678"
        turn_id = uuid.uuid4().hex[:16]
        private_final = f"PRIVATE_{invalid_lifecycle}_SUCCESS_MUST_NOT_REPLAY"
        run_count = 0
        app = _create_app(auth_adapter)

        async def _mock_run_agent(**kwargs):
            nonlocal run_count
            run_count += 1
            if invalid_lifecycle == "open-tool":
                kwargs["tool_progress_callback"](
                    "turn.started", delivery_id, turn_id, None,
                )
                kwargs["tool_start_callback"](
                    "call-left-open",
                    "mystand_query",
                    {"query": "private"},
                )
            kwargs["stream_delta_callback"](private_final)
            ledger = AgentCallUsageLedger(provider="test", model="test")
            ledger.set_status("completed")
            return (
                {
                    "final_response": private_final,
                    "completed": True,
                    "failed": False,
                    "partial": False,
                    "interrupted": False,
                    "messages": [],
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "agent_calls": ledger.to_dict(),
                },
            )

        headers = _mystand_stream_headers(delivery_id)
        request_body = {
            "model": "test",
            "messages": [{"role": "user", "content": "invalid lifecycle"}],
            "stream": True,
        }
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                auth_adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                first = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                first_body = await first.text()
                second = await cli.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                )
                second_body = await second.text()

        assert run_count == 1
        assert private_final not in first_body + second_body
        assert '"finish_reason": "error"' in first_body
        assert '"finish_reason": "error"' in second_body
        assert "event: xiaoban.error" in first_body
        assert "event: xiaoban.error" in second_body
        first_progress = _xiaoban_progress_payloads(first_body)
        if invalid_lifecycle == "open-tool":
            assert [
                payload.get("type")
                for payload in first_progress
                if payload.get("type")
            ] == ["turn.started", "turn.failed"]
            assert any(
                payload.get("toolCallId") == "call-left-open"
                and payload.get("status") == "failed"
                for payload in first_progress
            )
        else:
            assert first_progress == []
        scoped_key = auth_adapter._scoped_idempotency_key(
            headers,
            delivery_id,
        )
        durable_record = durable_cache.durable_record(scoped_key)
        assert durable_record["state"] == "failed"
        assert durable_record["usage"]["status"] == "failed"
        assert durable_cache._store[scoped_key]["resp"][0]["failed"] is True
        assert durable_cache.load_stream_replay(
            scoped_key,
            auth_adapter._chat_idempotency_fingerprint(
                request_body,
                headers,
            ),
        ) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exit_mode", ["normal", "exception"])
    async def test_stream_tool_lifecycle_closes_orphan_when_agent_exits(
        self,
        adapter,
        exit_mode,
    ):
        """Every started call gets one failed terminal event on agent exit."""
        app = _create_app(adapter)
        call_id = f"call_orphan_{exit_mode}"

        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                start_cb = kwargs.get("tool_start_callback")
                if start_cb:
                    start_cb(call_id, "mystand_query", {"query": "unfinished"})
                if exit_mode == "exception":
                    raise RuntimeError("provider failed after tool start")
                return (
                    {"final_response": "done", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run unfinished tool"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        statuses = []
        for line in body.splitlines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            if payload.get("toolCallId") == call_id:
                statuses.append(payload.get("status"))

        assert statuses == ["running", "failed"]

    @pytest.mark.asyncio
    async def test_stream_tool_lifecycle_closes_open_call_on_task_cancel(self, adapter):
        """Cancelling the API task closes each open tool before queue EOS."""
        app = _create_app(adapter)
        started = asyncio.Event()
        captured_items = []

        async def _mock_run_agent(**kwargs):
            start_cb = kwargs.get("tool_start_callback")
            if start_cb:
                start_cb("call_cancelled", "mystand_query", {"query": "slow"})
            started.set()
            await asyncio.Future()

        async def _cancel_stream_writer(*args, **kwargs):
            stream_q = args[4]
            agent_task = args[5]
            await asyncio.wait_for(started.wait(), timeout=1)
            agent_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await agent_task
            await asyncio.sleep(0)
            while not stream_q.empty():
                captured_items.append(stream_q.get_nowait())
            return web.Response(status=200, text="cancelled")

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent),
                patch.object(
                    adapter,
                    "_write_sse_chat_completion",
                    side_effect=_cancel_stream_writer,
                ),
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "cancel slow tool"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200

        progress = [
            item[1]
            for item in captured_items
            if isinstance(item, tuple)
            and len(item) == 2
            and item[0] == "__tool_progress__"
        ]
        assert [
            (item.get("toolCallId"), item.get("status"))
            for item in progress
        ] == [
            ("call_cancelled", "running"),
            ("call_cancelled", "failed"),
        ]
        assert captured_items[-1] is None

    @pytest.mark.asyncio
    async def test_stream_tool_lifecycle_skips_internal_and_orphan_completes(self, adapter):
        """Internal tools (``_thinking``-style) and ``completed`` events
        without a prior matching ``running`` must produce no lifecycle
        events on the wire — otherwise clients would see orphaned
        ``status: completed`` updates they cannot correlate."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # Internal tool — must be filtered.
                if ts_cb:
                    ts_cb("call_internal_1", "_thinking", {})
                if tc_cb:
                    tc_cb("call_internal_1", "_thinking", {}, "")
                # Completion without start — orphan, must be dropped.
                if tc_cb:
                    tc_cb("call_orphan_1", "web_search", {}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("ok.")
                return (
                    {"final_response": "ok.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "ok"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            # Neither the internal call_id nor the orphan call_id should
            # surface as a lifecycle payload on the wire.
            assert "call_internal_1" not in body
            assert "call_orphan_1" not in body
            assert "event: xiaoban.tool.progress" not in body

    @pytest.mark.asyncio
    async def test_no_user_message_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "system", "content": "You are helpful."}],
                },
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_successful_completion(self, adapter):
        """Test a successful chat completion with mocked agent."""
        mock_result = {
            "final_response": "Hello! How can I help you today?",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "chat.completion"
            assert data["id"].startswith("chatcmpl-")
            assert data["model"] == "xiaoban-agent"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
            assert data["choices"][0]["finish_reason"] == "stop"
            assert "usage" in data

    @pytest.mark.asyncio
    async def test_system_prompt_extracted(self, adapter):
        """System messages from the client are passed as ephemeral_system_prompt."""
        mock_result = {
            "final_response": "I am a pirate! Arrr!",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [
                            {"role": "system", "content": "You are a pirate."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            # Check that _run_agent was called with the system prompt
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("ephemeral_system_prompt") == "You are a pirate."
            assert call_kwargs.kwargs.get("user_message") == "Hello"

    @pytest.mark.asyncio
    async def test_conversation_history_passed(self, adapter):
        """Previous user/assistant messages become conversation_history."""
        mock_result = {"final_response": "3", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [
                            {"role": "user", "content": "1+1=?"},
                            {"role": "assistant", "content": "2"},
                            {"role": "user", "content": "Now add 1 more"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["user_message"] == "Now add 1 more"
            assert len(call_kwargs["conversation_history"]) == 2
            assert call_kwargs["conversation_history"][0] == {"role": "user", "content": "1+1=?"}
            assert call_kwargs["conversation_history"][1] == {"role": "assistant", "content": "2"}

    @pytest.mark.asyncio
    async def test_agent_error_returns_500(self, adapter):
        """Agent exception returns 500."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = RuntimeError("Provider failed")
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

            assert resp.status == 500
            data = await resp.json()
            assert "Provider failed" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_stable_session_id_across_turns(self, adapter):
        """Same conversation (same first user message) produces the same session_id."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        session_ids = []
        async with TestClient(TestServer(app)) as cli:
            # Turn 1: single user message
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )
                session_ids.append(mock_run.call_args.kwargs["session_id"])

            # Turn 2: same first message, conversation grew
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                            {"role": "user", "content": "How are you?"},
                        ],
                    },
                )
                session_ids.append(mock_run.call_args.kwargs["session_id"])

        assert session_ids[0] == session_ids[1], "Session ID should be stable across turns"
        assert session_ids[0].startswith("api-"), "Derived session IDs should have api- prefix"

    @pytest.mark.asyncio
    async def test_different_conversations_get_different_session_ids(self, adapter):
        """Different first messages produce different session_ids."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        session_ids = []
        async with TestClient(TestServer(app)) as cli:
            for first_msg in ["Hello", "Goodbye"]:
                with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                    mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                    await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "xiaoban-agent",
                            "messages": [{"role": "user", "content": first_msg}],
                        },
                    )
                    session_ids.append(mock_run.call_args.kwargs["session_id"])

        assert session_ids[0] != session_ids[1]


# ---------------------------------------------------------------------------
# _derive_chat_session_id unit tests
# ---------------------------------------------------------------------------


class TestDeriveChatSessionId:
    def test_deterministic(self):
        """Same inputs always produce the same session ID."""
        a = _derive_chat_session_id("sys", "hello")
        b = _derive_chat_session_id("sys", "hello")
        assert a == b

    def test_prefix(self):
        assert _derive_chat_session_id(None, "hi").startswith("api-")

    def test_different_system_prompt(self):
        a = _derive_chat_session_id("You are a pirate.", "Hello")
        b = _derive_chat_session_id("You are a robot.", "Hello")
        assert a != b

    def test_different_first_message(self):
        a = _derive_chat_session_id(None, "Hello")
        b = _derive_chat_session_id(None, "Goodbye")
        assert a != b

    def test_none_system_prompt(self):
        """None system prompt doesn't crash."""
        sid = _derive_chat_session_id(None, "test")
        assert isinstance(sid, str) and len(sid) > 4


# ---------------------------------------------------------------------------
# /v1/responses endpoint
# ---------------------------------------------------------------------------


class TestResponsesEndpoint:
    @pytest.mark.asyncio
    async def test_missing_input_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_successful_response_with_string_input(self, adapter):
        """String input is wrapped in a user message."""
        mock_result = {
            "final_response": "Paris is the capital of France.",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "What is the capital of France?",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "response"
            assert data["id"].startswith("resp_")
            assert data["status"] == "completed"
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"
            assert data["output"][0]["content"][0]["type"] == "output_text"
            assert data["output"][0]["content"][0]["text"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_failed_run_with_human_explanation_is_not_marked_completed(
        self,
        adapter,
    ):
        mock_result = {
            "final_response": (
                "资料读取没有取得可用结果，仍缺少完成请求所需的内容。"
            ),
            "messages": [],
            "api_calls": 1,
            "completed": False,
            "failed": True,
            "error": "trusted work did not complete",
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new_callable=AsyncMock,
            ) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {
                        "input_tokens": 1,
                        "output_tokens": 2,
                        "total_tokens": 3,
                    },
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "读取资料并给建议",
                    },
                )

                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "failed"
                assert data["error"]["message"] == (
                    "trusted work did not complete"
                )
                assert data["output"][-1]["content"][0]["text"] == (
                    mock_result["final_response"]
                )

    @pytest.mark.asyncio
    async def test_successful_response_with_array_input(self, adapter):
        """Array input with role/content objects."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": [
                            {"role": "user", "content": "Hello"},
                            {"role": "user", "content": "What is 2+2?"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # Last message is user_message, rest are history
            assert call_kwargs["user_message"] == "What is 2+2?"
            assert len(call_kwargs["conversation_history"]) == 1

    @pytest.mark.asyncio
    async def test_instructions_as_ephemeral_prompt(self, adapter):
        """The instructions field maps to ephemeral_system_prompt."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Hello",
                        "instructions": "Talk like a pirate.",
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Talk like a pirate."

    @pytest.mark.asyncio
    async def test_previous_response_id_chaining(self, adapter):
        """Test that responses can be chained via previous_response_id."""
        mock_result_1 = {
            "final_response": "2",
            "messages": [{"role": "assistant", "content": "2"}],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result_1, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "What is 1+1?"},
                )

            assert resp1.status == 200
            data1 = await resp1.json()
            response_id = data1["id"]

            # Second request chaining from the first
            mock_result_2 = {
                "final_response": "3",
                "messages": [{"role": "assistant", "content": "3"}],
                "api_calls": 1,
            }

            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result_2, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": response_id,
                    },
                )

            assert resp2.status == 200
            # The conversation_history should contain the full history from the first response
            call_kwargs = mock_run.call_args.kwargs
            assert len(call_kwargs["conversation_history"]) > 0
            assert call_kwargs["user_message"] == "Now add 1 more"

    @pytest.mark.asyncio
    async def test_previous_response_id_stores_full_agent_transcript_once(self, adapter):
        """Chained Responses storage must not append result["messages"] twice."""
        first_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "2",
                        "messages": list(first_history),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp1 = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "What is 1+1?"},
                )

            assert resp1.status == 200
            resp1_data = await resp1.json()
            stored_first = adapter._response_store.get(resp1_data["id"])
            assert stored_first["conversation_history"] == first_history

            second_history = first_history + [
                {"role": "user", "content": "Now add 1 more"},
                {"role": "assistant", "content": "3"},
            ]
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(second_history),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": resp1_data["id"],
                    },
                )

            assert resp2.status == 200
            resp2_data = await resp2.json()
            stored_second = adapter._response_store.get(resp2_data["id"])
            stored_history = stored_second["conversation_history"]
            assert stored_history == second_history
            assert stored_history.count(first_history[0]) == 1
            assert stored_history.count({"role": "user", "content": "Now add 1 more"}) == 1

    @pytest.mark.asyncio
    async def test_previous_response_id_outputs_only_current_turn_items(self, adapter):
        """Response output must not replay previous tool artifacts."""
        prior_history = [
            {"role": "user", "content": "Read old file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_old",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"old.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": '{"content":"old"}',
            },
            {"role": "assistant", "content": "old"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        full_agent_transcript = prior_history + [
            {"role": "user", "content": "Read new file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_new",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"new.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_new",
                "content": '{"content":"new"}',
            },
            {"role": "assistant", "content": "new"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "new",
                        "messages": list(full_agent_transcript),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Read new file",
                        "previous_response_id": "resp_prev",
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        output_json = json.dumps(data["output"])
        assert "call_new" in output_json
        assert "call_old" not in output_json
        assert "old.txt" not in output_json

    @pytest.mark.asyncio
    async def test_previous_response_id_preserves_session(self, adapter):
        """Chained responses via previous_response_id reuse the same session_id."""
        mock_result = {
            "final_response": "ok",
            "messages": [{"role": "assistant", "content": "ok"}],
            "api_calls": 1,
        }
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request — establishes a session
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp1 = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "Hello"},
                )
            assert resp1.status == 200
            first_session_id = mock_run.call_args.kwargs["session_id"]
            data1 = await resp1.json()
            response_id = data1["id"]

            # Second request — chains from the first
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Follow up",
                        "previous_response_id": response_id,
                    },
                )
            assert resp2.status == 200
            second_session_id = mock_run.call_args.kwargs["session_id"]

            # Session must be the same across the chain
            assert first_session_id == second_session_id

    @pytest.mark.asyncio
    async def test_invalid_previous_response_id_returns_404(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "xiaoban-agent",
                    "input": "follow up",
                    "previous_response_id": "resp_nonexistent",
                },
            )
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_store_false_does_not_store(self, adapter):
        """When store=false, the response is NOT stored."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Hello",
                        "store": False,
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            # The response has an ID but it shouldn't be retrievable
            assert adapter._response_store.get(data["id"]) is None

    @pytest.mark.asyncio
    async def test_store_string_false_does_not_store(self, adapter):
        """Quoted false must preserve ephemeral store=false semantics."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Hello",
                        "store": "false",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert adapter._response_store.get(data["id"]) is None

    @pytest.mark.asyncio
    async def test_instructions_inherited_from_previous(self, adapter):
        """If no instructions provided, carry forward from previous response."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request with instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Hello",
                        "instructions": "Be a pirate",
                    },
                )

            data1 = await resp1.json()
            resp_id = data1["id"]

            # Second request without instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Tell me more",
                        "previous_response_id": resp_id,
                    },
                )

            assert resp2.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Be a pirate"

    @pytest.mark.asyncio
    async def test_agent_error_returns_500(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.side_effect = RuntimeError("Boom")
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "Hello"},
                )

            assert resp.status == 500

    @pytest.mark.asyncio
    async def test_idempotency_key_reuse_with_different_request_returns_409(self, adapter):
        app = _create_app(adapter)
        key = f"responses-conflict-{uuid.uuid4().hex}"
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                first = await cli.post(
                    "/v1/responses",
                    headers={"Idempotency-Key": key},
                    json={"model": "test", "input": "first request"},
                )
                second = await cli.post(
                    "/v1/responses",
                    headers={"Idempotency-Key": key},
                    json={"model": "test", "input": "different request"},
                )
                second_data = await second.json()

        assert first.status == 200
        assert second.status == 409
        assert second_data["error"]["code"] == "idempotency_conflict"
        assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_input_type_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "xiaoban-agent", "input": 42},
            )
            assert resp.status == 400


class TestResponsesStreaming:
    @pytest.mark.asyncio
    async def test_stream_true_returns_responses_sse(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello")
                    cb(" world")
                return (
                    {"final_response": "Hello world", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                body = await resp.text()
                assert "event: response.created" in body
                assert "event: response.output_text.delta" in body
                assert "event: response.output_text.done" in body
                assert "event: response.completed" in body
                assert '"sequence_number":' in body
                assert '"logprobs": []' in body
                assert "Hello" in body
                assert " world" in body

    @pytest.mark.asyncio
    async def test_stream_failed_run_keeps_human_text_and_failed_terminal(
        self,
        adapter,
    ):
        explanation = (
            "读取已经尝试过，但服务仍不可用，所需资料没有取得。"
        )

        async def _mock_run_agent(**kwargs):
            return (
                {
                    "final_response": explanation,
                    "messages": [],
                    "api_calls": 1,
                    "completed": False,
                    "failed": True,
                    "error": "trusted work did not complete",
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                side_effect=_mock_run_agent,
            ):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "读取资料",
                        "stream": True,
                    },
                )
                body = await resp.text()

        assert resp.status == 200
        decoded_events = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        assert explanation in json.dumps(decoded_events, ensure_ascii=False)
        assert "event: response.failed" in body
        assert "event: response.completed" not in body

    @pytest.mark.asyncio
    async def test_stream_string_false_returns_json_response(self, adapter):
        """Quoted false must not route Responses API requests into SSE mode."""
        mock_result = {
            "final_response": "Paris is the capital of France.",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "What is the capital of France?",
                        "stream": "false",
                    },
                )

            assert resp.status == 200
            assert "text/event-stream" not in resp.headers.get("Content-Type", "")
            data = await resp.json()
            assert data["object"] == "response"
            assert data["output"][0]["content"][0]["text"] == mock_result["final_response"]

    @pytest.mark.asyncio
    async def test_stream_task_done_callback_enqueues_eos_for_responses(self, adapter):
        """Regression guard for #24451 on /v1/responses streaming path."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            class _FakeTask:
                def __init__(self):
                    self.callbacks = []

                def add_done_callback(self, cb):
                    self.callbacks.append(cb)

            fake_task = _FakeTask()

            def _fake_ensure_future(coro):
                # We short-circuit task scheduling in this unit test.
                coro.close()
                return fake_task

            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new=AsyncMock(
                        return_value=(
                            {"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        )
                    ),
                ),
                patch("gateway.platforms.api_server.asyncio.ensure_future", side_effect=_fake_ensure_future),
                patch.object(adapter, "_write_sse_responses", new_callable=AsyncMock) as mock_write_sse,
            ):
                mock_write_sse.return_value = web.Response(status=200, text="ok")
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200

            assert len(fake_task.callbacks) == 1
            stream_q = mock_write_sse.call_args.kwargs["stream_q"]
            assert stream_q.empty()
            fake_task.callbacks[0](fake_task)
            assert stream_q.get_nowait() is None

    @pytest.mark.asyncio
    async def test_stream_emits_function_call_and_output_items(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                start_cb = kwargs.get("tool_start_callback")
                complete_cb = kwargs.get("tool_complete_callback")
                text_cb = kwargs.get("stream_delta_callback")
                if start_cb:
                    start_cb("call_123", "read_file", {"path": "/tmp/test.txt"})
                if complete_cb:
                    complete_cb("call_123", "read_file", {"path": "/tmp/test.txt"}, '{"content":"hello"}')
                if text_cb:
                    text_cb("Done.")
                return (
                    {
                        "final_response": "Done.",
                        "messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"/tmp/test.txt"}',
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "call_123",
                                "content": '{"content":"hello"}',
                            },
                        ],
                        "api_calls": 1,
                    },
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "read the file", "stream": True},
                )
                assert resp.status == 200
                body = await resp.text()
                assert "event: response.output_item.added" in body
                assert "event: response.output_item.done" in body
                assert body.count("event: response.output_item.done") >= 2
                assert '"type": "function_call"' in body
                assert '"type": "function_call_output"' in body
                assert '"call_id": "call_123"' in body
                assert '"name": "read_file"' in body
                assert '"output": [{"type": "input_text", "text": "{\\"content\\":\\"hello\\"}"}]' in body

    @pytest.mark.asyncio
    async def test_streamed_response_is_stored_for_get(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Stored response")
                return (
                    {"final_response": "Stored response", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "store this", "stream": True},
                )
                body = await resp.text()
                response_id = None
                for line in body.splitlines():
                    if line.startswith("data: "):
                        try:
                            payload = json.loads(line[len("data: "):])
                        except json.JSONDecodeError:
                            continue
                        if payload.get("type") == "response.completed":
                            response_id = payload["response"]["id"]
                            break
                assert response_id

                get_resp = await cli.get(f"/v1/responses/{response_id}")
                assert get_resp.status == 200
                data = await get_resp.json()
                assert data["id"] == response_id
                assert data["status"] == "completed"
                assert data["output"][-1]["content"][0]["text"] == "Stored response"

    @pytest.mark.asyncio
    async def test_streamed_previous_response_id_stores_full_agent_transcript_once(self, adapter):
        prior_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )

        expected_history = prior_history + [
            {"role": "user", "content": "Now add 1 more"},
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("3")
                return (
                    {
                        "final_response": "3",
                        "messages": list(expected_history),
                        "api_calls": 1,
                    },
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                        "stream": True,
                    },
                )
                body = await resp.text()

        assert resp.status == 200
        response_id = None
        for line in body.splitlines():
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "response.completed":
                    response_id = payload["response"]["id"]
                    break

        assert response_id
        stored_history = adapter._response_store.get(response_id)["conversation_history"]
        assert stored_history == expected_history
        assert stored_history.count(prior_history[0]) == 1
        assert stored_history.count({"role": "user", "content": "Now add 1 more"}) == 1

    @pytest.mark.asyncio
    async def test_stream_cancelled_persists_incomplete_snapshot(self, adapter):
        """Server-side asyncio.CancelledError (shutdown, request timeout) must
        still leave an ``incomplete`` snapshot in ResponseStore so
        GET /v1/responses/{id} and previous_response_id chaining keep
        working.  Regression for PR #15171 follow-up.

        Calls _write_sse_responses directly so the test can await the
        handler to completion (TestClient disconnection races the server
        handler, which makes end-to-end assertion on the final stored
        snapshot flaky).
        """
        # Build a minimal fake request + stream queue the writer understands.
        fake_request = MagicMock()
        fake_request.headers = {}

        written_payloads: list = []

        class _FakeStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                written_payloads.append(payload)

        # Patch web.StreamResponse for the duration of the writer call.
        import gateway.platforms.api_server as api_mod
        import queue as _q

        stream_q: _q.Queue = _q.Queue()

        async def _agent_coro():
            # Feed one partial delta into the stream queue...
            stream_q.put("partial output")
            # ...then give the drain loop a moment to pick it up before
            # raising CancelledError to simulate a server-side cancel.
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError()

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_FakeStreamResponse()):
            with pytest.raises(asyncio.CancelledError):
                await adapter._write_sse_responses(
                    request=fake_request,
                    response_id=response_id,
                    model="xiaoban-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=[None],
                    conversation_history=[],
                    user_message="will be cancelled",
                    instructions=None,
                    conversation=None,
                    store=True,
                    session_id=None,
                )

        # The in_progress snapshot was persisted on response.created,
        # and the CancelledError handler must have updated it to
        # ``incomplete`` with the partial text it saw.
        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must be retrievable after cancellation"
        assert stored["response"]["status"] == "incomplete"
        # Partial text captured before cancel should be preserved.
        output_text = "".join(
            part.get("text", "")
            for item in stored["response"].get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
        )
        assert "partial output" in output_text

    @pytest.mark.asyncio
    async def test_stream_client_disconnect_persists_incomplete_snapshot(self, adapter):
        """Client disconnect (ConnectionResetError) during streaming must
        persist an ``incomplete`` snapshot in ResponseStore.  Regression
        for PR #15171."""
        fake_request = MagicMock()
        fake_request.headers = {}

        write_call_count = {"n": 0}

        class _DisconnectingStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                # First two writes succeed (prepare + response.created).
                # On the third write (a text delta), the "client"
                # disconnects — simulate with ConnectionResetError.
                write_call_count["n"] += 1
                if write_call_count["n"] >= 3:
                    raise ConnectionResetError("simulated client disconnect")

        import gateway.platforms.api_server as api_mod
        import queue as _q

        stream_q: _q.Queue = _q.Queue()
        stream_q.put("some streamed text")
        stream_q.put(None)  # EOS sentinel

        async def _agent_coro():
            await asyncio.sleep(0.01)
            return ({"final_response": "", "messages": [], "api_calls": 0},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_DisconnectingStreamResponse()):
            await adapter._write_sse_responses(
                request=fake_request,
                response_id=response_id,
                model="xiaoban-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=[None],
                conversation_history=[],
                user_message="will disconnect",
                instructions=None,
                conversation=None,
                store=True,
                session_id=None,
            )

        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must survive client disconnect"
        assert stored["response"]["status"] == "incomplete"


# ---------------------------------------------------------------------------
# Auth on endpoints
# ---------------------------------------------------------------------------


class TestEndpointAuth:
    @pytest.mark.asyncio
    async def test_chat_completions_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_responses_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={"model": "test", "input": "hi"},
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_models_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_health_does_not_require_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_platform_enum_has_api_server(self):
        assert Platform.API_SERVER.value == "api_server"

    def test_env_override_enables_api_server(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert Platform.API_SERVER in config.platforms
        assert config.platforms[Platform.API_SERVER].enabled is True

    def test_env_override_with_key(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_KEY", "sk-mykey")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert Platform.API_SERVER in config.platforms
        assert config.platforms[Platform.API_SERVER].extra.get("key") == "sk-mykey"

    def test_env_override_port_and_host(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv("API_SERVER_PORT", "9999")
        monkeypatch.setenv("API_SERVER_HOST", "0.0.0.0")
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("port") == 9999
        assert config.platforms[Platform.API_SERVER].extra.get("host") == "0.0.0.0"

    def test_env_override_cors_origins(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv(
            "API_SERVER_CORS_ORIGINS",
            "http://localhost:3000, http://127.0.0.1:3000",
        )
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("cors_origins") == [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_api_server_in_connected_platforms(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(enabled=True)
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER in connected

    def test_api_server_not_in_connected_when_disabled(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(enabled=False)
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER not in connected


# ---------------------------------------------------------------------------
# Multiple system messages
# ---------------------------------------------------------------------------


class TestMultipleSystemMessages:
    @pytest.mark.asyncio
    async def test_multiple_system_messages_concatenated(self, adapter):
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "system", "content": "Be concise."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            prompt = call_kwargs["ephemeral_system_prompt"]
            assert "You are helpful." in prompt
            assert "Be concise." in prompt


# ---------------------------------------------------------------------------
# send() method queues API-server session events
# ---------------------------------------------------------------------------


class TestSendMethod:
    @pytest.mark.asyncio
    async def test_send_queues_session_event(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        result = await adapter.send("chat1", "hello")
        assert result.success is True
        events = adapter._session_event_snapshot("chat1", since=0)
        assert len(events) == 1
        assert events[0]["message"]["content"] == "hello"


# ---------------------------------------------------------------------------
# GET /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestGetResponse:
    @pytest.mark.asyncio
    async def test_get_stored_response(self, adapter):
        """GET returns a previously stored response."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Create a response first
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            response_id = data["id"]

            # Now GET it
            resp2 = await cli.get(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_not_found(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_get_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/responses/resp_any")
            assert resp.status == 401


# ---------------------------------------------------------------------------
# DELETE /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestDeleteResponse:
    @pytest.mark.asyncio
    async def test_delete_stored_response(self, adapter):
        """DELETE removes a stored response and returns confirmation."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "Hi"},
                )

            data = await resp.json()
            response_id = data["id"]

            # Delete it
            resp2 = await cli.delete(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["deleted"] is True

            # Verify it's gone
            resp3 = await cli.get(f"/v1/responses/{response_id}")
            assert resp3.status == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/v1/responses/resp_nonexistent")
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_delete_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/v1/responses/resp_any")
            assert resp.status == 401


# ---------------------------------------------------------------------------
# Tool calls in output
# ---------------------------------------------------------------------------


class TestToolCallsInOutput:
    @pytest.mark.asyncio
    async def test_tool_calls_in_output(self, adapter):
        """When agent returns tool calls, they appear as function_call items."""
        mock_result = {
            "final_response": "The result is 42.",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "6*7"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "42",
                },
                {
                    "role": "assistant",
                    "content": "The result is 42.",
                },
            ],
            "api_calls": 2,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "What is 6*7?"},
                )

            assert resp.status == 200
            data = await resp.json()
            output = data["output"]

            # Should have: function_call, function_call_output, message
            assert len(output) == 3
            assert output[0]["type"] == "function_call"
            assert output[0]["name"] == "calculator"
            assert output[0]["arguments"] == '{"expression": "6*7"}'
            assert output[0]["call_id"] == "call_abc123"
            assert output[1]["type"] == "function_call_output"
            assert output[1]["call_id"] == "call_abc123"
            assert output[1]["output"] == "42"
            assert output[2]["type"] == "message"
            assert output[2]["content"][0]["text"] == "The result is 42."

    @pytest.mark.asyncio
    async def test_no_tool_calls_still_works(self, adapter):
        """Without tool calls, output is just a message."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "Hello"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"


# ---------------------------------------------------------------------------
# Usage / token counting
# ---------------------------------------------------------------------------


class TestUsageCounting:
    @pytest.mark.asyncio
    async def test_responses_usage(self, adapter):
        """Responses API returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "xiaoban-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["input_tokens"] == 100
            assert data["usage"]["output_tokens"] == 50
            assert data["usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_chat_completions_usage(self, adapter):
        """Chat completions returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "xiaoban-agent",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["prompt_tokens"] == 200
            assert data["usage"]["completion_tokens"] == 80
            assert data["usage"]["total_tokens"] == 280


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    @pytest.mark.asyncio
    async def test_truncation_auto_limits_history(self, adapter):
        """With truncation=auto, history over 100 messages is trimmed."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        # Pre-seed a stored response with a long history
        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(150)]
        adapter._response_store.put("resp_prev", {
            "response": {"id": "resp_prev", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_prev",
                        "truncation": "auto",
                    },
                )

        assert resp.status == 200
        call_kwargs = mock_run.call_args.kwargs
        # History should be truncated to 100
        assert len(call_kwargs["conversation_history"]) <= 100

    @pytest.mark.asyncio
    async def test_no_truncation_keeps_full_history(self, adapter):
        """Without truncation=auto, long history is passed as-is."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(150)]
        adapter._response_store.put("resp_prev2", {
            "response": {"id": "resp_prev2", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "xiaoban-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_prev2",
                    },
                )

        assert resp.status == 200
        call_kwargs = mock_run.call_args.kwargs
        assert len(call_kwargs["conversation_history"]) == 150


# ---------------------------------------------------------------------------
# Response-side truncation / failure handling (issue #22496)
# ---------------------------------------------------------------------------


class TestChatCompletionsAgentIncomplete:
    """When the agent run yields a partial / failed result, the API server
    must NOT pretend it succeeded. Either signal truncation via
    finish_reason='length' (with the partial text), or 502 with an OpenAI
    error envelope (no usable text). Issue #22496."""

    @pytest.mark.asyncio
    async def test_truncation_with_partial_text_uses_length_finish_reason(self, adapter):
        """Partial text + truncation marker → finish_reason='length', 200 OK,
        plus xiaoban extras + headers."""
        mock_result = {
            "final_response": "Here is part one of the answer",
            "completed": False,
            "partial": True,
            "error": "Response truncated due to output length limit",
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "tell me everything"}]},
                )
            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["finish_reason"] == "length"
            assert data["choices"][0]["message"]["content"] == "Here is part one of the answer"
            assert data["xiaoban"]["partial"] is True
            assert data["xiaoban"]["completed"] is False
            assert data["xiaoban"]["error_code"] == "output_truncated"
            assert resp.headers.get("X-Xiaoban-Completed") == "false"
            assert resp.headers.get("X-Xiaoban-Partial") == "true"

    @pytest.mark.asyncio
    async def test_failure_with_no_text_returns_502_error_envelope(self, adapter):
        """No usable assistant text + failure → 502 with OpenAI error envelope.

        Pre-fix behavior: the failure string ('Response remained truncated...')
        was substituted into message.content with finish_reason='stop',
        making API clients think the agent had answered.
        """
        mock_result = {
            "final_response": None,
            "completed": False,
            "partial": True,
            "failed": True,
            "error": "Response remained truncated after 3 continuation attempts",
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "x"}]},
                )
            # Hard fail: SDK clients will raise on this status
            assert resp.status == 502
            data = await resp.json()
            assert data["error"]["code"] == "agent_incomplete"
            assert "truncated" in data["error"]["message"].lower()
            assert data["error"]["xiaoban"]["partial"] is True
            assert data["error"]["xiaoban"]["failed"] is True
            assert resp.headers.get("X-Xiaoban-Completed") == "false"

    @pytest.mark.asyncio
    async def test_normal_completion_unchanged(self, adapter):
        """Sanity: a completed-True result still returns finish_reason='stop'
        and no xiaoban extras (preserves the existing happy-path contract)."""
        mock_result = {
            "final_response": "All good.",
            "completed": True,
            "partial": False,
            "failed": False,
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["finish_reason"] == "stop"
            assert data["choices"][0]["message"]["content"] == "All good."
            assert "xiaoban" not in data
            assert "X-Xiaoban-Completed" not in resp.headers


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_origin_allowed_for_non_browser_client(self, adapter):
        assert adapter._origin_allowed("") is True

    def test_origin_rejected_by_default(self, adapter):
        assert adapter._origin_allowed("http://evil.example") is False

    def test_origin_allowed_for_allowlist_match(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._origin_allowed("http://localhost:3000") is True

    def test_cors_headers_for_origin_disabled_by_default(self, adapter):
        assert adapter._cors_headers_for_origin("http://localhost:3000") is None

    def test_cors_headers_for_origin_matches_allowlist(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        headers = adapter._cors_headers_for_origin("http://localhost:3000")
        assert headers is not None
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert "POST" in headers["Access-Control-Allow-Methods"]

    def test_cors_headers_for_origin_rejects_unknown_origin(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._cors_headers_for_origin("http://evil.example") is None

    @pytest.mark.asyncio
    async def test_cors_headers_not_present_by_default(self, adapter):
        """CORS is disabled unless explicitly configured."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_browser_origin_rejected_by_default(self, adapter):
        """Browser-originated requests are rejected unless explicitly allowed."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://evil.example"})
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_cors_options_preflight_rejected_by_default(self, adapter):
        """Browser preflight is rejected unless CORS is explicitly configured."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://evil.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_cors_headers_present_for_allowed_origin(self):
        """Allowed origins receive explicit CORS headers."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://localhost:3000"})
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")
            assert "DELETE" in resp.headers.get("Access-Control-Allow-Methods", "")

    @pytest.mark.asyncio
    async def test_cors_allows_idempotency_key_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Idempotency-Key",
                },
            )
            assert resp.status == 200
            assert "Idempotency-Key" in resp.headers.get("Access-Control-Allow-Headers", "")

    @pytest.mark.asyncio
    async def test_cors_sets_vary_origin_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://localhost:3000"})
            assert resp.status == 200
            assert resp.headers.get("Vary") == "Origin"

    @pytest.mark.asyncio
    async def test_cors_options_preflight_allowed_for_configured_origin(self):
        """Configured origins can complete browser preflight."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "Authorization" in resp.headers.get("Access-Control-Allow-Headers", "")


    @pytest.mark.asyncio
    async def test_cors_preflight_sets_max_age(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Max-Age") == "600"
# ---------------------------------------------------------------------------
# Conversation parameter
# ---------------------------------------------------------------------------


class TestConversationParameter:
    @pytest.mark.asyncio
    async def test_conversation_creates_new(self, adapter):
        """First request with a conversation name works (new conversation)."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post("/v1/responses", json={
                    "input": "hi",
                    "conversation": "my-chat",
                })
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "completed"
                # Conversation mapping should be set
                assert adapter._response_store.get_conversation("my-chat") is not None

    @pytest.mark.asyncio
    async def test_conversation_chains_automatically(self, adapter):
        """Second request with same conversation name chains to first."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "First response", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # First request
                resp1 = await cli.post("/v1/responses", json={
                    "input": "hello",
                    "conversation": "test-conv",
                })
                assert resp1.status == 200
                data1 = await resp1.json()
                resp1_id = data1["id"]

                # Second request — should chain
                mock_run.return_value = (
                    {"final_response": "Second response", "messages": [], "api_calls": 1},
                    {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                )
                resp2 = await cli.post("/v1/responses", json={
                    "input": "follow up",
                    "conversation": "test-conv",
                })
                assert resp2.status == 200

                # The second call should have received conversation history from the first
                assert mock_run.call_count == 2
                second_call_kwargs = mock_run.call_args_list[1]
                history = second_call_kwargs.kwargs.get("conversation_history",
                          second_call_kwargs[1].get("conversation_history", []) if len(second_call_kwargs) > 1 else [])
                # History should be non-empty (contains messages from first response)
                assert len(history) > 0

    @pytest.mark.asyncio
    async def test_conversation_and_previous_response_id_conflict(self, adapter):
        """Cannot use both conversation and previous_response_id."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/responses", json={
                "input": "hi",
                "conversation": "my-chat",
                "previous_response_id": "resp_abc123",
            })
            assert resp.status == 400
            data = await resp.json()
            assert "Cannot use both" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_separate_conversations_are_isolated(self, adapter):
        """Different conversation names have independent histories."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Response A", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Conversation A
                await cli.post("/v1/responses", json={"input": "conv-a msg", "conversation": "conv-a"})
                # Conversation B
                mock_run.return_value = (
                    {"final_response": "Response B", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "conv-b msg", "conversation": "conv-b"})

                # They should have different response IDs in the mapping
                assert adapter._response_store.get_conversation("conv-a") != adapter._response_store.get_conversation("conv-b")

    @pytest.mark.asyncio
    async def test_conversation_store_false_no_mapping(self, adapter):
        """If store=false, conversation mapping is not updated."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Ephemeral", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post("/v1/responses", json={
                    "input": "hi",
                    "conversation": "ephemeral-chat",
                    "store": False,
                })
                assert resp.status == 200
                # Conversation mapping should NOT be set since store=false
                assert adapter._response_store.get_conversation("ephemeral-chat") is None

    @pytest.mark.asyncio
    async def test_conversation_reuse_after_eviction_no_404(self, adapter):
        """After eviction clears a conversation mapping, reusing that name starts fresh (no 404)."""
        adapter._response_store = ResponseStore(max_size=1)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "First", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Create conversation -> resp stored
                resp1 = await cli.post("/v1/responses", json={
                    "input": "hello",
                    "conversation": "my-chat",
                })
                assert resp1.status == 200

                # Evict by adding another response
                mock_run.return_value = (
                    {"final_response": "Other", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "other"})

                # Conversation mapping should have been cleaned by eviction
                assert adapter._response_store.get_conversation("my-chat") is None

                # Reuse conversation name — should start fresh, not 404
                mock_run.return_value = (
                    {"final_response": "Restarted", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp3 = await cli.post("/v1/responses", json={
                    "input": "hello again",
                    "conversation": "my-chat",
                })
                assert resp3.status == 200


# ---------------------------------------------------------------------------
# X-Xiaoban-Session-Id header (session continuity)
# ---------------------------------------------------------------------------


class TestSessionIdHeader:
    @pytest.mark.asyncio
    async def test_new_session_response_includes_session_id_header(self, adapter):
        """Without X-Xiaoban-Session-Id, a new session is created and returned in the header."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "Hi"}]},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Xiaoban-Session-Id") is not None

    @pytest.mark.asyncio
    async def test_provided_session_id_is_used_and_echoed(self, auth_adapter):
        """When X-Xiaoban-Session-Id is provided, it's passed to the agent and echoed in the response."""
        mock_result = {"final_response": "Continuing!", "messages": [], "api_calls": 1}
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "previous message"},
            {"role": "assistant", "content": "previous reply"},
        ]
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Xiaoban-Session-Id": "my-session-123", "Authorization": "Bearer sk-secret"},
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "Continue"}]},
                )

            assert resp.status == 200
            assert resp.headers.get("X-Xiaoban-Session-Id") == "my-session-123"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["session_id"] == "my-session-123"

    @pytest.mark.asyncio
    async def test_provided_session_id_loads_history_from_db(self, auth_adapter):
        """When X-Xiaoban-Session-Id is provided, history comes from SessionDB not request body."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        db_history = [
            {"role": "user", "content": "stored message 1"},
            {"role": "assistant", "content": "stored reply 1"},
        ]
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = db_history
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Xiaoban-Session-Id": "existing-session", "Authorization": "Bearer sk-secret"},
                    # Request body has different history — should be ignored
                    json={
                        "model": "xiaoban-agent",
                        "messages": [
                            {"role": "user", "content": "old msg from client"},
                            {"role": "assistant", "content": "old reply from client"},
                            {"role": "user", "content": "new question"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # History must come from DB, not from the request body
            assert call_kwargs["conversation_history"] == db_history
            assert call_kwargs["user_message"] == "new question"

    @pytest.mark.asyncio
    async def test_db_failure_falls_back_to_empty_history(self, auth_adapter):
        """If SessionDB raises, history falls back to empty and request still succeeds."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        # Simulate DB failure: _session_db is None and SessionDB() constructor raises
        auth_adapter._session_db = None
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run, \
                 patch("xiaoban_state.SessionDB", side_effect=Exception("DB unavailable")):
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Xiaoban-Session-Id": "some-session", "Authorization": "Bearer sk-secret"},
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "Hi"}]},
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["conversation_history"] == []
            assert call_kwargs["session_id"] == "some-session"


# ---------------------------------------------------------------------------
# X-Xiaoban-Session-Key header (long-term memory scoping)
# ---------------------------------------------------------------------------


class TestSessionKeyHeader:
    """The session key is a stable per-channel identifier that scopes
    long-term memory (e.g. Honcho) independently of the transcript-scoped
    session_id.  A third-party Web UI passes one stable key per assistant
    channel and rotates session_id on /new, matching the native
    gateway's session_key / session_id split.
    """

    @pytest.mark.asyncio
    async def test_session_key_passed_to_agent_and_echoed(self, auth_adapter):
        """X-Xiaoban-Session-Key reaches _run_agent as gateway_session_key and is echoed back."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Xiaoban-Session-Key": "webui:user-42",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Xiaoban-Session-Key") == "webui:user-42"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["gateway_session_key"] == "webui:user-42"

    @pytest.mark.asyncio
    async def test_session_key_independent_of_session_id(self, auth_adapter):
        """Both headers coexist: key scopes memory, id scopes transcript."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = []
        auth_adapter._session_db = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Xiaoban-Session-Key": "channel-abc",
                        "X-Xiaoban-Session-Id": "transcript-xyz",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Xiaoban-Session-Key") == "channel-abc"
            assert resp.headers.get("X-Xiaoban-Session-Id") == "transcript-xyz"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["gateway_session_key"] == "channel-abc"
            assert call_kwargs["session_id"] == "transcript-xyz"

    @pytest.mark.asyncio
    async def test_session_key_absent_yields_none(self, auth_adapter):
        """Omitting the header passes gateway_session_key=None and doesn't echo."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-secret"},
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            assert "X-Xiaoban-Session-Key" not in resp.headers
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["gateway_session_key"] is None

    @pytest.mark.asyncio
    async def test_session_key_rejected_without_api_key(self, adapter):
        """Without API_SERVER_KEY, accepting a caller-supplied memory scope is unsafe — reject with 403."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"X-Xiaoban-Session-Key": "whatever"},
                json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 403

    @pytest.mark.asyncio
    async def test_session_key_rejects_control_chars(self, auth_adapter):
        """Header injection via \\r\\n must be rejected by the server-side validator.

        Note: aiohttp client refuses to SEND a header containing CR/LF
        (that check fires before the request leaves the client), so we
        can't reach this code path through TestClient.  Test the helper
        directly instead with a raw request that bypasses client-side
        validation.
        """
        mock_request = MagicMock()
        mock_request.headers = {"X-Xiaoban-Session-Key": "bad\rvalue"}
        key, err = auth_adapter._parse_session_key_header(mock_request)
        assert key is None
        assert err is not None
        assert err.status == 400

    @pytest.mark.asyncio
    async def test_session_key_rejects_oversized(self, auth_adapter):
        """Session keys longer than the cap are rejected."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"X-Xiaoban-Session-Key": "x" * 1000, "Authorization": "Bearer sk-secret"},
                json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_session_key_threads_into_create_agent(self, auth_adapter):
        """End-to-end: verify AIAgent(gateway_session_key=...) receives the key via _create_agent."""
        captured_kwargs = {}

        def _fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok", "messages": []}
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent", side_effect=_fake_create_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Xiaoban-Session-Key": "agent:main:webui:dm:user-7",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "xiaoban-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            # _create_agent must be called with gateway_session_key threaded through
            assert captured_kwargs.get("gateway_session_key") == "agent:main:webui:dm:user-7"

    @pytest.mark.asyncio
    async def test_responses_endpoint_accepts_session_key(self, auth_adapter):
        """Responses API honors the same X-Xiaoban-Session-Key contract."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    headers={
                        "X-Xiaoban-Session-Key": "webui:chan-1",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "xiaoban-agent", "input": "hello", "store": False},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Xiaoban-Session-Key") == "webui:chan-1"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["gateway_session_key"] == "webui:chan-1"

    @pytest.mark.asyncio
    async def test_capabilities_advertises_session_key_header(self, adapter):
        """GET /v1/capabilities should advertise the new header so clients can feature-detect."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["session_key_header"] == "X-Xiaoban-Session-Key"
