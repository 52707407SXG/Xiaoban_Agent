"""真实语义回归：CompletionGuard 只审业务执行，不审普通自然语言。

这些用例覆盖 2026-07-26 正式站事故：My Stand Web 的普通对话被统一
替换成“没有真正查到站内资料”。边界必须来自服务器可信意图与真实
Action 生命周期，不能来自回答里的关键词、数字或日期。
"""

from __future__ import annotations

import pytest

from gateway.platforms.api_server import (
    _guard_evidence_backed_response,
    _should_buffer_stream_deltas,
)
from xiaoban.trusted_runtime.turns import begin_turn
from xiaoban.trusted_runtime.types import TrustedIdentity


ACCOUNT = "semantic-user"
REQUEST_ID = "xbd_semantic_request"
MESSAGE_ID = "semantic-message"


def _chat_result(
    user_message: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> dict:
    turn = begin_turn(
        channel="web",
        user_message=user_message,
        conversation_history=history or [],
        identity=TrustedIdentity(
            account_id=ACCOUNT,
            data_scope="mystand",
            source="server_session",
        ),
        request_id=REQUEST_ID,
        message_id=MESSAGE_ID,
    )
    return {
        "_mystand_request": True,
        "_mystand_user_id": ACCOUNT,
        "_mystand_request_id": REQUEST_ID,
        "_mystand_message_id": MESSAGE_ID,
        "_trusted_turn": turn,
        "messages": [
            *(history or []),
            {"role": "user", "content": user_message},
        ],
    }


@pytest.mark.parametrize(
    ("user_message", "answer"),
    [
        (
            "你好，简单介绍一下你自己。",
            "你好，我是站小伴，是 My Stand 的原生 Agent，可以帮你查询和核验站内资料，也能陪你正常聊天。",
        ),
        (
            "现在是哪一年？",
            "现在是 2026 年。",
        ),
        (
            "你能全天帮我吗？",
            "可以，我可以 24 小时响应；涉及真实资料时会按你的权限执行。",
        ),
        (
            "房源笔记怎么用？",
            "房源笔记用于整理楼盘和房源信息；打开对应页面后即可新建、编辑和检索。",
        ),
    ],
)
def test_plain_mystand_chat_is_not_rewritten_by_business_guard(
    user_message: str,
    answer: str,
) -> None:
    guarded = _guard_evidence_backed_response(
        answer,
        user_message=user_message,
        conversation_history=[],
        result=_chat_result(user_message),
    )

    assert guarded == answer


def test_old_business_history_does_not_turn_new_chat_into_work() -> None:
    history = [
        {"role": "user", "content": "帮我查一下滨江一号的房源资料"},
        {"role": "assistant", "content": "上轮没有查到可验证结果。"},
    ]
    user_message = "先不查了，谢谢。再介绍一下你自己。"
    answer = "不客气。我是站小伴，负责协助你使用 My Stand，也可以正常交流。"

    guarded = _guard_evidence_backed_response(
        answer,
        user_message=user_message,
        conversation_history=history,
        result=_chat_result(user_message, history=history),
    )

    assert guarded == answer


def test_mystand_chat_streams_without_waiting_for_completion_guard() -> None:
    prompt = (
        "【本轮可信意图与索引证据】\n"
        "意图=offsite-chat；索引=none；写闸=不需要；状态=not-needed。"
    )

    assert not _should_buffer_stream_deltas(
        "你好，介绍一下你自己。",
        mystand_request=True,
        system_prompt=prompt,
    )


def test_only_structured_resource_requirement_buffers_until_verified() -> None:
    prompt = (
        "【本轮可信意图与索引证据】\n"
        "意图=resource-read；索引=resource；写闸=不需要；状态=available。"
    )

    assert not _should_buffer_stream_deltas(
        "查一下这份资料",
        mystand_request=True,
        system_prompt=prompt,
    )
    assert _should_buffer_stream_deltas(
        "查一下这份资料",
        mystand_request=True,
        system_prompt=prompt,
        fact_requirement={"schema": "mystand.fact-requirement.v1"},
    )
