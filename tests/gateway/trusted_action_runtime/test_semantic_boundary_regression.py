"""K4 真实语义回归：普通模型答卷不被代码改写。

这些用例覆盖 2026-07-26 正式站事故：My Stand Web 的普通对话被统一
替换成“没有真正查到站内资料”。边界必须来自服务器可信意图与真实
Action 生命周期，不能来自回答里的关键词、数字或日期。
"""

from __future__ import annotations

import pytest

from gateway.platforms.api_server import (
    _finalize_mystand_egress_result,
)
from xiaoban.trusted_runtime.turns import begin_turn
from xiaoban.trusted_runtime.types import TrustedIdentity


ACCOUNT = "semantic-user"
REQUEST_ID = "xbd_semantic_request"
MESSAGE_ID = "semantic-message"


def _chat_result(
    user_message: str,
    answer: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> dict:
    turn = begin_turn(
        channel="web",
        user_message=user_message,
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
        "final_response": answer,
        "completed": True,
        "failed": False,
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
def test_plain_mystand_chat_keeps_the_model_answer(
    user_message: str,
    answer: str,
) -> None:
    visible = _finalize_mystand_egress_result(
        _chat_result(user_message, answer),
        user_message=user_message,
        conversation_history=[],
    )

    assert visible == answer


def test_old_business_history_does_not_turn_new_chat_into_work() -> None:
    history = [
        {"role": "user", "content": "帮我查一下滨江一号的房源资料"},
        {"role": "assistant", "content": "上轮没有查到可验证结果。"},
    ]
    user_message = "先不查了，谢谢。再介绍一下你自己。"
    answer = "不客气。我是站小伴，负责协助你使用 My Stand，也可以正常交流。"

    visible = _finalize_mystand_egress_result(
        _chat_result(user_message, answer, history=history),
        user_message=user_message,
        conversation_history=history,
    )

    assert visible == answer
