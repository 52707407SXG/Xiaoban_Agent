from gateway.platforms.api_server import (
    _guard_evidence_backed_response,
    _sanitize_user_visible_text,
)


def test_sanitize_user_visible_text_redacts_local_paths_and_file_urls():
    text = (
        "Do not cite file:///root/secret.md or /opt/mystand-api/mystand.sqlite; "
        "use https://example.com/source instead."
    )

    sanitized = _sanitize_user_visible_text(text)

    assert "file://" not in sanitized
    assert "/root/" not in sanitized
    assert "/opt/" not in sanitized
    assert "本地文件链接" in sanitized
    assert "本地路径" in sanitized
    assert "https://example.com/source" in sanitized


def test_evidence_guard_blocks_wechat_summary_without_tool_result():
    guarded = _guard_evidence_backed_response(
        "这篇文章主要讲 Hermes 的 MoA 混合 Agent 模式。",
        user_message="总结这篇公众号：https://mp.weixin.qq.com/s/pbHlRqN_w1RLXnC_IgC8Ag",
        conversation_history=[],
        result={"messages": []},
    )

    assert guarded == "我还没有成功读取到这个链接的正文，所以不能总结或分析里面的内容。"


def test_evidence_guard_allows_wechat_summary_with_parser_result():
    result = {
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
        ]
    }

    guarded = _guard_evidence_backed_response(
        "这篇文章主要讲 Hermes 的 MoA 混合 Agent 模式。",
        user_message="总结这篇公众号：https://mp.weixin.qq.com/s/pbHlRqN_w1RLXnC_IgC8Ag",
        conversation_history=[],
        result=result,
    )

    assert guarded == "这篇文章主要讲 Hermes 的 MoA 混合 Agent 模式。"


def test_evidence_guard_blocks_unsupported_mentioned_source_claim():
    result = {
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
                "content": '{"success": true, "title": "Hermes MoA", "text": "本文介绍 Hermes 的 MoA 混合 Agent 模式和多个 AI 模型协作。"}',
            },
        ]
    }

    guarded = _guard_evidence_backed_response(
        "文章认为 2027 年房价会明显上涨。",
        user_message="你看看这篇文章，分析里面提到的2027年房价走势如何：https://mp.weixin.qq.com/s/pbHlRqN_w1RLXnC_IgC8Ag",
        conversation_history=[],
        result=result,
    )

    assert guarded == "我已读取到这个链接，但正文里没有找到你说的这项内容，所以不能按文章内容展开分析。"


def test_evidence_guard_allows_source_absence_answer():
    result = {
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
                "content": '{"success": true, "title": "Hermes MoA", "text": "本文介绍 Hermes 的 MoA 混合 Agent 模式和多个 AI 模型协作。"}',
            },
        ]
    }

    guarded = _guard_evidence_backed_response(
        "这篇文章没有提到 2027 年房价走势，它讲的是 Hermes MoA。",
        user_message="你看看这篇文章，分析里面提到的2027年房价走势如何：https://mp.weixin.qq.com/s/pbHlRqN_w1RLXnC_IgC8Ag",
        conversation_history=[],
        result=result,
    )

    assert guarded == "这篇文章没有提到 2027 年房价走势，它讲的是 Hermes MoA。"


def test_evidence_guard_blocks_image_description_without_image_or_vision():
    guarded = _guard_evidence_backed_response(
        "图里是一张楼盘海报。",
        user_message="这张图片里是什么？",
        conversation_history=[],
        result={"messages": []},
    )

    assert guarded == "我现在没有成功看到这张图片的内容，所以不能描述画面或识别图片细节。"


def test_evidence_guard_allows_image_description_with_image_input():
    user_message = [
        {"type": "text", "text": "这张图片里是什么？"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    guarded = _guard_evidence_backed_response(
        "图里是一张楼盘海报。",
        user_message=user_message,
        conversation_history=[],
        result={"messages": []},
    )

    assert guarded == "图里是一张楼盘海报。"
