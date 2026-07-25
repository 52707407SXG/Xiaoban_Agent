from gateway.platforms.api_server import (
    _build_mystand_runtime_integrity_reminder,
    _guard_evidence_backed_response,
    _mystand_index_has_candidates,
    _resolve_mystand_initial_tool_choice,
    _sanitize_user_visible_text,
    _select_mystand_index_candidate,
    _should_buffer_stream_deltas,
    _trusted_mystand_module_id,
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


def test_mystand_stream_buffers_all_text_until_integrity_guard_runs():
    assert _should_buffer_stream_deltas(
        "屏山县属于哪里？",
        mystand_request=True,
    )
    assert not _should_buffer_stream_deltas(
        "屏山县属于哪里？",
        mystand_request=False,
    )


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


def test_evidence_guard_does_not_treat_knowledge_graph_as_image_request():
    for question in (
        "知识图谱是什么？",
        "你知道咱们站的知识图谱吗？",
        "如果给你权限，你能帮我组知识图谱吗？",
        "知识图谱怎么用",
        "看看知识图谱",
        "图谱里有哪些功能",
        "图形中心怎么用",
        "图表怎么看",
    ):
        guarded = _guard_evidence_backed_response(
            "My Stand 已有知识图谱功能，我先按站内说明讲清楚。",
            user_message=question,
            conversation_history=[],
            result={"messages": []},
        )
        assert guarded == "My Stand 已有知识图谱功能，我先按站内说明讲清楚。"


def test_evidence_guard_still_treats_explicit_image_requests_as_visual():
    for question in (
        "请帮我识图",
        "帮我看图",
        "知识图谱旁边这张图是什么",
    ):
        guarded = _guard_evidence_backed_response(
            "这是一张楼盘平面图。",
            user_message=question,
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


def _write_turn(operation, result_payload, *, user_message="确认写入"):
    return {
        "_mystand_request": True,
        "messages": [
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "function": {
                            "name": "mystand_authorization_write",
                            "arguments": (
                                '{"operation":"'
                                + operation
                                + '","idempotency_key":"idem-1"}'
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_write",
                "content": result_payload,
            },
        ],
    }


def test_integrity_guard_blocks_exact_incident_false_write_success_without_tool_call():
    guarded = _guard_evidence_backed_response(
        "刚哥，四项全部写入成功，已经落进特征卡，刷新一下就能看到。",
        user_message="确认",
        conversation_history=[
            {
                "role": "assistant",
                "content": "四项预览全都通了，等你确认。",
            }
        ],
        result={
            "_mystand_request": True,
            "messages": [
                {"role": "user", "content": "确认"},
                {
                    "role": "assistant",
                    "content": "刚哥，四项全部写入成功，已经落进特征卡。",
                },
            ],
        },
    )

    assert guarded == (
        "这次没有实际执行可验证的写入，所以不能说已经写入；"
        "当前资料没有确认发生变化。"
    )


def test_integrity_guard_does_not_reuse_verified_receipt_from_an_older_turn():
    guarded = _guard_evidence_backed_response(
        "已经写入成功。",
        user_message="确认",
        conversation_history=[],
        result={
            "_mystand_request": True,
            "messages": [
                {"role": "user", "content": "确认写入上一项"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "old_call",
                            "function": {
                                "name": "mystand_authorization_write",
                                "arguments": '{"operation":"commit_write"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "old_call",
                    "content": '{"ok":true,"verified":true}',
                },
                {"role": "assistant", "content": "上一项写入成功。"},
                {"role": "user", "content": "确认"},
                {"role": "assistant", "content": "已经写入成功。"},
            ],
        },
    )

    assert guarded == (
        "这次没有实际执行可验证的写入，所以不能说已经写入；"
        "当前资料没有确认发生变化。"
    )


def test_integrity_guard_blocks_preview_being_reported_as_committed():
    guarded = _guard_evidence_backed_response(
        "已经正式写入，刷新就能看到。",
        user_message="把这项写进去",
        conversation_history=[],
        result=_write_turn(
            "preview_write",
            '{"ok":true,"previewToken":"preview-1","status":200}',
            user_message="把这项写进去",
        ),
    )

    assert guarded == (
        "这次只完成了写入预览，还没有正式写入；"
        "当前资料没有发生已确认的变化。"
    )


def test_integrity_guard_blocks_failed_commit_being_reported_as_success():
    guarded = _guard_evidence_backed_response(
        "四项全部写入成功。",
        user_message="确认",
        conversation_history=[],
        result=_write_turn(
            "commit_write",
            '{"ok":false,"status":409,"error":"version conflict"}',
            user_message="确认",
        ),
    )

    assert guarded == (
        "这次写入没有成功，当前资料没有确认发生变化。"
        "我不能把失败说成完成。"
    )


def test_integrity_guard_allows_success_only_with_current_turn_verified_receipt():
    guarded = _guard_evidence_backed_response(
        "四项全部写入成功。",
        user_message="确认",
        conversation_history=[],
        result=_write_turn(
            "commit_write",
            (
                '{"ok":true,"status":200,"receiptVersion":'
                '"authorization-write-receipt-v2","verified":true}'
            ),
            user_message="确认",
        ),
    )

    assert guarded == "四项全部写入成功。"


def test_integrity_guard_rejects_unversioned_verified_flag():
    guarded = _guard_evidence_backed_response(
        "四项全部写入成功。",
        user_message="确认",
        conversation_history=[],
        result=_write_turn(
            "commit_write",
            '{"ok":true,"status":200,"verified":true}',
            user_message="确认",
        ),
    )

    assert guarded == (
        "这次写入没有成功，当前资料没有确认发生变化。"
        "我不能把失败说成完成。"
    )


def test_integrity_guard_keeps_honest_failure_reply():
    guarded = _guard_evidence_backed_response(
        "这次没有写入成功，我不能说已经完成。",
        user_message="确认",
        conversation_history=[],
        result=_write_turn(
            "commit_write",
            '{"ok":false,"status":404,"error":"not found"}',
            user_message="确认",
        ),
    )

    assert guarded == "这次没有写入成功，我不能说已经完成。"


def test_runtime_integrity_reminder_triggers_on_confirmation_after_write_context():
    reminder = _build_mystand_runtime_integrity_reminder(
        "确认",
        [
            {
                "role": "assistant",
                "content": "这是写入预览，确认后再正式提交。",
            },
            {
                "role": "tool",
                "name": "mystand_authorization_write",
                "content": '{"ok":false,"error":"write_resource_not_available"}',
            },
        ],
    )

    assert "本轮诚信强制提醒" in reminder
    assert "历史聊天里的成功自述不是证据" in reminder
    assert "verified=true" in reminder


def test_runtime_integrity_reminder_triggers_under_emotional_pressure():
    reminder = _build_mystand_runtime_integrity_reminder(
        "别再糊弄我，赶紧把它写进去",
        [],
    )

    assert "情绪或催促越强" in reminder
    assert "不得为了安抚用户编造进展" in reminder


def test_runtime_integrity_reminder_stays_off_for_unrelated_plain_question():
    assert (
        _build_mystand_runtime_integrity_reminder(
            "屏山县属于哪里？",
            [],
        )
        == ""
    )


def test_read_reply_is_not_replaced_by_write_guard_words():
    guarded = _guard_evidence_backed_response(
        "这条是已删除的历史记录，不代表本轮执行了删除。",
        user_message="这个档案以前删除了吗？",
        conversation_history=[
            {
                "role": "assistant",
                "content": "之前讨论过写入，但这不是当前任务。",
            },
        ],
        result={"_mystand_request": True, "messages": []},
    )

    assert guarded == "这条是已删除的历史记录，不代表本轮执行了删除。"


def test_delete_history_record_remains_a_write_request():
    reminder = _build_mystand_runtime_integrity_reminder(
        "删除这条历史记录",
        [],
    )

    assert "本轮诚信强制提醒" in reminder


def test_exact_auth_forces_authorization_evidence_tool():
    assert (
        _resolve_mystand_initial_tool_choice(
            "读取 AUTH-74D760C1-3EA2F5BA-23ACBEF4-BC1819E6",
            "",
        )
        == "mystand_authorization"
    )
    assert (
        _resolve_mystand_initial_tool_choice(
            "读取 AUTH-ABC12345",
            "",
        )
        == "mystand_authorization"
    )


def test_resource_intent_forces_index_before_content_read():
    assert (
        _resolve_mystand_initial_tool_choice(
            "看看游雪梅今年的结算情况",
            "【本轮可信意图与索引证据】\n意图=resource-read；索引=resource；状态=available。",
        )
        == "mystand_resource_index"
    )


def test_resource_index_candidate_detection_requires_nonempty_ok_page():
    assert _mystand_index_has_candidates(
        '{"ok":true,"items":[{"resourceUid":"resource-1"}]}'
    )
    assert not _mystand_index_has_candidates('{"ok":true,"items":[]}')
    assert not _mystand_index_has_candidates(
        '{"ok":false,"items":[{"resourceUid":"resource-1"}]}'
    )


def test_required_mystand_evidence_failure_blocks_model_story():
    guarded = _guard_evidence_backed_response(
        "档案存在，但小伴可读开关没有打开。",
        user_message="读取 AUTH-74D760C1-3EA2F5BA-23ACBEF4-BC1819E6",
        conversation_history=[],
        result={
            "_mystand_request": True,
            "_mystand_required_evidence_tool": "mystand_authorization",
            "messages": [],
        },
    )

    assert guarded == (
        "这轮没有取得可验证的 My Stand 站内资料结果，所以我不能判断资料内容、"
        "权限状态或是否完成。"
    )


def test_structured_ok_tool_result_wins_over_incidental_failure_digits():
    guarded = _guard_evidence_backed_response(
        "已按本轮结果读取。",
        user_message="读取 AUTH-ABC12345",
        conversation_history=[],
        result={
            "_mystand_request": True,
            "_mystand_required_evidence_tool": "mystand_authorization",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "read-1",
                            "function": {
                                "name": "mystand_authorization",
                                "arguments": '{"operation":"resolve"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "read-1",
                    "content": (
                        '{"ok":true,"key":{"canWrite":false},'
                        '"content":"地址3401号"}'
                    ),
                },
            ],
        },
    )

    assert guarded == "已按本轮结果读取。"


def test_resource_index_alone_cannot_prove_business_content():
    guarded = _guard_evidence_backed_response(
        "查到了，结算业绩是 32105.68 元。",
        user_message="按名称查游雪梅2026年财务档案",
        conversation_history=[],
        result={
            "_mystand_request": True,
            "_mystand_required_evidence_groups": [
                ["mystand_resource_index"],
                ["mystand_authorization", "mystand_query"],
            ],
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "index-1",
                            "function": {
                                "name": "mystand_resource_index",
                                "arguments": '{"operation":"list_resources"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "index-1",
                    "content": '{"ok":true,"items":[]}',
                },
            ],
        },
    )

    assert guarded == (
        "这轮没有取得可验证的 My Stand 站内资料结果，所以我不能判断资料内容、"
        "权限状态或是否完成。"
    )


def test_named_resource_content_without_index_is_blocked():
    guarded = _guard_evidence_backed_response(
        "查到了，结算业绩是 32105.68 元。",
        user_message="按名称查游雪梅2026年财务档案",
        conversation_history=[],
        result={
            "_mystand_request": True,
            "_mystand_required_evidence_groups": [
                ["mystand_resource_index"],
                ["mystand_authorization", "mystand_query"],
            ],
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "read-1",
                            "function": {
                                "name": "mystand_authorization",
                                "arguments": '{"operation":"resolve"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "read-1",
                    "content": '{"ok":true,"content":"32105.68"}',
                },
            ],
        },
    )

    assert guarded == (
        "这轮没有取得可验证的 My Stand 站内资料结果，所以我不能判断资料内容、"
        "权限状态或是否完成。"
    )


def test_executed_authorization_not_found_returns_precise_verified_failure():
    guarded = _guard_evidence_backed_response(
        "我猜权限可能没开。",
        user_message="AUTH-INVALID-123456 看看这个",
        conversation_history=[],
        result={
            "_mystand_request": True,
            "_mystand_required_evidence_groups": [["mystand_authorization"]],
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "read-failed-1",
                        "function": {
                            "name": "mystand_authorization",
                            "arguments": '{"operation":"resolve"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "read-failed-1",
                    "content": (
                        '{"ok":false,"status":404,'
                        '"code":"authorization_not_found"}'
                    ),
                },
            ],
        },
    )

    assert guarded == "没有找到这份资料，或者这个站内 ID 已失效。"


def test_trusted_index_context_selects_unique_named_resource():
    prompt = (
        "untrusted moduleId text\n"
        "【本轮可信意图与索引证据】\n"
        '{"intent":{"moduleId":"finance-ledger"}}'
    )
    result = (
        '{"ok":true,"items":['
        '{"resourceUid":"one","safeLabel":"覃滔 2026年个人业务档案","canRead":true},'
        '{"resourceUid":"two","safeLabel":"游雪梅 2026年个人业务档案","canRead":true}'
        "]}"
    )

    assert _trusted_mystand_module_id(prompt) == "finance-ledger"
    selected = _select_mystand_index_candidate(
        result,
        "给我查一下覃滔今年的总业绩",
    )
    assert selected["resourceUid"] == "one"
