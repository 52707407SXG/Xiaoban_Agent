"""事故结构脱敏 fixture：无证据业务陈述重放素材。

人物、楼盘、金额、状态全部为虚构脱敏值（周某/滨江一号/5600 元等），
不复制任何真实客户、房源、电话、金额或私密正文。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 出口唯一允许的失败语义文案（与现网已验证的失败口径一致）
DENIED_MESSAGE = "当前没有权限让小伴读取这份资料。"
NOT_FOUND_MESSAGE = "没有找到这份资料，或者这个站内 ID 已失效。"
AMBIGUOUS_MESSAGE = "这份资料目前无法唯一定位，请补充更完整的资料名称。"
ERROR_MESSAGE = "站内资料读取暂时没有接稳，请稍后再试。"

FABRICATED_ANSWER = "查到了，滨江一号3栋802的业主是周某，月供 5600 元，目前状态正常。"
FABRICATED_TOKENS = ("周某", "5600")

# WORK 强制最小索引前置：业务读取前必须先有服务端 IndexReceipt。
INDEX_CALL = (
    "call_idx",
    "mystand_resource_index",
    {"operation": "list_resources", "module_id": "finance-ledger"},
    {
        "ok": True,
        "items": [
            {
                "resourceUid": "res-demo-1",
                "safeLabel": "游某 2026年个人业务档案",
                "canRead": True,
            }
        ],
    },
)


def tool_turn(
    user_message: str,
    calls: Sequence[Tuple[str, str, Dict[str, Any], Optional[Any]]] = (),
    *,
    user_id: str = "user-a",
    prior_messages: Optional[List[Dict[str, Any]]] = None,
    evidence_required: bool = True,
) -> Dict[str, Any]:
    """Build an OpenAI-style result dict for one My Stand turn.

    ``calls`` items are ``(call_id, tool_name, arguments, receipt)``; a
    ``receipt`` of ``None`` simulates a call whose result never came back.
    """
    messages: List[Dict[str, Any]] = list(prior_messages or [])
    messages.append({"role": "user", "content": user_message})
    for call_id, name, arguments, receipt in calls:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        if receipt is not None:
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call_id,
                    "content": receipt
                    if isinstance(receipt, str)
                    else json.dumps(receipt, ensure_ascii=False),
                }
            )
    return {
        "_mystand_request": True,
        "_mystand_user_id": user_id,
        "_mystand_evidence_required": evidence_required,
        "messages": messages,
    }


# --- 事故场景 1：零工具调用却一本正经给出人名、金额、状态 ---
SCENARIO_ZERO_CALL_FABRICATION = {
    "user_message": "帮我看看滨江一号3栋802的业主和月供",
    "answer": FABRICATED_ANSWER,
    "result": tool_turn("帮我看看滨江一号3栋802的业主和月供"),
}

# --- 事故场景 2：工具全部失败后仍编造业务结果 ---
SCENARIO_ALL_TOOLS_FAILED = {
    "user_message": "帮我看看滨江一号3栋802的业主和月供",
    "answer": FABRICATED_ANSWER,
    "result": tool_turn(
        "帮我看看滨江一号3栋802的业主和月供",
        [
            INDEX_CALL,
            (
                "call_q1",
                "mystand_authorization",
                {"operation": "resolve", "resource_uid": "res-demo-1"},
                {"ok": False, "status": 500, "error": "internal_bridge_error"},
            ),
        ],
    ),
}

# --- 事故场景 3：索引成功、业务动作失败，把索引结果说成业务结果 ---
SCENARIO_INDEX_OK_BUSINESS_FAILED = {
    "user_message": "查一下周某今年的结算业绩",
    "answer": "查到了，周某今年结算业绩是 32105.68 元。",
    "forged_tokens": ("32105.68",),
    "result": tool_turn(
        "查一下周某今年的结算业绩",
        [
            (
                "call_idx",
                "mystand_resource_index",
                {"operation": "list_resources", "module_id": "finance-ledger"},
                {
                    "ok": True,
                    "items": [
                        {
                            "resourceUid": "res-demo-1",
                            "safeLabel": "周某 2026年个人业务档案",
                            "canRead": True,
                        }
                    ],
                },
            ),
            (
                "call_read",
                "mystand_authorization",
                {"operation": "resolve", "resource_uid": "res-demo-1"},
                {"ok": False, "status": 500, "error": "internal_bridge_error"},
            ),
        ],
    ),
}

# --- 事故场景 4：同一问题连续追问后反复补充新假事实 ---
SCENARIO_FOLLOWUP_NEW_FABRICATION = {
    "user_message": "那他家第二套呢？",
    "answer": "第二套在滨江一号5栋，月供 8200 元，也是正常还款。",
    "forged_tokens": ("8200", "5栋"),
    "conversation_history": [
        {"role": "user", "content": "帮我看看滨江一号3栋802的业主和月供"},
        {"role": "assistant", "content": "这套的月供情况我还在查。"},
    ],
    "result": tool_turn("那他家第二套呢？"),
}

# --- 事故场景 5：用户质疑后声称"我重新核验了"，但没有新的真实验证 ---
SCENARIO_FAKE_REVERIFICATION = {
    "user_message": "你确定吗？",
    "answer": "我重新核验过了，业主确实是周某，没错。",
    "forged_tokens": ("周某",),
    "conversation_history": [
        {"role": "user", "content": "滨江一号3栋802业主是谁"},
        {"role": "assistant", "content": "之前说是周某。"},
    ],
    "result": tool_turn("你确定吗？"),
}

# --- 事故场景 6：自然业务问法没有触发旧正则或 required evidence group ---
SCENARIO_NATURAL_PHRASING_BYPASS = {
    "user_message": "小张这个月提成多少",
    "answer": "小张这个月提成是 12800 元。",
    "forged_tokens": ("12800",),
    "result": tool_turn("小张这个月提成多少"),
}

# --- 事故场景 7：旧历史事实污染当前回合 ---
_old_turn_messages = [
    {"role": "user", "content": "查一下游某今年业绩"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_old",
                "type": "function",
                "function": {
                    "name": "mystand_query",
                    "arguments": '{"operation":"read","entities":["游某"]}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "name": "mystand_query",
        "tool_call_id": "call_old",
        "content": '{"ok":true,"content":"业绩 99999 元"}',
    },
    {"role": "assistant", "content": "游某今年业绩是 99999 元。"},
]
SCENARIO_STALE_HISTORY_POLLUTION = {
    "user_message": "那周某今年业绩呢？",
    "answer": "周某今年业绩是 99999 元。",
    "forged_tokens": ("99999",),
    "result": tool_turn(
        "那周某今年业绩呢？",
        prior_messages=_old_turn_messages,
    ),
}

# --- 事故场景 8：ActionResult 各失败状态不得转成业务成功 ---
SCENARIO_RESULT_STATUSES = {
    "user_message": "帮我看看滨江一号3栋802的业主和月供",
    "answer": FABRICATED_ANSWER,
    "cases": {
        "empty": {
            "receipt": {"ok": True, "content": ""},
            "blocked": True,
        },
        "error": {
            "receipt": {"ok": False, "status": 500, "error": "internal"},
            "message": ERROR_MESSAGE,
        },
        "denied": {
            "receipt": {"ok": False, "status": 403, "code": "read_not_authorized"},
            "message": DENIED_MESSAGE,
        },
        "ambiguous": {
            "receipt": {"ok": False, "status": 409, "code": "ambiguous_resource"},
            "message": AMBIGUOUS_MESSAGE,
        },
        "not_found": {
            "receipt": {"ok": False, "status": 404, "code": "resource_not_found"},
            "message": NOT_FOUND_MESSAGE,
        },
    },
}


def result_status_turn(receipt: Any) -> Dict[str, Any]:
    return tool_turn(
        SCENARIO_RESULT_STATUSES["user_message"],
        [
            INDEX_CALL,
            (
                "call_s1",
                "mystand_authorization",
                {"operation": "resolve", "resource_uid": "res-demo-1"},
                receipt,
            ),
        ],
    )


# --- 事故场景 9a：伪造 callId 的回执不得成为证据 ---
SCENARIO_FORGED_CALL_ID = {
    "user_message": "帮我看看滨江一号3栋802的业主和月供",
    "answer": FABRICATED_ANSWER,
    "result": {
        "_mystand_request": True,
        "_mystand_user_id": "user-a",
        "_mystand_evidence_required": True,
        "messages": [
            {"role": "user", "content": "帮我看看滨江一号3栋802的业主和月供"},
            {
                "role": "tool",
                "name": "mystand_query",
                "tool_call_id": "call_ghost",
                "content": '{"ok":true,"content":"业主是周某，月供 5600 元"}',
            },
        ],
    },
}

# --- 事故场景 9b：跨回合 callId 的证据不得进入本轮 ---
SCENARIO_CROSS_TURN_CALL_ID = {
    "user_message": "那周某的呢？",
    "answer": "周某的月供也是 5600 元。",
    "forged_tokens": ("5600",),
    "result": tool_turn(
        "那周某的呢？",
        prior_messages=[
            {"role": "user", "content": "帮我看看滨江一号3栋802的月供"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_prev",
                        "type": "function",
                        "function": {
                            "name": "mystand_query",
                            "arguments": '{"operation":"read"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "mystand_query",
                "tool_call_id": "call_prev",
                "content": '{"ok":true,"content":"月供 5600 元"}',
            },
            {"role": "assistant", "content": "3栋802的月供是 5600 元。"},
        ],
    ),
}

# --- 事故场景 9c：跨账号 evidence 不得进入本轮 ---
SCENARIO_CROSS_ACCOUNT_EVIDENCE = {
    "user_message": "帮我看看滨江一号3栋802的业主和月供",
    "answer": FABRICATED_ANSWER,
    "result": tool_turn(
        "帮我看看滨江一号3栋802的业主和月供",
        [
            INDEX_CALL,
            (
                "call_x1",
                "mystand_authorization",
                {"operation": "resolve", "resource_uid": "res-demo-1"},
                {
                    "ok": True,
                    "accountId": "user-b",
                    "content": "业主是周某，月供 5600 元",
                },
            ),
        ],
        user_id="user-a",
    ),
}

# --- 事故场景 10：现有正常闲聊不得被误伤成生硬工作流 ---
SCENARIO_PLAIN_CHAT = {
    "user_message": "今天有点累，陪我聊两句",
    "answer": "辛苦了，先喝口水歇一下，我陪你聊会儿。",
    "result": tool_turn(
        "今天有点累，陪我聊两句",
        evidence_required=False,
    ),
}

# --- 正向控制：本轮真实证据支持的业务回答必须放行 ---
SCENARIO_EVIDENCE_BACKED_ANSWER = {
    "user_message": "查一下游某今年的结算业绩",
    "answer": "游某今年结算业绩是 32105.68 元。",
    "result": tool_turn(
        "查一下游某今年的结算业绩",
        [
            INDEX_CALL,
            (
                "call_ok",
                "mystand_authorization",
                {"operation": "resolve", "resource_uid": "res-demo-1"},
                {"ok": True, "content": "游某今年结算业绩是 32105.68 元。"},
            ),
        ],
    ),
}

# --- 反向控制：公开回答以本轮证据投影为准，模型口径不一致不得出站 ---
SCENARIO_FACT_MISMATCH = {
    "user_message": "查一下游某今年的结算业绩",
    "answer": "游某今年结算业绩是 32105.68 元。",
    "forged_tokens": ("32105.68",),
    "result": tool_turn(
        "查一下游某今年的结算业绩",
        [
            INDEX_CALL,
            (
                "call_ok",
                "mystand_authorization",
                {"operation": "resolve", "resource_uid": "res-demo-1"},
                {"ok": True, "content": "游某 2026 结算业绩 12345.00 元"},
            ),
        ],
    ),
}

# --- 正向控制：诚实承认失败/无权限的回答必须原样放行 ---
SCENARIO_HONEST_DENIAL_ADMISSION = {
    "user_message": "帮我看看滨江一号3栋802的业主和月供",
    "answer": "这份资料目前没有授权给小伴读取，你可以在资料页打开可读开关。",
    "result": result_status_turn({"ok": False, "status": 403, "code": "read_not_authorized"}),
}
