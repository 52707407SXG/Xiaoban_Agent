"""Hard privacy boundary for outbound web research tools."""

from __future__ import annotations

import json
import re
import unicodedata
from urllib.parse import unquote

from gateway.session_context import (
    get_session_env,
    get_session_user_message,
    mark_mystand_private_query_turn,
    mystand_private_query_turn_active,
    mystand_private_taint_persistence_failed,
)

_CURRENT_MYSTAND_PRIVATE_TOOL_NAMES = frozenset(
    {
        "mystand_query",
        "mystand_authorization_write",
    }
)
_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LANDLINE_RE = re.compile(r"(?<!\d)0\d{2,3}\d{7,8}(?!\d)")
_IDENTITY_CARD_RE = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"
)
_EXACT_BUILDING_ROOM_RE = re.compile(
    r"(?:\d{1,3}|[零〇一二两三四五六七八九十百]{1,5})\s*"
    r"(?:栋|幢|座|号楼)"
    r"[^，,。；;\n]{0,24}?"
    r"(?:"
    r"(?:\d{1,3}|[零〇一二两三四五六七八九十百]{1,5})\s*单元"
    r"[^，,。；;\n]{0,16}?"
    r")?"
    r"(?:\d{3,5}|[零〇一二两三四五六七八九十]{3,8})\s*(?:室|号|户)?"
)
_HYPHEN_ROOM_RE = re.compile(
    r"(?<!\d)\d{1,3}\s*[-－]\s*\d{1,3}\s*[-－]\s*\d{3,5}(?!\d)"
)
_EXPLICIT_ADDRESS_RE = re.compile(
    r"(?:详细地址|家庭住址|居住地址|门牌号|房号|室号|房间号)\s*[:：]?\s*\S{2,80}"
)
_PRIVATE_CONTENT_RE = re.compile(
    r"(?:"
    r"身份证|银行卡|开户行|征信|家庭成员|家庭情况|经济情况|"
    r"跟进记录|沟通记录|客户资料|业主资料|租客资料|"
    r"业主姓名|客户姓名|联系人电话|联系电话|手机号码|"
    r"家庭住址|居住地址|详细地址|隐私资料|私密资料|内部资料"
    r")"
)
_LABELED_NAME_RE = re.compile(
    r"(?:姓名|名字|业主|客户|联系人|户主|房东|租客)\s*"
    r"(?:为|是|叫|[:：])?\s*"
    r"(?:[\u4e00-\u9fff·]{2,12}|[A-Za-z][A-Za-z .'-]{1,60})"
)
_HONORIFIC_NAME_RE = re.compile(
    r"(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{2,4}(?:先生|女士)(?![\u4e00-\u9fff])"
)
_COMMON_CHINESE_SURNAMES = (
    "赵钱孙李周吴郑王冯陈蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜"
    "谢邹苏潘范彭鲁韦马方任袁柳鲍史唐薛雷贺倪汤罗郝安常傅齐康"
    "伍余顾孟黄萧尹姚邵汪毛宋熊纪舒项董梁杜阮蓝季贾江郭梅林钟"
    "徐高夏蔡田樊胡霍万卢莫石崔龚程陆翁段侯白廖龙叶黎牛温庄阎"
    "乔曾关游"
)
_BARE_CHINESE_NAME_RE = re.compile(
    rf"^[{_COMMON_CHINESE_SURNAMES}][\u4e00-\u9fff]{{1,2}}$"
)
_CHINESE_NAME_QUERY_RE = re.compile(
    rf"(?:查|查询|搜索|找|看看|关于)\s*(?:一下)?\s*"
    rf"[{_COMMON_CHINESE_SURNAMES}][\u4e00-\u9fff]{{1,2}}"
    r"(?=$|[\s,，。；;]|(?:的)?(?:电话|资料|情况|新闻|信息|记录|家庭|住址))"
)


def _normalized_text(value) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value)
    for _ in range(2):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    return text[:20_000]


def contains_private_egress_data(value) -> bool:
    """Detect personally identifying or record-specific text before egress."""
    text = _normalized_text(value)
    if not text:
        return False
    compact_phone = re.sub(r"[\s()（）+.-]+", "", text)
    return any(
        pattern.search(candidate) is not None
        for pattern, candidate in (
            (_MOBILE_RE, compact_phone),
            (_LANDLINE_RE, compact_phone),
            (_IDENTITY_CARD_RE, compact_phone),
            (_EMAIL_RE, text),
            (_EXACT_BUILDING_ROOM_RE, text),
            (_HYPHEN_ROOM_RE, text),
            (_EXPLICIT_ADDRESS_RE, text),
            (_PRIVATE_CONTENT_RE, text),
            (_LABELED_NAME_RE, text),
            (_HONORIFIC_NAME_RE, text),
        )
    )


def contains_likely_person_name(value) -> bool:
    """Conservatively recognize bare Chinese names in authenticated My Stand."""
    text = _normalized_text(value).strip()
    if not text:
        return False
    return (
        _BARE_CHINESE_NAME_RE.fullmatch(text) is not None
        or _CHINESE_NAME_QUERY_RE.search(text) is not None
    )


def web_egress_block_result(values) -> str | None:
    """Return a synthetic tool error when outbound research is unsafe."""
    platform = get_session_env(
        "XIAOBAN_SESSION_PLATFORM",
        "",
    ).strip().lower()
    user_id = get_session_env("XIAOBAN_SESSION_USER_ID", "").strip()
    is_mystand_session = platform == "api_server" and bool(user_id)
    if (
        is_mystand_session
        and mystand_private_taint_persistence_failed()
    ):
        blocked = True
    elif mystand_private_query_turn_active():
        blocked = True
    else:
        blocked = any(
            contains_private_egress_data(value)
            or (is_mystand_session and contains_likely_person_name(value))
            for value in values
        )
        if not blocked:
            current_message = get_session_user_message()
            blocked = bool(
                is_mystand_session
                and contains_private_egress_data(current_message)
            )
            if (
                not blocked
                and is_mystand_session
                and contains_likely_person_name(current_message)
            ):
                blocked = True
    if not blocked:
        return None
    return json.dumps(
        {
            "success": False,
            "code": "private_data_egress_blocked",
            "error": (
                "Blocked: private My Stand or personally identifying data "
                "cannot be sent to external web services."
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def mark_mystand_private_batch(tool_calls) -> bool:
    """Mark a whole private My Stand tool batch before any call can run."""
    for tool_call in tool_calls or []:
        if isinstance(tool_call, dict):
            function = tool_call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
            else:
                name = tool_call.get("name")
        else:
            function = getattr(tool_call, "function", None)
            name = getattr(function, "name", "")
        if str(name or "").strip() in _CURRENT_MYSTAND_PRIVATE_TOOL_NAMES:
            mark_mystand_private_query_turn()
            return True
    return False


# Compatibility for callers introduced with the original query-only boundary.
mark_mystand_query_batch = mark_mystand_private_batch
