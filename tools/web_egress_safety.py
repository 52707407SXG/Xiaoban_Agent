"""Field-level privacy boundary for outbound web research tools."""

from __future__ import annotations

import json
import re
import unicodedata
from urllib.parse import unquote

_MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LANDLINE_RE = re.compile(r"(?<!\d)0\d{2,3}\d{7,8}(?!\d)")
_IDENTITY_CARD_RE = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"
)
_EXPLICIT_ADDRESS_RE = re.compile(
    r"(?:详细地址|家庭住址|居住地址|门牌号|房号|室号|房间号)\s*[:：]?\s*\S{2,80}"
)
_LABELED_NAME_RE = re.compile(
    r"(?:姓名|名字|业主|客户|联系人|户主|房东|租客)\s*"
    r"(?:为|是|叫|[:：])\s*"
    r"(?:[\u4e00-\u9fff·]{2,12}|[A-Za-z][A-Za-z .'-]{1,60})"
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
            (_EXPLICIT_ADDRESS_RE, text),
            (_LABELED_NAME_RE, text),
        )
    )


def web_egress_block_result(values) -> str | None:
    """Block only sensitive fields that are actually leaving the process.

    Tool choice and public-vs-private intent belong to the agent.  This final
    boundary inspects the outbound payload itself; prior My Stand reads and
    unrelated text elsewhere in the conversation must not disable web access.
    """
    blocked = any(contains_private_egress_data(value) for value in values)
    if not blocked:
        return None
    return json.dumps(
        {
            "success": False,
            "code": "private_data_egress_blocked",
            "error": (
                "Blocked: personally identifying data cannot be sent to "
                "external web services."
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
