"""Guard against raw My Stand source-code disclosure."""

from __future__ import annotations

RAW_SOURCE_REQUEST_MARKERS = (
    "完整源码",
    "全部代码",
    "源代码发给我",
    "paste the source",
    "full source",
    "dump source",
)


def looks_like_raw_source_request(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in RAW_SOURCE_REQUEST_MARKERS)


def source_disclosure_response() -> str:
    return (
        "这部分我不能直接贴出 My Stand 的原始源码。"
        "我可以帮你解释这个功能怎么用、业务逻辑是什么、入口在哪里，"
        "也可以给有权限的开发者整理改造建议。"
    )

