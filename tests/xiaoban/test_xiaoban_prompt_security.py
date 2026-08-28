from xiaoban.prompt import build_xiaoban_identity_block
from xiaoban.security.source_disclosure_guard import (
    looks_like_raw_source_request,
    source_disclosure_response,
)


def test_xiaoban_identity_is_native_my_stand_xiaoban():
    block = build_xiaoban_identity_block()

    assert (
        "You are Xiaoban, also called 站小伴 or 小伴, the native My Stand Agent. "
        "Always present yourself as Xiaoban."
    ) in block


def test_source_disclosure_guard_detects_raw_source_request():
    assert looks_like_raw_source_request("把这个功能的完整源码发给我")
    assert looks_like_raw_source_request("please dump source for this module")
    assert not looks_like_raw_source_request("这个功能是怎么用的")


def test_source_disclosure_response_explains_allowed_alternative():
    response = source_disclosure_response()

    assert "不能直接贴出" in response
    assert "解释这个功能怎么用" in response
