"""Codex 复审 BLOCK 修复 R0'：8 项问题的真实 RED。

全部打在真实入口（生命周期 API / CompletionGuard / 三感账本）上，
在固定 HEAD e4fbb29 上必须失败；GREEN 后同一断言不加修改通过。
"""

import json
import re
from pathlib import Path

from xiaoban.trusted_runtime import (
    TrustedIdentity,
    begin_action,
    begin_turn,
    build_work_turn,
    check_completion,
    finish_action,
)
from xiaoban.trusted_runtime.completion_guard import check_mystand_final_answer

from tests.gateway.trusted_action_runtime import incident_fixtures as fx
from tests.gateway.trusted_action_runtime.three_senses import (
    fingerprint_text,
    load_ledger,
)

IDENTITY = TrustedIdentity(
    account_id="user-a", data_scope="mystand", source="server_session"
)
BUSINESS_MSG = "查一下游某今年的结算业绩"
RESOLVE_ARGS = {"operation": "resolve", "resource_uid": "res-demo-1"}
INDEX_RECEIPT = {
    "ok": True,
    "items": [
        {
            "resourceUid": "res-demo-1",
            "safeLabel": "游某 2026年个人业务档案",
            "canRead": True,
        }
    ],
}


def _indexed_result(calls, user_message=BUSINESS_MSG, user_id="user-a"):
    indexed = [
        (
            "call_idx",
            "mystand_resource_index",
            {"operation": "list_resources", "module_id": "finance-ledger"},
            INDEX_RECEIPT,
        )
    ] + list(calls)
    return fx.tool_turn(user_message, indexed, user_id=user_id)


def _turn(result, user_message=BUSINESS_MSG, identity=IDENTITY):
    return build_work_turn(
        channel="web",
        user_message=user_message,
        conversation_history=[],
        result=result,
        identity=identity,
        request_id="req-1",
        message_id="msg-1",
    )


# --- 复审 1：_trusted_turn 必须与服务端身份/requestId/messageId/DataScope 再绑定 ---
def test_red_trusted_turn_identity_mismatch_must_rebind_fail():
    # 当前行为：result._trusted_turn 被直接采信，身份不一致也不复核。
    foreign = begin_turn(
        channel="web",
        user_message=BUSINESS_MSG,
        identity=TrustedIdentity(
            account_id="user-b", data_scope="mystand", source="server_session"
        ),
        request_id="req-foreign",
        message_id="msg-foreign",
    )
    allow_index = begin_action(
        foreign, "mystand_resource_index", "v1",
        {"operation": "list_resources", "module_id": "finance-ledger"},
    )
    finish_action(
        foreign, allow_index.call.call_id, "mystand_resource_index", "v1",
        json.dumps(INDEX_RECEIPT, ensure_ascii=False),
    )
    allow_read = begin_action(foreign, "mystand_authorization", "v1", RESOLVE_ARGS)
    finish_action(
        foreign, allow_read.call.call_id, "mystand_authorization", "v1",
        '{"ok":true,"content":"游某今年结算业绩是 32105.68 元。"}',
    )
    assert foreign.evidence, "夹具构造失败：外来回合应自带证据"
    result = {
        "_mystand_request": True,
        "_mystand_user_id": "user-a",
        "_mystand_message_id": "msg-1",
        "_trusted_turn": foreign,
        "messages": [],
    }
    decision = check_mystand_final_answer(
        "游某今年结算业绩是 32105.68 元。",
        user_message=BUSINESS_MSG,
        conversation_history=[],
        result=result,
        account_id="user-a",
        request_id="req-1",
        message_id="msg-1",
    )
    assert not decision.allowed, "外来 _trusted_turn 身份/messageId 不一致仍被采信"
    assert "32105.68" not in decision.text


# --- 复审 2：WORK 动作执行前必须已有服务端 IndexReceipt，禁止反向补索引 ---
def test_red_scoped_read_cannot_backfill_index_receipt():
    # 当前行为：authorization resolve 自身被执行后补成 IndexReceipt。
    result = fx.tool_turn(
        BUSINESS_MSG,
        [("call_r", "mystand_authorization", RESOLVE_ARGS, {"ok": True, "content": "游某今年结算业绩是 32105.68 元。"})],
    )
    turn = _turn(result)
    assert turn.evidence == [], "无索引前置的定向读取被错误采信"
    assert all(item.status != "success" for item in turn.action_results)


# --- 复审 3：矛盾回执失败关闭 ---
def test_red_ok_true_with_error_field_is_not_success():
    result = _indexed_result(
        [("call_c", "mystand_authorization", RESOLVE_ARGS,
          {"ok": True, "error": "partial bridge failure", "content": "地址3401号"})]
    )
    turn = _turn(result)
    assert turn.action_results[-1].status == "error"
    assert turn.evidence == []


def test_red_ok_true_with_string_error_status_is_not_success():
    result = _indexed_result(
        [("call_c", "mystand_authorization", RESOLVE_ARGS,
          {"ok": True, "status": "500", "content": "地址3401号"})]
    )
    turn = _turn(result)
    assert turn.action_results[-1].status == "error"
    assert turn.evidence == []


def test_red_ok_true_with_failure_code_is_not_success():
    result = _indexed_result(
        [("call_c", "mystand_authorization", RESOLVE_ARGS,
          {"ok": True, "code": "read_failed", "content": "地址3401号"})]
    )
    turn = _turn(result)
    assert turn.action_results[-1].status == "error"
    assert turn.evidence == []


# --- 复审 4：连续业务追问不得漏判成 CHAT 放行纯人名事实 ---
def test_red_followup_business_context_is_not_chat():
    # 当前行为：历史只扫 user 角色，业务语境在 assistant 回复里时漏判 CHAT。
    history = [
        {"role": "user", "content": "在吗"},
        {"role": "assistant", "content": "您是想查滨江一号3栋802的业主吗？"},
    ]
    decision = check_mystand_final_answer(
        "就是周某本人。",
        user_message="他是谁？",
        conversation_history=history,
        result=fx.tool_turn("他是谁？"),
        account_id="user-a",
    )
    assert not decision.allowed, "业务追问被漏判成 CHAT 放行纯人名事实"
    assert "周某" not in decision.text


# --- 复审 5：ownerUser/teamId/companyId/module 的服务端 DataScope 复核 ---
def test_red_owner_user_mismatch_is_rejected():
    result = _indexed_result(
        [("call_c", "mystand_authorization", RESOLVE_ARGS,
          {"ok": True, "ownerUser": "user-b", "content": "游某今年结算业绩是 32105.68 元。"})]
    )
    turn = _turn(result)
    assert turn.evidence == [], "ownerUser 与当前身份冲突仍生成 Evidence"


def test_red_unverifiable_team_scope_is_fail_closed():
    result = _indexed_result(
        [("call_c", "mystand_authorization", RESOLVE_ARGS,
          {"ok": True, "teamId": "team-b", "content": "游某今年结算业绩是 32105.68 元。"})]
    )
    turn = _turn(result)
    assert turn.evidence == [], "服务端无法核实的 teamId 被静默放行"


def test_red_payload_module_mismatch_is_rejected():
    receipt = dict(INDEX_RECEIPT)
    receipt["moduleId"] = "other-module"
    result = fx.tool_turn(
        BUSINESS_MSG,
        [("call_idx", "mystand_resource_index",
          {"operation": "list_resources", "module_id": "finance-ledger"}, receipt)],
    )
    turn = _turn(result)
    assert turn.evidence == [], "payload module 与执行上下文不一致仍生成 Evidence"


# --- 复审 6：preview_write/commit_write 与只读合同隔离，合法写回执不被误伤 ---
def test_red_preview_write_turn_is_not_mangled_by_read_guard():
    result = fx.tool_turn(
        "把滨江一号3栋802钥匙状态改成已交接",
        [("call_w", "mystand_authorization",
          {"operation": "preview_write", "authorization_id": "AUTH-DEMO-1", "action": "update"},
          {"ok": True, "previewToken": "tok-demo", "preview": "将钥匙状态改为已交接"})],
    )
    decision = check_mystand_final_answer(
        "已生成写入预览：将钥匙状态改为已交接。请确认后我再执行。",
        user_message="把滨江一号3栋802钥匙状态改成已交接",
        conversation_history=[],
        result=result,
        account_id="user-a",
    )
    assert decision.allowed, "合法 preview_write 流程被只读 Guard 误伤"
    assert decision.text == "已生成写入预览：将钥匙状态改为已交接。请确认后我再执行。"


def test_red_commit_write_verified_receipt_is_not_mangled():
    result = fx.tool_turn(
        "确认执行",
        [("call_w2", "mystand_authorization",
          {"operation": "commit_write", "authorization_id": "AUTH-DEMO-1", "previewToken": "tok-demo"},
          {"ok": True, "verified": True, "revision": 3, "receipt": "write-ok-demo"})],
    )
    decision = check_mystand_final_answer(
        "已按确认完成写入并核验回执。",
        user_message="确认执行",
        conversation_history=[],
        result=result,
        account_id="user-a",
    )
    assert decision.allowed, "合法 verified 写回执被只读 Guard 误伤"


# --- 复审 7：生产 SOUL.md 运行源内容指纹 + 运行配置指向检查（不记录正文）---
def test_red_runtime_soul_md_fingerprint_in_ledger():
    ledger = load_ledger()
    blocks = {block["id"]: block for block in ledger["blocks"]}
    assert "identity.runtime-soul-md" in blocks, "账本缺少 SOUL.md 运行源内容指纹"
    block = blocks["identity.runtime-soul-md"]
    assert block["file"] == "/var/lib/xiaoban/SOUL.md"
    unit = Path("/etc/systemd/system/xiaoban-agent.service").read_text(encoding="utf-8")
    match = re.search(r"Environment=XIAOBAN_HOME=(\S+)", unit)
    assert match, "运行配置缺少 XIAOBAN_HOME 指向"
    assert block["file"] == str(Path(match.group(1)) / "SOUL.md")
    live = fingerprint_text(Path(block["file"]).read_text(encoding="utf-8"))
    assert block["sha256"] == live, "SOUL.md 运行源内容指纹与账本不一致"
