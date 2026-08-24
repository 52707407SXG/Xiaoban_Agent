from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from plugins.memory.holographic.scope import open_scoped_memory_store
from xiaoban.mystand_account_memory import (
    OWNER_JOURNAL_CATEGORY,
    OWNER_PROFILE_CATEGORY,
    SERVICE_NOTEBOOK_CATEGORY,
    build_account_memory_context,
    list_account_documents,
    record_account_turn,
)


def _open(tmp_path, user: str):
    return open_scoped_memory_store(
        secret="account-memory-test-secret",
        site_id="mystand-preview",
        user_id=user,
        xiaoban_home=tmp_path,
    )


def test_notebook_account_has_exactly_one_bounded_idempotent_document(tmp_path):
    store = _open(tmp_path, "broker-zhang")
    try:
        first = record_account_turn(
            store,
            tier="notebook",
            turn_id="delivery-broker-0001",
            user_message="请把今天带看城南一号的事项整理一下",
            assistant_message="已整理带看事项，并列出明天的跟进重点。",
            account_label="张三",
            occurred_at="2026-08-21T08:00:00Z",
        )
        replay = record_account_turn(
            store,
            tier="notebook",
            turn_id="delivery-broker-0001",
            user_message="请把今天带看城南一号的事项整理一下",
            assistant_message="已整理带看事项，并列出明天的跟进重点。",
            account_label="张三",
            occurred_at="2026-08-21T08:00:00Z",
        )
        documents = list_account_documents(store, tier="notebook")
        all_facts = store.list_facts(limit=100)
    finally:
        store.close()

    assert first["documents"] == 1
    assert first["recorded"] is True
    assert replay["recorded"] is False
    assert len(documents) == 1
    assert len(all_facts) == 1
    assert documents[0]["category"] == SERVICE_NOTEBOOK_CATEGORY
    assert "服务小本（张三）" in documents[0]["content"]
    assert documents[0]["content"].count("带看") == 2


def test_owner_gets_profile_and_long_term_journal_only(tmp_path):
    store = _open(tmp_path, "owner-user-001")
    try:
        result = record_account_turn(
            store,
            tier="owner",
            turn_id="delivery-owner-0001",
            user_message="我是刚哥，以后请先讲结论，我正在负责 My Stand。",
            assistant_message="明白，后续会先给结论再说明依据。",
            account_label="刚哥",
            occurred_at="2026-08-21T09:00:00Z",
        )
        documents = list_account_documents(store, tier="owner")
        context, count = build_account_memory_context(store, tier="owner")
    finally:
        store.close()

    assert result["documents"] == 2
    assert {item["category"] for item in documents} == {
        OWNER_PROFILE_CATEGORY,
        OWNER_JOURNAL_CATEGORY,
    }
    assert count == 2
    assert "我是刚哥" in context
    assert "主账号长期事项" in context
    assert "低优先级数据" in context


def test_sensitive_turn_is_redacted_and_accounts_never_share_documents(tmp_path):
    first = _open(tmp_path, "broker-a")
    second = _open(tmp_path, "broker-b")
    try:
        record_account_turn(
            first,
            tier="notebook",
            turn_id="delivery-sensitive-0001",
            user_message="客户电话是 13800001234，密码是 abc，请帮我记住",
            assistant_message="已经处理客户电话 13800001234。",
            account_label="经纪人甲",
        )
        first_documents = list_account_documents(first, tier="notebook")
        second_documents = list_account_documents(second, tier="notebook")
    finally:
        first.close()
        second.close()

    assert len(first_documents) == 1
    assert "13800001234" not in first_documents[0]["content"]
    assert "abc" not in first_documents[0]["content"]
    assert first_documents[0]["content"].count("[敏感字段已脱敏]") == 3
    assert "请帮我记住" in first_documents[0]["content"]
    assert second_documents == []


def test_notebook_keeps_only_latest_sixteen_turns(tmp_path):
    store = _open(tmp_path, "broker-bounded")
    try:
        for index in range(20):
            record_account_turn(
                store,
                tier="notebook",
                turn_id=f"delivery-bounded-{index:04d}",
                user_message=f"整理第 {index} 项跟进",
                assistant_message=f"第 {index} 项已整理",
                account_label="边界测试账号",
            )
        documents = list_account_documents(store, tier="notebook")
    finally:
        store.close()

    lines = documents[0]["content"].splitlines()
    assert len([line for line in lines if line.startswith("-")]) == 16
    assert "第 0 项" not in documents[0]["content"]
    assert "第 19 项" in documents[0]["content"]
    assert len(documents[0]["content"]) <= 5200


def test_concurrent_first_turns_still_create_one_notebook_document(tmp_path):
    def record(index: int) -> None:
        store = _open(tmp_path, "broker-concurrent")
        try:
            record_account_turn(
                store,
                tier="notebook",
                turn_id=f"delivery-concurrent-{index:04d}",
                user_message=f"整理并发事项 {index}",
                assistant_message=f"并发事项 {index} 已整理",
                account_label="并发账号",
            )
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record, range(12)))

    store = _open(tmp_path, "broker-concurrent")
    try:
        documents = list_account_documents(store, tier="notebook")
        all_facts = store.list_facts(limit=100)
    finally:
        store.close()

    assert len(documents) == 1
    assert len(all_facts) == 1
    assert len([
        line for line in documents[0]["content"].splitlines()
        if line.startswith("-")
    ]) == 12
