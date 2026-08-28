from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from plugins.memory.holographic.scope import open_scoped_memory_store
from xiaoban.mystand_account_memory import (
    OWNER_JOURNAL_CATEGORY,
    OWNER_PROFILE_CATEGORY,
    SERVICE_NOTEBOOK_CATEGORY,
    _date_label,
    build_account_memory_context,
    list_account_documents,
    manage_account_memory,
)


def _open(tmp_path, user: str):
    return open_scoped_memory_store(
        secret="account-memory-test-secret",
        site_id="mystand-preview",
        user_id=user,
        xiaoban_home=tmp_path,
    )


def test_skip_is_a_structured_noop_and_does_not_interpret_business_words(tmp_path):
    store = _open(tmp_path, "broker-skip")
    try:
        result = manage_account_memory(
            store,
            tier="notebook",
            action="skip",
            content="请记住长期客户、业主、佣金和底价",
            account_label="经纪人",
        )
        documents = list_account_documents(store, tier="notebook")
    finally:
        store.close()

    assert result["action"] == "skip"
    assert result["recorded"] is False
    assert documents == []


def test_notebook_upsert_is_idempotent_bounded_and_account_scoped(tmp_path):
    alice = _open(tmp_path, "broker-alice")
    bob = _open(tmp_path, "broker-bob")
    try:
        for index in range(20):
            manage_account_memory(
                alice,
                tier="notebook",
                action="upsert",
                target="notebook",
                content=f"服务事项 {index}",
                account_label="经纪人甲",
            )
        manage_account_memory(
            alice,
            tier="notebook",
            action="upsert",
            target="notebook",
            content="服务事项 19",
            account_label="经纪人甲",
        )
        alice_documents = list_account_documents(alice, tier="notebook")
        bob_documents = list_account_documents(bob, tier="notebook")
    finally:
        alice.close()
        bob.close()

    content = alice_documents[0]["content"]
    lines = [line for line in content.splitlines() if line.startswith("-")]
    assert alice_documents[0]["category"] == SERVICE_NOTEBOOK_CATEGORY
    assert len(lines) == 16
    assert content.count("服务事项 19") == 1
    assert "服务事项 0" not in content
    assert len(content) <= 5200
    assert bob_documents == []


def test_structured_summary_is_field_redacted_and_keeps_valid_resource_refs(tmp_path):
    reference_id = "AUTH-AAAAAAAA-BBBBBBBB-CCCCCCCC-DDDDDDDD"
    store = _open(tmp_path, "broker-private")
    try:
        result = manage_account_memory(
            store,
            tier="notebook",
            action="upsert",
            target="notebook",
            content=(
                "客户姓名：张三；客户电话：13800001234；"
                "佣金：12000元；结论：按资料继续跟进"
            ),
            resource_refs=[{
                "referenceId": reference_id,
                "sourceType": "business-archive",
            }],
            account_label="经纪人",
        )
    finally:
        store.close()

    content = result["documents"][0]["content"]
    assert "张三" not in content
    assert "13800001234" not in content
    assert "12000" not in content
    assert reference_id in content
    assert "按资料继续跟进" in content


def test_invalid_resource_ref_is_rejected_instead_of_silently_saved(tmp_path):
    store = _open(tmp_path, "broker-invalid-ref")
    try:
        with pytest.raises(ValueError, match="resource reference"):
            manage_account_memory(
                store,
                tier="notebook",
                action="upsert",
                target="notebook",
                content="只保存安全摘要",
                resource_refs=[{
                    "referenceId": "../../private-file",
                    "sourceType": "business-archive",
                }],
            )
    finally:
        store.close()


def test_owner_can_add_correct_merge_and_delete_structured_memories(tmp_path):
    store = _open(tmp_path, "52707407")
    try:
        manage_account_memory(
            store,
            tier="owner",
            action="upsert",
            target="profile",
            content="回答时先讲过程",
            account_label="刚哥",
        )
        manage_account_memory(
            store,
            tier="owner",
            action="correct",
            target="profile",
            old_text="先讲过程",
            content="回答时先讲结论",
            account_label="刚哥",
        )
        manage_account_memory(
            store,
            tier="owner",
            action="upsert",
            target="profile",
            content="答复保持简洁",
            account_label="刚哥",
        )
        manage_account_memory(
            store,
            tier="owner",
            action="upsert",
            target="profile",
            old_text="答复保持简洁",
            content="回答简洁并保留必要证据",
            account_label="刚哥",
        )
        manage_account_memory(
            store,
            tier="owner",
            action="upsert",
            target="profile",
            content="回答简洁并保留必要证据",
            account_label="刚哥",
        )
        manage_account_memory(
            store,
            tier="owner",
            action="upsert",
            target="journal",
            content="持续维护 My Stand 发布基线",
            account_label="刚哥",
        )
        result = manage_account_memory(
            store,
            tier="owner",
            action="forget",
            target="profile",
            old_text="回答简洁并保留必要证据",
            account_label="刚哥",
        )
    finally:
        store.close()

    documents = {item["category"]: item for item in result["documents"]}
    profile = documents[OWNER_PROFILE_CATEGORY]["content"]
    journal = documents[OWNER_JOURNAL_CATEGORY]["content"]
    assert "先讲过程" not in profile
    assert "先讲结论" in profile
    assert "答复保持简洁" not in profile
    assert "回答简洁并保留必要证据" not in profile
    assert "持续维护 My Stand 发布基线" in journal


def test_concurrent_upserts_still_create_one_bounded_notebook_document(tmp_path):
    def upsert(index: int) -> None:
        store = _open(tmp_path, "broker-concurrent")
        try:
            manage_account_memory(
                store,
                tier="notebook",
                action="upsert",
                target="notebook",
                content=f"并发服务事项 {index}",
                account_label="并发账号",
            )
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(upsert, range(12)))

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


def test_expired_memory_uses_shanghai_boundary_and_is_pruned_by_skip(tmp_path):
    assert _date_label("2026-08-26T16:30:00Z") == "2026-08-27"
    store = _open(tmp_path, "broker-expiry")
    try:
        store.add_fact(
            "服务小本（临时）\n- 临时跟进〔有效至 2020-01-01〕\n- 长期跟进",
            category=SERVICE_NOTEBOOK_CATEGORY,
        )
        before = list_account_documents(store, tier="notebook")
        result = manage_account_memory(
            store,
            tier="notebook",
            action="skip",
            account_label="临时",
        )
        persisted = store.list_facts(category=SERVICE_NOTEBOOK_CATEGORY, limit=2)
    finally:
        store.close()

    assert "临时跟进" not in before[0]["content"]
    assert result["removedExpired"] == 1
    assert "临时跟进" not in persisted[0]["content"]
    assert "长期跟进" in persisted[0]["content"]


def test_notebook_cannot_manage_owner_profile(tmp_path):
    store = _open(tmp_path, "broker-boundary")
    try:
        with pytest.raises(ValueError, match="cannot manage owner"):
            manage_account_memory(
                store,
                tier="notebook",
                action="upsert",
                target="profile",
                content="不应写入",
            )
    finally:
        store.close()


def test_owner_memory_context_contains_only_owner_documents(tmp_path):
    store = _open(tmp_path, "52707407-context")
    try:
        manage_account_memory(
            store,
            tier="owner",
            action="upsert",
            target="profile",
            content="回答先给结论",
            account_label="刚哥",
        )
        manage_account_memory(
            store,
            tier="owner",
            action="upsert",
            target="journal",
            content="持续维护发布基线",
            account_label="刚哥",
        )
        context, count = build_account_memory_context(store, tier="owner")
    finally:
        store.close()

    assert count == 2
    assert "回答先给结论" in context
    assert "持续维护发布基线" in context
    assert "低优先级数据" in context
