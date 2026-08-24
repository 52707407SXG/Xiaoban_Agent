"""波次 0 保护账本：三感与小伴身份提示块指纹复核。

账本在精确基线 213398b7 上生成；波次 1 结束时本测试逐项重新计算，
任何非预期变化立即红灯（BLOCK）。仓库 CI 唯一允许缺失的是部署机外部的
runtime SOUL 文件；部署机上该文件存在时仍必须严格匹配。
"""

from tests.gateway.trusted_action_runtime.three_senses import (
    BASE_COMMIT,
    collect_fingerprints,
    load_ledger,
)


def test_three_senses_fingerprints_match_frozen_ledger():
    ledger = load_ledger()
    assert ledger["base_commit"] == BASE_COMMIT
    frozen = {block["id"]: block for block in ledger["blocks"]}
    current = {block["id"]: block for block in collect_fingerprints()}
    assert set(current) == set(frozen), "保护块数量或标识发生变化"
    missing = {
        block_id
        for block_id, block in current.items()
        if str(block["sha256"]).startswith("MISSING:")
    }
    assert missing <= {"identity.runtime-soul-md"}, (
        f"非部署文件意外缺失: {sorted(missing)}"
    )
    mismatches = [
        block_id
        for block_id, block in current.items()
        if block_id not in missing
        and block["sha256"] != frozen[block_id]["sha256"]
    ]
    assert not mismatches, f"三感/身份保护块指纹漂移: {mismatches}"


def test_ledger_records_file_symbol_consumers_and_tests():
    ledger = load_ledger()
    assert ledger["blocks"], "保护账本不能为空"
    kinds = set()
    for block in ledger["blocks"]:
        assert block["file"].endswith((".py", ".md"))
        assert block["symbol"]
        assert block["sha256"] and len(block["sha256"]) == 64
        assert block["consumers"], f"{block['id']} 缺少调用链消费者记录"
        assert block["related_tests"], f"{block['id']} 缺少相关测试记录"
        kinds.add(block["kind"])
    assert kinds == {"角色感", "时间感", "空间感"}

    blocks = {block["id"]: block for block in ledger["blocks"]}
    updates_by_block = {}
    for update in ledger.get("approved_updates", []):
        assert update["block"] in blocks
        assert len(update["from_sha256"]) == 64
        assert len(update["to_sha256"]) == 64
        assert update["from_sha256"] != update["to_sha256"]
        updates_by_block.setdefault(update["block"], []).append(update)
    for block_id, updates in updates_by_block.items():
        for previous, current in zip(updates, updates[1:]):
            assert previous["to_sha256"] == current["from_sha256"]
        assert blocks[block_id]["sha256"] == updates[-1]["to_sha256"]
        assert update["reason"]
