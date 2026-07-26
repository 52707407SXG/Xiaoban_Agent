"""波次 0 保护账本：三感与小伴身份提示块指纹复核。

账本在精确基线 213398b7 上生成；波次 1 结束时本测试逐项重新计算，
任何非预期变化立即红灯（BLOCK）。
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
    mismatches = [
        block_id
        for block_id, block in current.items()
        if block["sha256"] != frozen[block_id]["sha256"]
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
