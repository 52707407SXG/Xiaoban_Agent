from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


def test_search_falls_back_for_hyphenated_marker_and_chinese_terms(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memory.db"), hrr_dim=128)
    try:
        store.add_fact(
            "50轮回测标记为 XIAOBAN-REGRESSION-50，完成后需能复述标记内容。",
            category="project",
        )
        store.add_fact(
            "小伴对 My Stand 代码的硬边界：只读理解，不操作代码，不外泄。",
            category="project",
        )
        retriever = FactRetriever(store=store, hrr_dim=128)

        marker_results = retriever.search("XIAOBAN-REGRESSION-50", limit=3)
        boundary_results = retriever.search("My Stand 代码 只读 不外泄", limit=3)

        assert any("XIAOBAN-REGRESSION-50" in item["content"] for item in marker_results)
        assert any("只读理解" in item["content"] for item in boundary_results)
    finally:
        store.close()
