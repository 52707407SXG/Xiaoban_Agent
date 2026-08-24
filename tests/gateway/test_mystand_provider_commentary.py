from gateway.platforms.mystand_provider_commentary import (
    MystandProviderCommentaryProjector,
)


def _projector():
    return MystandProviderCommentaryProjector(
        summary_builder=lambda value: " ".join(str(value or "").split())[:240],
        progress_schema="xiaoban.progress.v2",
    )


STARTED_TURN = {
    "requestId": "xbd_123",
    "turnId": "abcdef0123456789",
}


def test_pure_final_stays_provisional_and_never_becomes_commentary():
    projector = _projector()

    assert projector.accept_delta(
        "这是最终答复。",
        started_turn=STARTED_TURN,
    ) is None

    assert projector.pending_final_chunks() == ["这是最终答复。"]
    assert projector.drain_final_chunks() == ["这是最终答复。"]
    assert projector.pending_final_chunks() == []


def test_tool_preamble_updates_one_real_event_from_running_to_completed():
    projector = _projector()
    projector.accept_delta("我先核对", started_turn=STARTED_TURN)

    first = projector.confirm_tool_generation(
        source="provider",
        provider_sequence=1,
        provider_event_at=1_700_000_000,
        stage="intent",
        started_turn=STARTED_TURN,
    )
    second = projector.accept_delta("未结算记录。", started_turn=STARTED_TURN)
    completed = projector.complete_tool_commentary(
        "我先核对未结算记录。",
        source="provider",
        provider_sequence=1,
        provider_event_at=1_700_000_000,
        stage="intent",
        started_turn=STARTED_TURN,
    )

    assert first["status"] == "running"
    assert second["status"] == "running"
    assert completed["status"] == "completed"
    assert first["eventId"] == second["eventId"] == completed["eventId"]
    assert completed["summary"] == "我先核对未结算记录。"
    assert completed["source"] == "provider"
    assert completed["stage"] == "intent"
    assert projector.close_tool_response(started_turn=STARTED_TURN) is None
    assert projector.pending_final_chunks() == []


def test_missing_provider_text_never_creates_a_synthetic_commentary_event():
    projector = _projector()

    assert projector.confirm_tool_generation(
        source="provider",
        provider_sequence=1,
        provider_event_at=None,
        stage="intent",
        started_turn=STARTED_TURN,
    ) is None
    assert projector.close_tool_response(started_turn=STARTED_TURN) is None


def test_conflicting_source_cannot_promote_provisional_final_text():
    projector = _projector()
    projector.accept_delta("保留为最终候选", started_turn=STARTED_TURN)

    assert projector.confirm_tool_generation(
        source="runtime",
        provider_sequence=1,
        provider_event_at=None,
        stage="intent",
        started_turn=STARTED_TURN,
    ) is None
    assert projector.pending_final_chunks() == ["保留为最终候选"]
