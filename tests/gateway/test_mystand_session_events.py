from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


SESSION_ID = "web:w-0123456789abcdef:52707407:c-0123456789abcdef0123456789abcdef01234567"


def test_mystand_async_event_survives_adapter_restart(tmp_path, monkeypatch):
    event_db = tmp_path / "session-events.sqlite3"
    monkeypatch.setenv("XIAOBAN_SESSION_EVENT_DB", str(event_db))
    first = APIServerAdapter(PlatformConfig())
    event = first._enqueue_session_event(
        SESSION_ID,
        "assistant.message",
        {"message": {"id": "message-one", "role": "assistant", "content": "早报完成"}},
    )

    second = APIServerAdapter(PlatformConfig())
    events = second._session_event_snapshot(SESSION_ID, 0)

    assert event_db.exists()
    assert event_db.stat().st_mode & 0o777 == 0o600
    assert len(events) == 1
    assert events[0]["id"] == event["id"]
    assert events[0]["message"]["content"] == "早报完成"
    assert second._session_event_snapshot(SESSION_ID, events[0]["seq"]) == []


def test_non_mystand_session_event_is_not_persisted(tmp_path, monkeypatch):
    event_db = tmp_path / "session-events.sqlite3"
    monkeypatch.setenv("XIAOBAN_SESSION_EVENT_DB", str(event_db))
    adapter = APIServerAdapter(PlatformConfig())
    adapter._enqueue_session_event(
        "ordinary-api-session",
        "assistant.message",
        {"message": {"id": "message-two", "role": "assistant", "content": "hello"}},
    )
    assert not event_db.exists()
