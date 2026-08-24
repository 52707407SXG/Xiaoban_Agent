"""State machine for real Provider commentary in My Stand streams.

The Provider may stream visible text before it reveals that the same assistant
response contains tool calls.  This module keeps those bytes provisional until
the structural tool-generation callback confirms their role.  It never creates
commentary text: every public summary is derived from Provider-authored bytes
through the caller's sanitizer.

Instances are request-local.  The owning gateway callback ledger serializes all
method calls with its lifecycle lock, so this class intentionally owns no lock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any, Callable, Mapping, Optional


@dataclass
class _ActiveCommentary:
    sequence: int
    provider_event_at: str
    stage: str
    chunks: list[str] = field(default_factory=list)
    last_summary: str = ""
    last_status: str = ""


class MystandProviderCommentaryProjector:
    """Separate tool preambles from final text and project typed updates."""

    def __init__(
        self,
        *,
        summary_builder: Callable[[Any], str],
        progress_schema: str,
    ) -> None:
        self._summary_builder = summary_builder
        self._progress_schema = progress_schema
        self._pending_chunks: list[str] = []
        self._sequence = 0
        self._active: Optional[_ActiveCommentary] = None

    @staticmethod
    def _turn_identity(
        started_turn: Optional[Mapping[str, Any]],
    ) -> Optional[tuple[str, str]]:
        if not isinstance(started_turn, Mapping):
            return None
        request_id = str(started_turn.get("requestId") or "")
        turn_id = str(started_turn.get("turnId") or "")
        if not request_id or not turn_id:
            return None
        return request_id, turn_id

    def _sequence_for(self, value: Any) -> int:
        try:
            supplied = int(value)
        except (TypeError, ValueError):
            supplied = 0
        if supplied > 0:
            self._sequence = max(self._sequence, supplied)
            return supplied
        self._sequence += 1
        return self._sequence

    @staticmethod
    def _event_at(value: Any) -> str:
        try:
            event_at = float(value) if value is not None else time.time()
            return datetime.fromtimestamp(
                event_at,
                tz=timezone.utc,
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except (TypeError, ValueError, OSError, OverflowError):
            return datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ).replace("+00:00", "Z")

    def _project(
        self,
        state: _ActiveCommentary,
        *,
        status: str,
        started_turn: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        identity = self._turn_identity(started_turn)
        if identity is None:
            return None
        summary = self._summary_builder("".join(state.chunks))
        if not summary:
            return None
        if state.last_summary == summary and state.last_status == status:
            return None
        state.last_summary = summary
        state.last_status = status
        request_id, turn_id = identity
        return {
            "progressSchema": self._progress_schema,
            "eventId": f"commentary-{turn_id}-{state.sequence}",
            "type": "assistant.commentary",
            "status": status,
            "stage": state.stage,
            "summary": summary,
            "source": "provider",
            "providerSequence": state.sequence,
            "providerEventAt": state.provider_event_at,
            "requestId": request_id,
            "turnId": turn_id,
        }

    def accept_delta(
        self,
        visible: str,
        *,
        started_turn: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Buffer an undecided delta or extend a confirmed tool preamble."""

        if not visible:
            return None
        if self._active is None:
            self._pending_chunks.append(visible)
            return None
        if self._active.last_status == "completed":
            return None
        self._active.chunks.append(visible)
        return self._project(
            self._active,
            status="running",
            started_turn=started_turn,
        )

    def confirm_tool_generation(
        self,
        *,
        source: Any,
        provider_sequence: Any,
        provider_event_at: Any,
        stage: str,
        started_turn: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Promote provisional bytes only after a real Provider tool signal."""

        if source != "provider" or self._turn_identity(started_turn) is None:
            return None
        sequence = self._sequence_for(provider_sequence)
        if self._active is not None:
            return None
        self._active = _ActiveCommentary(
            sequence=sequence,
            provider_event_at=self._event_at(provider_event_at),
            stage=stage,
            chunks=list(self._pending_chunks),
        )
        self._pending_chunks.clear()
        return self._project(
            self._active,
            status="running",
            started_turn=started_turn,
        )

    def complete_tool_commentary(
        self,
        text: Any,
        *,
        source: Any,
        provider_sequence: Any,
        provider_event_at: Any,
        stage: str,
        started_turn: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Settle one structurally confirmed tool response's canonical text."""

        if source not in {None, "provider"}:
            return None
        if self._turn_identity(started_turn) is None:
            return None
        raw_summary = str(text or "")
        if not self._summary_builder(raw_summary):
            return None
        try:
            supplied_sequence = int(provider_sequence)
        except (TypeError, ValueError):
            supplied_sequence = 0
        if self._active is not None and (
            supplied_sequence <= 0
            or self._active.sequence == supplied_sequence
        ):
            state = self._active
        else:
            state = _ActiveCommentary(
                sequence=self._sequence_for(provider_sequence),
                provider_event_at=self._event_at(provider_event_at),
                stage=stage,
            )
            self._active = state
        state.chunks = [raw_summary]
        self._pending_chunks.clear()
        return self._project(
            state,
            status="completed",
            started_turn=started_turn,
        )

    def close_tool_response(
        self,
        *,
        started_turn: Optional[Mapping[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Close one tool-call response and discard it from final-answer bytes."""

        payload = None
        if self._active is not None:
            payload = self._project(
                self._active,
                status="completed",
                started_turn=started_turn,
            )
        self._pending_chunks.clear()
        self._active = None
        return payload

    def drain_final_chunks(self) -> list[str]:
        """Return and clear bytes from the final non-tool Provider response."""

        chunks = list(self._pending_chunks)
        self._pending_chunks.clear()
        return chunks

    def pending_final_chunks(self) -> list[str]:
        """Return a copy for final settlement validation without mutation."""

        return list(self._pending_chunks)
