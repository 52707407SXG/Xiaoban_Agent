"""Cancellation and final-commit fences for one true-MoA request."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

class TrueMoACancelController:
    """Shared cancellation and terminal fence for one true-MoA request."""

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._state = "running"
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._dispatch_keys: set[str] = set()
        self._reserved_final_commit_key: str | None = None
        self._reserved_final_commit_thread_id: int | None = None
        self._reserved_final_commit_deadline: float | None = None

    @property
    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def register_cancel_callback(self, key: str, callback: Callable[[], None]) -> None:
        call_now = False
        with self._lock:
            if self._event.is_set():
                call_now = True
            else:
                self._callbacks[key] = callback
        if call_now:
            _dispatch_cancel_callback_async(callback)

    def unregister_cancel_callback(self, key: str) -> None:
        with self._lock:
            self._callbacks.pop(key, None)

    def try_begin_dispatch(self, key: str) -> bool:
        """Atomically linearize one downstream dispatch against cancellation.

        A caller that wins this gate is considered already dispatched; a
        concurrent ``cancel`` that wins first makes the gate return ``False``.
        Keys are one-shot, preventing a retry from reusing the same paid/tool
        slot outside the fixed ledger.
        """

        dispatch_key = str(key or "").strip()
        if not dispatch_key:
            return False
        with self._lock:
            if self._state != "running" or dispatch_key in self._dispatch_keys:
                return False
            self._dispatch_keys.add(dispatch_key)
            return True

    def cancel(self) -> bool:
        return self._terminate("cancelled")

    def fail(self) -> bool:
        return self._terminate("failed")

    def complete(self) -> bool:
        with self._lock:
            if self._state == "completed":
                return True
            if self._state != "running":
                return False
            if self._reserved_final_commit_key is not None:
                return False
            self._state = "completed"
            return True

    def reserve_final_commit(
        self,
        key: str,
        *,
        deadline_monotonic: float | None = None,
    ) -> bool:
        """Reserve public completion for the current gateway thread.

        The final executor runs in a daemon worker so the gateway can enforce
        a hard total deadline even when a provider ignores interruption.  That
        worker may stage a candidate response, but it must never make the
        request publicly ``completed`` before the gateway has received the
        complete payload and applied CompletionGuard. Reserving the one-shot
        key, current thread identity, and optional monotonic deadline makes
        that hand-off the only legal final commit point.
        """

        commit_key = str(key or "").strip()
        if not commit_key:
            return False
        deadline = None
        if deadline_monotonic is not None:
            try:
                deadline = float(deadline_monotonic)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(deadline) or deadline <= 0:
                return False
        current_thread_id = threading.get_ident()
        with self._lock:
            if self._state != "running":
                return False
            if self._reserved_final_commit_key is None:
                self._reserved_final_commit_key = commit_key
                self._reserved_final_commit_thread_id = current_thread_id
                self._reserved_final_commit_deadline = deadline
                return True
            return (
                self._reserved_final_commit_key == commit_key
                and self._reserved_final_commit_thread_id == current_thread_id
                and self._reserved_final_commit_deadline == deadline
            )

    def try_commit_final(self, key: str) -> bool:
        """Atomically commit the final response against a concurrent stop.

        This is the user-visible terminal linearization point. If stop or the
        reserved deadline wins first, no response may be appended or persisted.
        If this commit wins, a later stop observes a completed request and must
        not create a stop tombstone that rewrites the completed result.
        """

        commit_key = str(key or "").strip()
        if not commit_key:
            return False
        with self._lock:
            if self._state != "running" or commit_key in self._dispatch_keys:
                return False
            if self._reserved_final_commit_key is not None and (
                commit_key != self._reserved_final_commit_key
                or threading.get_ident()
                != self._reserved_final_commit_thread_id
            ):
                return False
            if (
                self._reserved_final_commit_deadline is not None
                and time.monotonic()
                >= self._reserved_final_commit_deadline
            ):
                return False
            self._dispatch_keys.add(commit_key)
            self._state = "completed"
            self._callbacks.clear()
            return True

    def _terminate(self, state: str) -> bool:
        callbacks: list[Callable[[], None]] = []
        with self._lock:
            if self._state != "running":
                return False
            self._state = state
            self._event.set()
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            _dispatch_cancel_callback_async(callback)
        return True


def _call_cancel_callback(callback: Callable[[], None]) -> None:
    try:
        callback()
    except BaseException:
        # Cancellation is best effort; the terminal fence remains authoritative.
        pass


def _dispatch_cancel_callback_async(callback: Callable[[], None]) -> None:
    """Run one best-effort transport abort outside the terminal owner thread."""

    threading.Thread(
        target=_call_cancel_callback,
        args=(callback,),
        name="xiaoban-true-moa-cancel-callback",
        daemon=True,
    ).start()
