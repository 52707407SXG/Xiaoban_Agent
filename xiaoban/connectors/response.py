"""Connector response shaping.

Production connectors should not return runtime internals, prompt context, or
tool outputs directly. Debug mode is explicit and local/dev only.
"""

from __future__ import annotations

from typing import Any, Literal

from .events import DurableReceiveResult

ResponseMode = Literal["production", "debug"]


def build_connector_response(
    receive_result: DurableReceiveResult,
    *,
    mode: ResponseMode = "production",
    runtime_result: dict[str, Any] | None = None,
    envelope: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    base = {
        "ok": error is None and receive_result.accepted,
        "status": receive_result.status,
        "deliveryKey": receive_result.delivery_key.stable_key,
    }
    if error:
        base["error"] = error
    if mode == "debug":
        base["runtimeResult"] = runtime_result or {}
        base["envelope"] = envelope or {}
    return base

