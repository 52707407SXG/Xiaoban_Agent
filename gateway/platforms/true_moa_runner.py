"""Thin async entrypoint for normal and fixed true-MoA Agent runs."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from gateway.platforms.api_server import (
    _MYSTAND_STREAM_DELIVERY_ID_RE,
    _merge_temporal_context,
    _mystand_index_has_candidates,
    _mystand_tool_result_failed,
    _resolve_mystand_initial_tool_choice,
    _run_mystand_preexecuted_evidence,
    logger,
)
from gateway.platforms.true_moa_runner_workflow import (
    TrueMoARunRequest,
    TrueMoARunnerTraceState,
    run_agent_sync,
)
from xiaoban.trusted_runtime.protocol_contract import (
    MYSTAND_COMPLETION_PROTOCOL,
)


def _mystand_index_followup_tool(
    *,
    completion_protocol: str,
    fact_requirement: Optional[Dict[str, Any]],
    resource_index_required: bool,
    valid_tool_names: set[str],
) -> str:
    """Choose the post-index tool without weakening the dynamic v2 path."""
    if (
        completion_protocol == MYSTAND_COMPLETION_PROTOCOL
        and fact_requirement is None
    ):
        return "mystand_query" if "mystand_query" in valid_tool_names else ""
    if (
        resource_index_required
        and "mystand_authorization" in valid_tool_names
    ):
        return "mystand_authorization"
    return ""


class TrueMoARunnerMixin:
    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
        gateway_session_key: Optional[str] = None,
        request_headers: Any = None,
        async_delivery: bool = False,
        fact_requirement: Optional[Dict[str, Any]] = None,
        completion_protocol: str = "",
        completion_binding: Optional[Dict[str, Any]] = None,
        dynamic_evidence_required: bool = False,
        true_moa_snapshot: Any = None,
        paid_call_usage_callback=None,
    ) -> tuple:
        """Create an Agent and run one thread-isolated conversation."""

        # Preserve the historical api_server monkeypatch surface used by
        # deterministic gateway tests and downstream embedders.
        from gateway.platforms import api_server as _api_server

        resolve_mystand_initial_tool_choice = (
            _api_server._resolve_mystand_initial_tool_choice
        )
        run_mystand_preexecuted_evidence = (
            _api_server._run_mystand_preexecuted_evidence
        )
        loop = asyncio.get_running_loop()
        effective_system_prompt = _merge_temporal_context(
            ephemeral_system_prompt,
            headers=request_headers,
        )
        request_user_id = self._header_value(
            request_headers,
            "X-Xiaoban-User-Id",
        )
        request_message_id = self._header_value(
            request_headers,
            "X-Xiaoban-Message-Id",
        )
        raw_request_delivery_id = self._header_value(
            request_headers,
            "X-Xiaoban-Delivery-Id",
        )
        request_delivery_id = raw_request_delivery_id
        if not _MYSTAND_STREAM_DELIVERY_ID_RE.fullmatch(
            request_delivery_id
        ):
            request_delivery_id = ""
        enabled_toolsets_override = (
            self._toolsets_for_request_headers(request_headers)
        )
        mystand_request = enabled_toolsets_override is not None
        from xiaoban.trusted_runtime.paid_call_policy import (
            SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
        )

        billing_policy_revision = self._header_value(
            request_headers,
            SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER,
        )
        durable_paid_call = bool(
            mystand_request
            and true_moa_snapshot is None
            and (
                raw_request_delivery_id
                or billing_policy_revision
            )
        )
        memory_identity = (
            self._mystand_memory_identity(request_headers)
            if mystand_request
            else None
        )
        metadata_trace = None
        if memory_identity and memory_identity[0] and self._api_key:
            try:
                from xiaoban.observability.mystand_metadata import (
                    MystandMetadataTrace,
                )

                metadata_trace = MystandMetadataTrace(
                    secret=self._api_key,
                    site_id=memory_identity[0],
                    user_id=memory_identity[1],
                )
            except Exception:
                logger.warning(
                    "My Stand metadata trace unavailable",
                    exc_info=False,
                )

        tool_started_at: dict[str, float] = {}
        trace_state = TrueMoARunnerTraceState(
            evidence_followup={
                "agent": None,
                "resource_index_required": False,
            },
        )

        def _traced_tool_start(
            tool_call_id,
            function_name,
            function_args,
        ):
            trace_state.tool_count += 1
            if tool_call_id:
                tool_started_at[str(tool_call_id)] = time.monotonic()
            if metadata_trace is not None:
                metadata_trace.safe_emit(
                    "tool_started",
                    status="running",
                    tool_name=str(function_name or "unknown"),
                )
            if tool_start_callback is not None:
                tool_start_callback(
                    tool_call_id,
                    function_name,
                    function_args,
                )

        def _traced_tool_complete(
            tool_call_id,
            function_name,
            function_args,
            function_result,
        ):
            started = (
                tool_started_at.pop(str(tool_call_id), None)
                if tool_call_id
                else None
            )
            duration_ms = (
                max(0, int((time.monotonic() - started) * 1000))
                if started
                else 0
            )
            tool_failed = _mystand_tool_result_failed(
                function_name,
                function_result,
            )
            if metadata_trace is not None:
                metadata_trace.safe_emit(
                    "tool_completed",
                    status="failed" if tool_failed else "completed",
                    tool_name=str(function_name or "unknown"),
                    tool_duration_ms=duration_ms,
                    success=not tool_failed,
                )
            if tool_complete_callback is not None:
                tool_complete_callback(
                    tool_call_id,
                    function_name,
                    function_args,
                    function_result,
                )
            if (
                function_name == "mystand_resource_index"
                and _mystand_index_has_candidates(function_result)
            ):
                evidence_agent = trace_state.evidence_followup["agent"]
                if evidence_agent is not None:
                    followup_tool = _mystand_index_followup_tool(
                        completion_protocol=completion_protocol,
                        fact_requirement=fact_requirement,
                        resource_index_required=trace_state.evidence_followup[
                            "resource_index_required"
                        ],
                        valid_tool_names=set(
                            evidence_agent.valid_tool_names
                        ),
                    )
                    if followup_tool:
                        evidence_agent._ephemeral_tool_choice = followup_tool

        run_request = TrueMoARunRequest(
            adapter=self,
            user_message=user_message,
            conversation_history=conversation_history,
            effective_system_prompt=effective_system_prompt,
            session_id=session_id,
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            traced_tool_start=_traced_tool_start,
            traced_tool_complete=_traced_tool_complete,
            agent_ref=agent_ref,
            gateway_session_key=gateway_session_key,
            request_headers=request_headers,
            async_delivery=async_delivery,
            fact_requirement=fact_requirement,
            completion_protocol=completion_protocol,
            completion_binding=completion_binding or {},
            dynamic_evidence_required=dynamic_evidence_required,
            true_moa_snapshot=true_moa_snapshot,
            paid_call_usage_callback=paid_call_usage_callback,
            request_user_id=request_user_id,
            request_message_id=request_message_id,
            request_delivery_id=request_delivery_id,
            enabled_toolsets_override=enabled_toolsets_override,
            mystand_request=mystand_request,
            durable_paid_call=durable_paid_call,
            memory_identity=memory_identity,
            metadata_trace=metadata_trace,
            trace_state=trace_state,
            resolve_mystand_initial_tool_choice=(
                resolve_mystand_initial_tool_choice
            ),
            run_mystand_preexecuted_evidence=(
                run_mystand_preexecuted_evidence
            ),
        )
        self._inflight_agent_runs += 1
        try:
            return await loop.run_in_executor(
                None,
                run_agent_sync,
                run_request,
            )
        finally:
            self._inflight_agent_runs -= 1


__all__ = [
    "TrueMoARunnerMixin",
    "_resolve_mystand_initial_tool_choice",
    "_run_mystand_preexecuted_evidence",
]
