"""HTTP stop, terminal usage recovery, and sealed-outcome ACK handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from xiaoban.trusted_runtime.protocol_contract import (
    MYSTAND_TRUE_MOA_USAGE_SCHEMA,
    TrustedRuntimeContractError,
    validate_trusted_runtime_contract_headers,
)

if TYPE_CHECKING:
    from aiohttp import web


class TrueMoAHttpHandlersMixin:
    @staticmethod
    def _trusted_runtime_contract_error(request, *, web, error_response):
        try:
            validate_trusted_runtime_contract_headers(request.headers)
        except TrustedRuntimeContractError:
            return web.json_response(
                error_response(
                    "My Stand and Xiaoban trusted runtime contracts do not match",
                    code=TrustedRuntimeContractError.code,
                ),
                status=409,
            )
        return None

    async def _handle_stop_idempotent_chat_completion(self, request: "web.Request") -> "web.Response":
        """Stop one trusted My Stand non-stream completion by delivery key."""
        from gateway.platforms.api_server import (
            InvalidToolsetPolicy,
            _idem_cache,
            _openai_error,
            web,
        )
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        policy_err = self._request_toolset_policy_error(request.headers)
        if policy_err is not None:
            return policy_err
        if not self._header_present(request.headers, "X-Xiaoban-Toolset-Policy"):
            return web.json_response(
                _openai_error("My Stand tool policy is required", code="mystand_policy_required"),
                status=403,
            )
        if not self._api_key:
            return web.json_response(
                _openai_error(
                    "My Stand requests require configured API authentication",
                    code="mystand_auth_unavailable",
                ),
                status=503,
            )
        contract_error = self._trusted_runtime_contract_error(
            request,
            web=web,
            error_response=_openai_error,
        )
        if contract_error is not None:
            return contract_error
        true_moa_snapshot, true_moa_error = self._true_moa_snapshot_error(
            request.headers,
            mystand_request=True,
            api_authenticated=True,
        )
        if true_moa_error is not None:
            return true_moa_error
        if not _idem_cache.durable_ready:
            return web.json_response(
                _openai_error(
                    (
                        "True MoA durable idempotency ledger is unavailable"
                        if true_moa_snapshot is not None
                        else "My Stand provider-call ledger is unavailable"
                    ),
                    code=(
                        "true_moa_durable_ledger_unavailable"
                        if true_moa_snapshot is not None
                        else "mystand_durable_ledger_unavailable"
                    ),
                ),
                status=503,
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)
        if not isinstance(body, dict):
            return web.json_response(_openai_error("JSON body must be an object"), status=400)
        raw_key = str(body.get("idempotency_key") or request.headers.get("Idempotency-Key") or "").strip()
        try:
            scoped_key = self._scoped_idempotency_key(request.headers, raw_key)
        except InvalidToolsetPolicy as exc:
            return web.json_response(_openai_error(str(exc), code="invalid_idempotency_key"), status=400)
        if not _idem_cache.stop(
            scoped_key,
            durable=True,
        ):
            return web.json_response(
                _openai_error("No active completion for this delivery", code="completion_not_running"),
                status=404,
            )
        return web.json_response({"ok": True, "status": "stopping"}, status=202)

    async def _trusted_chat_control_context(self, request: "web.Request"):
        """Resolve one owner- and fingerprint-bound live chat control bridge."""

        from gateway.platforms.api_server import (
            InvalidToolsetPolicy,
            _CHAT_CONTROL_ID_RE,
            _MYSTAND_STREAM_FINGERPRINT_RE,
            _idem_cache,
            _openai_error,
            web,
        )

        auth_err = self._check_auth(request)
        if auth_err:
            return None, None, auth_err
        policy_err = self._request_toolset_policy_error(request.headers)
        if policy_err is not None:
            return None, None, policy_err
        if not self._header_present(request.headers, "X-Xiaoban-Toolset-Policy"):
            return None, None, web.json_response(
                _openai_error(
                    "My Stand tool policy is required",
                    code="mystand_policy_required",
                ),
                status=403,
            )
        if not self._api_key:
            return None, None, web.json_response(
                _openai_error(
                    "My Stand requests require configured API authentication",
                    code="mystand_auth_unavailable",
                ),
                status=503,
            )
        contract_error = self._trusted_runtime_contract_error(
            request,
            web=web,
            error_response=_openai_error,
        )
        if contract_error is not None:
            return None, None, contract_error
        _snapshot, true_moa_error = self._true_moa_snapshot_error(
            request.headers,
            mystand_request=True,
            api_authenticated=True,
        )
        if true_moa_error is not None:
            return None, None, true_moa_error
        if not _idem_cache.durable_ready:
            return None, None, web.json_response(
                _openai_error(
                    "My Stand provider-call ledger is unavailable",
                    code="mystand_durable_ledger_unavailable",
                ),
                status=503,
            )
        try:
            body = await request.json()
        except Exception:
            return None, None, web.json_response(
                _openai_error("Invalid JSON"),
                status=400,
            )
        if not isinstance(body, dict):
            return None, None, web.json_response(
                _openai_error("JSON body must be an object"),
                status=400,
            )
        control_id = str(body.get("controlId") or "").strip()
        if not _CHAT_CONTROL_ID_RE.fullmatch(control_id):
            return None, None, web.json_response(
                _openai_error(
                    "A valid controlId is required",
                    code="invalid_control_id",
                ),
                status=400,
            )
        identity_error, delivery_id = self._stream_delivery_identity_error(
            request.headers
        )
        if identity_error is not None:
            return None, None, identity_error
        raw_key = str(body.get("idempotency_key") or "").strip()
        if not raw_key or raw_key != delivery_id:
            return None, None, web.json_response(
                _openai_error(
                    "Control delivery identity does not match the trusted headers",
                    code="invalid_control_delivery_identity",
                ),
                status=400,
            )
        request_fingerprint = self._header_value(
            request.headers,
            "X-Xiaoban-Request-Fingerprint",
        ).lower()
        if not _MYSTAND_STREAM_FINGERPRINT_RE.fullmatch(request_fingerprint):
            return None, None, web.json_response(
                _openai_error(
                    "A valid request fingerprint is required",
                    code="invalid_control_fingerprint",
                ),
                status=400,
            )
        try:
            scoped_key = self._scoped_idempotency_key(
                request.headers,
                raw_key,
            )
        except InvalidToolsetPolicy as exc:
            return None, None, web.json_response(
                _openai_error(str(exc), code="invalid_idempotency_key"),
                status=400,
            )
        agent_ref = _idem_cache.active_agent_ref(
            scoped_key,
            request_fingerprint,
        )
        bridge = (
            agent_ref[3]
            if isinstance(agent_ref, list) and len(agent_ref) > 3
            else None
        )
        if bridge is None:
            return None, None, web.json_response(
                _openai_error(
                    "No active completion for this control",
                    code="chat_control_not_active",
                ),
                status=409,
            )
        return body, bridge, None

    async def _handle_chat_completion_approval(
        self,
        request: "web.Request",
    ) -> "web.Response":
        """Resolve one exact approval in an active trusted chat stream."""

        from gateway.platforms.api_server import (
            _CHAT_APPROVAL_ID_RE,
            _ChatControlConflict,
            _openai_error,
            web,
        )

        body, bridge, error = await self._trusted_chat_control_context(request)
        if error is not None:
            return error
        approval_id = str(body.get("approvalId") or "").strip()
        choice = str(body.get("choice") or "").strip().lower()
        if not _CHAT_APPROVAL_ID_RE.fullmatch(approval_id):
            return web.json_response(
                _openai_error(
                    "A valid approvalId is required",
                    code="invalid_approval_id",
                ),
                status=400,
            )
        if choice not in {"once", "session", "deny"}:
            return web.json_response(
                _openai_error(
                    "Invalid approval choice; expected one of: once, session, deny",
                    code="invalid_approval_choice",
                ),
                status=400,
            )
        try:
            receipt = bridge.respond(
                control_id=str(body["controlId"]).strip(),
                approval_id=approval_id,
                choice=choice,
            )
        except _ChatControlConflict as exc:
            return web.json_response(
                _openai_error(str(exc), code=exc.code),
                status=409,
            )
        except Exception:
            return web.json_response(
                _openai_error(
                    "Chat approval control failed closed",
                    code="chat_control_failed",
                ),
                status=500,
            )
        return web.json_response(receipt, status=202)

    async def _handle_chat_completion_steer(
        self,
        request: "web.Request",
    ) -> "web.Response":
        """Inject a plaintext-free-receipt supplement into an active tool turn."""

        from gateway.platforms.api_server import (
            _CHAT_APPROVAL_ID_RE,
            _CHAT_STEER_MAX_BYTES,
            _CHAT_STEER_MAX_CHARS,
            _canonical_chat_steer_message,
            _ChatControlConflict,
            _openai_error,
            web,
        )

        body, bridge, error = await self._trusted_chat_control_context(request)
        if error is not None:
            return error
        message = body.get("message")
        cleaned_message = (
            _canonical_chat_steer_message(message)
            if isinstance(message, str)
            else ""
        )
        raw_approval_id = body.get("approvalId")
        approval_id = str(raw_approval_id or "").strip()
        if (
            not isinstance(message, str)
            or not cleaned_message
            or len(cleaned_message) > _CHAT_STEER_MAX_CHARS
            or len(cleaned_message.encode("utf-8")) > _CHAT_STEER_MAX_BYTES
        ):
            return web.json_response(
                _openai_error(
                    "Steer message must be non-empty and within the size limit",
                    code="invalid_steer_message",
                ),
                status=400,
            )
        if approval_id and not _CHAT_APPROVAL_ID_RE.fullmatch(approval_id):
            return web.json_response(
                _openai_error(
                    "approvalId is invalid",
                    code="invalid_approval_id",
                ),
                status=400,
            )
        try:
            receipt = bridge.steer(
                control_id=str(body["controlId"]).strip(),
                message=cleaned_message,
                approval_id=approval_id or None,
            )
        except _ChatControlConflict as exc:
            return web.json_response(
                _openai_error(str(exc), code=exc.code),
                status=409,
            )
        except Exception:
            return web.json_response(
                _openai_error(
                    "Chat steer control failed closed",
                    code="chat_control_failed",
                ),
                status=500,
            )
        return web.json_response(receipt, status=202)

    async def _handle_chat_completion_usage(self, request: "web.Request") -> "web.Response":
        """Recover the terminal true-MoA usage ledger after SSE cancellation."""
        from gateway.platforms.api_server import (
            InvalidToolsetPolicy,
            _idem_cache,
            _openai_error,
            web,
        )

        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        policy_err = self._request_toolset_policy_error(request.headers)
        if policy_err is not None:
            return policy_err
        if not self._header_present(request.headers, "X-Xiaoban-Toolset-Policy"):
            return web.json_response(
                _openai_error(
                    "My Stand tool policy is required",
                    code="mystand_policy_required",
                ),
                status=403,
            )
        if not self._api_key:
            return web.json_response(
                _openai_error(
                    "My Stand requests require configured API authentication",
                    code="mystand_auth_unavailable",
                ),
                status=503,
            )
        contract_error = self._trusted_runtime_contract_error(
            request,
            web=web,
            error_response=_openai_error,
        )
        if contract_error is not None:
            return contract_error
        true_moa_snapshot, true_moa_error = self._true_moa_snapshot_error(
            request.headers,
            mystand_request=True,
            api_authenticated=True,
        )
        if true_moa_error is not None:
            return true_moa_error
        if not _idem_cache.durable_ready:
            return web.json_response(
                _openai_error(
                    "True MoA durable idempotency ledger is unavailable",
                    code="true_moa_durable_ledger_unavailable",
                ),
                status=503,
            )
        try:
            body = await request.json()
        except Exception:
            return web.json_response(_openai_error("Invalid JSON"), status=400)
        if not isinstance(body, dict):
            return web.json_response(
                _openai_error("JSON body must be an object"),
                status=400,
            )
        raw_key = str(
            body.get("idempotency_key")
            or request.headers.get("Idempotency-Key")
            or ""
        ).strip()
        action = str(body.get("action") or "recover").strip().lower()
        if action not in {"recover", "ack"}:
            return web.json_response(
                _openai_error(
                    "Unsupported true MoA usage action",
                    code="invalid_true_moa_usage_action",
                ),
                status=400,
            )
        if action == "ack" and true_moa_snapshot is None:
            return web.json_response(
                _openai_error(
                    "ACK is available only for true MoA outcomes",
                    code="true_moa_snapshot_required",
                ),
                status=400,
            )
        try:
            scoped_key = self._scoped_idempotency_key(request.headers, raw_key)
            outcome_binding = (
                self._true_moa_outcome_binding(
                    request.headers,
                    snapshot=true_moa_snapshot,
                    delivery_id=raw_key,
                )
                if true_moa_snapshot is not None
                else None
            )
        except InvalidToolsetPolicy as exc:
            return web.json_response(
                _openai_error(
                    str(exc),
                    code="invalid_true_moa_outcome_binding",
                ),
                status=400,
            )
        record = _idem_cache.durable_record(scoped_key)
        if record is None:
            return web.json_response(
                _openai_error(
                    "No completion ledger for this delivery",
                    code="completion_ledger_not_found",
                ),
                status=404,
            )
        record_state = str(record.get("state") or "")
        if (
            record_state == "running"
            and _idem_cache._has_running_usage_receipt(record)
        ):
            # A replacement process can start while the dead owner's durable
            # lease is still valid.  The first poll must wait for that fence,
            # but every later poll retries the claim so expiry cannot strand a
            # running receipt forever.
            record = (
                _idem_cache.terminalize_orphaned_usage(scoped_key)
                or record
            )
            record_state = str(record.get("state") or "")
            if (
                record_state == "running"
                and _idem_cache._has_running_usage_receipt(record)
            ):
                return web.json_response(
                    {
                        "ok": True,
                        "status": "running_draining",
                        "final": False,
                        "usage": record.get("usage"),
                        "terminalState": "running",
                        "outcomeStatus": str(
                            record.get("outcomeState") or "none"
                        ),
                        "settlementBlocked": True,
                    },
                    status=202,
                )
        if record_state == "stopped":
            if (
                _idem_cache.has_active_usage_drain(scoped_key)
                and _idem_cache._has_running_usage_receipt(record)
            ):
                return web.json_response(
                    {
                        "ok": True,
                        "status": "stopped_draining",
                        "final": False,
                        "usage": record.get("usage"),
                        "terminalState": "stopped",
                        "outcomeStatus": str(
                            record.get("outcomeState") or "none"
                        ),
                        "settlementBlocked": True,
                    },
                    status=202,
                )
            # A replacement process has no provider worker that can finish a
            # pre-restart usage drain.  Atomically close only those orphaned
            # running receipts before projecting recovery; exact late usage
            # may still merge, but no call can remain running forever.
            record = (
                _idem_cache.terminalize_orphaned_stopped_usage(scoped_key)
                or record
            )
            if _idem_cache._has_running_usage_receipt(record):
                return web.json_response(
                    {
                        "ok": True,
                        "status": "stopped_draining",
                        "final": False,
                        "usage": record.get("usage"),
                        "terminalState": "stopped",
                        "outcomeStatus": str(
                            record.get("outcomeState") or "none"
                        ),
                        "settlementBlocked": True,
                    },
                    status=202,
                )
        elif record_state == "interrupted":
            record = (
                _idem_cache.terminalize_orphaned_usage(scoped_key)
                or record
            )
            if (
                str(record.get("state") or "") == "interrupted"
                and (
                    _idem_cache.has_active_execution(scoped_key)
                    or _idem_cache._has_running_usage_receipt(record)
                )
            ):
                return web.json_response(
                    {
                        "ok": True,
                        "status": "interrupted_draining",
                        "final": False,
                        "usage": record.get("usage"),
                        "terminalState": "interrupted",
                        "outcomeStatus": str(
                            record.get("outcomeState") or "none"
                        ),
                        "settlementBlocked": True,
                    },
                    status=202,
                )
        elif (
            record_state == "failed"
            and _idem_cache._has_running_usage_receipt(record)
        ):
            record = (
                _idem_cache.terminalize_orphaned_usage(scoped_key)
                or record
            )
            if _idem_cache._has_running_usage_receipt(record):
                return web.json_response(
                    {
                        "ok": True,
                        "status": "failed_draining",
                        "final": False,
                        "usage": record.get("usage"),
                        "terminalState": "failed",
                        "outcomeStatus": str(
                            record.get("outcomeState") or "none"
                        ),
                        "settlementBlocked": True,
                    },
                    status=202,
                )
        state = str(record.get("state") or "")
        usage = record.get("usage")
        is_true_moa = bool(
            isinstance(usage, dict)
            and usage.get("schema") == MYSTAND_TRUE_MOA_USAGE_SCHEMA
        )
        # A process can die after the completed-ledger callback is durably
        # persisted but before save_completed_outcome atomically seals the
        # user-visible result.  The durable execution state intentionally
        # stays provisional until that seal.  Do not report this crash gap as
        # "running" forever after restart: surface the completed usage below
        # as settlement-blocked with the missing-outcome error.
        provisional_completed_usage = bool(
            isinstance(usage, dict)
            and usage.get("status") == "completed"
        )
        if (
            state in {"claimed", "running"}
            and not provisional_completed_usage
        ):
            return web.json_response(
                {"ok": True, "status": "running", "final": False},
                status=202,
            )
        if state == "stopped" and not isinstance(usage, dict):
            return web.json_response({
                "ok": True,
                "status": "stopped_before_start",
                "final": True,
                "usage": None,
                "outcomeStatus": str(
                    record.get("outcomeState") or "none"
                ),
            })
        if not isinstance(usage, dict):
            return web.json_response(
                _openai_error(
                    "Completion has no true MoA usage ledger",
                    code="true_moa_usage_unavailable",
                ),
                status=409,
            )
        usage_settlement_blocked = any(
            not isinstance(call, dict)
            or call.get("status") == "running"
            or (
                call.get("startedAtMs") is not None
                and call.get("status")
                not in {"not_started", "not_dispatched"}
                and (
                    call.get("usageStatus") != "reported"
                    or not all(
                        isinstance(call.get(name), int)
                        and not isinstance(call.get(name), bool)
                        for name in (
                            "inputTokens",
                            "outputTokens",
                            "totalTokens",
                            "cachedInputTokens",
                        )
                    )
                )
            )
            for call in (usage.get("calls") or ())
        )
        outcome_state = str(record.get("outcomeState") or "none")

        from xiaoban.trusted_runtime.true_moa_durable import (
            TrueMoAOutcomeBindingError,
            TrueMoAOutcomeUnavailableError,
        )

        if action == "ack":
            outcome_id = body.get("outcome_id")
            if not isinstance(outcome_id, str):
                return web.json_response(
                    _openai_error(
                        "True MoA outcome id is required",
                        code="true_moa_outcome_id_required",
                    ),
                    status=400,
                )
            try:
                acknowledgment = _idem_cache.acknowledge_outcome(
                    scoped_key,
                    binding=outcome_binding,
                    outcome_id=outcome_id,
                )
            except TrueMoAOutcomeBindingError:
                return web.json_response(
                    _openai_error(
                        "True MoA outcome binding did not verify",
                        code="true_moa_outcome_binding_invalid",
                    ),
                    status=409,
                )
            except TrueMoAOutcomeUnavailableError:
                return web.json_response(
                    _openai_error(
                        "True MoA completed outcome is unavailable",
                        code="true_moa_outcome_unavailable",
                    ),
                    status=409,
                )
            return web.json_response({
                "ok": True,
                "status": acknowledgment,
                "final": True,
                "usage": usage,
                "terminalState": state,
                "outcomeStatus": "acked",
                "settlementBlocked": usage_settlement_blocked,
            })

        recovered_outcome = None
        outcome_unavailable = False
        if outcome_state == "sealed":
            try:
                recovered_outcome = _idem_cache.recover_outcome(
                    scoped_key,
                    binding=outcome_binding,
                )
            except TrueMoAOutcomeBindingError:
                return web.json_response(
                    _openai_error(
                        "True MoA outcome binding did not verify",
                        code="true_moa_outcome_binding_invalid",
                    ),
                    status=409,
                )
            except TrueMoAOutcomeUnavailableError:
                outcome_unavailable = True
        expects_outcome = bool(
            is_true_moa
            and state != "stopped"
            and (
                state == "completed"
                or usage.get("status") == "completed"
            )
        )
        outcome_settlement_blocked = bool(
            expects_outcome
            and outcome_state != "acked"
            and not isinstance(recovered_outcome, dict)
        )
        settlement_blocked = bool(
            usage_settlement_blocked or outcome_settlement_blocked
        )
        response_payload: Dict[str, Any] = {
            "ok": True,
            "status": str(
                "settlement_blocked"
                if settlement_blocked
                else "acknowledged"
                if outcome_state == "acked"
                else "cancelled"
                if state == "stopped"
                else usage.get("status")
                or state
            ),
            "final": True,
            "usage": usage,
            "terminalState": state,
            "outcomeStatus": (
                "unavailable"
                if outcome_unavailable
                else outcome_state
            ),
            "settlementBlocked": settlement_blocked,
        }
        if outcome_unavailable:
            response_payload["errorCode"] = (
                "true_moa_outcome_unavailable"
            )
        elif outcome_settlement_blocked:
            response_payload["errorCode"] = (
                "true_moa_outcome_unavailable"
            )
        if isinstance(recovered_outcome, dict):
            public_outcome: Dict[str, Any] = {
                "finalResponse": recovered_outcome["finalResponse"],
                "outputDigest": recovered_outcome["outputDigest"],
            }
            response_payload["outcome"] = public_outcome
            response_payload["outcomeId"] = recovered_outcome["outcomeId"]
            response_payload["retentionOverdue"] = bool(
                recovered_outcome.get("retentionOverdue")
            )
        return web.json_response(response_payload)
