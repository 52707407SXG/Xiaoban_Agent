"""Process-local replay cache backed by the durable true-MoA fence."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import threading
import time
from typing import Any, Dict, Mapping, Optional

from gateway.platforms.true_moa_stop_projection import (
    IdempotencyConflictError,
    _cancel_chat_agent_ref,
    _stopped_chat_completion_response,
)

class _IdempotencyCache:
    """Fast in-process replay cache backed by a plaintext-free durable fence."""

    def __init__(
        self,
        max_items: int = 1000,
        ttl_seconds: int = 300,
        durable_path: str = "",
        *,
        outcome_keys: Optional[Mapping[str, bytes]] = None,
        active_outcome_key_id: str = "",
        outcome_ttl_seconds: Optional[int] = None,
    ):
        from collections import OrderedDict
        self._store = OrderedDict()
        self._inflight: Dict[tuple[str, str], "asyncio.Task[Any]"] = {}
        self._agent_refs: Dict[tuple[str, str], list] = {}
        self._stopped: Dict[str, float] = {}
        self._usage_drains: set[str] = set()
        self._closed_usage_drain_owners: Dict[str, float] = {}
        self._usage_drains_lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max = max_items
        self._durable = None
        self._durable_error = None
        if str(durable_path or "").strip():
            try:
                from xiaoban.trusted_runtime.true_moa_durable import (
                    TrueMoADurableStore,
                )

                self._durable = TrueMoADurableStore(
                    str(durable_path).strip(),
                    outcome_keys=outcome_keys,
                    active_outcome_key_id=(
                        str(active_outcome_key_id or "") or None
                    ),
                    outcome_ttl_seconds=outcome_ttl_seconds,
                )
            except Exception as exc:
                # Normal requests stay available.  A true-MoA request checks
                # durable_ready before any provider dispatch and receives a
                # stable fail-closed 503 without reflecting filesystem detail.
                self._durable_error = exc

    @property
    def durable_ready(self) -> bool:
        return self._durable is not None

    @property
    def outcome_ready(self) -> bool:
        return bool(
            self._durable is not None
            and self._durable.outcome_ready
        )

    @staticmethod
    def _durable_response(
        record: Dict[str, Any],
        *,
        outcome: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        usage_ledger = copy.deepcopy(record.get("usage"))
        if isinstance(usage_ledger, dict):
            is_true_moa = (
                usage_ledger.get("schema")
                == "mystand.true-moa.usage.v1"
            )
            if record.get("state") == "interrupted":
                usage_ledger["status"] = "failed"
                projection_timestamp = int(time.time() * 1000)
                for collection_name in ("slots", "calls"):
                    collection = usage_ledger.get(collection_name)
                    if not isinstance(collection, list):
                        continue
                    for receipt in collection:
                        if not isinstance(receipt, dict):
                            continue
                        if receipt.get("status") == "reserved":
                            receipt["status"] = "not_dispatched"
                            receipt["endedAtMs"] = max(
                                int(receipt.get("startedAtMs") or 0),
                                projection_timestamp,
                            )
                            receipt["errorCategory"] = (
                                "provider_dispatch_fence_closed"
                            )
                        elif receipt.get("status") == "running":
                            receipt["status"] = "failed"
                            receipt["endedAtMs"] = max(
                                int(receipt.get("startedAtMs") or 0),
                                projection_timestamp,
                            )
                            receipt["errorCategory"] = (
                                "agent_restart_outcome_unknown"
                            )
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            receipts = usage_ledger.get("calls")
            if not isinstance(receipts, list):
                receipts = usage_ledger.get("slots") or []
            for receipt in receipts:
                if not isinstance(receipt, dict):
                    continue
                input_tokens += max(0, int(receipt.get("inputTokens") or 0))
                output_tokens += max(0, int(receipt.get("outputTokens") or 0))
                total_tokens += max(0, int(receipt.get("totalTokens") or 0))
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                (
                    "true_moa"
                    if is_true_moa
                    else "agent_calls"
                ): usage_ledger,
            }
        else:
            usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
        if (
            isinstance(outcome, Mapping)
            and outcome.get("completed") is True
            and isinstance(outcome.get("finalResponse"), str)
            and isinstance(outcome.get("outputDigest"), str)
        ):
            result: Dict[str, Any] = {
                "final_response": outcome["finalResponse"],
                "messages": [],
                "completed": True,
                "failed": False,
                "_mystand_request": True,
                "_mystand_egress_finalized": True,
                "_mystand_egress_output_digest": outcome["outputDigest"],
                "_true_moa_outcome_id": outcome.get("outcomeId"),
                **(
                    {"_true_moa_usage": usage_ledger}
                    if isinstance(usage_ledger, dict)
                    else {}
                ),
            }
            verification = outcome.get("trustedVerification")
            if isinstance(verification, Mapping):
                result["_mystand_trusted_verification"] = dict(
                    verification
                )
            completion_protocol = outcome.get("completionProtocol")
            if completion_protocol == "dynamic-evidence-v2":
                result["_mystand_completion_protocol"] = (
                    completion_protocol
                )
            return result, usage
        return (
            {
                "final_response": "",
                "messages": [],
                "completed": False,
                "failed": True,
                "interrupted": True,
                "error": "durable completion replay has no public output",
                **(
                    {
                        (
                            "_true_moa_usage"
                            if usage_ledger.get("schema")
                            == "mystand.true-moa.usage.v1"
                            else "_agent_call_usage"
                        ): usage_ledger
                    }
                    if isinstance(usage_ledger, dict)
                    else {}
                ),
            },
            usage,
        )

    def _purge(self):
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v["ts"] > self._ttl]
        for k in expired:
            self._store.pop(k, None)
        while len(self._store) > self._max:
            self._store.popitem(last=False)
        self._stopped = {
            key: expires_at
            for key, expires_at in self._stopped.items()
            if expires_at > now
        }
        while len(self._stopped) > self._max:
            oldest = min(self._stopped, key=self._stopped.get)
            self._stopped.pop(oldest, None)
        with self._usage_drains_lock:
            self._closed_usage_drain_owners = {
                key: expires_at
                for key, expires_at
                in self._closed_usage_drain_owners.items()
                if expires_at > now
            }
            while len(self._closed_usage_drain_owners) > self._max:
                oldest = min(
                    self._closed_usage_drain_owners,
                    key=self._closed_usage_drain_owners.get,
                )
                self._closed_usage_drain_owners.pop(oldest, None)

    def lookup_state(
        self,
        key: str,
        fingerprint: str,
        *,
        durable: bool = False,
    ) -> str:
        """Return missing, reusable, or conflict for a scoped request key."""
        self._purge()
        item = self._store.get(key)
        if item:
            return "reusable" if item["fp"] == fingerprint else "conflict"
        for (existing_key, existing_fingerprint), task in self._inflight.items():
            if task.done():
                continue
            if existing_key == key:
                return "reusable" if existing_fingerprint == fingerprint else "conflict"
        if durable and self._durable is not None:
            durable_record = self._durable.get(key)
            if durable_record is not None:
                return (
                    "reusable"
                    if durable_record["fingerprint"] in {"", fingerprint}
                    else "conflict"
                )
        return "missing"

    def claim(
        self,
        key: str,
        fingerprint: str,
        *,
        durable: bool = False,
    ) -> str:
        """Bind a stable identity before work starts, returning its prior state."""
        state = self.lookup_state(key, fingerprint, durable=durable)
        if state == "missing":
            self._store[key] = {
                "resp": None,
                "fp": fingerprint,
                "ts": time.time(),
            }
            if durable and self._durable is not None:
                durable_state = self._durable.claim(
                    key,
                    fingerprint,
                    kind="binding",
                )
                if durable_state == "conflict":
                    self._store.pop(key, None)
                    return "conflict"
            self._purge()
        return state

    @staticmethod
    def _completed_outcome_payload(
        result: Any,
    ) -> Dict[str, Any]:
        from xiaoban.trusted_runtime.true_moa_durable import (
            TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
        )

        if (
            not isinstance(result, dict)
            or result.get("completed") is not True
            or result.get("failed")
            or result.get("partial")
            or result.get("interrupted")
            or result.get("_mystand_egress_finalized") is not True
        ):
            raise RuntimeError(
                "true MoA completed outcome was not finalized"
            )
        final_response = result.get("final_response")
        output_digest = result.get("_mystand_egress_output_digest")
        if (
            not isinstance(final_response, str)
            or not isinstance(output_digest, str)
            or hashlib.sha256(final_response.encode("utf-8")).hexdigest()
            != output_digest
        ):
            raise RuntimeError(
                "true MoA finalized outcome digest mismatch"
            )
        payload: Dict[str, Any] = {
            "schema": TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
            "completed": True,
            "finalResponse": final_response,
            "outputDigest": output_digest,
            "factGuardRequired": isinstance(
                result.get("_mystand_fact_requirement"),
                dict,
            ),
        }
        verification = result.get("_mystand_trusted_verification")
        completion_protocol = str(
            result.get("_mystand_completion_protocol") or ""
        )
        verification_is_v2 = bool(
            isinstance(verification, dict)
            and verification.get("schema")
            == "mystand.xiaoban-completion-verification.v2"
        )
        trusted_turn = result.get("_trusted_turn")
        dynamic_action_ids = {
            str(getattr(item, "action_id", "") or "")
            for item in (
                list(getattr(trusted_turn, "action_calls", None) or [])
                + list(getattr(trusted_turn, "action_results", None) or [])
            )
        }
        dynamic_actions_used = bool(
            dynamic_action_ids.intersection(
                {
                    "mystand_resource_index",
                    "mystand_query",
                    "mystand_authorization",
                }
            )
        )
        if verification_is_v2:
            if completion_protocol != "dynamic-evidence-v2":
                raise RuntimeError(
                    "true MoA dynamic completion protocol is missing"
                )
            payload["completionProtocol"] = completion_protocol
        elif completion_protocol not in {"", "dynamic-evidence-v2"}:
            raise RuntimeError(
                "true MoA dynamic completion protocol is invalid"
            )
        elif completion_protocol and dynamic_actions_used:
            raise RuntimeError(
                "true MoA dynamic completion receipt is invalid"
            )
        if isinstance(verification, dict):
            payload["trustedVerification"] = dict(verification)
        return payload

    async def get_or_set(
        self,
        key: str,
        fingerprint: str,
        compute_coro,
        agent_ref=None,
        *,
        durable: bool = False,
        outcome_binding: Optional[Mapping[str, Any]] = None,
    ):
        state = self.lookup_state(key, fingerprint, durable=durable)
        if state == "conflict":
            raise IdempotencyConflictError("idempotency key was reused with a different request")
        item = self._store.get(key)
        if item and item.get("resp") is not None:
            return item["resp"]
        if key in self._stopped and agent_ref is not None:
            while len(agent_ref) < 2:
                agent_ref.append(False)
            agent_ref[1] = True

        inflight_key = (key, fingerprint)
        task = self._inflight.get(inflight_key)
        if task is not None:
            return await asyncio.shield(task)
        durable_store = self._durable if durable else None
        if durable_store is not None:
            durable_record = durable_store.get(key)
            if durable_record is not None:
                if durable_record["fingerprint"] not in {"", fingerprint}:
                    raise IdempotencyConflictError(
                        "idempotency key was reused with a different request"
                    )
                if not durable_record["fingerprint"]:
                    durable_state = durable_store.claim(
                        key,
                        fingerprint,
                        kind="execution",
                    )
                    if durable_state == "conflict":
                        raise IdempotencyConflictError(
                            "idempotency key was reused with a different request"
                        )
                    durable_record = durable_store.get(key) or durable_record
                if (
                    durable_record.get("state")
                    in {"interrupted", "stopped"}
                    or (
                        durable_record.get("state") == "failed"
                        and self._has_running_usage_receipt(
                            durable_record
                        )
                    )
                ):
                    durable_record = (
                        self.terminalize_orphaned_usage(key)
                        or durable_record
                    )
                outcome = None
                if (
                    durable_record.get("state") == "completed"
                    and outcome_binding is not None
                ):
                    try:
                        outcome = durable_store.recover_completed_outcome(
                            key,
                            binding=outcome_binding,
                        )
                    except Exception as exc:
                        from xiaoban.trusted_runtime.true_moa_durable import (
                            TrueMoAOutcomeUnavailableError,
                        )

                        if not isinstance(
                            exc,
                            TrueMoAOutcomeUnavailableError,
                        ):
                            raise
                return self._durable_response(
                    durable_record,
                    outcome=outcome,
                )
            durable_state = durable_store.claim(
                key,
                fingerprint,
                kind="execution",
            )
            if durable_state != "missing":
                durable_record = durable_store.get(key)
                if durable_record is not None:
                    if (
                        durable_record.get("state")
                        in {"interrupted", "stopped"}
                        or (
                            durable_record.get("state") == "failed"
                            and self._has_running_usage_receipt(
                                durable_record
                            )
                        )
                    ):
                        durable_record = (
                            self.terminalize_orphaned_usage(key)
                            or durable_record
                        )
                    outcome = None
                    if (
                        durable_record.get("state") == "completed"
                        and outcome_binding is not None
                    ):
                        try:
                            outcome = durable_store.recover_completed_outcome(
                                key,
                                binding=outcome_binding,
                            )
                        except Exception as exc:
                            from xiaoban.trusted_runtime.true_moa_durable import (
                                TrueMoAOutcomeUnavailableError,
                            )

                            if not isinstance(
                                exc,
                                TrueMoAOutcomeUnavailableError,
                            ):
                                raise
                    return self._durable_response(
                        durable_record,
                        outcome=outcome,
                    )
        if task is None:
            with self._usage_drains_lock:
                self._closed_usage_drain_owners.pop(key, None)

            async def _compute_and_store():
                if durable_store is not None:
                    durable_store.set_state(key, state="running")
                try:
                    resp = await compute_coro()
                    if key in self._stopped:
                        # Keep only the plaintext-free stop projection.  The
                        # tombstone still wins delivery, while actual usage
                        # remains recoverable for settlement.
                        resp = _stopped_chat_completion_response(resp)
                    raw_result, raw_usage = (
                        resp
                        if isinstance(resp, tuple) and len(resp) == 2
                        else (resp, {})
                    )
                    ledger = (
                        raw_usage.get("true_moa")
                        if isinstance(raw_usage, dict)
                        else None
                    )
                    if (
                        not isinstance(ledger, dict)
                        and isinstance(raw_usage, dict)
                    ):
                        ledger = raw_usage.get("agent_calls")
                    if (
                        not isinstance(ledger, dict)
                        and isinstance(raw_result, dict)
                    ):
                        ledger = (
                            raw_result.get("_true_moa_usage")
                            or raw_result.get("_agent_call_usage")
                        )
                    if durable_store is not None:
                        if isinstance(ledger, dict):
                            ledger_status = str(
                                ledger.get("status") or ""
                            )
                            is_true_moa = (
                                ledger.get("schema")
                                == "mystand.true-moa.usage.v1"
                            )
                            if (
                                ledger_status == "completed"
                                and key not in self._stopped
                                and is_true_moa
                            ):
                                if outcome_binding is None:
                                    raise RuntimeError(
                                        "true MoA outcome binding is required"
                                    )
                                outcome_id = (
                                    durable_store.save_completed_outcome(
                                        key,
                                        fingerprint,
                                        ledger,
                                        self._completed_outcome_payload(
                                            raw_result
                                        ),
                                        binding=outcome_binding,
                                    )
                                )
                                if isinstance(raw_result, dict):
                                    raw_result[
                                        "_true_moa_outcome_id"
                                    ] = outcome_id
                            else:
                                durable_store.save_usage(
                                    key,
                                    fingerprint,
                                    ledger,
                                    state=(
                                        "stopped"
                                        if key in self._stopped
                                        or ledger_status == "cancelled"
                                        else "completed"
                                        if ledger_status == "completed"
                                        else "failed"
                                        if ledger_status == "failed"
                                        else "interrupted"
                                    ),
                                )
                        else:
                            raise RuntimeError(
                                "durable terminal usage ledger is required"
                            )
                    import time as _t

                    self._store[key] = {
                        "resp": resp,
                        "fp": fingerprint,
                        "ts": _t.time(),
                    }
                    self._purge()
                    return resp
                except BaseException:
                    if durable_store is not None:
                        try:
                            durable_store.set_state(
                                key,
                                state="interrupted",
                            )
                        except Exception:
                            pass
                    raise

            task = asyncio.create_task(_compute_and_store())
            self._inflight[inflight_key] = task
            if agent_ref is not None:
                self._agent_refs[inflight_key] = agent_ref

            def _clear_inflight(done_task: "asyncio.Task[Any]") -> None:
                if self._inflight.get(inflight_key) is done_task:
                    agent_ref = self._agent_refs.get(inflight_key)
                    self._inflight.pop(inflight_key, None)
                    if self._local_usage_owner_finished(agent_ref):
                        with self._usage_drains_lock:
                            self._closed_usage_drain_owners[key] = (
                                time.time() + self._ttl
                            )
                            self._usage_drains.discard(key)
                    self._agent_refs.pop(inflight_key, None)

            task.add_done_callback(_clear_inflight)

        return await asyncio.shield(task)

    def stop(self, key: str, *, durable: bool = False) -> bool:
        """Interrupt the one in-flight computation for a scoped key."""
        self._purge()
        if key in self._store:
            return False
        matches = [item for item in self._inflight if item[0] == key]
        if durable:
            # The SQLite transition is the true-MoA stop linearization point.
            # A completed/failed terminal state that committed first must leave
            # the live controller and provider untouched.  Once the durable
            # stop wins, install the process-local fence before any cancellation
            # callback can observe or return a late result.
            if self._durable is None or not self._durable.mark_stopped(key):
                return False
            self._stopped[key] = time.time() + self._ttl
            for inflight_key in matches:
                agent_ref = self._agent_refs.get(inflight_key)
                if agent_ref is not None:
                    _cancel_chat_agent_ref(
                        agent_ref,
                        "Stop requested via My Stand delivery",
                    )
            return True

        accepted = not matches
        for inflight_key in matches:
            agent_ref = self._agent_refs.get(inflight_key)
            if agent_ref is None:
                accepted = True
                continue
            accepted = _cancel_chat_agent_ref(
                agent_ref,
                "Stop requested via My Stand delivery",
            ) or accepted
        if not accepted:
            return False
        # Record a short-lived tombstone even before the completion request
        # reaches this process. This closes the create/stop HTTP race without
        # cancelling an unrelated account or attempt (the key is HMAC-scoped).
        self._stopped[key] = time.time() + self._ttl
        return True

    def is_stopped(self, key: str) -> bool:
        """Return True while a stop tombstone is active for a scoped key."""
        self._purge()
        return key in self._stopped

    def result_state(self, key: str) -> tuple[str, Any]:
        """Return missing/running/stopped/completed plus a cached response."""

        self._purge()
        if any(
            existing_key == key and not task.done()
            for (existing_key, _), task in self._inflight.items()
        ):
            return "running", None
        item = self._store.get(key)
        if key in self._stopped:
            return (
                "stopped",
                item.get("resp") if item is not None else None,
            )
        if item is not None and item.get("resp") is not None:
            return "completed", item["resp"]
        if self._durable is not None:
            durable = self._durable.get(key)
            if durable is not None:
                if durable["state"] in {"claimed", "running"}:
                    return "running", None
                if durable["state"] == "stopped" and durable.get("usage") is None:
                    return "stopped", None
                durable_response = self._durable_response(durable)
                return str(durable["state"]), durable_response
        return "missing", None

    def durable_record(self, key: str) -> Optional[Dict[str, Any]]:
        if self._durable is None:
            return None
        return self._durable.get(key)

    @staticmethod
    def _has_running_usage_receipt(record: Any) -> bool:
        usage = record.get("usage") if isinstance(record, dict) else None
        return bool(
            isinstance(usage, dict)
            and any(
                isinstance(call, dict)
                and call.get("status") in {"reserved", "running"}
                for call in (usage.get("calls") or ())
            )
        )

    def has_active_usage_drain(self, key: str) -> bool:
        with self._usage_drains_lock:
            return key in self._usage_drains

    def has_active_execution(self, key: str) -> bool:
        return any(
            existing_key == key and not task.done()
            for (existing_key, _), task in self._inflight.items()
        )

    @staticmethod
    def _local_usage_owner_finished(agent_ref: Any) -> bool:
        """Return true only when an attached local ledger has no live call."""

        agent = (
            agent_ref[0]
            if isinstance(agent_ref, (list, tuple)) and agent_ref
            else None
        )
        if agent is None:
            return False
        ledgers = []
        seen_ids: set[int] = set()
        for attribute in (
            "_paid_call_usage_ledger",
            "_true_moa_usage_ledger",
        ):
            ledger = getattr(agent, attribute, None)
            if ledger is None or id(ledger) in seen_ids:
                continue
            seen_ids.add(id(ledger))
            ledgers.append(ledger)
        if not ledgers:
            return False
        for ledger in ledgers:
            to_dict = getattr(ledger, "to_dict", None)
            if not callable(to_dict):
                return False
            try:
                snapshot = to_dict()
            except BaseException:
                return False
            calls = (
                snapshot.get("calls")
                if isinstance(snapshot, dict)
                else None
            )
            if not isinstance(calls, list):
                return False
            if any(
                isinstance(call, dict)
                and call.get("status") in {"reserved", "running"}
                for call in calls
            ):
                return False
        return True

    def _sync_usage_drain_owner(self, key: str) -> None:
        if self._durable is None:
            return
        has_running_receipt = self._has_running_usage_receipt(
            self._durable.get(key),
        )
        with self._usage_drains_lock:
            if key in self._closed_usage_drain_owners:
                self._usage_drains.discard(key)
            elif has_running_receipt:
                self._usage_drains.add(key)
            else:
                self._usage_drains.discard(key)

    def terminalize_orphaned_stopped_usage(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Compatibility wrapper for the general orphan recovery fence."""

        return self.terminalize_orphaned_usage(key)

    def terminalize_orphaned_usage(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Close durable active receipts only when this process has no owner."""

        if self._durable is None:
            return None
        with self._usage_drains_lock:
            if key in self._usage_drains:
                return self._durable.get(key)
        if self.has_active_execution(key):
            return self._durable.get(key)
        self._durable.terminalize_orphaned_running_calls(key)
        return self._durable.get(key)

    def recover_outcome(
        self,
        key: str,
        *,
        binding: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if self._durable is None:
            raise RuntimeError("true MoA durable ledger is unavailable")
        return self._durable.recover_completed_outcome(
            key,
            binding=binding,
        )

    def acknowledge_outcome(
        self,
        key: str,
        *,
        binding: Mapping[str, Any],
        outcome_id: str,
    ) -> str:
        if self._durable is None:
            raise RuntimeError("true MoA durable ledger is unavailable")
        return self._durable.acknowledge_completed_outcome(
            key,
            binding=binding,
            outcome_id=outcome_id,
        )

    def persist_usage(
        self,
        key: str,
        fingerprint: str,
        ledger: Dict[str, Any],
    ) -> None:
        if self._durable is None:
            raise RuntimeError("true MoA durable ledger is unavailable")
        state = str(ledger.get("status") or "running")
        is_true_moa = (
            ledger.get("schema") == "mystand.true-moa.usage.v1"
        )
        durable_state = (
            "stopped"
            if state == "cancelled"
            else "completed"
            if state == "completed" and not is_true_moa
            else "failed"
            if state == "failed"
            else "running"
        )
        incoming_has_running_receipt = self._has_running_usage_receipt(
            {"usage": ledger},
        )
        with self._usage_drains_lock:
            was_active = key in self._usage_drains
            owner_is_closed = key in self._closed_usage_drain_owners
            if incoming_has_running_receipt and not owner_is_closed:
                # Register before SQLite exposes the running receipt.  A
                # concurrent recovery request can therefore never observe a
                # durable running call without its same-process owner.
                self._usage_drains.add(key)
        try:
            self._durable.save_usage(
                key,
                fingerprint,
                ledger,
                state=durable_state,
            )
        except BaseException:
            with self._usage_drains_lock:
                if key in self._closed_usage_drain_owners:
                    self._usage_drains.discard(key)
                elif was_active:
                    self._usage_drains.add(key)
                else:
                    self._usage_drains.discard(key)
            raise
        # Provider workers outlive the outer asyncio request after a stop.
        # The durable merge is authoritative here: a stale callback cannot
        # re-register ownership after a newer terminal receipt has landed.
        self._sync_usage_drain_owner(key)
