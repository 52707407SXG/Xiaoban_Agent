"""Topology-neutral fixed policy for paid provider dispatches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from xiaoban_cli.model_normalize import normalize_model_for_provider
from xiaoban.trusted_runtime.protocol_contract import (
    MYSTAND_NORMAL_BILLING_POLICY_REVISION,
    MYSTAND_NORMAL_CALL_LIMIT,
    MYSTAND_NORMAL_INPUT_MAX_BYTES,
    MYSTAND_NORMAL_MODEL,
    MYSTAND_NORMAL_OUTPUT_MAX_TOKENS,
    MYSTAND_NORMAL_PROVIDER,
    MYSTAND_NORMAL_ROLE,
)


@dataclass(frozen=True)
class PaidCallBudget:
    """Provider payload and physical-call ceilings for one paid role."""

    policy_id: str
    input_max_bytes: int | None
    output_max_tokens: int
    call_limit: int


@dataclass(frozen=True)
class FixedPaidCallPolicy(PaidCallBudget):
    """One server-owned provider route plus its prepaid hard ceilings."""

    provider: str
    model: str
    role: str


class PaidCallPolicyError(RuntimeError):
    """A fixed paid-call policy was violated before provider dispatch."""

    def __init__(self, code: str):
        self.code = str(code or "paid_call_policy_invalid")
        super().__init__(self.code)


LEGACY_SIGNED_MYSTAND_AGENT_POLICY_REVISION = (
    "deepseek-v4-pro-20260729-v1"
)
LEGACY_SIGNED_MYSTAND_AGENT_POLICY = FixedPaidCallPolicy(
    policy_id="mystand.signed-normal-paid-call.v1",
    provider="deepseek",
    model="deepseek-v4-pro",
    role="agent",
    input_max_bytes=131072,
    output_max_tokens=4096,
    call_limit=8,
)
SIGNED_MYSTAND_AGENT_POLICY = FixedPaidCallPolicy(
    policy_id="mystand.signed-normal-paid-call.v2",
    provider=MYSTAND_NORMAL_PROVIDER,
    model=MYSTAND_NORMAL_MODEL,
    role=MYSTAND_NORMAL_ROLE,
    input_max_bytes=MYSTAND_NORMAL_INPUT_MAX_BYTES,
    output_max_tokens=MYSTAND_NORMAL_OUTPUT_MAX_TOKENS,
    call_limit=MYSTAND_NORMAL_CALL_LIMIT,
)
SIGNED_MYSTAND_AGENT_POLICY_REVISION = (
    MYSTAND_NORMAL_BILLING_POLICY_REVISION
)
SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER = (
    "X-Xiaoban-Agent-Billing-Policy-Revision"
)
SIGNED_MYSTAND_AGENT_POLICY_REGISTRY = MappingProxyType(
    {
        LEGACY_SIGNED_MYSTAND_AGENT_POLICY_REVISION: (
            LEGACY_SIGNED_MYSTAND_AGENT_POLICY
        ),
        SIGNED_MYSTAND_AGENT_POLICY_REVISION: (
            SIGNED_MYSTAND_AGENT_POLICY
        ),
    }
)


def resolve_signed_mystand_agent_policy(
    observed: Any,
) -> FixedPaidCallPolicy:
    """Resolve one immutable revision without rewriting older policies."""

    revision = str(observed or "").strip()
    policy = SIGNED_MYSTAND_AGENT_POLICY_REGISTRY.get(revision)
    if policy is None:
        raise PaidCallPolicyError(
            "signed_mystand_billing_policy_revision_mismatch"
        )
    return policy


def enforce_signed_mystand_policy_revision(observed: Any) -> str:
    """Bind the API reservation policy to this runtime's dispatch policy."""

    revision = str(observed or "").strip()
    resolve_signed_mystand_agent_policy(revision)
    return revision


def normalize_fixed_route(
    policy: FixedPaidCallPolicy,
    *,
    provider: Any,
    model: Any,
) -> tuple[str, str]:
    """Normalize an observed route against the policy's provider dialect."""

    return (
        str(provider or "").strip().lower(),
        normalize_model_for_provider(
            str(model or ""),
            policy.provider,
        ),
    )


def enforce_fixed_paid_call_route(
    policy: FixedPaidCallPolicy,
    *,
    provider: Any,
    model: Any,
    error_code: str = "paid_call_fixed_route_mismatch",
) -> tuple[str, str]:
    """Fail closed when runtime configuration drifts from a paid route."""

    normalized = normalize_fixed_route(
        policy,
        provider=provider,
        model=model,
    )
    if normalized != (policy.provider, policy.model):
        raise PaidCallPolicyError(error_code)
    return normalized


def enforce_paid_call_dispatch_budget(
    policy: PaidCallBudget,
    *,
    payload: Any,
    error_prefix: str = "paid_call_cost_cap",
) -> int:
    """Validate one serialized provider payload before physical dispatch."""

    if not isinstance(payload, Mapping):
        raise PaidCallPolicyError(f"{error_prefix}_input_payload_invalid")
    output_limits: list[int] = []
    for field in (
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
    ):
        if field not in payload:
            continue
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise PaidCallPolicyError(
                f"{error_prefix}_output_token_cap_invalid"
            )
        output_limits.append(value)
    if (
        not output_limits
        or len(set(output_limits)) != 1
        or output_limits[0] > policy.output_max_tokens
    ):
        raise PaidCallPolicyError(
            f"{error_prefix}_output_token_cap_exceeded"
        )
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PaidCallPolicyError(
            f"{error_prefix}_input_payload_invalid"
        ) from exc
    if (
        policy.input_max_bytes is not None
        and len(encoded) > policy.input_max_bytes
    ):
        raise PaidCallPolicyError(
            f"{error_prefix}_input_byte_cap_exceeded"
        )
    return len(encoded)


def serialize_openai_chat_request_body(payload: Any) -> bytes:
    """Build the exact JSON body produced by the installed OpenAI SDK.

    The strict My Stand route uses ``chat.completions.create``.  Reusing that
    generated SDK resource keeps request-option handling, typed transforms,
    ``extra_body`` flattening, and JSON encoding identical to the physical
    provider call without opening a network connection.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("OpenAI chat payload must be a mapping")

    from openai._base_client import _merge_mappings, openapi_dumps
    from openai.resources.chat.completions.completions import Completions

    class _CaptureClient:
        def __init__(self) -> None:
            self.request: tuple[tuple[Any, ...], dict[str, Any]] | None = None

        def post(self, *args: Any, **kwargs: Any) -> None:
            self.request = (args, kwargs)

        get = post
        patch = post
        put = post
        delete = post
        get_api_list = post

    capture = _CaptureClient()
    Completions(capture).create(**dict(payload))
    if capture.request is None:
        raise RuntimeError("OpenAI SDK did not build a chat request")
    _, request_kwargs = capture.request
    body = request_kwargs.get("body")
    options = request_kwargs.get("options") or {}
    extra_body = options.get("extra_json")
    if extra_body is not None:
        if body is None:
            body = extra_body
        elif isinstance(body, Mapping) and isinstance(extra_body, Mapping):
            body = _merge_mappings(body, extra_body)
        else:
            raise TypeError("OpenAI extra_body cannot be merged into request body")
    return openapi_dumps(body)


def serialize_openai_responses_request_body(payload: Any) -> bytes:
    """Build the exact JSON body produced by the installed Responses SDK."""

    if not isinstance(payload, Mapping):
        raise TypeError("OpenAI Responses payload must be a mapping")

    from openai._base_client import _merge_mappings, openapi_dumps
    from openai.resources.responses.responses import Responses

    class _CaptureClient:
        def __init__(self) -> None:
            self.request: tuple[tuple[Any, ...], dict[str, Any]] | None = None

        def post(self, *args: Any, **kwargs: Any) -> None:
            self.request = (args, kwargs)

        get = post
        patch = post
        put = post
        delete = post
        get_api_list = post

    capture = _CaptureClient()
    request_payload = dict(payload)
    request_payload["stream"] = True
    Responses(capture).create(**request_payload)
    if capture.request is None:
        raise RuntimeError("OpenAI SDK did not build a Responses request")
    _, request_kwargs = capture.request
    body = request_kwargs.get("body")
    options = request_kwargs.get("options") or {}
    extra_body = options.get("extra_json")
    if extra_body is not None:
        if body is None:
            body = extra_body
        elif isinstance(body, Mapping) and isinstance(extra_body, Mapping):
            body = _merge_mappings(body, extra_body)
        else:
            raise TypeError("OpenAI extra_body cannot be merged into request body")
    return openapi_dumps(body)


def enforce_openai_responses_paid_call_dispatch_budget(
    policy: PaidCallBudget,
    *,
    payload: Any,
    configured_output_max_tokens: Any,
    error_prefix: str = "paid_call_cost_cap",
) -> int:
    """Validate a Responses request, including Codex's logical output cap."""

    if not isinstance(payload, Mapping):
        raise PaidCallPolicyError(f"{error_prefix}_input_payload_invalid")
    if configured_output_max_tokens is not None and (
        isinstance(configured_output_max_tokens, bool)
        or not isinstance(configured_output_max_tokens, int)
        or configured_output_max_tokens <= 0
    ):
        raise PaidCallPolicyError(f"{error_prefix}_output_token_cap_invalid")
    if (
        configured_output_max_tokens is not None
        and configured_output_max_tokens > policy.output_max_tokens
    ):
        raise PaidCallPolicyError(f"{error_prefix}_output_token_cap_exceeded")
    extra_body = payload.get("extra_body")
    controlled_fields = frozenset(
        {
            "model",
            "input",
            "instructions",
            "max_output_tokens",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream",
        }
    )
    if isinstance(extra_body, Mapping) and controlled_fields.intersection(
        extra_body
    ):
        raise PaidCallPolicyError(
            f"{error_prefix}_protected_field_override"
        )
    try:
        encoded = serialize_openai_responses_request_body(payload)
        effective_payload = json.loads(encoded)
    except Exception as exc:
        raise PaidCallPolicyError(
            f"{error_prefix}_input_payload_invalid"
        ) from exc
    if not isinstance(effective_payload, Mapping):
        raise PaidCallPolicyError(f"{error_prefix}_input_payload_invalid")
    wire_output_limit = effective_payload.get("max_output_tokens")
    if (
        wire_output_limit is not None
        and wire_output_limit != configured_output_max_tokens
    ):
        raise PaidCallPolicyError(f"{error_prefix}_output_token_cap_exceeded")
    if isinstance(policy, FixedPaidCallPolicy):
        effective_model = normalize_model_for_provider(
            str(effective_payload.get("model") or ""),
            policy.provider,
        )
        if effective_model != policy.model:
            raise PaidCallPolicyError(
                f"{error_prefix}_fixed_route_mismatch"
            )
    if (
        policy.input_max_bytes is not None
        and len(encoded) > policy.input_max_bytes
    ):
        raise PaidCallPolicyError(
            f"{error_prefix}_input_byte_cap_exceeded"
        )
    return len(encoded)


def enforce_openai_chat_paid_call_dispatch_budget(
    policy: PaidCallBudget,
    *,
    payload: Any,
    error_prefix: str = "paid_call_cost_cap",
) -> int:
    """Validate a paid OpenAI-compatible chat request by its wire JSON body."""

    if not isinstance(payload, Mapping):
        raise PaidCallPolicyError(f"{error_prefix}_input_payload_invalid")
    extra_body = payload.get("extra_body")
    controlled_fields = frozenset(
        {
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream",
        }
    )
    if isinstance(extra_body, Mapping) and controlled_fields.intersection(
        extra_body
    ):
        raise PaidCallPolicyError(
            f"{error_prefix}_protected_field_override"
        )
    try:
        encoded = serialize_openai_chat_request_body(payload)
        effective_payload = json.loads(encoded)
    except Exception as exc:
        raise PaidCallPolicyError(
            f"{error_prefix}_input_payload_invalid"
        ) from exc
    if not isinstance(effective_payload, Mapping):
        raise PaidCallPolicyError(f"{error_prefix}_input_payload_invalid")

    output_limits: list[int] = []
    for field in (
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
    ):
        if field not in effective_payload:
            continue
        value = effective_payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise PaidCallPolicyError(
                f"{error_prefix}_output_token_cap_invalid"
            )
        output_limits.append(value)
    if (
        not output_limits
        or len(set(output_limits)) != 1
        or output_limits[0] > policy.output_max_tokens
    ):
        raise PaidCallPolicyError(
            f"{error_prefix}_output_token_cap_exceeded"
        )
    if isinstance(policy, FixedPaidCallPolicy):
        effective_model = normalize_model_for_provider(
            str(effective_payload.get("model") or ""),
            policy.provider,
        )
        if effective_model != policy.model:
            raise PaidCallPolicyError(
                f"{error_prefix}_fixed_route_mismatch"
            )
    if (
        policy.input_max_bytes is not None
        and len(encoded) > policy.input_max_bytes
    ):
        raise PaidCallPolicyError(
            f"{error_prefix}_input_byte_cap_exceeded"
        )
    return len(encoded)


__all__ = [
    "FixedPaidCallPolicy",
    "LEGACY_SIGNED_MYSTAND_AGENT_POLICY",
    "LEGACY_SIGNED_MYSTAND_AGENT_POLICY_REVISION",
    "PaidCallBudget",
    "PaidCallPolicyError",
    "SIGNED_MYSTAND_AGENT_POLICY",
    "SIGNED_MYSTAND_AGENT_POLICY_REVISION",
    "SIGNED_MYSTAND_AGENT_POLICY_REVISION_HEADER",
    "SIGNED_MYSTAND_AGENT_POLICY_REGISTRY",
    "enforce_fixed_paid_call_route",
    "enforce_openai_chat_paid_call_dispatch_budget",
    "enforce_openai_responses_paid_call_dispatch_budget",
    "enforce_paid_call_dispatch_budget",
    "enforce_signed_mystand_policy_revision",
    "normalize_fixed_route",
    "resolve_signed_mystand_agent_policy",
    "serialize_openai_chat_request_body",
    "serialize_openai_responses_request_body",
]
