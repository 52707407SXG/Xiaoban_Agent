"""Permission checks for MMCC tool and context-provider access."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .manifest import MMCCContextProvider, MMCCManifest, MMCCTool

DecisionStatus = Literal["allow", "deny", "confirm"]


@dataclass(frozen=True)
class XiaobanPrincipal:
    site_id: str
    user_id: str
    role: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[str] = field(default_factory=frozenset)
    is_owner: bool = False
    channel: str = "web"


@dataclass(frozen=True)
class ToolInvocationContext:
    conversation_id: str
    message_id: str
    task_id: str | None = None
    idempotency_key: str | None = None
    approval_mode: str = "readonly"


@dataclass(frozen=True)
class PolicyDecision:
    status: DecisionStatus
    reason: str
    required_capability: str | None = None

    @property
    def allowed(self) -> bool:
        return self.status == "allow"


def can_access_capability(
    manifest: MMCCManifest,
    capability: str,
    principal: XiaobanPrincipal,
) -> PolicyDecision:
    if capability not in principal.capabilities:
        return PolicyDecision("deny", "capability_not_granted", capability)

    permission = manifest.permission_for(capability)
    if permission is None:
        return PolicyDecision("deny", "capability_has_no_permission_rule", capability)

    if permission.roles and principal.role not in permission.roles:
        return PolicyDecision("deny", "role_not_allowed", capability)

    if permission.scopes and not (set(permission.scopes) & set(principal.scopes)):
        return PolicyDecision("deny", "scope_not_allowed", capability)

    return PolicyDecision("allow", "ok", capability)


def can_call_tool(
    manifest: MMCCManifest,
    tool: MMCCTool,
    principal: XiaobanPrincipal,
    invocation: ToolInvocationContext,
) -> PolicyDecision:
    capability_decision = can_access_capability(manifest, tool.required_capability, principal)
    if not capability_decision.allowed:
        return capability_decision

    if tool.side_effect in {"write", "external"}:
        if not invocation.message_id:
            return PolicyDecision("deny", "message_id_required", tool.required_capability)
        if tool.idempotency == "required" and not invocation.idempotency_key:
            return PolicyDecision("deny", "idempotency_key_required", tool.required_capability)
        if invocation.approval_mode not in {"ask_once", "ask_per_action", "allow_this_session", "owner_only"}:
            return PolicyDecision("confirm", "write_or_external_action_requires_confirmation", tool.required_capability)
        if invocation.approval_mode == "owner_only" and not principal.is_owner:
            return PolicyDecision("deny", "owner_required", tool.required_capability)

    return PolicyDecision("allow", "ok", tool.required_capability)


def can_use_context_provider(
    manifest: MMCCManifest,
    provider: MMCCContextProvider,
    principal: XiaobanPrincipal,
) -> PolicyDecision:
    return can_access_capability(manifest, provider.required_capability, principal)

