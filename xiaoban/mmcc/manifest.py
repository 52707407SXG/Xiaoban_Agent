"""Data model for Mystand Module Capability Contract v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MMCC_CONTRACT_VERSION = "mystand.module-capability.v0.1"

SideEffect = Literal["read", "write", "external"]
Idempotency = Literal["required", "recommended", "none"]
Scope = Literal["self", "team", "company", "public"]


@dataclass(frozen=True)
class MMCCPermission:
    capability: str
    roles: tuple[str, ...] = ()
    scopes: tuple[Scope, ...] = ()
    default_grant: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MMCCPermission":
        return cls(
            capability=str(raw.get("capability", "")).strip(),
            roles=tuple(str(role) for role in raw.get("roles", ()) or ()),
            scopes=tuple(raw.get("scopes", ()) or ()),
            default_grant=bool(raw.get("defaultGrant", False)),
        )


@dataclass(frozen=True)
class MMCCTool:
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    required_capability: str
    side_effect: SideEffect
    output_schema: dict[str, Any] | None = None
    idempotency: Idempotency = "none"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MMCCTool":
        return cls(
            tool_name=str(raw.get("toolName", "")).strip(),
            description=str(raw.get("description", "")).strip(),
            input_schema=dict(raw.get("inputSchema") or {}),
            output_schema=dict(raw["outputSchema"]) if isinstance(raw.get("outputSchema"), dict) else None,
            required_capability=str(raw.get("requiredCapability", "")).strip(),
            idempotency=raw.get("idempotency", "none"),
            side_effect=raw.get("sideEffect", "read"),
        )

    @property
    def needs_idempotency(self) -> bool:
        return self.side_effect in {"write", "external"} and self.idempotency != "none"


@dataclass(frozen=True)
class MMCCContextProvider:
    provider_id: str
    description: str
    required_capability: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MMCCContextProvider":
        return cls(
            provider_id=str(raw.get("providerId", "")).strip(),
            description=str(raw.get("description", "")).strip(),
            required_capability=str(raw.get("requiredCapability", "")).strip(),
        )


@dataclass(frozen=True)
class MMCCManifest:
    contract_version: str
    module_id: str
    version: str
    display_name: str
    owner: str
    package_name: str | None = None
    category: str | None = None
    capabilities_provides: tuple[str, ...] = ()
    capabilities_requires: tuple[str, ...] = ()
    permissions: tuple[MMCCPermission, ...] = ()
    data_scopes: tuple[dict[str, Any], ...] = ()
    events_emits: tuple[str, ...] = ()
    events_subscribes: tuple[str, ...] = ()
    tools: tuple[MMCCTool, ...] = ()
    context_providers: tuple[MMCCContextProvider, ...] = ()
    migrations: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MMCCManifest":
        capabilities = raw.get("capabilities") or {}
        events = raw.get("events") or {}
        agent = raw.get("agent") or {}

        return cls(
            contract_version=str(raw.get("contractVersion", "")).strip(),
            module_id=str(raw.get("moduleId", "")).strip(),
            package_name=raw.get("packageName"),
            version=str(raw.get("version", "")).strip(),
            display_name=str(raw.get("displayName", "")).strip(),
            owner=str(raw.get("owner", "")).strip(),
            category=raw.get("category"),
            capabilities_provides=tuple(str(item) for item in capabilities.get("provides", ()) or ()),
            capabilities_requires=tuple(str(item) for item in capabilities.get("requires", ()) or ()),
            permissions=tuple(MMCCPermission.from_dict(item) for item in raw.get("permissions", ()) or ()),
            data_scopes=tuple(dict(item) for item in raw.get("dataScopes", ()) or ()),
            events_emits=tuple(str(item) for item in events.get("emits", ()) or ()),
            events_subscribes=tuple(str(item) for item in events.get("subscribes", ()) or ()),
            tools=tuple(MMCCTool.from_dict(item) for item in agent.get("tools", ()) or ()),
            context_providers=tuple(
                MMCCContextProvider.from_dict(item)
                for item in agent.get("contextProviders", ()) or ()
            ),
            migrations=dict(raw.get("migrations") or {}),
            raw=dict(raw),
        )

    def tool_full_name(self, tool: MMCCTool) -> str:
        return f"{self.module_id}.{tool.tool_name}"

    def permission_for(self, capability: str) -> MMCCPermission | None:
        for permission in self.permissions:
            if permission.capability == capability:
                return permission
        return None

