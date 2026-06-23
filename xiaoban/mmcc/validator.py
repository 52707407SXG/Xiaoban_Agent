"""Validation for Mystand Module Capability Contract manifests."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .manifest import MMCC_CONTRACT_VERSION, MMCCManifest

_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_TOOL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9-]*(\.[a-z][a-z0-9_-]*)+$")
_VALID_OWNERS = {"core", "official", "community", "private"}
_VALID_SIDE_EFFECTS = {"read", "write", "external"}
_VALID_IDEMPOTENCY = {"required", "recommended", "none"}
_VALID_SCOPES = {"self", "team", "company", "public"}


@dataclass(frozen=True)
class MMCCValidationError(Exception):
    errors: tuple[str, ...]

    def __str__(self) -> str:
        return "; ".join(self.errors)


def validate_manifest(manifest: MMCCManifest) -> MMCCManifest:
    """Validate *manifest* and return it.

    Raises MMCCValidationError with all discovered problems instead of failing
    on the first one. This keeps module author feedback useful.
    """

    errors: list[str] = []

    if manifest.contract_version != MMCC_CONTRACT_VERSION:
        errors.append(f"contractVersion must be {MMCC_CONTRACT_VERSION!r}")
    if not _ID_RE.match(manifest.module_id):
        errors.append("moduleId must be kebab-case and start with a lowercase letter")
    if not manifest.version:
        errors.append("version is required")
    if not manifest.display_name:
        errors.append("displayName is required")
    if manifest.owner not in _VALID_OWNERS:
        errors.append("owner must be one of core, official, community, private")

    provided = set(manifest.capabilities_provides)
    permission_caps = {permission.capability for permission in manifest.permissions}

    for capability in provided | permission_caps | set(manifest.capabilities_requires):
        if not _CAPABILITY_RE.match(capability):
            errors.append(f"invalid capability name: {capability}")

    for permission in manifest.permissions:
        if not permission.capability:
            errors.append("permission.capability is required")
        unknown_scopes = set(permission.scopes) - _VALID_SCOPES
        if unknown_scopes:
            errors.append(f"permission {permission.capability} has invalid scopes: {sorted(unknown_scopes)}")

    seen_tools: set[str] = set()
    for tool in manifest.tools:
        if not _TOOL_RE.match(tool.tool_name):
            errors.append(f"toolName must be snake_case and start lowercase: {tool.tool_name}")
        if tool.tool_name in seen_tools:
            errors.append(f"duplicate toolName: {tool.tool_name}")
        seen_tools.add(tool.tool_name)
        if not tool.description:
            errors.append(f"tool {tool.tool_name} description is required")
        if not isinstance(tool.input_schema, dict) or not tool.input_schema:
            errors.append(f"tool {tool.tool_name} inputSchema is required")
        if not tool.required_capability:
            errors.append(f"tool {tool.tool_name} requiredCapability is required")
        elif tool.required_capability not in provided and tool.required_capability not in permission_caps:
            errors.append(f"tool {tool.tool_name} requiredCapability is not declared: {tool.required_capability}")
        if tool.side_effect not in _VALID_SIDE_EFFECTS:
            errors.append(f"tool {tool.tool_name} sideEffect is invalid: {tool.side_effect}")
        if tool.idempotency not in _VALID_IDEMPOTENCY:
            errors.append(f"tool {tool.tool_name} idempotency is invalid: {tool.idempotency}")
        if tool.side_effect in {"write", "external"} and tool.idempotency == "none":
            errors.append(f"tool {tool.tool_name} with sideEffect={tool.side_effect} must declare idempotency")

    seen_providers: set[str] = set()
    for provider in manifest.context_providers:
        if not _TOOL_RE.match(provider.provider_id):
            errors.append(f"context providerId must be snake_case and start lowercase: {provider.provider_id}")
        if provider.provider_id in seen_providers:
            errors.append(f"duplicate context providerId: {provider.provider_id}")
        seen_providers.add(provider.provider_id)
        if not provider.description:
            errors.append(f"context provider {provider.provider_id} description is required")
        if not provider.required_capability:
            errors.append(f"context provider {provider.provider_id} requiredCapability is required")
        elif provider.required_capability not in provided and provider.required_capability not in permission_caps:
            errors.append(
                f"context provider {provider.provider_id} requiredCapability is not declared: "
                f"{provider.required_capability}"
            )

    if errors:
        raise MMCCValidationError(tuple(errors))
    return manifest

