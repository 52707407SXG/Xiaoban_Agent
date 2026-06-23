"""Build authorized My Stand module context snippets from MMCC manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .manifest import MMCCContextProvider, MMCCManifest
from .policy import XiaobanPrincipal, can_use_context_provider


@dataclass(frozen=True)
class AuthorizedContextProvider:
    module_id: str
    provider: MMCCContextProvider
    source: str
    generated_at: str


@dataclass(frozen=True)
class MMCCContextIndex:
    principal_user_id: str
    principal_role: str
    providers: tuple[dict[str, Any], ...]
    generated_at: str


def select_authorized_context_providers(
    manifests: list[MMCCManifest],
    principal: XiaobanPrincipal,
) -> list[AuthorizedContextProvider]:
    generated_at = datetime.now(timezone.utc).isoformat()
    selected: list[AuthorizedContextProvider] = []
    for manifest in manifests:
        for provider in manifest.context_providers:
            decision = can_use_context_provider(manifest, provider, principal)
            if not decision.allowed:
                continue
            selected.append(
                AuthorizedContextProvider(
                    module_id=manifest.module_id,
                    provider=provider,
                    source=f"{manifest.module_id}.{provider.provider_id}",
                    generated_at=generated_at,
                )
            )
    return selected


def build_mmcc_context_index(
    manifests: list[MMCCManifest],
    principal: XiaobanPrincipal,
) -> MMCCContextIndex:
    """Build a compact context-provider index for prompt assembly.

    This does not execute providers or fetch module records. It only tells the
    runtime which authorized summaries are available, so the main prompt stays
    small and cannot accidentally receive data from disabled/unauthorized
    modules.
    """

    selected = select_authorized_context_providers(manifests, principal)
    generated_at = selected[0].generated_at if selected else datetime.now(timezone.utc).isoformat()
    manifests_by_id = {manifest.module_id: manifest for manifest in manifests}
    provider_rows = []
    for item in selected:
        manifest = manifests_by_id[item.module_id]
        permission = manifest.permission_for(item.provider.required_capability)
        provider_rows.append(
            {
                "moduleId": item.module_id,
                "providerId": item.provider.provider_id,
                "source": item.source,
                "description": item.provider.description,
                "requiredCapability": item.provider.required_capability,
                "permissionRoles": permission.roles if permission else (),
                "permissionScopes": permission.scopes if permission else (),
                "generatedAt": item.generated_at,
                "freshness": "provider-not-executed",
            }
        )
    return MMCCContextIndex(
        principal_user_id=principal.user_id,
        principal_role=principal.role,
        providers=tuple(provider_rows),
        generated_at=generated_at,
    )
