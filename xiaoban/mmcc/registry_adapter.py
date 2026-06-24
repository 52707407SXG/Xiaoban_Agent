"""Adapt MMCC tools into Xiaoban-compatible tool schemas."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .manifest import MMCCManifest, MMCCTool
from .policy import ToolInvocationContext, XiaobanPrincipal, can_call_tool


@dataclass(frozen=True)
class MMCCRegisteredTool:
    name: str
    schema: dict[str, Any]
    manifest: MMCCManifest
    tool: MMCCTool


def build_tool_schema(manifest: MMCCManifest, tool: MMCCTool) -> dict[str, Any]:
    return {
        "name": manifest.tool_full_name(tool),
        "description": tool.description,
        "parameters": tool.input_schema,
    }


def list_registered_tools(manifests: list[MMCCManifest]) -> list[MMCCRegisteredTool]:
    tools: list[MMCCRegisteredTool] = []
    for manifest in manifests:
        for tool in manifest.tools:
            tools.append(
                MMCCRegisteredTool(
                    name=manifest.tool_full_name(tool),
                    schema=build_tool_schema(manifest, tool),
                    manifest=manifest,
                    tool=tool,
                )
            )
    return tools


def filter_enabled_manifests(
    manifests: list[MMCCManifest],
    enabled_module_ids: set[str] | frozenset[str] | None,
) -> list[MMCCManifest]:
    """Return manifests that are enabled for the current site/profile.

    ``None`` means the caller has not supplied an enabled-module view yet, so
    all validated manifests are considered visible. An empty set means no
    module tools are visible.
    """

    if enabled_module_ids is None:
        return list(manifests)
    return [manifest for manifest in manifests if manifest.module_id in enabled_module_ids]


def register_mmcc_tools(
    registry: Any,
    manifests: list[MMCCManifest],
    *,
    principal_resolver: Callable[[dict[str, Any]], XiaobanPrincipal],
    gateway_call: Callable[[str, str, dict[str, Any]], dict[str, Any]],
    toolset: str = "xiaoban-mmcc",
    enabled_module_ids: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Register MMCC tools into a Xiaoban-compatible ToolRegistry instance.

    The registry object is intentionally structural: it only needs a
    ``register(...)`` method matching ``tools.registry.ToolRegistry``. This keeps
    the Xiaoban layer testable without importing the full runtime.
    """

    registered: list[str] = []
    visible_manifests = filter_enabled_manifests(manifests, enabled_module_ids)
    for item in list_registered_tools(visible_manifests):
        registry.register(
            name=item.name,
            toolset=toolset,
            schema=item.schema,
            handler=make_gateway_handler(
                item.manifest,
                item.tool,
                principal_resolver=principal_resolver,
                gateway_call=gateway_call,
            ),
            description=item.tool.description,
        )
        registered.append(item.name)
    return registered


def make_gateway_handler(
    manifest: MMCCManifest,
    tool: MMCCTool,
    *,
    principal_resolver: Callable[[dict[str, Any]], XiaobanPrincipal],
    gateway_call: Callable[[str, str, dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], str]:
    """Create a Xiaoban registry handler for an MMCC module tool.

    The handler enforces Xiaoban policy before delegating to a future My Stand
    module tool gateway. It never imports a module's internal implementation.
    """

    def handler(args: dict[str, Any], **_: Any) -> str:
        principal = principal_resolver(args)
        invocation = ToolInvocationContext(
            conversation_id=str(args.get("conversation_id", "")),
            message_id=str(args.get("message_id", "")),
            task_id=args.get("task_id"),
            idempotency_key=args.get("idempotencyKey") or args.get("idempotency_key"),
            approval_mode=str(args.get("approval_mode", "readonly")),
        )
        decision = can_call_tool(manifest, tool, principal, invocation)
        if not decision.allowed:
            return json.dumps(
                {
                    "ok": False,
                    "status": decision.status,
                    "reason": decision.reason,
                    "requiredCapability": decision.required_capability,
                },
                ensure_ascii=False,
            )
        result = gateway_call(manifest.module_id, tool.tool_name, args)
        return json.dumps({"ok": True, "result": result}, ensure_ascii=False)

    return handler
