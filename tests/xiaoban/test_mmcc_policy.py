from xiaoban.mmcc.manifest import MMCCManifest
from xiaoban.mmcc.policy import (
    ToolInvocationContext,
    XiaobanPrincipal,
    can_call_tool,
)
from xiaoban.mmcc.validator import validate_manifest


def _manifest():
    return validate_manifest(
        MMCCManifest.from_dict(
            {
                "contractVersion": "mystand.module-capability.v0.1",
                "moduleId": "works-processing",
                "version": "0.1.0",
                "displayName": "作品加工",
                "owner": "official",
                "capabilities": {"provides": ["works.read", "works.write"]},
                "permissions": [
                    {"capability": "works.read", "roles": ["owner", "staff"], "scopes": ["self", "team"]},
                    {"capability": "works.write", "roles": ["owner"], "scopes": ["self"]},
                ],
                "agent": {
                    "tools": [
                        {
                            "toolName": "generate_post",
                            "description": "生成作品加工草稿。",
                            "inputSchema": {"type": "object"},
                            "requiredCapability": "works.write",
                            "sideEffect": "write",
                            "idempotency": "required",
                        }
                    ],
                    "contextProviders": [],
                },
            }
        )
    )


def test_known_tool_name_does_not_bypass_capability():
    manifest = _manifest()
    tool = manifest.tools[0]
    principal = XiaobanPrincipal(
        site_id="site-1",
        user_id="user-1",
        role="owner",
        scopes=frozenset({"self"}),
        capabilities=frozenset({"works.read"}),
    )

    decision = can_call_tool(
        manifest,
        tool,
        principal,
        ToolInvocationContext(conversation_id="c1", message_id="m1", idempotency_key="i1"),
    )

    assert not decision.allowed
    assert decision.reason == "capability_not_granted"


def test_write_tool_requires_idempotency_key():
    manifest = _manifest()
    tool = manifest.tools[0]
    principal = XiaobanPrincipal(
        site_id="site-1",
        user_id="user-1",
        role="owner",
        scopes=frozenset({"self"}),
        capabilities=frozenset({"works.write"}),
        is_owner=True,
    )

    decision = can_call_tool(
        manifest,
        tool,
        principal,
        ToolInvocationContext(conversation_id="c1", message_id="m1"),
    )

    assert not decision.allowed
    assert decision.reason == "idempotency_key_required"


def test_write_tool_requires_confirmation_after_idempotency_passes():
    manifest = _manifest()
    tool = manifest.tools[0]
    principal = XiaobanPrincipal(
        site_id="site-1",
        user_id="user-1",
        role="owner",
        scopes=frozenset({"self"}),
        capabilities=frozenset({"works.write"}),
        is_owner=True,
    )

    decision = can_call_tool(
        manifest,
        tool,
        principal,
        ToolInvocationContext(conversation_id="c1", message_id="m1", idempotency_key="i1"),
    )

    assert decision.status == "confirm"
    assert decision.reason == "write_or_external_action_requires_confirmation"


def test_owner_approved_write_can_pass():
    manifest = _manifest()
    tool = manifest.tools[0]
    principal = XiaobanPrincipal(
        site_id="site-1",
        user_id="52707407",
        role="owner",
        scopes=frozenset({"self"}),
        capabilities=frozenset({"works.write"}),
        is_owner=True,
    )

    decision = can_call_tool(
        manifest,
        tool,
        principal,
        ToolInvocationContext(
            conversation_id="c1",
            message_id="m1",
            idempotency_key="i1",
            approval_mode="allow_this_session",
        ),
    )

    assert decision.allowed

