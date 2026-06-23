import pytest

from xiaoban.mmcc.manifest import MMCCManifest
from xiaoban.mmcc.registry_adapter import list_registered_tools
from xiaoban.mmcc.validator import MMCCValidationError, validate_manifest


def _valid_manifest_raw():
    return {
        "contractVersion": "mystand.module-capability.v0.1",
        "moduleId": "event-center",
        "version": "0.1.0",
        "displayName": "事件中心",
        "owner": "core",
        "capabilities": {
            "provides": ["event.read", "event.create"],
            "requires": [],
        },
        "permissions": [
            {
                "capability": "event.read",
                "roles": ["owner", "manager", "staff"],
                "scopes": ["self", "team", "company"],
                "defaultGrant": True,
            },
            {
                "capability": "event.create",
                "roles": ["owner", "manager"],
                "scopes": ["self", "team"],
                "defaultGrant": False,
            },
        ],
        "agent": {
            "tools": [
                {
                    "toolName": "read_due_events",
                    "description": "读取当前用户可见的到期事件摘要。",
                    "inputSchema": {"type": "object", "properties": {}},
                    "requiredCapability": "event.read",
                    "sideEffect": "read",
                    "idempotency": "none",
                },
                {
                    "toolName": "create_event",
                    "description": "创建一个新的 My Stand 事件。",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                    "requiredCapability": "event.create",
                    "sideEffect": "write",
                    "idempotency": "required",
                },
            ],
            "contextProviders": [
                {
                    "providerId": "current_due_events",
                    "description": "当前用户待处理事件摘要。",
                    "requiredCapability": "event.read",
                }
            ],
        },
    }


def test_valid_manifest_registers_module_prefixed_tools():
    manifest = validate_manifest(MMCCManifest.from_dict(_valid_manifest_raw()))

    registered = list_registered_tools([manifest])

    assert [tool.name for tool in registered] == [
        "event-center.read_due_events",
        "event-center.create_event",
    ]


def test_wrong_contract_version_is_rejected():
    raw = _valid_manifest_raw()
    raw["contractVersion"] = "wrong"

    with pytest.raises(MMCCValidationError) as exc:
        validate_manifest(MMCCManifest.from_dict(raw))

    assert "contractVersion" in str(exc.value)


def test_tool_without_declared_capability_is_rejected():
    raw = _valid_manifest_raw()
    raw["agent"]["tools"][0]["requiredCapability"] = "finance.read"

    with pytest.raises(MMCCValidationError) as exc:
        validate_manifest(MMCCManifest.from_dict(raw))

    assert "requiredCapability is not declared" in str(exc.value)


def test_write_tool_without_idempotency_is_rejected():
    raw = _valid_manifest_raw()
    raw["agent"]["tools"][1]["idempotency"] = "none"

    with pytest.raises(MMCCValidationError) as exc:
        validate_manifest(MMCCManifest.from_dict(raw))

    assert "must declare idempotency" in str(exc.value)

