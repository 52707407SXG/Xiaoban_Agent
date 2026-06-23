from xiaoban.mmcc.context_builder import select_authorized_context_providers
from xiaoban.mmcc.manifest import MMCCManifest
from xiaoban.mmcc.policy import XiaobanPrincipal
from xiaoban.mmcc.validator import validate_manifest


def test_context_provider_requires_authorized_capability():
    manifest = validate_manifest(
        MMCCManifest.from_dict(
            {
                "contractVersion": "mystand.module-capability.v0.1",
                "moduleId": "help-center",
                "version": "0.1.0",
                "displayName": "帮助中心",
                "owner": "core",
                "capabilities": {"provides": ["help.read"]},
                "permissions": [
                    {"capability": "help.read", "roles": ["owner", "staff"], "scopes": ["company"]}
                ],
                "agent": {
                    "tools": [],
                    "contextProviders": [
                        {
                            "providerId": "module_help_summary",
                            "description": "当前模块帮助摘要。",
                            "requiredCapability": "help.read",
                        }
                    ],
                },
            }
        )
    )

    allowed = XiaobanPrincipal(
        site_id="site-1",
        user_id="u1",
        role="staff",
        scopes=frozenset({"company"}),
        capabilities=frozenset({"help.read"}),
    )
    denied = XiaobanPrincipal(
        site_id="site-1",
        user_id="u2",
        role="visitor",
        scopes=frozenset({"public"}),
        capabilities=frozenset(),
    )

    assert [item.source for item in select_authorized_context_providers([manifest], allowed)] == [
        "help-center.module_help_summary"
    ]
    assert select_authorized_context_providers([manifest], denied) == []

