from agent.tool_outcome_lineage import (
    RUNTIME_LINKAGE_ARGUMENT,
    action_fingerprint,
    normalize_lineage_action,
    sanitize_runtime_linkage_arguments,
    split_runtime_linkage,
    unresolved_failure_ids,
)


def test_model_authored_runtime_linkage_is_stripped_without_authority():
    handler_args, linkage = split_runtime_linkage(
        {
            "query": "feature card",
            RUNTIME_LINKAGE_ARGUMENT: {
                "recovery_of": "failed-call",
                "attempt_group_id": "sources",
                "completion_policy": "any_of",
            },
        }
    )

    assert handler_args == {"query": "feature card"}
    assert linkage == {}


def test_tool_search_nested_runtime_linkage_uses_underlying_identity(
    monkeypatch,
):
    from tools import tool_search

    monkeypatch.setattr(
        tool_search,
        "resolve_underlying_call",
        lambda args: (
            str(args["name"]),
            dict(args["arguments"]),
            None,
        ),
    )
    tool_name, handler_args, linkage = normalize_lineage_action(
        "tool_call",
        {
            "name": "deferred_lookup",
            "arguments": {
                "query": "feature card",
                RUNTIME_LINKAGE_ARGUMENT: {
                    "recovery_of": "failed-call",
                },
            },
        },
    )

    assert tool_name == "deferred_lookup"
    assert handler_args == {"query": "feature card"}
    assert linkage == {}


def test_unresolved_tool_search_bridge_still_strips_nested_metadata():
    handler_args, linkage = sanitize_runtime_linkage_arguments(
        "tool_call",
        {
            "name": "not-resolved",
            "arguments": {
                "query": "feature card",
                RUNTIME_LINKAGE_ARGUMENT: {
                    "recovery_of": "failed-call",
                },
            },
        },
    )

    assert handler_args == {
        "name": "not-resolved",
        "arguments": {"query": "feature card"},
    }
    assert linkage == {}


def test_action_fingerprint_ignores_runtime_linkage_but_not_action_identity():
    original = action_fingerprint(
        "memory",
        {"content": "A", "target": "user"},
    )
    exact_retry = action_fingerprint(
        "memory",
        {
            "target": "user",
            "content": "A",
            RUNTIME_LINKAGE_ARGUMENT: {"recovery_of": "first"},
        },
    )
    different_action = action_fingerprint(
        "memory",
        {"content": "B", "target": "memory"},
    )

    assert original == exact_retry
    assert original != different_action


def test_unrelated_success_does_not_clear_failure():
    outcomes = [
        {
            "batch_id": 1,
            "tool_call_id": "save-a",
            "failed": True,
            "material": True,
        },
        {
            "batch_id": 2,
            "tool_call_id": "save-b",
            "failed": False,
            "material": True,
        },
    ]

    assert unresolved_failure_ids(outcomes) == ["save-a"]


def test_exact_recovery_clears_only_the_bound_failure():
    same_action = action_fingerprint("lookup", {"query": "same"})
    outcomes = [
        {
            "batch_id": 1,
            "tool_call_id": "primary",
            "failed": True,
            "material": True,
            "action_fingerprint": same_action,
        },
        {
            "batch_id": 2,
            "tool_call_id": "alternate",
            "failed": False,
            "material": True,
            "recovery_of": "primary",
            "recovery_authority": "exact-action",
            "action_fingerprint": same_action,
        },
    ]

    assert unresolved_failure_ids(outcomes) == []


def test_model_predeclared_group_and_true_event_id_have_no_authority():
    outcomes = [
        {
            "batch_id": 1,
            "tool_call_id": "external-source",
            "failed": True,
            "material": True,
            "action_fingerprint": action_fingerprint(
                "web_search",
                {"query": "same material"},
            ),
            "attempt_group_id": "model-chosen-group",
            "completion_policy": "any_of",
        },
        {
            "batch_id": 2,
            "tool_call_id": "session-source",
            "failed": False,
            "material": True,
            "action_fingerprint": action_fingerprint(
                "session_search",
                {"query": "same material"},
            ),
            "recovery_of": "external-source",
            "recovery_authority": "predeclared-any-of",
            "attempt_group_id": "model-chosen-group",
            "completion_policy": "any_of",
        },
    ]

    assert unresolved_failure_ids(outcomes) == ["external-source"]


def test_server_grant_clears_only_when_runtime_authenticates_grant_id():
    session_fingerprint = action_fingerprint(
        "session_search",
        {"query": "same material"},
    )
    outcomes = [
        {
            "batch_id": 1,
            "tool_call_id": "external-source",
            "failed": True,
            "material": True,
            "action_fingerprint": action_fingerprint(
                "web_search",
                {"query": "same material"},
            ),
        },
        {
            "batch_id": 2,
            "tool_call_id": "session-source",
            "failed": False,
            "material": True,
            "action_fingerprint": session_fingerprint,
            "recovery_of": "external-source",
            "recovery_authority": "server-grant",
            "recovery_grant_id": "opaque-runtime-grant",
        },
    ]

    valid_grant = {
        "schema": "xiaoban.recovery-grant.v1",
        "grant_id": "opaque-runtime-grant",
        "turn_id": "turn-a",
        "target_event_id": "external-source",
        "allowed_action_fingerprint": session_fingerprint,
        "max_uses": 1,
        "used": True,
    }
    assert unresolved_failure_ids(outcomes) == ["external-source"]
    assert unresolved_failure_ids(
        outcomes,
        server_grants={"opaque-runtime-grant": valid_grant},
        expected_turn_id="turn-a",
    ) == []
    for invalid_binding in (
        {"target_event_id": "different-failure"},
        {"allowed_action_fingerprint": "different-action"},
        {"turn_id": "different-turn"},
        {"max_uses": 2},
        {"used": False},
    ):
        assert unresolved_failure_ids(
            outcomes,
            server_grants={
                "opaque-runtime-grant": {
                    **valid_grant,
                    **invalid_binding,
                }
            },
            expected_turn_id="turn-a",
        ) == ["external-source"]


def test_model_cannot_forge_server_grant_authority():
    outcomes = [
        {
            "batch_id": 1,
            "tool_call_id": "required-a",
            "failed": True,
            "material": True,
        },
        {
            "batch_id": 2,
            "tool_call_id": "unrelated-b",
            "failed": False,
            "material": True,
            "recovery_of": "required-a",
            "recovery_authority": "server-grant",
            "recovery_grant_id": "model-invented",
        },
    ]

    assert unresolved_failure_ids(
        outcomes,
        server_grants={
            "different-real-grant": {
                "schema": "xiaoban.recovery-grant.v1",
                "grant_id": "different-real-grant",
                "turn_id": "turn-a",
                "target_event_id": "required-a",
                "allowed_action_fingerprint": "",
                "max_uses": 1,
                "used": True,
            }
        },
        expected_turn_id="turn-a",
    ) == ["required-a"]


def test_recovery_reference_is_later_batch_and_single_use():
    same_action = action_fingerprint("lookup", {"query": "same"})
    outcomes = [
        {
            "batch_id": 1,
            "tool_call_id": "original",
            "failed": True,
            "material": True,
            "action_fingerprint": same_action,
        },
        {
            # Same-batch self-declared linkage is invalid.
            "batch_id": 1,
            "tool_call_id": "same-batch",
            "failed": False,
            "material": True,
            "recovery_of": "original",
            "recovery_authority": "exact-action",
            "action_fingerprint": same_action,
        },
        {
            "batch_id": 2,
            "tool_call_id": "first-recovery",
            "failed": True,
            "material": True,
            "recovery_of": "original",
            "recovery_authority": "exact-action",
            "action_fingerprint": same_action,
        },
        {
            # Reusing the already-consumed original reference cannot erase
            # the failed recovery attempt.
            "batch_id": 3,
            "tool_call_id": "reused-reference",
            "failed": False,
            "material": True,
            "recovery_of": "original",
            "recovery_authority": "exact-action",
            "action_fingerprint": same_action,
        },
    ]

    assert unresolved_failure_ids(outcomes) == ["first-recovery"]
