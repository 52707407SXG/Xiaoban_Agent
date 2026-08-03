"""Reserved model metadata is stripped before physical tool dispatch."""

import json

import pytest

from agent.tool_outcome_lineage import (
    RUNTIME_LINKAGE_ARGUMENT,
    sanitize_runtime_linkage_arguments,
    split_runtime_linkage,
)


def test_model_authored_runtime_metadata_is_stripped_without_authority():
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


@pytest.mark.parametrize("value", [None, "not-an-object", ["x"]])
def test_non_object_arguments_fail_to_an_empty_handler_shape(value):
    assert split_runtime_linkage(value) == ({}, {})


@pytest.mark.parametrize("nested_as_json", [False, True])
def test_tool_search_bridge_strips_nested_runtime_metadata(nested_as_json):
    nested = {
        "query": "feature card",
        RUNTIME_LINKAGE_ARGUMENT: {
            "recovery_of": "failed-call",
        },
    }
    handler_args, linkage = sanitize_runtime_linkage_arguments(
        "tool_call",
        {
            "name": "deferred_lookup",
            "arguments": (
                json.dumps(nested)
                if nested_as_json
                else nested
            ),
        },
    )

    assert handler_args == {
        "name": "deferred_lookup",
        "arguments": {"query": "feature card"},
    }
    assert linkage == {}


def test_non_bridge_tool_keeps_nested_business_arguments_unchanged():
    arguments = {
        "resource": {
            "name": "feature card",
            RUNTIME_LINKAGE_ARGUMENT: {"ignored": "nested business value"},
        },
        RUNTIME_LINKAGE_ARGUMENT: {"recovery_of": "failed-call"},
    }
    handler_args, linkage = sanitize_runtime_linkage_arguments(
        "mystand_query",
        arguments,
    )

    assert handler_args == {"resource": arguments["resource"]}
    assert linkage == {}
