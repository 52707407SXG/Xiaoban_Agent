"""Tests for shared tool result classification helpers."""

import json

import pytest

from agent.tool_result_classification import (
    file_mutation_result_landed,
    tool_result_failed,
)


def test_write_file_with_nested_lint_error_counts_as_landed():
    result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })

    assert file_mutation_result_landed("write_file", result) is True


def test_patch_with_nested_lsp_diagnostics_counts_as_landed():
    result = json.dumps({
        "success": True,
        "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
        "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
    })

    assert file_mutation_result_landed("patch", result) is True


def test_top_level_file_mutation_error_does_not_count_as_landed():
    result = json.dumps({"success": True, "error": "post-write verification failed"})

    assert file_mutation_result_landed("patch", result) is False


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"ok": False, "status": 403, "code": "denied"}, True),
        ({"ok": True, "is_error": True}, True),
        ({"success": False, "message": "rate limited"}, True),
        ({"status": "timeout"}, True),
        ({"exit_code": 7}, True),
        ({"ok": True, "failed": False}, False),
        ({"data": {"failed": True, "error": "business value"}}, False),
        ("Error executing tool 'read_file': missing", True),
        ('{"ok":true,"failed":false}', False),
    ],
)
def test_shared_tool_failure_contract(result, expected):
    assert tool_result_failed("generic_tool", result) is expected
