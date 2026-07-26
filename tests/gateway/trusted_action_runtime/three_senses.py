"""三感（角色感/时间感/空间感）与小伴身份保护块指纹账本工具。

波次 0 冻结现场：对受保护块记录文件、符号、规范化 SHA-256、相关测试与
调用链消费者。波次 1 结束时由测试逐项重新计算，任何非预期变化立即红灯。

规范化规则：CRLF/CR 统一为 LF，逐行去尾部空白，去首尾空行，再 SHA-256。
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
LEDGER_PATH = Path(__file__).with_name("three-senses-ledger.json")
BASE_COMMIT = "213398b7abe4a42b453f92c1a0f1398b4c1a5371"


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip("\n")


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(_normalize(value).encode("utf-8")).hexdigest()


def _constant_loader(module_name: str, symbol: str) -> Callable[[], str]:
    def load() -> str:
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, symbol)
        if not isinstance(value, str):
            raise TypeError(f"{module_name}.{symbol} is not a string constant")
        return fingerprint_text(value)

    return load


def _function_loader(module_name: str, symbol: str) -> Callable[[], str]:
    def load() -> str:
        import importlib

        module = importlib.import_module(module_name)
        func = getattr(module, symbol)
        return fingerprint_text(inspect.getsource(func))

    return load


# 生产 SOUL.md 运行源（systemd xiaoban-agent.service: XIAOBAN_HOME=/var/lib/xiaoban，
# 经 agent/prompt_builder.py::load_soul_md 注入系统提示）。只记录规范化指纹，
# 不记录、不输出正文。
RUNTIME_SOUL_MD_PATH = "/var/lib/xiaoban/SOUL.md"


def _runtime_file_loader(path: str) -> Callable[[], str]:
    def load() -> str:
        try:
            return fingerprint_text(Path(path).read_text(encoding="utf-8"))
        except OSError:
            return f"MISSING:{path}"

    return load


# (id, 类别, 文件, 符号, loader, 调用链消费者, 相关现有测试)
_BLOCK_SPECS = [
    (
        "identity.default-agent-identity",
        "角色感",
        "agent/prompt_builder.py",
        "DEFAULT_AGENT_IDENTITY",
        _constant_loader("agent.prompt_builder", "DEFAULT_AGENT_IDENTITY"),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "identity.soul-md-loader",
        "角色感",
        "agent/prompt_builder.py",
        "load_soul_md",
        _function_loader("agent.prompt_builder", "load_soul_md"),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "identity.runtime-soul-md",
        "角色感",
        RUNTIME_SOUL_MD_PATH,
        "(runtime-file)",
        _runtime_file_loader(RUNTIME_SOUL_MD_PATH),
        [
            "agent/prompt_builder.py::load_soul_md",
            "agent/system_prompt.py::build_system_prompt_parts",
            "systemd:xiaoban-agent.service Environment=XIAOBAN_HOME",
        ],
        ["tests/gateway/trusted_action_runtime/test_three_senses_fingerprints.py"],
    ),
    (
        "identity.native-identity",
        "角色感",
        "xiaoban/prompt.py",
        "XIAOBAN_NATIVE_IDENTITY",
        _constant_loader("xiaoban.prompt", "XIAOBAN_NATIVE_IDENTITY"),
        ["xiaoban/prompt.py::build_xiaoban_identity_block", "scripts/xiaoban_smoke.py"],
        ["tests/gateway/trusted_action_runtime/test_three_senses_fingerprints.py"],
    ),
    (
        "identity.reply-style",
        "角色感",
        "xiaoban/prompt.py",
        "XIAOBAN_REPLY_STYLE",
        _constant_loader("xiaoban.prompt", "XIAOBAN_REPLY_STYLE"),
        ["xiaoban/prompt.py::build_xiaoban_identity_block"],
        ["tests/gateway/trusted_action_runtime/test_three_senses_fingerprints.py"],
    ),
    (
        "identity.security-boundary",
        "角色感",
        "xiaoban/prompt.py",
        "XIAOBAN_SECURITY_BOUNDARY",
        _constant_loader("xiaoban.prompt", "XIAOBAN_SECURITY_BOUNDARY"),
        ["xiaoban/prompt.py::build_xiaoban_identity_block"],
        ["tests/gateway/trusted_action_runtime/test_three_senses_fingerprints.py"],
    ),
    (
        "identity.identity-block-builder",
        "角色感",
        "xiaoban/prompt.py",
        "build_xiaoban_identity_block",
        _function_loader("xiaoban.prompt", "build_xiaoban_identity_block"),
        ["scripts/xiaoban_smoke.py"],
        ["tests/gateway/trusted_action_runtime/test_three_senses_fingerprints.py"],
    ),
    (
        "policy.operating-policy",
        "角色感",
        "agent/xiaoban_operating_policy.py",
        "XIAOBAN_OPERATING_POLICY",
        _constant_loader("agent.xiaoban_operating_policy", "XIAOBAN_OPERATING_POLICY"),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "policy.mystand-feature-reasoning",
        "角色感",
        "agent/xiaoban_operating_policy.py",
        "XIAOBAN_MYSTAND_FEATURE_REASONING_POLICY",
        _constant_loader(
            "agent.xiaoban_operating_policy",
            "XIAOBAN_MYSTAND_FEATURE_REASONING_POLICY",
        ),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "policy.mystand-security-boundary",
        "角色感",
        "agent/xiaoban_operating_policy.py",
        "XIAOBAN_MYSTAND_SECURITY_BOUNDARY_POLICY",
        _constant_loader(
            "agent.xiaoban_operating_policy",
            "XIAOBAN_MYSTAND_SECURITY_BOUNDARY_POLICY",
        ),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "policy.agentic-workflow",
        "角色感",
        "agent/xiaoban_operating_policy.py",
        "XIAOBAN_AGENTIC_WORKFLOW_POLICY",
        _constant_loader(
            "agent.xiaoban_operating_policy", "XIAOBAN_AGENTIC_WORKFLOW_POLICY"
        ),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "policy.verification-backfill",
        "角色感",
        "agent/xiaoban_operating_policy.py",
        "XIAOBAN_VERIFICATION_BACKFILL_POLICY",
        _constant_loader(
            "agent.xiaoban_operating_policy",
            "XIAOBAN_VERIFICATION_BACKFILL_POLICY",
        ),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "policy.explicit-learn",
        "角色感",
        "agent/xiaoban_operating_policy.py",
        "XIAOBAN_EXPLICIT_LEARN_POLICY",
        _constant_loader(
            "agent.xiaoban_operating_policy", "XIAOBAN_EXPLICIT_LEARN_POLICY"
        ),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "policy.advanced-moa",
        "角色感",
        "agent/xiaoban_operating_policy.py",
        "XIAOBAN_ADVANCED_MOA_POLICY",
        _constant_loader(
            "agent.xiaoban_operating_policy", "XIAOBAN_ADVANCED_MOA_POLICY"
        ),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "policy.deepseek-execution-guidance",
        "角色感",
        "agent/xiaoban_operating_policy.py",
        "XIAOBAN_DEEPSEEK_EXECUTION_GUIDANCE",
        _constant_loader(
            "agent.xiaoban_operating_policy",
            "XIAOBAN_DEEPSEEK_EXECUTION_GUIDANCE",
        ),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
    (
        "time.api-temporal-context",
        "时间感",
        "gateway/platforms/api_server.py",
        "_build_api_temporal_context",
        _function_loader(
            "gateway.platforms.api_server", "_build_api_temporal_context"
        ),
        ["gateway/platforms/api_server.py::_merge_temporal_context"],
        ["tests/gateway/test_api_server.py"],
    ),
    (
        "time.merge-temporal-context",
        "时间感",
        "gateway/platforms/api_server.py",
        "_merge_temporal_context",
        _function_loader(
            "gateway.platforms.api_server", "_merge_temporal_context"
        ),
        [
            "gateway/platforms/api_server.py::_run_agent",
            "gateway/platforms/api_server.py::_handle_runs",
        ],
        ["tests/gateway/test_api_server.py"],
    ),
    (
        "time.format-message-timestamp",
        "时间感",
        "gateway/message_timestamps.py",
        "format_message_timestamp",
        _function_loader("gateway.message_timestamps", "format_message_timestamp"),
        ["gateway/message_timestamps.py::render_user_content_with_timestamp"],
        ["tests/gateway/test_message_timestamps.py"],
    ),
    (
        "time.render-user-content-with-timestamp",
        "时间感",
        "gateway/message_timestamps.py",
        "render_user_content_with_timestamp",
        _function_loader(
            "gateway.message_timestamps", "render_user_content_with_timestamp"
        ),
        ["gateway/run.py::GatewayRunner._handle_message"],
        ["tests/gateway/test_message_timestamps.py"],
    ),
    (
        "space.environment-hints",
        "空间感",
        "agent/prompt_builder.py",
        "build_environment_hints",
        _function_loader("agent.prompt_builder", "build_environment_hints"),
        ["agent/system_prompt.py::build_system_prompt_parts"],
        ["tests/agent/test_system_prompt.py"],
    ),
]


def collect_fingerprints() -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for block_id, kind, path, symbol, loader, consumers, related_tests in _BLOCK_SPECS:
        blocks.append(
            {
                "id": block_id,
                "kind": kind,
                "file": path,
                "symbol": symbol,
                "sha256": loader(),
                "consumers": list(consumers),
                "related_tests": list(related_tests),
            }
        )
    return blocks


def load_ledger() -> Dict[str, Any]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def main() -> None:
    ledger = {
        "base_commit": BASE_COMMIT,
        "normalization": "crlf->lf; rstrip each line; strip leading/trailing blank lines; sha256 utf-8",
        "blocks": collect_fingerprints(),
    }
    LEDGER_PATH.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {LEDGER_PATH} with {len(ledger['blocks'])} blocks")


if __name__ == "__main__":
    main()
