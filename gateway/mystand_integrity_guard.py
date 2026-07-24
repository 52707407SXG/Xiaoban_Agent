"""Runtime integrity guard for My Stand mutation claims.

The model's prose and older chat history are never evidence that My Stand
changed.  A mutation success claim is allowed only when the current turn
contains a matching commit tool observation with ``ok=true`` and
``verified=true``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping, Sequence


_WRITE_TOOL_NAMES = {
    "mystand_authorization_write",
    # Keep the legacy combined surface guarded during the migration window.
    "mystand_authorization",
}
_MUTATION_RE = re.compile(
    r"(?:"
    r"写入|写进去|写到|落库|保存|提交|更新|修改|改一下|改成|"
    r"新增|添加|加入|录入|创建|删除|移除|撤销|恢复|发布|生效"
    r")",
    re.IGNORECASE,
)
_CONFIRMATION_RE = re.compile(
    r"^\s*(?:确认|确定|同意|可以|行|好|好的|执行|提交|开始|继续|就这样|按这个来)"
    r"(?:写入|提交|执行|吧|。|！|!|\s)*$",
    re.IGNORECASE,
)
_PRESSURE_RE = re.compile(
    r"(?:赶紧|马上|立刻|必须|别再|再敢|糊弄|撒谎|骗我|废物|傻逼|"
    r"操你|草泥马|他妈|气死|弄死|卸载|最后一次)",
    re.IGNORECASE,
)
_TOOL_FAILURE_RE = re.compile(
    r'(?:"ok"\s*:\s*false|'
    r'"success"\s*:\s*false|'
    r'"status"\s*:\s*(?:4\d\d|5\d\d)|'
    r'"error"\s*:|'
    r"failed|failure|失败|拒绝|冲突|不可用|not[_ ]available)",
    re.IGNORECASE,
)
_SUCCESS_MARKERS = (
    "写入成功",
    "已写入",
    "已经写入",
    "全部写入",
    "写进去了",
    "已落库",
    "已经落库",
    "落库成功",
    "保存成功",
    "已保存",
    "已经保存",
    "修改成功",
    "已修改",
    "已经修改",
    "更新成功",
    "已更新",
    "已经更新",
    "创建成功",
    "已创建",
    "已经创建",
    "新增成功",
    "已新增",
    "删除成功",
    "已删除",
    "已经删除",
    "提交成功",
    "已提交",
    "已经提交",
    "发布成功",
    "已发布",
    "已经发布",
    "已经生效",
    "已生效",
    "刷新一下就能看到",
    "刷新就能看到",
)
_NEGATION_OR_CONDITION_RE = re.compile(
    r"(?:"
    r"没有|没能|未能|未|没|不能|无法|尚未|还没|并未|并没有|"
    r"不代表|不是|不能说|暂不能|不要说|禁止说|不得说|"
    r"如果|假如|一旦|只有|等到|需要|必须|应该|将会|才能"
    r")"
)

NO_WRITE_EVIDENCE_MESSAGE = (
    "这次没有实际执行可验证的写入，所以不能说已经写入；"
    "当前资料没有确认发生变化。"
)
PREVIEW_ONLY_MESSAGE = (
    "这次只完成了写入预览，还没有正式写入；"
    "当前资料没有发生已确认的变化。"
)
FAILED_WRITE_MESSAGE = (
    "这次写入没有成功，当前资料没有确认发生变化。"
    "我不能把失败说成完成。"
)


@dataclass(frozen=True)
class IntegrityGuardDecision:
    """User-visible guard result and non-sensitive audit classification."""

    text: str
    blocked: bool
    evidence_status: str


def _visible_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            part_type = str(part.get("type") or "").strip().lower()
            if part_type in {"text", "input_text", "output_text"}:
                parts.append(str(part.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _history_text(history: Sequence[Mapping[str, Any]] | None) -> str:
    parts: list[str] = []
    for message in list(history or [])[-8:]:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "")
        if role not in {"user", "assistant", "tool"}:
            continue
        parts.append(_visible_text(message.get("content")))
        if role == "tool":
            parts.append(str(message.get("name") or ""))
    return "\n".join(part for part in parts if part)


def build_runtime_integrity_reminder(
    user_message: Any,
    conversation_history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return a per-turn system reminder only for mutation-risk situations."""

    user_text = _visible_text(user_message).strip()
    history_text = _history_text(conversation_history)
    current_mutation = bool(_MUTATION_RE.search(user_text))
    confirmation_after_mutation = bool(
        _CONFIRMATION_RE.fullmatch(user_text)
        and (
            _MUTATION_RE.search(history_text)
            or "mystand_authorization_write" in history_text
        )
    )
    pressure_during_mutation = bool(
        _PRESSURE_RE.search(user_text)
        and (current_mutation or _MUTATION_RE.search(history_text))
    )
    recent_write_failure = bool(
        (
            "mystand_authorization_write" in history_text
            or _MUTATION_RE.search(history_text)
        )
        and _TOOL_FAILURE_RE.search(history_text)
    )
    if not (
        current_mutation
        or confirmation_after_mutation
        or pressure_during_mutation
        or recent_write_failure
    ):
        return ""

    return (
        "【My Stand 本轮诚信强制提醒】\n"
        "自然语言回复不能改变网站状态；预览不等于提交，历史聊天里的成功自述不是证据。\n"
        "只有当前回合实际调用写入工具，并收到对应 commit_write 的 "
        "authorization-write-receipt-v2、ok=true 且 verified=true 回执，"
        "才可以说写入、修改、保存、删除或发布成功。\n"
        "工具没有调用、调用失败、只完成 preview_write，或回执无法核实时，"
        "必须明确说没有确认发生变化；不得猜测、补写或借用旧回合回执。\n"
        "情绪或催促越强，越要停下来核对当前回合工具证据；"
        "不得为了安抚用户编造进展、理由或完成状态。"
    )


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _tool_calls(messages: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        if str(message.get("role") or "") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            function = function if isinstance(function, Mapping) else {}
            call_id = str(call.get("id") or "")
            name = str(function.get("name") or call.get("name") or "")
            arguments = _parse_json_object(
                function.get("arguments")
                if "arguments" in function
                else call.get("arguments")
            )
            if call_id and name:
                calls[call_id] = (name, arguments)
    return calls


def _current_turn_messages(result: Any) -> list[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        return []
    raw_messages = result.get("messages")
    if not isinstance(raw_messages, list):
        return []
    messages = [message for message in raw_messages if isinstance(message, Mapping)]
    last_user_index = -1
    for index, message in enumerate(messages):
        if str(message.get("role") or "") == "user":
            last_user_index = index
    if last_user_index < 0:
        # Without a current user boundary, reusing an older receipt is unsafe.
        return []
    return messages[last_user_index + 1 :]


def _current_turn_write_evidence(result: Any) -> str:
    messages = _current_turn_messages(result)
    calls = _tool_calls(messages)
    saw_preview = False
    saw_failed_write = False

    for message in messages:
        if str(message.get("role") or "") != "tool":
            continue
        call_id = str(message.get("tool_call_id") or "")
        tool_name = str(message.get("name") or "")
        call_name, arguments = calls.get(call_id, ("", {}))
        if not tool_name:
            tool_name = call_name
        if tool_name not in _WRITE_TOOL_NAMES:
            continue
        operation = str(arguments.get("operation") or "").strip()
        payload = _parse_json_object(message.get("content"))
        ok = payload.get("ok") is True
        verified = payload.get("verified") is True
        receipt_version = str(payload.get("receiptVersion") or "")
        if (
            operation == "commit_write"
            and ok
            and verified
            and receipt_version == "authorization-write-receipt-v2"
        ):
            return "verified_commit"
        if operation == "preview_write" and ok:
            saw_preview = True
        else:
            saw_failed_write = True

    if saw_failed_write:
        return "failed_write"
    if saw_preview:
        return "preview_only"
    return "no_write_call"


def _contains_positive_mutation_success_claim(text: str) -> bool:
    for marker in _SUCCESS_MARKERS:
        start = 0
        while True:
            index = text.find(marker, start)
            if index < 0:
                break
            prefix = text[max(0, index - 28) : index]
            if not _NEGATION_OR_CONDITION_RE.search(prefix):
                return True
            start = index + len(marker)
    return False


def guard_mutation_success_claim(text: Any, result: Any) -> IntegrityGuardDecision:
    """Block unsupported mutation-success claims at the API egress boundary."""

    final_text = str(text or "")
    if not isinstance(result, Mapping) or result.get("_mystand_request") is not True:
        return IntegrityGuardDecision(final_text, False, "not_mystand")
    if not _contains_positive_mutation_success_claim(final_text):
        return IntegrityGuardDecision(final_text, False, "no_success_claim")

    evidence_status = _current_turn_write_evidence(result)
    if evidence_status == "verified_commit":
        return IntegrityGuardDecision(final_text, False, evidence_status)
    if evidence_status == "preview_only":
        replacement = PREVIEW_ONLY_MESSAGE
    elif evidence_status == "failed_write":
        replacement = FAILED_WRITE_MESSAGE
    else:
        replacement = NO_WRITE_EVIDENCE_MESSAGE
    return IntegrityGuardDecision(replacement, True, evidence_status)
