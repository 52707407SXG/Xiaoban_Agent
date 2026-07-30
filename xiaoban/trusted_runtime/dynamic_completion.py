"""Dynamic-evidence-v2 completion projection and verification.

This module owns the v2-only completion contract.  The public guard keeps
protocol routing and delegates deterministic projection, binding validation,
and receipt construction here.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from xiaoban.trusted_runtime.fact_contract import canonical_digest
from xiaoban.trusted_runtime.turns import serialize_allowed_facts
from xiaoban.trusted_runtime.types import (
    ACTION_OUTPUT_CONTRACTS,
    CompletionDecision,
    MYSTAND_COMPLETION_BINDING_FIELDS,
    MYSTAND_COMPLETION_PROTOCOL_V2,
    MYSTAND_COMPLETION_VERIFICATION_SCHEMA_V2,
    WorkTurn,
)


NO_EVIDENCE_MESSAGE = (
    "这轮我没有真正查到站内资料，所以不能给出具体的资料内容、数值或状态。"
)
_MAX_COMPLETION_TEXT = 4_000
_DYNAMIC_ACTION_IDS = frozenset(ACTION_OUTPUT_CONTRACTS)
DYNAMIC_READ_NOT_DISPATCHED = "read_not_dispatched_after_index"
DYNAMIC_ACTION_NOT_DISPATCHED = "action_not_dispatched"
DYNAMIC_READ_PRECONDITION_NOT_MET = "read_precondition_not_met"
DYNAMIC_ACTION_RESULT_MISSING = "action_result_missing"
DYNAMIC_INDEX_INCOMPLETE = "index_incomplete"
_HARD_PREACTION_ERRORS = frozenset(
    {
        "duplicate_call_id",
        "missing_datascope",
        "missing_identity",
        "missing_turn_id",
        "not_in_catalog",
        "preaction_error",
        "unknown_action",
        "write_isolated",
    }
)
_TRANSIENT_TIMEOUT_CODES = frozenset(
    {
        "deadline_exceeded",
        "gateway_timeout",
        "handler_timeout",
        "provider_timeout",
        "read_timeout",
        "request_timeout",
        "timed_out",
        "timeout",
        "upstream_timeout",
    }
)
_TRANSIENT_UNAVAILABLE_CODES = frozenset(
    {
        "connection_error",
        "connection_failed",
        "econnrefused",
        "econnreset",
        "mystand_authorization_transport_failed",
        "mystand_query_transport_failed",
        "network_unavailable",
        "provider_unavailable",
        "service_unavailable",
        "upstream_unavailable",
    }
)
_TRANSIENT_RECOVERY_CODES = (
    _TRANSIENT_TIMEOUT_CODES | _TRANSIENT_UNAVAILABLE_CODES
)
_PRESENTATION_UNAVAILABLE_CODES = _TRANSIENT_UNAVAILABLE_CODES | frozenset(
    {
        # These codes are safe to explain as an unavailable site-data
        # connection, but they are deliberately not recoverable: retrying a
        # missing bridge/configuration cannot fix it and only spends another
        # paid call.
        "mystand_authorization_unavailable",
        "mystand_query_unavailable",
        "mystand_resource_index_transport_failed",
        "mystand_resource_index_unavailable",
    }
)
_FAILURE_INCOMPLETE_RE = re.compile(
    r"(?:"
    r"(?:没有|没能|未能|无法|不能|尚未|还没有|还没|暂时无法|暂时不能)"
    r"[^。！？；，,]{0,24}"
    r"(?:完成|办完|处理完|查完|读完|继续|回答|拿到|查到|读到|读取|返回)"
    r"|(?:失败|中断|超时|没成功|未成功|被拒绝|无权读取|没有读取权限)"
    r"|(?:未完成|没有完成|还未完成|还没完成)"
    r")"
)
_FAILURE_INTERNAL_RE = re.compile(
    r"(?:"
    r"系统提示|固定回复|动态证据|证据回执|回执|内部协议|协议校验|"
    r"实例|运行环境|状态码|错误码|明确点击重试|点击重试|"
    r"mystand_(?:query|resource_index|authorization)|"
    r"\b(?:tool|function|api|gateway|delivery|receipt|protocol|"
    r"status(?:\s*code)?|error(?:_code)?|exception|traceback)\b|"
    r"(?:^|[\s(])/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._-]+|"
    r"<[/!]?[a-zA-Z][^>]*>|```|[{}]"
    r")",
    re.IGNORECASE,
)
_FAILURE_UNBOUND_FACT_RE = re.compile(
    r"(?:"
    r"[0-9０-９]|[￥¥$€£%％]|"
    r"[零〇一二三四五六七八九十百千万亿两]+(?:多|余)?"
    r"(?:元|块|个|位|名|套|份|笔|条|项|户|人|家)|"
    r"(?:查询结果|数据|资料|事实|结论|答案)"
    r"[^。！？；，,]{0,12}(?:显示|表明|是|为)|"
    r"(?:还有|共有|总共|合计|总计)"
    r")"
)
_FAILURE_BUSINESS_SUBJECT_RE = re.compile(
    r"(?:"
    r"提成|佣金|业绩|金额|余额|结算|到账|房源|客源|客户|业主|"
    r"手机号|电话号码?|成交|合同|账本|流水"
    r")"
)
_FAILURE_POSITIVE_RESULT_RE = re.compile(
    r"(?:"
    r"(?:已经|已)(?:查到|读到|拿到|取得|完成|办完|处理完|"
    r"读取完成|查询完成|成功|确认)|"
    r"(?:工资|薪资|商铺|店铺|房源|记录|老板|领导|客户|业主)"
    r"[^。！？；，,]{0,16}"
    r"(?:发放|出租|删除|批准|成交|到账|完成|成功)"
    r")"
)
_FAILURE_NEGATED_CAUSE_RE = re.compile(
    r"(?:"
    r"(?:没有|没|并无|不存在)(?:任何)?(?:问题|错误|失败|异常)|"
    r"(?:一切|状态|结果)?(?:正常|无误|没问题)|"
    r"(?:已经|已)(?:确认|核实)"
    r")"
)
_FAILURE_PERMISSION_CAUSE_RE = re.compile(
    r"(?:权限|授权|无权|拒绝|禁止|访问条件)"
)
_FAILURE_TIMEOUT_CAUSE_RE = re.compile(
    r"(?:超时|网络|连接|断线|服务不可用|服务中断)"
)
_FAILURE_NOT_FOUND_RE = re.compile(
    r"(?:没有找到|没找到|未找到|找不到|无法定位|不能确定|"
    r"不够明确|唯一匹配|存在歧义)"
)
_FAILURE_NO_PROGRESS_RE = re.compile(
    r"(?:"
    r"(?:定位|找到|核对).{0,18}(?:资料|范围|目录|候选)"
    r"|(?:资料|范围|目录|候选).{0,18}(?:定位|找到|核对)"
    r")"
)
_FAILURE_NO_READ_RE = re.compile(
    r"(?:"
    r"没有|没能|未能|尚未|还没|无法"
    r").{0,20}(?:继续读取|读取|读到|拿到|取得|完成查询|查完)"
)
_FAILURE_NOT_STARTED_RE = re.compile(
    r"(?:"
    r"(?:没有|没能|未能|尚未|还没).{0,18}"
    r"(?:开始|发起|实际处理|执行)"
    r"|(?:实际处理|执行).{0,18}(?:没有|没能|未能)(?:开始|发起)"
    r")"
)
_FAILURE_FIRST_PERSON_PREFIX = (
    r"(?:抱歉[，,]|不好意思[，,])?"
    r"(?:我(?:这次|本轮)?|(?:这次|本轮)我)"
)
_FAILURE_INCOMPLETE_TAIL = (
    r"(?:所以|因此|目前|现在)?"
    r"(?:"
    r"(?:这项|这次|本轮)?(?:任务|查询)"
    r"(?:还|仍然|仍|尚)?(?:没有|没|没能|未能|未)"
    r"(?:完成|办完|处理完|查完)"
    r"|(?:我)?(?:目前|暂时|现在)?(?:还)?(?:无法|不能|没法)"
    r"(?:完成(?:这项|这次|本轮)?任务|"
    r"给(?:你)?(?:可靠|准确|明确)?(?:答复|结果|答案)|"
    r"确认(?:最终)?结果|继续回答)"
    r")"
)
_FAILURE_NO_PROGRESS_INDEX_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}"
    r"(?:已经|已)?(?:完成|做完)(?:了)?资料目录"
    r"(?:查询|核对)"
    r"[，,；;。](?:但|不过)?(?:这次|本轮)?我?"
    r"(?:没有|没能|未能|还没|尚未)继续"
    r"(?:读取|读到|拿到|取得)(?:到)?"
    r"(?:能回答问题的|可用于回答的|可回答的)?"
    r"(?:内容|正文|结果)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_NOT_STARTED_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}"
    r"(?:没有|没能|未能|还没|尚未)"
    r"(?:发起|开始)(?:实际)?(?:处理|执行)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_NOT_FOUND_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}"
    r"(?:没有找到|没找到|未找到|找不到|无法定位|不能确定)"
    r"(?:能够|可以)?(?:唯一)?(?:匹配的)?(?:相关)?"
    r"(?:目标|资料|记录|对象|候选|内容)"
    r"(?:"
    r"[，,；;。](?:需要|请)你补充(?:更)?(?:准确|具体|明确)的"
    r"(?:名称|范围|信息)"
    rf"|[，,；;。]{_FAILURE_INCOMPLETE_TAIL}"
    r")[。！？!?]?$"
)
_FAILURE_DENIED_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}"
    r"(?:没有|没能|未能|无法)"
    r"(?:取得|获得|通过)?(?:完成任务所需的)?"
    r"(?:读取|访问)?(?:权限|授权|访问条件)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_EMPTY_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}(?:已经|已)?"
    r"(?:发起|完成)?(?:了)?(?:读取|查询|处理)"
    r"[，,；;。](?:但|不过)?我?"
    r"(?:没有|没能|未能|无法)"
    r"(?:读到|拿到|取得)(?:可用|有效|完整)?(?:的)?"
    r"(?:内容|结果|正文|资料)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_GENERIC_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}(?:已经|已)?"
    r"(?:(?:发起|开始|尝试)(?:了)?(?:实际)?(?:处理|执行|查询|读取)|处理)"
    r"[，,；;。](?:但|不过)?(?:处理|执行|查询|读取)?"
    r"(?:没有成功|未成功|失败|出了问题|返回了错误)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_TIMEOUT_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}(?:已经|已)?"
    r"(?:发起|开始|尝试)(?:了)?(?:实际)?(?:处理|执行|查询|读取)"
    r"[，,；;。](?:但|不过)?(?:等待|处理|执行|查询|读取)?"
    r"(?:结果)?(?:超时|超过等待时间)(?:了)?"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_UNAVAILABLE_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}(?:已经|已)?"
    r"(?:发起|开始|尝试)(?:了)?(?:实际)?(?:处理|执行|查询|读取)"
    r"[，,；;。](?:但|不过)?"
    r"(?:连接失败|网络中断|服务暂时不可用|读取服务暂时不可用)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_CANCELLED_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}(?:已经|已)?"
    r"(?:发起|开始)(?:了)?(?:实际)?(?:处理|执行|查询|读取)"
    r"[，,；;。](?:但|不过)?(?:随后)?"
    r"(?:被停止|被取消|已经停止|已经取消)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_READ_PRECONDITION_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}"
    r"(?:没有|没能|未能)先(?:完成|做好)"
    r"(?:资料)?(?:目录查询|资料定位|前置准备)"
    r"[，,；;。](?:所以|因此)?(?:正文)?读取"
    r"(?:没有|没能|未能)(?:发起|开始)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_RESULT_MISSING_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}(?:的)?(?:处理|查询|读取)请求"
    r"(?:已经|已)(?:生成|准备好|登记)"
    r"[，,；;。](?:但|不过)?(?:没有|没能|未能)"
    r"(?:形成|收到|拿到|取得)(?:完整|最终|可以确认|可确认)?"
    r"(?:的)?(?:结果|返回内容)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_INDEX_INCOMPLETE_FULL_RE = re.compile(
    rf"^{_FAILURE_FIRST_PERSON_PREFIX}(?:已经|已)?"
    r"(?:发起|完成)(?:了)?资料目录(?:查询|核对)"
    r"[，,；;。](?:但|不过)?(?:返回|拿到|取得)(?:的)?"
    r"(?:目录|结果|内容)(?:不完整|无法确认)"
    rf"[，,；;。]{_FAILURE_INCOMPLETE_TAIL}[。！？!?]?$"
)
_FAILURE_NEGATIVE_CLAUSE_RE = re.compile(
    r"(?:"
    r"失败|错误|出错|问题|异常|超时|不完整|"
    r"(?:没有|无)响应|"
    r"(?:服务|网络|连接).{0,8}(?:不可用|中断|失败)|"
    r"(?:停止|取消)|权限|授权|无权|拒绝|禁止|"
    r"(?:还)?(?:没有|没|未)(?:找到|查到|读到|等到|拿到|"
    r"取得|完成|成功|通过)|"
    r"(?:无法|不能|没法|没能|未能).{0,12}"
    r"(?:完成|继续|答复|确认|回答|读取|查询|发起|开始)"
    r")"
)
_FAILURE_DIRECT_ASSERTION_RE = re.compile(
    r"(?:"
    r"(?:遇到|遇到了|出现|出现了|发生|发生了)"
    r"(?:错误|问题|异常)|"
    r"失败|出错|错误|有问题|出了?问题|异常|超时|不完整|"
    r"(?:没有|无)响应|(?:暂时)?不可用|被(?:停止|取消)|"
    r"(?:还)?(?:没有|没|未)(?:成功|找到|查到|读到|等到|拿到|"
    r"取得|完成|通过)|"
    r"(?:暂时)?(?:没能|未能|无法|不能|没法)"
    r"(?:完成|继续|答复|确认|回答|读取|查询|发起|开始|确定|定位)|"
    r"(?:没有|没能|未能|无法|不能|没法)?(?:权限|授权)"
    r")"
)
_FAILURE_DIRECT_ASSERTION_PREFIX_RE = re.compile(
    r"(?:"
    r"(?:这次|本轮)?我(?:这次|本轮)?|"
    r"(?:处理|执行|查询|读取|读|查|连接|网络|服务|目录|索引|"
    r"任务|请求)(?:时|后来|随后)?|"
    r"(?:等待(?:结果)?|遇到|遇到了|出现|出现了|发生|发生了|"
    r"出了|返回|返回了)"
    r")$"
)
_FAILURE_FIRST_PERSON_EXECUTION_BINDING_RE = re.compile(
    r"(?:(?:这次|本轮)?我|我(?:这次|本轮)?)"
    r"[^，,；;。！？!?]{0,14}"
    r"(?:"
    r"(?:已经|已)?(?:发起|开始|尝试|处理|执行|查询|查|读取|"
    r"读|等待)|"
    r"(?:还)?(?:没有|没|未)(?:找到|查到|读到|等到|拿到|"
    r"取得|完成|成功|权限)|"
    r"(?:没能|未能|无法|不能|没法)(?:取得|获得|找到|定位|"
    r"确定|完成|继续|答复|确认|回答|读取|查询|发起|开始|给)|"
    r"(?:权限|授权)"
    r")"
)
_FAILURE_CURRENT_EXECUTION_BINDING_RE = re.compile(
    r"^(?:这次|本轮)(?:的)?(?:实际)?"
    r"(?:处理|执行|查询|读取|任务|请求)"
)
_FAILURE_UNBOUND_SCOPE_RE = re.compile(
    r"(?:项目|计划|申请|订单|工单|审批)"
)
_FAILURE_SAFE_CONTINUATION_FULL_RE = re.compile(
    r"^(?:"
    r"(?:等待(?:结果)?|连接|网络|(?:读取)?服务|处理|执行|查询|"
    r"读取).{0,10}(?:超时|失败|中断|不可用|(?:没有|无)响应|"
    r"没成功|未成功|出了?问题|返回了?错误|被停止|被取消)|"
    r"(?:一直)?(?:没有|没|没能|未能)(?:继续)?"
    r"(?:等到|查到|读到|拿到|取得).{0,18}"
    r"(?:结果|内容|正文|资料)|"
    r"(?:暂时)?(?:无法|不能|没法|没能|未能)(?:继续)?"
    r"(?:完成(?:你的请求|这项任务|任务)?|"
    r"给你(?:可靠|准确|明确)?(?:答复|结果|答案)|"
    r"确认(?:最终)?(?:结果)?|回答(?:你的问题)?|继续回答)|"
    r"(?:这项|这次|本轮)?(?:任务|查询|请求)"
    r"(?:还|仍然|仍|尚)?(?:没有|没|未能|未)?完成|"
    r"(?:没能|未能)(?:继续)?完成|"
    r"(?:处理|执行|查询|读取)(?:后来)?被(?:停止|取消)"
    r")了?$"
)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        candidate = vars(value)
    except TypeError:
        return {}
    return candidate if isinstance(candidate, Mapping) else {}


def _canonical_digest(value: Any) -> str:
    return canonical_digest(value)


def evidence_receipt_digest(items: Sequence[Any]) -> str:
    return _canonical_digest(
        [
            {
                "evidence_id": item.evidence_id,
                "turn_id": item.turn_id,
                "call_id": item.call_id,
                "action_id": item.action_id,
                "datascope_fingerprint": item.datascope_fingerprint,
                "status": item.status,
                "allowed_facts": item.allowed_facts,
                "record_refs": item.record_refs,
                "input_digest": item.input_digest,
                "output_digest": item.output_digest,
                "requirement_digest": item.requirement_digest,
                "coverage_digest": item.coverage_digest,
                "verification_status": item.verification_status,
            }
            for item in sorted(
                items,
                key=lambda item: (
                    str(item.call_id),
                    str(item.action_id),
                    str(item.evidence_id),
                ),
            )
        ]
    )


def _completion_binding_valid(turn: WorkTurn) -> bool:
    binding = _mapping(getattr(turn, "completion_binding", None))
    identity = turn.identity
    return bool(
        turn.completion_protocol == MYSTAND_COMPLETION_PROTOCOL_V2
        and identity is not None
        and set(binding) == MYSTAND_COMPLETION_BINDING_FIELDS
        and binding.get("user_id") == identity.account_id
        and binding.get("datascope_fingerprint")
        == identity.datascope_fingerprint
        and binding.get("delivery_id") == turn.request_id
        and binding.get("message_id") == turn.message_id
        and isinstance(binding.get("session_id"), str)
        and binding.get("session_id")
        and isinstance(binding.get("attempt"), int)
        and not isinstance(binding.get("attempt"), bool)
        and binding.get("attempt") >= 1
        and re.fullmatch(
            r"[a-f0-9]{64}",
            str(binding.get("request_fingerprint") or ""),
        )
        and re.fullmatch(
            r"[a-f0-9]{64}",
            str(binding.get("invocation_fingerprint") or ""),
        )
    )


def _completion_receipt(
    turn: WorkTurn,
    *,
    completion_kind: str,
    action_count: int,
    evidence_count: int,
    output: str,
    decision: str,
) -> dict[str, Any]:
    binding = _mapping(turn.completion_binding)
    return {
        "schema": MYSTAND_COMPLETION_VERIFICATION_SCHEMA_V2,
        "completion_kind": completion_kind,
        "binding_verified": True,
        "semantic_verified": False,
        "delivery_id": str(binding.get("delivery_id") or ""),
        "request_id": turn.request_id,
        "attempt": binding.get("attempt"),
        "message_id": turn.message_id,
        "request_fingerprint": str(
            binding.get("request_fingerprint") or ""
        ),
        "invocation_fingerprint": str(
            binding.get("invocation_fingerprint") or ""
        ),
        "datascope_fingerprint": str(
            binding.get("datascope_fingerprint") or ""
        ),
        "action_count": action_count,
        "evidence_count": evidence_count,
        "output_digest": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "decision": decision,
        "verified_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
    }


def _dynamic_index_binding(
    turn: WorkTurn,
) -> tuple[str, list[Mapping[str, Any]]]:
    """Bind a model-chosen read to this turn's complete server index."""
    receipt = turn.index_receipt
    identity = turn.identity
    if (
        receipt is None
        or identity is None
        or receipt.status != "found"
        or receipt.request_id != turn.request_id
        or receipt.actor_fingerprint != identity.datascope_fingerprint
        or receipt.scope_summary != "mystand_resource_index"
        or not receipt.source_call_id
        or receipt.has_more is not False
        or receipt.resource_count <= 0
        or receipt.resource_count != len(set(receipt.matched_resource_refs))
        or receipt.resource_refs_digest
        != _canonical_digest(sorted(set(receipt.matched_resource_refs)))
    ):
        return "", []
    matching_calls = [
        call
        for call in turn.action_calls
        if call.call_id == receipt.source_call_id
        and call.action_id == "mystand_resource_index"
        and call.version == "v1"
    ]
    matching_results = [
        result
        for result in turn.action_results
        if result.call_id == receipt.source_call_id
        and result.action_id == "mystand_resource_index"
        and result.status == "success"
    ]
    if len(matching_calls) != 1 or len(matching_results) != 1:
        return "", []
    call = matching_calls[0]
    result = matching_results[0]
    if (
        result.started_at != call.requested_at
        or result.finished_at != receipt.loaded_at
        or not result.raw_text
    ):
        return "", []
    try:
        raw_payload = json.loads(result.raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return "", []
    raw_items = (
        raw_payload.get("items")
        if isinstance(raw_payload, Mapping)
        else None
    )
    if (
        not isinstance(raw_payload, Mapping)
        or raw_payload.get("schema")
        not in {
            "mystand.resource-index.page.v1",
            "mystand.resource-index.complete.v1",
        }
        or raw_payload.get("ok") is not True
        or raw_payload.get("hasMore") is not False
        or raw_payload.get("nextCursor") not in (None, "")
        or not isinstance(raw_items, list)
        or not raw_items
        or _canonical_digest(raw_payload)
        != _canonical_digest(result.normalized_payload)
    ):
        return "", []
    listed_refs: list[str] = []
    projected_items: list[Mapping[str, Any]] = []
    for item in raw_items:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("resourceUid"), str)
            or not item.get("resourceUid")
            or not isinstance(item.get("safeLabel"), str)
            or not item.get("safeLabel").strip()
            or not isinstance(item.get("resourceType"), str)
            or not item.get("resourceType").strip()
            or not isinstance(item.get("canRead"), bool)
            or not isinstance(item.get("locked"), bool)
        ):
            return "", []
        listed_refs.append(item["resourceUid"])
        projected_items.append(item)
    if (
        len(listed_refs) != len(set(listed_refs))
        or sorted(listed_refs)
        != sorted(set(receipt.matched_resource_refs))
        or len(listed_refs) != receipt.resource_count
    ):
        return "", []
    digest = _canonical_digest(
        {
            "receipt": _mapping(receipt),
            "action_call": _mapping(call),
            "action_result": {
                **_mapping(result),
                "raw_text": None,
                "output_digest": hashlib.sha256(
                    result.raw_text.encode("utf-8")
                ).hexdigest(),
            },
        }
    )
    return digest, projected_items


def _hard_runtime_violation(turn: WorkTurn) -> bool:
    if turn.orphaned_receipts or turn.rejected_cross_account:
        return True
    return any(
        result.error_code in _HARD_PREACTION_ERRORS
        for result in turn.action_results
    )


def _validated_terminal_text(final_text: str) -> str:
    text = str(final_text or "")
    if not text.strip() or len(text) > _MAX_COMPLETION_TEXT or "\x00" in text:
        return ""
    return text


def _natural_failure_clauses_are_execution_bound(
    clauses: Sequence[str],
) -> bool:
    """Reject negative facts that are not about this turn's own execution."""
    execution_bound = False
    semantic_units = [
        unit
        for clause in clauses
        for unit in re.split(r"(?=(?:但|不过|所以|因此))", clause)
        if unit
    ]
    for clause in semantic_units:
        body = re.sub(
            r"^(?:抱歉|不好意思|但|不过|所以|因此|目前|现在)",
            "",
            clause,
        )
        explicitly_bound = bool(
            _FAILURE_FIRST_PERSON_EXECUTION_BINDING_RE.search(body)
            or _FAILURE_CURRENT_EXECUTION_BINDING_RE.search(body)
        )
        has_negative_status = bool(_FAILURE_NEGATIVE_CLAUSE_RE.search(body))
        if explicitly_bound:
            if (
                has_negative_status
                and _FAILURE_UNBOUND_SCOPE_RE.search(body)
            ):
                return False
            for assertion in _FAILURE_DIRECT_ASSERTION_RE.finditer(body):
                if not _FAILURE_DIRECT_ASSERTION_PREFIX_RE.search(
                    body[:assertion.start()]
                ):
                    return False
            execution_bound = True
            continue
        if not has_negative_status:
            continue
        if (
            not execution_bound
            or _FAILURE_UNBOUND_SCOPE_RE.search(body)
            or not _FAILURE_SAFE_CONTINUATION_FULL_RE.fullmatch(body)
        ):
            return False
    return True


def _validated_natural_failure_text(
    final_text: str,
    *,
    failure_class: str,
    failure_reason: str = "",
) -> bool:
    """Accept only a whole, status-bound explanation with no free-form tail."""
    text = re.sub(r"\s+", "", str(final_text or "").strip())
    if (
        not text
        or _FAILURE_INTERNAL_RE.search(text)
        or _FAILURE_UNBOUND_FACT_RE.search(text)
        or _FAILURE_BUSINESS_SUBJECT_RE.search(text)
    ):
        return False
    if failure_class == "no_progress":
        patterns = {
            DYNAMIC_ACTION_NOT_DISPATCHED: _FAILURE_NOT_STARTED_FULL_RE,
            DYNAMIC_READ_NOT_DISPATCHED: _FAILURE_NO_PROGRESS_INDEX_FULL_RE,
            DYNAMIC_READ_PRECONDITION_NOT_MET:
                _FAILURE_READ_PRECONDITION_FULL_RE,
            DYNAMIC_ACTION_RESULT_MISSING: _FAILURE_RESULT_MISSING_FULL_RE,
            DYNAMIC_INDEX_INCOMPLETE: _FAILURE_INDEX_INCOMPLETE_FULL_RE,
        }
        pattern = patterns.get(failure_reason)
    elif failure_class in {"not_found", "ambiguous"}:
        pattern = _FAILURE_NOT_FOUND_FULL_RE
    elif failure_class == "denied":
        pattern = _FAILURE_DENIED_FULL_RE
    elif failure_class == "empty":
        pattern = _FAILURE_EMPTY_FULL_RE
    elif failure_reason == "timeout":
        pattern = _FAILURE_TIMEOUT_FULL_RE
    elif failure_reason == "unavailable":
        pattern = _FAILURE_UNAVAILABLE_FULL_RE
    elif failure_class == "cancelled":
        pattern = _FAILURE_CANCELLED_FULL_RE
    else:
        pattern = _FAILURE_GENERIC_FULL_RE
    if pattern and pattern.fullmatch(text):
        return True

    # Models naturally vary word order and connective words.  Keep the
    # category and safety binding strict, but do not require one memorized
    # sentence grammar.
    if (
        len(text) > 180
        or "我" not in text
        or _FAILURE_POSITIVE_RESULT_RE.search(text)
        or _FAILURE_NEGATED_CAUSE_RE.search(text)
    ):
        return False
    clauses = [
        clause
        for clause in re.split(r"[，,；;。！？!?]+", text)
        if clause
    ]
    if not 1 <= len(clauses) <= 4:
        return False
    if not _natural_failure_clauses_are_execution_bound(clauses):
        return False

    incomplete = bool(
        _FAILURE_INCOMPLETE_RE.search(text)
        or re.search(
            r"(?:无法|不能|没法|没能).{0,16}"
            r"(?:答复|确认|完成|继续)",
            text,
        )
    )
    needs_detail = bool(
        re.search(
            r"(?:请|需要).{0,10}"
            r"(?:(?:补充|提供).{0,12}(?:名称|范围|信息|资料|对象)"
            r"|(?:再)?说(?:得)?(?:更)?具体(?:一点)?)",
            text,
        )
    )
    if not incomplete and not needs_detail:
        return False

    category_checks = {
        DYNAMIC_ACTION_NOT_DISPATCHED: lambda: bool(
            _FAILURE_NOT_STARTED_RE.search(text)
        ),
        DYNAMIC_READ_NOT_DISPATCHED: lambda: bool(
            _FAILURE_NO_PROGRESS_RE.search(text)
            and _FAILURE_NO_READ_RE.search(text)
        ),
        DYNAMIC_READ_PRECONDITION_NOT_MET: lambda: bool(
            re.search(r"(?:定位|目录|前置)", text)
            and re.search(
                r"(?:读取|查询).{0,12}(?:没有|没能|未能|无法)"
                r"(?:发起|开始|继续)",
                text,
            )
        ),
        DYNAMIC_ACTION_RESULT_MISSING: lambda: bool(
            re.search(r"(?:请求|处理).{0,12}(?:生成|登记|发起)", text)
            and re.search(
                r"(?:没有|没能|未能).{0,12}(?:结果|返回)",
                text,
            )
        ),
        DYNAMIC_INDEX_INCOMPLETE: lambda: bool(
            re.search(r"(?:目录|索引).{0,12}(?:不完整|无法确认)", text)
        ),
        "not_found": lambda: bool(_FAILURE_NOT_FOUND_RE.search(text)),
        "ambiguous": lambda: bool(_FAILURE_NOT_FOUND_RE.search(text)),
        "denied": lambda: bool(_FAILURE_PERMISSION_CAUSE_RE.search(text)),
        "empty": lambda: bool(
            _FAILURE_NO_READ_RE.search(text)
            or re.search(
                r"(?:没有|没能|未能).{0,12}"
                r"(?:查到|读到|拿到).{0,12}"
                r"(?:回答|答复|确认|内容|结果|资料)",
                text,
            )
        ),
        "timeout": lambda: bool(
            re.search(r"(?:超时|等待时间|没等到|没有等到)", text)
        ),
        "unavailable": lambda: bool(
            re.search(
                r"(?:连接失败|网络中断|服务(?:暂时)?不可用|"
                r"读取服务不可用|服务(?:没有|无)响应)",
                text,
            )
        ),
        "cancelled": lambda: bool(re.search(r"(?:停止|取消)", text)),
        "execution_error": lambda: bool(
            re.search(r"(?:失败|错误|问题|没成功|未成功)", text)
        ),
        "mixed": lambda: bool(
            re.search(r"(?:失败|错误|问题|没成功|未成功)", text)
        ),
    }
    category = (
        failure_reason if failure_class == "no_progress" else failure_reason
    )
    category_check = category_checks.get(category)
    if category_check is None or not category_check():
        return False

    common_clause = re.compile(
        r"(?:"
        r"^(?:抱歉|不好意思)$|"
        r"(?:我|这次|本轮).{0,28}"
        r"(?:发起|开始|尝试|处理|执行|查询|查|读取)|"
        r"(?:没有|没|没能|未能|无法|不能|尚未|还没|暂时无法|暂时不能)|"
        r"(?:失败|错误|问题|超时|停止|取消|权限|授权|连接|网络|"
        r"服务不可用|服务暂时不可用|响应|不完整|找不到|未找到|没找到)|"
        r"(?:请|需要).{0,12}(?:补充|提供|再说|说得)"
        r")"
    )
    return all(common_clause.search(clause) for clause in clauses)


def _failure_reason_category(
    failure_class: str,
    failures: Sequence[Any],
) -> str:
    """Reduce handler-controlled codes to a safe user-facing cause class."""
    if failure_class != "error":
        return failure_class
    error_codes = {
        str(result.error_code or "").strip().lower()
        for result in failures
        if str(result.error_code or "").strip()
    }
    if error_codes and error_codes <= _TRANSIENT_TIMEOUT_CODES:
        return "timeout"
    if error_codes and error_codes <= _PRESENTATION_UNAVAILABLE_CODES:
        return "unavailable"
    return "execution_error"


def dynamic_transient_recovery_plan(
    turn: WorkTurn,
) -> Optional[dict[str, Any]]:
    """Allow one caller-controlled recovery turn for a transient read failure.

    This function only authenticates the current failure.  The conversation
    loop owns the one-shot budget, so a second failed physical call always
    proceeds to the normal failure finalizer.
    """
    if (
        turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2
        or turn.fact_requirement is not None
        or turn.evidence
        or _hard_runtime_violation(turn)
    ):
        return None
    lifecycle = _validated_failure_lifecycle(
        turn,
        include_single_preaction=True,
    )
    if lifecycle is None:
        return None
    _, failures = lifecycle
    if len(failures) != 1:
        return None
    failure = failures[0]
    failed_calls = [
        call
        for call in turn.action_calls
        if call.call_id == failure.call_id
        and call.action_id == failure.action_id
    ]
    if len(failed_calls) != 1:
        return None
    failed_call = failed_calls[0]
    error_code = str(failure.error_code or "").strip().lower()
    contract = ACTION_OUTPUT_CONTRACTS.get(failure.action_id)
    _, index_items = _dynamic_index_binding(turn)
    if (
        failure.status != "error"
        or contract is None
        # A failed index cannot safely "change path" because no trusted scope
        # exists yet; retrying it would still need another read + finalizer and
        # can only add cost.  Recovery is therefore limited to one failed read
        # after a complete owner-bound index.
        or contract.kind != "read"
        or not index_items
        or error_code not in _TRANSIENT_RECOVERY_CODES
    ):
        return None
    reason = (
        "timeout"
        if error_code in _TRANSIENT_TIMEOUT_CODES
        else "unavailable"
    )
    safe_scope = [
        {
            "resourceUid": str(item["resourceUid"]),
            "safeLabel": str(item["safeLabel"]),
            "resourceType": str(item["resourceType"]),
            "canRead": bool(item["canRead"]),
            "locked": bool(item["locked"]),
        }
        for item in index_items
    ]
    indexed_by_ref = {
        str(item.get("resourceUid") or ""): item
        for item in index_items
        if str(item.get("resourceUid") or "")
    }
    retry_refs: list[str] = []
    for key in ("resource_uid", "authorization_id"):
        value = failed_call.arguments.get(key)
        if isinstance(value, str) and value.strip():
            retry_refs.append(value.strip())
    for key in (
        "record_refs",
        "recordRefs",
        "resource_refs",
        "resourceRefs",
    ):
        values = failed_call.arguments.get(key)
        if isinstance(values, list):
            retry_refs.extend(
                value.strip()
                for value in values
                if isinstance(value, str) and value.strip()
            )
    retry_refs = sorted(set(retry_refs))
    if (
        not retry_refs
        or _indexed_read_refs(
            payload={},
            required_refs=retry_refs,
            indexed_by_ref=indexed_by_ref,
        )
        != retry_refs
    ):
        return None
    return {
        "reason": reason,
        "state": (
            "上一次只读处理等待超时"
            if reason == "timeout"
            else "上一次只读处理遇到暂时不可用"
        ),
        "safe_scope": safe_scope,
        "retry": {
            "action_id": failed_call.action_id,
            "version": failed_call.version,
            "arguments": dict(failed_call.arguments),
            "arguments_digest": _canonical_digest(
                failed_call.arguments
            ),
        },
    }


def dynamic_transient_recovery_tool_call_valid(
    turn: WorkTurn,
    *,
    action_id: str,
    arguments: Mapping[str, Any],
) -> bool:
    """Accept only the exact server-recorded read selected for one recovery."""
    plan = dynamic_transient_recovery_plan(turn)
    retry = plan.get("retry") if plan else None
    return bool(
        isinstance(retry, Mapping)
        and str(action_id or "") == str(retry.get("action_id") or "")
        and isinstance(arguments, Mapping)
        and _canonical_digest(dict(arguments))
        == str(retry.get("arguments_digest") or "")
    )


def dynamic_failure_presentation(turn: WorkTurn) -> Optional[dict[str, str]]:
    """Project one truthful, prompt-safe failure state from runtime facts."""
    no_progress = _validated_no_progress_failure(turn)
    if no_progress is not None:
        failure_class = "no_progress"
        failure_reason = str(no_progress["reason"])
    else:
        lifecycle = _validated_failure_lifecycle(
            turn,
            include_single_preaction=True,
        )
        if lifecycle is None:
            return None
        _, failures = lifecycle
        statuses = sorted({str(result.status) for result in failures})
        failure_class = statuses[0] if len(statuses) == 1 else "mixed"
        failure_reason = _failure_reason_category(failure_class, failures)
    presentations = {
        DYNAMIC_READ_NOT_DISPATCHED: (
            "资料目录查询已完成，但没有继续读取正文",
            "我完成了资料目录查询，但没有继续读取到能回答问题的内容，"
            "所以这项任务还没有完成。",
        ),
        DYNAMIC_ACTION_NOT_DISPATCHED: (
            "没有发起实际处理",
            "我这次没有发起实际处理，所以这项任务还没有完成。",
        ),
        DYNAMIC_READ_PRECONDITION_NOT_MET: (
            "没有先完成资料定位，因此正文读取没有发起",
            "我这次没能先完成资料定位，所以正文读取没有发起，"
            "这项任务还没有完成。",
        ),
        DYNAMIC_ACTION_RESULT_MISSING: (
            "处理请求已生成，但没有形成可确认结果",
            "我这次的处理请求已经生成，但没有形成可以确认的结果，"
            "所以这项任务还没有完成。",
        ),
        DYNAMIC_INDEX_INCOMPLETE: (
            "资料目录查询已发起，但返回的目录不完整",
            "我这次发起了资料目录查询，但返回的目录不完整，"
            "所以这项任务还没有完成。",
        ),
        "empty": (
            "读取已发起，但没有取得可回答内容",
            "我这次已经发起读取，但没能读到可用内容，"
            "所以这项任务还没有完成。",
        ),
        "not_found": (
            "没有找到能够唯一匹配的资料",
            "我这次没有找到能够唯一匹配的资料，"
            "需要你补充更准确的名称。",
        ),
        "denied": (
            "没有取得完成任务所需的读取权限",
            "我这次没能取得完成任务所需的读取权限，"
            "所以这项任务还没有完成。",
        ),
        "ambiguous": (
            "读取目标无法唯一确认",
            "我这次不能确定唯一匹配的资料，"
            "需要你补充更准确的名称。",
        ),
        "timeout": (
            "实际处理已发起，但等待结果超时",
            "我这次已经发起实际处理，但等待结果超时，"
            "所以这项任务还没有完成。",
        ),
        "unavailable": (
            "实际处理已发起，但读取服务暂时不可用",
            "我这次已经发起实际处理，但读取服务暂时不可用，"
            "所以这项任务还没有完成。",
        ),
        "cancelled": (
            "实际处理已发起，但随后被停止",
            "我这次已经发起实际处理，但随后被停止，"
            "所以这项任务还没有完成。",
        ),
        "mixed": (
            "实际处理已发起，但其中有步骤没有成功",
            "我这次已经发起实际处理，但处理没有成功，"
            "所以这项任务还没有完成。",
        ),
        "execution_error": (
            "实际处理已发起，但执行返回了错误",
            "我这次已经发起实际处理，但执行返回了错误，"
            "所以这项任务还没有完成。",
        ),
    }
    state, example = presentations.get(
        failure_reason,
        presentations["execution_error"],
    )
    return {
        "failure_class": failure_class,
        "failure_reason": failure_reason,
        "state": state,
        "example": example,
    }


def _matched_action_lifecycle(
    turn: WorkTurn,
) -> Optional[list[tuple[Any, Any]]]:
    calls: dict[str, Any] = {}
    results: dict[str, Any] = {}
    for call in turn.action_calls:
        if call.call_id in calls:
            return None
        calls[call.call_id] = call
    for result in turn.action_results:
        if result.call_id not in calls:
            continue
        if result.call_id in results:
            return None
        results[result.call_id] = result
    if not calls or set(calls) != set(results):
        return None
    matched: list[tuple[Any, Any]] = []
    for call_id in sorted(calls):
        call = calls[call_id]
        result = results[call_id]
        contract = ACTION_OUTPUT_CONTRACTS.get(call.action_id)
        if (
            contract is None
            or contract.version != call.version
            or result.action_id != call.action_id
            or result.started_at != call.requested_at
        ):
            return None
        matched.append((call, result))
    return matched


def _record_refs_for_paths(
    paths: Sequence[str],
    payload: Mapping[str, Any],
) -> list[str]:
    refs: list[str] = []
    for path in paths:
        if path == "recordRefs[]":
            refs.extend(
                str(value)
                for value in payload.get("recordRefs") or []
                if isinstance(value, str) and value
            )
        elif path == "resource.resourceUid":
            resource = payload.get("resource")
            if isinstance(resource, Mapping) and resource.get("resourceUid"):
                refs.append(str(resource["resourceUid"]))
        elif payload.get(path):
            refs.append(str(payload[path]))
    return refs


def _expected_read_record_refs(
    contract: Any,
    payload: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> list[str]:
    refs = _record_refs_for_paths(contract.record_ref_paths, payload)
    for key in ("resource_uid", "authorization_id"):
        if arguments.get(key):
            refs.append(str(arguments[key]))
    return sorted(set(refs))


def _reference_payload_valid(
    contract: Any,
    payload: Mapping[str, Any],
) -> bool:
    for path in contract.record_ref_paths:
        if path == "recordRefs[]" and "recordRefs" in payload:
            values = payload.get("recordRefs")
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, str) or not value
                    for value in values
                )
                or values != sorted(set(values))
            ):
                return False
        elif path == "resource.resourceUid":
            resource = payload.get("resource")
            if (
                isinstance(resource, Mapping)
                and "resourceUid" in resource
                and (
                    not isinstance(resource.get("resourceUid"), str)
                    or not resource.get("resourceUid")
                )
            ):
                return False
        elif path in payload and (
            not isinstance(payload.get(path), str)
            or not payload.get(path)
        ):
            return False
    return True


def _argument_resource_binding_valid(
    payload: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> bool:
    bindings = (
        (
            "resource_uid",
            [
                payload.get("resourceUid"),
                (
                    payload.get("resource", {}).get("resourceUid")
                    if isinstance(payload.get("resource"), Mapping)
                    else None
                ),
            ],
        ),
        ("authorization_id", [payload.get("authorizationId")]),
    )
    for argument_key, candidates in bindings:
        expected = arguments.get(argument_key)
        observed = [
            value
            for value in candidates
            if value not in (None, "")
        ]
        if expected not in (None, "") and observed and any(
            value != expected for value in observed
        ):
            return False
    return True


def _required_index_refs(
    contract: Any,
    payload: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> list[str]:
    index_paths = tuple(
        path
        for path in contract.record_ref_paths
        if "resource" in path.lower() or path == "recordRefs[]"
    )
    refs = _record_refs_for_paths(index_paths, payload)
    if arguments.get("resource_uid"):
        refs.append(str(arguments["resource_uid"]))
    return sorted(set(refs))


def _indexed_read_refs(
    *,
    payload: Mapping[str, Any],
    required_refs: Sequence[str],
    indexed_by_ref: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if any(ref not in indexed_by_ref for ref in required_refs):
        return []
    indexed_refs = sorted(set(required_refs))
    if not indexed_refs:
        resource = payload.get("resource")
        if not isinstance(resource, Mapping):
            return []
        display_name = resource.get("display_name")
        resource_type = resource.get("type")
        if (
            not isinstance(display_name, str)
            or not display_name.strip()
            or not isinstance(resource_type, str)
            or not resource_type.strip()
        ):
            return []
        matches = [
            ref
            for ref, item in indexed_by_ref.items()
            if item.get("safeLabel") == display_name
            and item.get("resourceType") == resource_type
        ]
        if len(matches) != 1:
            return []
        indexed_refs = matches
    if any(
        indexed_by_ref[ref].get("canRead") is not True
        or indexed_by_ref[ref].get("locked") is not False
        or indexed_by_ref[ref].get("status") == "locked"
        for ref in indexed_refs
    ):
        return []
    return indexed_refs


def _validated_read_evidence(
    turn: WorkTurn,
    matched: Sequence[tuple[Any, Any]],
    index_items: Sequence[Mapping[str, Any]],
) -> Optional[tuple[list[Any], list[str]]]:
    identity = turn.identity
    if identity is None:
        return None
    indexed_by_ref = {
        str(item.get("resourceUid")): item for item in index_items
    }
    evidence_by_call: dict[str, list[Any]] = {}
    for item in turn.evidence:
        evidence_by_call.setdefault(item.call_id, []).append(item)
    verified: list[Any] = []
    public_refs: set[str] = set()
    successful_reads = [
        (call, result)
        for call, result in matched
        if ACTION_OUTPUT_CONTRACTS[call.action_id].kind == "read"
        and result.status == "success"
    ]
    if not successful_reads:
        return None
    for call, result in successful_reads:
        contract = ACTION_OUTPUT_CONTRACTS[call.action_id]
        items = evidence_by_call.get(call.call_id, [])
        if len(items) != 1 or not result.raw_text:
            return None
        item = items[0]
        try:
            payload = json.loads(result.raw_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping):
            return None
        if (
            not _reference_payload_valid(contract, payload)
            or not _argument_resource_binding_valid(
                payload,
                call.arguments,
            )
        ):
            return None
        expected_refs = _expected_read_record_refs(
            contract,
            payload,
            call.arguments,
        )
        required_refs = _required_index_refs(
            contract,
            payload,
            call.arguments,
        )
        indexed_refs = _indexed_read_refs(
            payload=payload,
            required_refs=required_refs,
            indexed_by_ref=indexed_by_ref,
        )
        if (
            not indexed_refs
            or _canonical_digest(payload)
            != _canonical_digest(result.normalized_payload)
            or item.turn_id != turn.turn_id
            or item.call_id != call.call_id
            or item.action_id != call.action_id
            or item.evidence_id
            != hashlib.sha256(
                f"{turn.turn_id}|{call.call_id}".encode("utf-8")
            ).hexdigest()[:16]
            or item.datascope_fingerprint
            != identity.datascope_fingerprint
            or item.status != "success"
            or item.verification_status != "verified"
            or item.verified_at != result.finished_at
            or item.allowed_facts
            != serialize_allowed_facts(call.action_id, dict(payload))
            or list(item.record_refs) != expected_refs
            or item.input_digest
            != hashlib.sha256(
                json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            or item.output_digest
            != hashlib.sha256(
                result.raw_text.encode("utf-8")
            ).hexdigest()
            or item.requirement_digest
            or item.coverage_digest
        ):
            return None
        verified.append(item)
        public_refs.update(indexed_refs)
    if len(verified) != len(turn.evidence):
        return None
    return verified, sorted(public_refs)


def _validated_transient_recovery_results(
    turn: WorkTurn,
    matched: Sequence[tuple[Any, Any]],
) -> Optional[list[Any]]:
    """Bind one transient failure to a later exact-target successful retry."""
    non_success = [
        (call, result)
        for call, result in matched
        if result.status != "success"
    ]
    if not non_success:
        return []
    if len(non_success) != 1:
        return None
    failed_call, failed_result = non_success[0]
    failed_contract = ACTION_OUTPUT_CONTRACTS.get(failed_call.action_id)
    error_code = str(failed_result.error_code or "").strip().lower()
    if (
        failed_result.status != "error"
        or failed_contract is None
        or failed_contract.kind != "read"
        or error_code not in _TRANSIENT_RECOVERY_CODES
    ):
        return None
    ordered_calls = {
        call.call_id: index
        for index, call in enumerate(turn.action_calls)
    }
    failed_position = ordered_calls.get(failed_call.call_id)
    failed_arguments_digest = _canonical_digest(failed_call.arguments)
    if failed_position is None:
        return None
    post_failure = [
        (call, result)
        for call, result in matched
        if (
            ordered_calls.get(call.call_id, -1) > failed_position
        )
    ]
    if len(post_failure) != 1:
        return None
    recovered_call, recovered_result = post_failure[0]
    recovered = (
        recovered_result.status == "success"
        and recovered_call.action_id == failed_call.action_id
        and recovered_call.version == failed_call.version
        and _canonical_digest(recovered_call.arguments)
        == failed_arguments_digest
    )
    return [failed_result] if recovered else None


def _dynamic_evidence_completion(
    turn: WorkTurn,
    final_text: str,
) -> CompletionDecision:
    """Authenticate successful reads while preserving the model's answer."""
    identity = turn.identity
    output = _validated_terminal_text(final_text)
    index_receipt_digest, index_items = _dynamic_index_binding(turn)
    matched = _matched_action_lifecycle(turn)
    if (
        identity is None
        or not output
        or not _completion_binding_valid(turn)
        or not index_receipt_digest
        or _hard_runtime_violation(turn)
        or matched is None
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_evidence_binding",
        )
    validated = _validated_read_evidence(turn, matched, index_items)
    if validated is None:
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_action_binding",
        )
    verified_evidence, record_refs = validated
    transient_failures = _validated_transient_recovery_results(
        turn,
        matched,
    )
    if transient_failures is None:
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_recovery_binding",
        )
    receipt = turn.index_receipt
    verification = {
        **_completion_receipt(
            turn,
            completion_kind="evidence-bound",
            action_count=len(matched),
            evidence_count=len(verified_evidence),
            output=output,
            decision="evidence_access_verified",
        ),
        "index_count": receipt.resource_count,
        "index_resource_refs_digest": receipt.resource_refs_digest,
        "index_has_more": receipt.has_more,
        "index_receipt_digest": index_receipt_digest,
        "record_refs": record_refs,
        "record_refs_digest": _canonical_digest(record_refs),
        "evidence_digest": evidence_receipt_digest(verified_evidence),
    }
    if transient_failures:
        verification.update(
            {
                "transient_failure_count": len(transient_failures),
                "transient_action_result_digest": _canonical_digest(
                    [
                        {
                            "call_id": result.call_id,
                            "action_id": result.action_id,
                            "status": result.status,
                            "error_code": result.error_code,
                            "payload_digest": _canonical_digest(
                                result.normalized_payload
                            ),
                            "started_at": result.started_at,
                            "finished_at": result.finished_at,
                        }
                        for result in transient_failures
                    ]
                ),
            }
        )
    return CompletionDecision(
        True,
        output,
        "evidence_access_verified",
        verification,
    )


def _dynamic_failure_completion(
    turn: WorkTurn,
    final_text: str,
) -> CompletionDecision:
    """Bind a natural explanation to real non-success execution state."""
    output = _validated_terminal_text(final_text)
    failure_lifecycle = _validated_failure_lifecycle(
        turn,
        include_single_preaction=True,
    )
    no_progress_failure = _validated_no_progress_failure(turn)
    if (
        getattr(turn, "completion_finalization", "") != "failure"
        or not output
        or not _completion_binding_valid(turn)
        or _hard_runtime_violation(turn)
        or turn.evidence
        or (
            failure_lifecycle is None
            and no_progress_failure is None
        )
        or getattr(
            turn,
            "completion_finalization_output_digest",
            "",
        )
        != hashlib.sha256(output.encode("utf-8")).hexdigest()
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_failure_binding",
        )
    if no_progress_failure is not None:
        action_count = no_progress_failure["action_count"]
        failure_class = "no_progress"
        action_result_digest = _canonical_digest(no_progress_failure)
        failed_action_count = 0
    else:
        action_count, failures = failure_lifecycle
        failure_statuses = sorted({result.status for result in failures})
        failure_class = (
            failure_statuses[0]
            if len(failure_statuses) == 1
            else "mixed"
        )
        action_result_digest = _canonical_digest(
            [
                {
                    "call_id": result.call_id,
                    "action_id": result.action_id,
                    "status": result.status,
                    "error_code": result.error_code,
                    "payload_digest": _canonical_digest(
                        result.normalized_payload
                    ),
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                }
                for result in failures
            ]
        )
        failed_action_count = len(failures)
    presentation = dynamic_failure_presentation(turn)
    if (
        presentation is None
        or presentation["failure_class"] != failure_class
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_failure_binding",
        )
    failure_reason = presentation["failure_reason"]
    if not _validated_natural_failure_text(
        output,
        failure_class=failure_class,
        failure_reason=failure_reason,
    ):
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_failure_presentation",
        )
    verification = {
        **_completion_receipt(
            turn,
            completion_kind="failure-bound",
            action_count=action_count,
            evidence_count=0,
            output=output,
            decision="execution_status_bound",
        ),
        "action_result_digest": action_result_digest,
        "failed_action_count": failed_action_count,
        "failure_class": failure_class,
    }
    return CompletionDecision(
        True,
        output,
        "execution_status_bound",
        verification,
    )


def _validated_failure_lifecycle(
    turn: WorkTurn,
    *,
    include_single_preaction: bool,
) -> Optional[tuple[int, list[Any]]]:
    """Return server-recorded failures without treating zero work as failure."""
    matched = _matched_action_lifecycle(turn)
    if turn.action_calls and matched is None:
        return None
    matched = matched or []
    _ = include_single_preaction  # retained for call-site compatibility
    matched_failures = [
        result
        for call, result in matched
        if ACTION_OUTPUT_CONTRACTS[call.action_id].kind in {"index", "read"}
        and result.status != "success"
    ]
    # A natural failure reply may describe only a handler that really ran and
    # returned a bound non-success result.  PreAction denials are protocol
    # errors/no-dispatch states, never user-facing execution evidence.
    failures = sorted(
        matched_failures,
        key=lambda result: (
            str(result.call_id),
            str(result.action_id),
            str(result.error_code),
        ),
    )
    if not failures:
        return None
    return len(matched), failures


def _validated_no_progress_failure(
    turn: WorkTurn,
) -> Optional[dict[str, Any]]:
    """Authenticate a server-observed no-dispatch execution failure."""
    reason = getattr(turn, "completion_execution_failure", "")
    if (
        reason not in {
            DYNAMIC_READ_NOT_DISPATCHED,
            DYNAMIC_ACTION_NOT_DISPATCHED,
            DYNAMIC_READ_PRECONDITION_NOT_MET,
            DYNAMIC_ACTION_RESULT_MISSING,
            DYNAMIC_INDEX_INCOMPLETE,
        }
        or turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2
        or turn.fact_requirement is not None
        or turn.evidence
        or _hard_runtime_violation(turn)
    ):
        return None
    if reason == DYNAMIC_ACTION_NOT_DISPATCHED:
        if (
            turn.interaction_kind != "WORK"
            or turn.index_receipt is not None
            or turn.action_calls
            or turn.action_results
        ):
            return None
        return {
            "schema": "mystand.dynamic-execution-failure.v1",
            "reason": DYNAMIC_ACTION_NOT_DISPATCHED,
            "action_count": 0,
        }
    if reason == DYNAMIC_READ_PRECONDITION_NOT_MET:
        denials = list(turn.action_results)
        if (
            turn.interaction_kind != "WORK"
            or turn.action_calls
            or not denials
            or turn.pre_action_denials != len(denials)
            or any(
                result.status != "denied"
                or result.error_code != "missing_index_receipt"
                or ACTION_OUTPUT_CONTRACTS.get(result.action_id) is None
                or ACTION_OUTPUT_CONTRACTS[result.action_id].kind != "read"
                for result in denials
            )
        ):
            return None
        return {
            "schema": "mystand.dynamic-execution-failure.v1",
            "reason": DYNAMIC_READ_PRECONDITION_NOT_MET,
            "action_count": 0,
            "denial_digest": _canonical_digest(
                [
                    {
                        "call_id": result.call_id,
                        "action_id": result.action_id,
                        "status": result.status,
                        "error_code": result.error_code,
                    }
                    for result in denials
                ]
            ),
        }
    if reason == DYNAMIC_ACTION_RESULT_MISSING:
        calls = list(turn.action_calls)
        if not calls:
            return None
        call_ids = {call.call_id for call in calls}
        result_ids = [result.call_id for result in turn.action_results]
        if (
            len(call_ids) != len(calls)
            or len(result_ids) != len(set(result_ids))
            or any(result_id not in call_ids for result_id in result_ids)
            or call_ids == set(result_ids)
        ):
            return None
        return {
            "schema": "mystand.dynamic-execution-failure.v1",
            "reason": DYNAMIC_ACTION_RESULT_MISSING,
            "action_count": len(calls),
            "lifecycle_digest": _canonical_digest(
                {
                    "calls": [
                        {
                            "call_id": call.call_id,
                            "action_id": call.action_id,
                            "version": call.version,
                        }
                        for call in calls
                    ],
                    "results": [
                        {
                            "call_id": result.call_id,
                            "action_id": result.action_id,
                            "status": result.status,
                            "error_code": result.error_code,
                        }
                        for result in turn.action_results
                    ],
                }
            ),
        }
    if reason == DYNAMIC_INDEX_INCOMPLETE:
        matched = _matched_action_lifecycle(turn)
        receipt = turn.index_receipt
        if (
            matched is None
            or not matched
            or receipt is None
            or receipt.status != "unavailable"
            or any(
                call.action_id != "mystand_resource_index"
                or result.status != "success"
                for call, result in matched
            )
        ):
            return None
        return {
            "schema": "mystand.dynamic-execution-failure.v1",
            "reason": DYNAMIC_INDEX_INCOMPLETE,
            "action_count": len(matched),
            "index_receipt_digest": _canonical_digest(_mapping(receipt)),
        }
    index_receipt_digest, _ = _dynamic_index_binding(turn)
    matched = _matched_action_lifecycle(turn)
    if (
        not index_receipt_digest
        or matched is None
        or not matched
        or any(
            call.action_id != "mystand_resource_index"
            or result.status != "success"
            for call, result in matched
        )
        or any(
            ACTION_OUTPUT_CONTRACTS.get(call.action_id)
            and ACTION_OUTPUT_CONTRACTS[call.action_id].kind == "read"
            for call in turn.action_calls
        )
    ):
        return None
    return {
        "schema": "mystand.dynamic-execution-failure.v1",
        "reason": reason,
        "action_count": len(matched),
        "index_receipt_digest": index_receipt_digest,
    }


def mark_dynamic_read_no_progress(turn: WorkTurn) -> bool:
    """Mark a complete index lookup that never dispatched the required read."""
    previous = getattr(turn, "completion_execution_failure", "")
    turn.completion_execution_failure = DYNAMIC_READ_NOT_DISPATCHED
    if _validated_no_progress_failure(turn) is not None:
        return True
    turn.completion_execution_failure = previous
    return False


def mark_dynamic_action_no_progress(turn: WorkTurn) -> bool:
    """Mark a trusted work turn where no site action was dispatched."""
    previous = getattr(turn, "completion_execution_failure", "")
    turn.completion_execution_failure = DYNAMIC_ACTION_NOT_DISPATCHED
    if _validated_no_progress_failure(turn) is not None:
        return True
    turn.completion_execution_failure = previous
    return False


def mark_dynamic_execution_no_progress(turn: WorkTurn) -> bool:
    """Authenticate every safe unfinished lifecycle before finalization."""
    previous = getattr(turn, "completion_execution_failure", "")
    candidates = (
        DYNAMIC_READ_NOT_DISPATCHED,
        DYNAMIC_READ_PRECONDITION_NOT_MET,
        DYNAMIC_ACTION_RESULT_MISSING,
        DYNAMIC_INDEX_INCOMPLETE,
        DYNAMIC_ACTION_NOT_DISPATCHED,
    )
    for reason in candidates:
        turn.completion_execution_failure = reason
        if _validated_no_progress_failure(turn) is not None:
            return True
    turn.completion_execution_failure = previous
    return False


def dynamic_finalization_mode(
    turn: WorkTurn,
    *,
    include_single_preaction: bool = False,
) -> str:
    """Derive whether the next paid call must be the no-tool final reply."""
    if (
        turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2
        or turn.fact_requirement is not None
        or not _completion_binding_valid(turn)
        or _hard_runtime_violation(turn)
    ):
        return ""
    matched = _matched_action_lifecycle(turn)
    if turn.evidence and matched is not None:
        index_receipt_digest, index_items = _dynamic_index_binding(turn)
        if (
            index_receipt_digest
            and _validated_read_evidence(turn, matched, index_items)
            is not None
        ):
            return "evidence"
    if not turn.evidence and _validated_failure_lifecycle(
        turn,
        include_single_preaction=include_single_preaction,
    ) is not None:
        return "failure"
    if not turn.evidence and _validated_no_progress_failure(turn) is not None:
        return "failure"
    return ""


def check_dynamic_completion(
    turn: WorkTurn,
    *,
    final_text: str,
    failure_message: str,
) -> Optional[CompletionDecision]:
    """Return None when legacy completion routing must continue."""
    if turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2:
        return None
    if turn.fact_requirement is not None:
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_completion_protocol_mixed",
        )
    if getattr(turn, "completion_finalization", "") == "not_executed":
        return CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_dynamic_not_executed",
        )
    if (
        getattr(turn, "completion_finalization", "") == "failure"
        and _validated_no_progress_failure(turn) is not None
    ):
        return _dynamic_failure_completion(turn, final_text)
    action_ids = {
        call.action_id for call in turn.action_calls
    } | {
        result.action_id for result in turn.action_results
    }
    if not action_ids.intersection(_DYNAMIC_ACTION_IDS):
        return None
    if turn.evidence:
        return _dynamic_evidence_completion(turn, final_text)
    if getattr(turn, "completion_finalization", "") == "failure":
        return _dynamic_failure_completion(turn, final_text)
    if not turn.action_calls:
        reason = "blocked_no_action_call"
    elif not turn.action_results:
        reason = "blocked_no_action_result"
    else:
        reason = "blocked_no_evidence"
    return CompletionDecision(False, failure_message, reason)


def validate_dynamic_result_protocol(
    result: Mapping[str, Any],
    signed_fact_requirement: Any,
) -> tuple[bool, Optional[CompletionDecision]]:
    """Validate top-level result routing before a trusted turn is accepted."""
    completion_protocol = str(
        result.get("_mystand_completion_protocol") or ""
    )
    dynamic_completion = (
        completion_protocol == MYSTAND_COMPLETION_PROTOCOL_V2
    )
    if completion_protocol and not dynamic_completion:
        return dynamic_completion, CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_completion_protocol",
        )
    if dynamic_completion and isinstance(signed_fact_requirement, Mapping):
        return dynamic_completion, CompletionDecision(
            False,
            NO_EVIDENCE_MESSAGE,
            "blocked_completion_protocol_mixed",
        )
    return dynamic_completion, None


def dynamic_result_turn_binding_valid(
    result: Mapping[str, Any],
    turn: WorkTurn,
) -> bool:
    return bool(
        turn.completion_protocol
        == str(result.get("_mystand_completion_protocol") or "")
        and _mapping(turn.completion_binding)
        == _mapping(result.get("_mystand_completion_binding"))
        and _completion_binding_valid(turn)
    )
