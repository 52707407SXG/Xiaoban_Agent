"""High-level semantic query bridge for authenticated My Stand API sessions."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from gateway.session_context import (
    get_session_env,
    get_session_user_message,
    mark_mystand_private_query_turn,
)
from tools.registry import registry
from xiaoban.trusted_runtime.fact_contract import (
    build_fact_query_plan,
    evidence_requirement_digest,
    normalized_fact_query_text,
)

_DEFAULT_API_URL = "http://127.0.0.1:18081"
_DEFAULT_ENV_FILE = "/opt/xiaoban-agent/.env"
_INTERNAL_QUERY_PATH = "/api/xiaoban/internal/query"
_MAX_RESPONSE_BYTES = 65_536
_INTERNAL_TOKEN_KEYS = (
    "MYSTAND_XIAOBAN_MYSTAND_API_TOKEN",
    "MYSTAND_XIAOBAN_GATEWAY_INTERNAL_TOKEN",
)
_CONTRACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "contracts"
    / "mystand-query-tool.v1.json"
)
MYSTAND_QUERY_CONTRACT = json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))
MYSTAND_QUERY_SCHEMA = MYSTAND_QUERY_CONTRACT["tool"]

_TOP_LEVEL_KEYS = {
    "operation",
    "query_kind",
    "module_id",
    "fact_paths",
    "query_args",
    "coverage_required",
    "resource",
    "entities",
    "fact_needs",
    "mode",
}
_RESOURCE_KEYS = {"name", "type_hint"}
_ENTITY_KEYS = {"kind", "value", "role"}
_RESOURCE_TYPE_HINTS = {
    "note",
    "knowledge-markdown",
    "knowledge-graph",
    "property-note",
    "business-archive",
    "profile-card",
    "property-data",
    "property-md",
    "finance-archive",
}
_ENTITY_KINDS = {
    "building",
    "unit",
    "room",
    "person",
    "estate",
    "document",
    "topic",
    "time",
}
_ENTITY_ROLES = {
    "resource",
    "locator",
    "subject",
    "time",
    "topic",
    "attribute",
}
_FACT_NEEDS = {
    "owner.name",
    "owner.phone",
    "owner.family",
    "owner.interests",
    "owner.economic",
    "relationship.communication",
    "relationship.followup",
    "property.parking",
    "property.area",
    "property.price.total",
    "property.price.unit",
    "property.rent",
    "document.content",
    "resource.summary",
    "graph.nodes",
    "graph.relations",
}
_TYPED_QUERY_KINDS = {
    "resource-read",
    "rank",
    "predicate",
    "count",
    "list",
    "aggregate",
}
_TYPED_QUERY_FIELDS = frozenset(
    {
        "query_kind",
        "module_id",
        "fact_paths",
        "query_args",
        "coverage_required",
    }
)
_SEMANTIC_QUERY_FIELDS = frozenset(
    {
        "operation",
        "resource",
        "entities",
        "fact_needs",
        "mode",
    }
)
_FORBIDDEN_TYPED_KEYS = {
    "owner",
    "owner_user",
    "user",
    "user_id",
    "authorization_id",
    "auth_id",
    "resource_uid",
    "source_id",
    "scope_fingerprint",
    "requirement_digest",
    "plan_id",
    "queryText",
}
_FACT_PATH_RE = re.compile(r"[a-z][a-z0-9_.-]{1,119}")
_MYSTAND_REFERENCE_RE = re.compile(
    r"(?:AUTH|OUT)-[A-Z0-9][A-Z0-9-]{5,}[A-Z0-9]",
    re.IGNORECASE,
)
_INTERNAL_IDENTIFIER_RE = re.compile(
    r"(?:"
    r"\b(?:AUTH|OUT|KGREF|RESOURCEUID|SOURCEID|OWNERUSER|USERID|INTERNALID)"
    r"[-_:\s=][A-Za-z0-9._:@/-]+"
    r"|"
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
    r")",
    re.IGNORECASE,
)
_CANDIDATE_LABEL_KEYS = ("label", "safeLabel", "displayName")
_CANDIDATE_NAME_KEYS = ("name", "resourceName")
_CANDIDATE_TYPE_KEYS = ("type", "resourceType", "typeHint")


def _read_env_file_value(path: str, key: str) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"")
    return ""


def _internal_token() -> str:
    for key in _INTERNAL_TOKEN_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            return value
    env_file = os.getenv("MYSTAND_XIAOBAN_GATEWAY_ENV_FILE", _DEFAULT_ENV_FILE)
    for key in _INTERNAL_TOKEN_KEYS:
        value = _read_env_file_value(env_file, key)
        if value:
            return value
    return ""


def _api_base_url() -> str:
    value = os.getenv(
        "MYSTAND_XIAOBAN_MYSTAND_API_URL",
        _DEFAULT_API_URL,
    ).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        return ""
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return ""
    return value


def check_mystand_query() -> bool:
    return bool(_internal_token() and _api_base_url())


def _json_result(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _error(
    message: str,
    *,
    code: str = "mystand_query_failed",
    status: int = 400,
    retryable: bool | None = None,
    correction: dict | None = None,
) -> str:
    payload = {
        "ok": False,
        "status": status,
        "code": code,
        "error": message,
    }
    if isinstance(retryable, bool):
        payload["retryable"] = retryable
    if isinstance(correction, dict) and correction:
        payload["correction"] = correction
    return _json_result(payload)


def _safe_header(value: str, limit: int = 200) -> str:
    text = str(value or "")
    if len(text) > limit or not re.fullmatch(r"[A-Za-z0-9._:@-]+", text):
        return ""
    return text


def _safe_public_text(value, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _INTERNAL_IDENTIFIER_RE.search(text):
        return ""
    return text[:limit]


def _safe_error_code(value, fallback: str) -> str:
    text = str(value or "").strip()
    if (
        re.fullmatch(r"[a-z0-9_.-]{1,120}", text, re.IGNORECASE)
        and not _INTERNAL_IDENTIFIER_RE.search(text)
    ):
        return text
    return fallback


def _safe_candidates(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    safe_items = []
    for item in value[:8]:
        if isinstance(item, str):
            label = _safe_public_text(item, limit=160)
            if label:
                safe_items.append({"label": label})
            continue
        if not isinstance(item, dict):
            continue
        safe_item = {}
        for output_key, source_keys, limit in (
            ("label", _CANDIDATE_LABEL_KEYS, 160),
            ("name", _CANDIDATE_NAME_KEYS, 160),
            ("type", _CANDIDATE_TYPE_KEYS, 80),
        ):
            for source_key in source_keys:
                text = _safe_public_text(item.get(source_key), limit=limit)
                if text:
                    safe_item[output_key] = text
                    break
        candidate_type = safe_item.get("type")
        if candidate_type and candidate_type not in _RESOURCE_TYPE_HINTS:
            safe_item.pop("type", None)
        if safe_item:
            safe_items.append(safe_item)
    return safe_items


def _http_error_result(status: int, parsed) -> str:
    data = parsed if isinstance(parsed, dict) else {}
    if status in {404, 409}:
        details = data.get("details")
        details = details if isinstance(details, dict) else {}
        clarification = _safe_public_text(
            data.get("clarification") or details.get("clarification"),
            limit=300,
        )
        candidates = _safe_candidates(
            data.get("candidates")
            if isinstance(data.get("candidates"), list)
            else details.get("candidates")
        )
        fallback_code = (
            "mystand_query_not_found"
            if status == 404
            else "mystand_query_ambiguous"
        )
        result = {
            "ok": False,
            "status": status,
            "code": _safe_error_code(data.get("code"), fallback_code),
            "error": (
                "没有找到匹配的资料。"
                if status == 404
                else (
                    "找到多项可能资料，需要补充信息。"
                    if candidates
                    else clarification
                    or "没有找到唯一资料，需要补充信息。"
                )
            ),
        }
        if clarification:
            result["clarification"] = clarification
        if candidates:
            result["candidates"] = candidates
        return _json_result(result)

    public_error = _safe_public_text(
        data.get("error") or data.get("message"),
        limit=300,
    )
    return _error(
        public_error or "My Stand 拒绝了这次资料查询。",
        code=_safe_error_code(
            data.get("code"),
            "mystand_query_rejected",
        ),
        status=status,
    )


def _current_session() -> dict:
    return {
        "platform": get_session_env(
            "XIAOBAN_SESSION_PLATFORM",
            "",
        ).strip().lower(),
        "user_id": get_session_env("XIAOBAN_SESSION_USER_ID", "").strip(),
        "message_id": get_session_env("XIAOBAN_SESSION_MESSAGE_ID", "").strip(),
        "session_id": get_session_env("XIAOBAN_SESSION_ID", "").strip(),
    }


def _post_internal(payload: dict, session: dict) -> str:
    base_url = _api_base_url()
    token = _internal_token()
    if not base_url or not token:
        return _error(
            "My Stand 资料查询暂时不可用，请稍后重试。",
            code="mystand_query_unavailable",
            status=503,
        )
    safe_user_id = _safe_header(session.get("user_id", ""))
    if not safe_user_id:
        return _error(
            "当前 My Stand 登录身份无效。",
            code="mystand_session_required",
            status=403,
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "X-Xiaoban-User-Id": safe_user_id,
    }
    try:
        from xiaoban.trusted_runtime.turns import current_turn
        from xiaoban.trusted_runtime.types import (
            MYSTAND_COMPLETION_BINDING_FIELDS,
            MYSTAND_COMPLETION_PROTOCOL_V2,
        )

        turn = current_turn()
    except Exception:
        turn = None
    if turn is not None and turn.completion_protocol:
        binding = (
            dict(turn.completion_binding)
            if isinstance(turn.completion_binding, dict)
            else {}
        )
        attempt = binding.get("attempt")
        if (
            turn.completion_protocol != MYSTAND_COMPLETION_PROTOCOL_V2
            or set(binding) != MYSTAND_COMPLETION_BINDING_FIELDS
            or binding.get("user_id") != safe_user_id
            or binding.get("session_id") != session.get("session_id")
            or binding.get("delivery_id") != turn.request_id
            or binding.get("message_id") != turn.message_id
            or binding.get("message_id") != session.get("message_id")
            or isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
            or not re.fullmatch(
                r"xbd_[0-9a-f]{40}",
                str(binding.get("delivery_id") or ""),
            )
            or not re.fullmatch(
                r"[a-f0-9]{64}",
                str(binding.get("request_fingerprint") or ""),
            )
            or not re.fullmatch(
                r"[a-f0-9]{64}",
                str(binding.get("invocation_fingerprint") or ""),
            )
            or not re.fullmatch(
                r"[a-f0-9]{16}",
                str(binding.get("datascope_fingerprint") or ""),
            )
        ):
            return _error(
                "本轮事实查询没有通过可信完成绑定。",
                code="trusted_completion_binding_required",
                status=409,
            )
        headers.update(
            {
                "X-Xiaoban-Completion-Protocol": (
                    MYSTAND_COMPLETION_PROTOCOL_V2
                ),
                "X-Xiaoban-Delivery-Id": binding["delivery_id"],
                "X-Xiaoban-Delivery-Attempt": str(attempt),
                "X-Xiaoban-Attempt": str(attempt),
                "X-Xiaoban-Message-Id": binding["message_id"],
                "X-Xiaoban-Request-Fingerprint": (
                    binding["request_fingerprint"]
                ),
                "X-Xiaoban-Invocation-Fingerprint": (
                    binding["invocation_fingerprint"]
                ),
                "X-Xiaoban-Datascope-Fingerprint": (
                    binding["datascope_fingerprint"]
                ),
            }
        )
    for header_name, session_key in (
        ("X-Xiaoban-Message-Id", "message_id"),
        ("X-Xiaoban-Session-Id", "session_id"),
    ):
        safe_value = _safe_header(session.get(session_key, ""))
        if safe_value:
            headers[header_name] = safe_value
    request = urllib.request.Request(
        f"{base_url}{_INTERNAL_QUERY_PATH}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                return _error(
                    "My Stand 返回的查询结果过大，已停止读取。",
                    code="mystand_query_result_too_large",
                    status=413,
                )
            parsed = json.loads(raw.decode("utf-8")) if raw else {"ok": True}
            if not isinstance(parsed, dict):
                return _error(
                    "My Stand 返回了无效的查询结果。",
                    code="mystand_query_invalid_result",
                    status=502,
                )
            return _json_result(parsed)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(_MAX_RESPONSE_BYTES)
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            parsed = {}
        return _http_error_result(int(exc.code), parsed)
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return _error(
            "My Stand 资料查询暂时没有接稳，请稍后重试。",
            code="mystand_query_transport_failed",
            status=502,
        )


def _text(
    value,
    *,
    field: str,
    minimum: int = 1,
    maximum: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是字符串")
    text = value.strip()
    if len(text) < minimum or len(text) > maximum:
        raise ValueError(f"{field} 长度不在允许范围内")
    return text


def _typed_value_is_safe(value, *, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if value is None or isinstance(value, (str, int, float, bool)):
        return not isinstance(value, str) or len(value) <= 500
    if isinstance(value, list):
        return len(value) <= 24 and all(
            _typed_value_is_safe(item, depth=depth + 1) for item in value
        )
    if isinstance(value, dict):
        return (
            len(value) <= 24
            and all(
                isinstance(key, str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,79}", key)
                and key not in _FORBIDDEN_TYPED_KEYS
                and _typed_value_is_safe(item, depth=depth + 1)
                for key, item in value.items()
            )
        )
    return False


def _validate_typed_plan(args: dict) -> dict:
    query_kind = args.get("query_kind")
    if query_kind not in _TYPED_QUERY_KINDS:
        raise ValueError("query_kind 不在允许范围内")
    module_id = args.get("module_id")
    if (
        not isinstance(module_id, str)
        or len(module_id) > 80
        or (
            module_id
            and not re.fullmatch(r"[A-Za-z0-9._:@-]{1,80}", module_id)
        )
        or (query_kind != "resource-read" and not module_id)
    ):
        raise ValueError("module_id 不在允许范围内")
    fact_paths = args.get("fact_paths")
    if (
        not isinstance(fact_paths, list)
        or not fact_paths
        or len(fact_paths) > 12
        or len(set(fact_paths)) != len(fact_paths)
        or any(
            not isinstance(path, str) or not _FACT_PATH_RE.fullmatch(path)
            for path in fact_paths
        )
    ):
        raise ValueError("fact_paths 不在允许范围内")
    query_args = args.get("query_args")
    if not isinstance(query_args, dict) or not _typed_value_is_safe(query_args):
        raise ValueError("query_args 不在允许范围内")
    coverage_required = args.get("coverage_required")
    if (
        not isinstance(coverage_required, bool)
        or coverage_required is (query_kind == "resource-read")
    ):
        raise ValueError("coverage_required 不在允许范围内")
    if query_kind == "resource-read":
        if fact_paths != ["content"]:
            raise ValueError("resource-read fact_paths 不在允许范围内")
        semantic_digest = query_args.get("semanticQueryDigest")
        stable_reference = query_args.get("stableReference")
        if (
            not isinstance(semantic_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", semantic_digest)
            or (
                stable_reference is not None
                and (
                    not isinstance(stable_reference, str)
                    or not _MYSTAND_REFERENCE_RE.fullmatch(stable_reference)
                )
            )
            or set(query_args) - {"semanticQueryDigest", "stableReference"}
        ):
            raise ValueError("resource-read query_args 不在允许范围内")
    elif query_kind == "rank":
        if (
            fact_paths != ["finance.performance.rank"]
            or set(query_args) != {"year", "rank"}
            or isinstance(query_args.get("year"), bool)
            or not isinstance(query_args.get("year"), int)
            or not 2000 <= query_args["year"] <= 2200
            or isinstance(query_args.get("rank"), bool)
            or not isinstance(query_args.get("rank"), int)
            or not 1 <= query_args["rank"] <= 10_000
        ):
            raise ValueError("rank query_args 不在允许范围内")
    elif query_kind in {"predicate", "count"}:
        if (
            fact_paths != [f"finance.performance.{query_kind}"]
            or set(query_args) != {"year", "field", "operator", "amount"}
            or isinstance(query_args.get("year"), bool)
            or not isinstance(query_args.get("year"), int)
            or not 2000 <= query_args["year"] <= 2200
            or query_args.get("field") != "yearlyAmount"
            or query_args.get("operator") not in {"gt", "gte"}
            or isinstance(query_args.get("amount"), bool)
            or not isinstance(query_args.get("amount"), (int, float))
            or query_args["amount"] < 0
        ):
            raise ValueError(f"{query_kind} query_args 不在允许范围内")
    return {
        "operation": "read",
        "query_kind": query_kind,
        "module_id": module_id,
        "fact_paths": list(fact_paths),
        "query_args": json.loads(json.dumps(query_args, ensure_ascii=False)),
        "coverage_required": coverage_required,
    }


def _validate_plan(args) -> dict:
    if not isinstance(args, dict):
        raise ValueError("查询参数必须是对象")
    if set(args) - _TOP_LEVEL_KEYS:
        raise ValueError("查询参数包含不允许的字段")
    if args.get("operation") != "read":
        raise ValueError("operation 不在允许范围内")
    if "query_kind" in args:
        typed_fields = {
            "operation",
            "query_kind",
            "module_id",
            "fact_paths",
            "query_args",
            "coverage_required",
        }
        if set(args) != typed_fields:
            raise ValueError("typed 查询参数包含不允许的字段")
        return _validate_typed_plan(args)
    if _TYPED_QUERY_FIELDS.intersection(args):
        raise ValueError("semantic 查询参数包含 typed 字段")

    resource = args.get("resource")
    normalized_resource = None
    if resource is not None:
        if not isinstance(resource, dict):
            raise ValueError("resource 必须是对象")
        if set(resource) - _RESOURCE_KEYS:
            raise ValueError("resource 包含不允许的字段")
        name = _text(
            resource.get("name"),
            field="resource.name",
            minimum=2,
            maximum=240,
        )
        normalized_resource = {"name": name}
        if "type_hint" in resource:
            type_hint = resource.get("type_hint")
            if (
                not isinstance(type_hint, str)
                or type_hint not in _RESOURCE_TYPE_HINTS
            ):
                raise ValueError("resource.type_hint 不在允许范围内")
            normalized_resource["type_hint"] = type_hint

    entities = args.get("entities", [])
    if not isinstance(entities, list) or len(entities) > 12:
        raise ValueError("entities 必须是不超过 12 项的数组")
    normalized_entities = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(f"entities[{index}] 必须是对象")
        if set(entity) - _ENTITY_KEYS:
            raise ValueError(f"entities[{index}] 包含不允许的字段")
        kind = entity.get("kind")
        if not isinstance(kind, str) or kind not in _ENTITY_KINDS:
            raise ValueError(f"entities[{index}].kind 不在允许范围内")
        normalized_entity = {
            "kind": kind,
            "value": _text(
                entity.get("value"),
                field=f"entities[{index}].value",
                maximum=160,
            ),
        }
        if "role" in entity:
            role = entity.get("role")
            if not isinstance(role, str) or role not in _ENTITY_ROLES:
                raise ValueError(f"entities[{index}].role 不在允许范围内")
            normalized_entity["role"] = role
        normalized_entities.append(normalized_entity)

    fact_needs = args.get("fact_needs")
    if (
        not isinstance(fact_needs, list)
        or not fact_needs
        or len(fact_needs) > 12
        or any(
            not isinstance(fact, str) or fact not in _FACT_NEEDS
            for fact in fact_needs
        )
        or len(set(fact_needs)) != len(fact_needs)
    ):
        raise ValueError("fact_needs 不在允许范围内")

    mode = args.get("mode", "facts")
    if not isinstance(mode, str) or mode not in {"facts", "summary"}:
        raise ValueError("mode 不在允许范围内")

    if normalized_resource is None and not any(
        entity["kind"] in {"person", "estate", "document", "topic"}
        for entity in normalized_entities
    ):
        raise ValueError("缺少可定位资料的语义主体")

    return {
        "operation": "read",
        **(
            {"resource": normalized_resource}
            if normalized_resource is not None
            else {}
        ),
        "entities": normalized_entities,
        "fact_needs": list(fact_needs),
        "mode": mode,
    }


def validate_mystand_query_plan(args) -> dict:
    """Validate and normalize one provider-visible query call.

    Recovery validation deliberately reuses the handler's exact parser so the
    model-facing schema, the pre-dispatch correction gate, and physical
    execution cannot drift into three different definitions of "valid".
    """
    return _validate_plan(args)


def validate_mystand_semantic_query_plan(args) -> dict:
    """Validate the exact semantic-only shape exposed in dynamic reads."""
    if not isinstance(args, dict):
        raise ValueError("查询参数必须是对象")
    if set(args) - _SEMANTIC_QUERY_FIELDS:
        raise ValueError("semantic 查询参数包含不允许的字段")
    if "resource" not in args:
        raise ValueError("semantic 查询参数缺少 resource")
    payload = _validate_plan(args)
    if "query_kind" in payload:
        raise ValueError("dynamic read requires semantic 查询参数")
    return payload


def _dynamic_semantic_query_active() -> bool:
    try:
        from xiaoban.trusted_runtime.turns import current_turn
        from xiaoban.trusted_runtime.types import (
            MYSTAND_COMPLETION_PROTOCOL_V2,
        )

        turn = current_turn()
    except Exception:
        return False
    return bool(
        turn is not None
        and turn.completion_protocol == MYSTAND_COMPLETION_PROTOCOL_V2
        and turn.fact_requirement is None
        and not str(turn.completion_finalization or "")
    )


_SEMANTIC_CORRECTION = {
    "allowed_fields": [
        "operation",
        "resource",
        "entities",
        "fact_needs",
        "mode",
    ],
    "required_fields": ["operation", "resource", "fact_needs"],
    "locator_rule": "preserve the indexed resource; entities may only narrow it",
    "max_calls": 1,
}


def mystand_query_tool_handler(args, **_kwargs):
    session = _current_session()
    if session["platform"] != "api_server" or not session["user_id"]:
        return _error(
            "该资料查询只允许 My Stand 已登录网页/API 会话使用。",
            code="mystand_session_required",
            status=403,
        )
    dynamic_semantic = _dynamic_semantic_query_active()
    try:
        payload = (
            validate_mystand_semantic_query_plan(args)
            if dynamic_semantic
            else validate_mystand_query_plan(args)
        )
    except ValueError as exc:
        return _error(
            str(exc),
            code="invalid_mystand_query_arguments",
            retryable=dynamic_semantic,
            correction=(
                dict(_SEMANTIC_CORRECTION)
                if dynamic_semantic
                else None
            ),
        )
    mark_mystand_private_query_turn()
    if "query_kind" in payload:
        try:
            from xiaoban.trusted_runtime.turns import current_turn

            turn = current_turn()
            requirement = (
                turn.fact_requirement
                if turn is not None
                and isinstance(turn.fact_requirement, dict)
                else None
            )
            signed_plan = build_fact_query_plan(requirement)
            expected_plan = (
                _validate_plan(signed_plan)
                if isinstance(signed_plan, dict)
                else None
            )
            if expected_plan != payload:
                raise ValueError("typed 查询未绑定本轮事实要求")
            binding = requirement.get("binding")
            if not isinstance(binding, dict):
                raise ValueError("typed 查询缺少可信绑定")
            requirement_digest = evidence_requirement_digest(
                requirement,
                canonical_fallback=turn.fact_requirement_digest,
            )
            plan_id = str(requirement.get("plan_id") or "")
            scope_fingerprint = str(
                binding.get("datascope_fingerprint") or ""
            )
            if (
                not re.fullmatch(r"[a-f0-9]{64}", requirement_digest)
                or not re.fullmatch(r"[A-Za-z0-9._:@-]{1,160}", plan_id)
                or not re.fullmatch(r"[a-f0-9]{16,64}", scope_fingerprint)
            ):
                raise ValueError("typed 查询可信字段无效")
            payload.update(
                {
                    "plan_id": plan_id,
                    "requirement_digest": requirement_digest,
                    "scope_fingerprint": scope_fingerprint,
                }
            )
        except (ValueError, TypeError):
            return _error(
                "本轮事实查询没有通过可信计划绑定。",
                code="trusted_fact_requirement_required",
                status=409,
            )
    raw_user_message = get_session_user_message()
    trusted_user_message = (
        normalized_fact_query_text(raw_user_message)
        if "query_kind" in payload
        else raw_user_message.strip()[:4_000]
    )
    if not trusted_user_message:
        return _error(
            "当前查询缺少可信用户消息。",
            code="trusted_query_text_required",
            status=409,
        )
    if "query_kind" not in payload or payload.get("query_kind") == "resource-read":
        payload["queryText"] = trusted_user_message
    return _post_internal(payload, session)


registry.register(
    name="mystand_query",
    toolset="mystand_query",
    schema=MYSTAND_QUERY_SCHEMA,
    handler=mystand_query_tool_handler,
    check_fn=check_mystand_query,
    requires_env=[],
    is_async=False,
    description="High-level semantic query for authorized My Stand data",
    emoji="🔎",
    max_result_size_chars=_MAX_RESPONSE_BYTES,
)
