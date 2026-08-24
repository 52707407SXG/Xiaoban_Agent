"""Tests for Xiaoban's authorized semantic and finance query bridge."""

import io
import json
import urllib.error

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from tools import mystand_query_tool as bridge

FACT_NEEDS = {
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
RESOURCE_TYPES = {
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


def _valid_plan():
    return {
        "operation": "read",
        "resource": {
            "name": "复地金融岛楼盘MD",
            "type_hint": "property-md",
        },
        "entities": [
            {"kind": "building", "value": "17", "role": "locator"},
            {"kind": "unit", "value": "1"},
            {"kind": "room", "value": "801"},
        ],
        "fact_needs": ["owner.name", "owner.phone"],
        "mode": "facts",
    }


def _finance_plan(kind):
    query_args = {
        "list": {"year": 2026},
        "rank": {"year": 2026, "rank": 3},
        "predicate": {
            "year": 2026,
            "field": "yearlyAmount",
            "operator": "gte",
            "amount": 1_000_000,
        },
        "count": {
            "year": 2026,
            "field": "yearlyAmount",
            "operator": "gt",
            "amount": 500_000,
        },
    }[kind]
    return {
        "operation": "read",
        "query_kind": kind,
        "module_id": "finance-ledger",
        "fact_paths": [f"finance.performance.{kind}"],
        "query_args": query_args,
        "coverage_required": True,
    }


def _settlement_confirmation_plan(month=7):
    return {
        "operation": "read",
        "query_kind": "list",
        "module_id": "finance-ledger",
        "fact_paths": ["finance.settlement_confirmation.unconfirmed"],
        "query_args": {"year": 2026, "month": month},
        "coverage_required": True,
    }


def _settlement_proof_confirmed_unsettled_plan(month=7):
    return {
        "operation": "read",
        "query_kind": "list",
        "module_id": "finance-ledger",
        "fact_paths": ["finance.settlement.proof_confirmed_unsettled"],
        "query_args": {"year": 2026, "month": month},
        "coverage_required": True,
    }


def _call(
    args,
    *,
    platform="api_server",
    user_id="ZYJ005",
    user_message="查复地金融岛17栋1单元801的业主姓名和电话",
):
    tokens = set_session_vars(
        platform=platform,
        user_id=user_id,
        message_id="msg-1",
        session_id="session-1",
        user_message=user_message,
    )
    try:
        return json.loads(bridge.mystand_query_tool_handler(args))
    finally:
        clear_session_vars(tokens)


def test_contract_exposes_semantic_and_strict_finance_aggregate_shapes():
    parameters = bridge.MYSTAND_QUERY_SCHEMA["parameters"]
    properties = parameters["properties"]

    assert set(properties) == {
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
    assert parameters["required"] == ["operation"]
    assert len(parameters["anyOf"]) == 8
    assert parameters["additionalProperties"] is False
    assert properties["operation"]["const"] == "read"
    assert set(properties["query_kind"]["enum"]) == {
        "rank",
        "list",
        "predicate",
        "count",
    }
    assert properties["module_id"]["const"] == "finance-ledger"
    assert properties["coverage_required"]["const"] is True
    finance_branches = {
        tuple(branch["properties"]["fact_paths"]["const"]): branch
        for branch in parameters["anyOf"]
        if "query_kind" in branch.get("properties", {})
    }
    assert set(finance_branches) == {
        ("finance.performance.rank",),
        ("finance.performance.list",),
        ("finance.performance.predicate",),
        ("finance.performance.count",),
        ("finance.settlement_confirmation.unconfirmed",),
        ("finance.settlement.proof_confirmed_unsettled",),
    }
    for fact_paths, branch in finance_branches.items():
        assert set(branch["required"]) == {
            "query_kind",
            "module_id",
            "fact_paths",
            "query_args",
            "coverage_required",
        }
        assert branch["properties"]["fact_paths"]["const"] == list(fact_paths)
        assert branch["properties"]["query_args"]["additionalProperties"] is False
    assert set(properties["fact_needs"]["items"]["enum"]) == FACT_NEEDS
    assert set(
        properties["resource"]["properties"]["type_hint"]["enum"]
    ) == RESOURCE_TYPES
    assert set(properties["entities"]["items"]["properties"]["kind"]["enum"]) == {
        "building",
        "unit",
        "room",
        "person",
        "estate",
        "document",
        "topic",
        "time",
    }
    assert "role" not in properties["entities"]["items"]["required"]
    assert "queryText" not in properties
    assert {
        "owner",
        "owner_user",
        "user",
        "user_id",
        "authorization_id",
        "auth_id",
        "resource_uid",
        "source_id",
    }.isdisjoint(properties)


def test_settlement_contract_tells_the_model_the_complete_first_call_shape():
    from tools.schema_sanitizer import sanitize_tool_schemas

    description = bridge.MYSTAND_QUERY_SCHEMA["description"]
    required_shape = 'query_args={"year": YYYY, "month": 1-12}'

    assert required_shape in description
    assert "Both integer year and integer month are required" in description
    for required_predicate in (
        "uploaded settlement proof",
        "broker confirmation",
        "manager confirmation",
        "still unsettled",
    ):
        assert required_predicate in description
    assert bridge.registry.get_entry("mystand_query").description == description
    provider_schema = sanitize_tool_schemas(
        [{"type": "function", "function": bridge.MYSTAND_QUERY_SCHEMA}]
    )[0]["function"]
    assert "anyOf" not in provider_schema["parameters"]
    assert required_shape in provider_schema["description"]
    assert "finance.settlement.proof_confirmed_unsettled" in provider_schema["description"]
    for required_predicate in (
        "uploaded settlement proof",
        "broker confirmation",
        "manager confirmation",
        "still unsettled",
    ):
        assert required_predicate in provider_schema["description"]
    assert "TodoResult" not in provider_schema["description"]
    assert "TOOL ORDERING" not in provider_schema["description"]


@pytest.mark.parametrize("kind", ["rank", "list", "predicate", "count"])
def test_handler_directly_dispatches_each_finance_aggregate_shape(
    monkeypatch,
    kind,
):
    calls = []

    def fake_post(payload, session):
        calls.append((payload, session))
        return json.dumps(
            {
                "schema": "mystand.query-result.v1",
                "ok": True,
                "facts": [{"path": f"finance.performance.{kind}"}],
                "coverage": {"complete": True},
            }
        )

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    plan = _finance_plan(kind)
    result = _call(plan, user_message="查今年的财务表现")

    assert result["ok"] is True
    assert calls == [
        (
            plan,
            {
                "platform": "api_server",
                "user_id": "ZYJ005",
                "message_id": "msg-1",
                "session_id": "session-1",
            },
        )
    ]


def test_handler_dispatches_settlement_confirmation_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda payload, session: (
            calls.append((payload, session))
            or json.dumps({"ok": True, "coverage": {"complete": True}})
        ),
    )
    plan = _settlement_confirmation_plan()

    result = _call(plan, user_message="查7月结算卡还有谁没点")

    assert result["ok"] is True
    assert calls[0][0] == plan
    assert len(calls) == 1


def test_handler_dispatches_compound_settlement_predicate_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda payload, session: (
            calls.append((payload, session))
            or json.dumps(
                {
                    "ok": True,
                    "facts": [
                        {"path": "finance.settlement.proof_confirmed_unsettled"}
                    ],
                    "coverage": {"complete": True},
                }
            )
        ),
    )
    plan = _settlement_proof_confirmed_unsettled_plan()

    result = _call(
        plan,
        user_message="查7月已上传结算凭证、经纪人确认、店长确认，但仍未结算的人员",
    )

    assert result["ok"] is True
    assert result["facts"] == [
        {"path": "finance.settlement.proof_confirmed_unsettled"}
    ]
    assert calls[0][0] == plan
    assert "finance.settlement_confirmation.unconfirmed" not in json.dumps(
        calls[0][0]
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "plan_factory",
    [_settlement_confirmation_plan, _settlement_proof_confirmed_unsettled_plan],
)
def test_handler_normalizes_year_month_suffixes_before_one_dispatch(
    monkeypatch,
    plan_factory,
):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda payload, session: (
            calls.append((payload, session))
            or json.dumps({"ok": True, "coverage": {"complete": True}})
        ),
    )
    plan = plan_factory("7月")
    plan["query_args"]["year"] = "2026年"

    result = _call(plan, user_message="查7月结算卡还有谁没点")

    assert result["ok"] is True
    assert calls[0][0]["query_args"] == {"year": 2026, "month": 7}
    assert len(calls) == 1


@pytest.mark.parametrize(
    "plan_factory",
    [_settlement_confirmation_plan, _settlement_proof_confirmed_unsettled_plan],
)
def test_handler_normalizes_consistent_settlement_period_before_one_dispatch(
    monkeypatch,
    plan_factory,
):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda payload, session: (
            calls.append((payload, session))
            or json.dumps({"ok": True, "coverage": {"complete": True}})
        ),
    )
    plan = plan_factory("7月")
    plan["query_args"]["year"] = "2026年7月"

    result = _call(plan, user_message="查7月结算卡还有谁没点")

    assert result["ok"] is True
    assert calls[0][0]["query_args"] == {"year": 2026, "month": 7}
    assert len(calls) == 1


@pytest.mark.parametrize(
    "plan_factory",
    [_settlement_confirmation_plan, _settlement_proof_confirmed_unsettled_plan],
)
def test_handler_rejects_conflicting_settlement_period_before_dispatch(
    monkeypatch,
    plan_factory,
):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args: calls.append(args) or json.dumps({"ok": True}),
    )
    plan = plan_factory(8)
    plan["query_args"]["year"] = "2026年7月"

    result = _call(plan)

    assert result["code"] == "invalid_mystand_query_arguments"
    assert calls == []


@pytest.mark.parametrize(
    "plan_factory",
    [_settlement_confirmation_plan, _settlement_proof_confirmed_unsettled_plan],
)
@pytest.mark.parametrize("month", [0, 13, True, 7.5])
def test_handler_rejects_invalid_settlement_month_before_dispatch(
    monkeypatch,
    month,
    plan_factory,
):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args: calls.append(args) or json.dumps({"ok": True}),
    )

    result = _call(plan_factory(month))

    assert result["code"] == "invalid_mystand_query_arguments"
    assert calls == []


def test_handler_rejects_finance_aggregate_shape_drift_before_dispatch(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args: calls.append(args) or json.dumps({"ok": True}),
    )

    invalid_plans = []

    invalid_kind = _finance_plan("list")
    invalid_kind["query_kind"] = "sum"
    invalid_plans.append(invalid_kind)

    invalid_module = _finance_plan("list")
    invalid_module["module_id"] = "profile"
    invalid_plans.append(invalid_module)

    invalid_path = _finance_plan("rank")
    invalid_path["fact_paths"] = ["finance.performance.list"]
    invalid_plans.append(invalid_path)

    invalid_coverage = _finance_plan("count")
    invalid_coverage["coverage_required"] = False
    invalid_plans.append(invalid_coverage)

    invalid_year = _finance_plan("list")
    invalid_year["query_args"] = {"year": 2101}
    invalid_plans.append(invalid_year)

    invalid_list = _finance_plan("list")
    invalid_list["query_args"] = {"year": 2026, "rank": 1}
    invalid_plans.append(invalid_list)

    invalid_rank = _finance_plan("rank")
    invalid_rank["query_args"] = {"year": 2026, "rank": 0}
    invalid_plans.append(invalid_rank)

    invalid_predicate = _finance_plan("predicate")
    invalid_predicate["query_args"] = {
        "year": 2026,
        "field": "yearlyAmount",
        "operator": "eq",
        "amount": 1,
    }
    invalid_plans.append(invalid_predicate)

    invalid_count = _finance_plan("count")
    invalid_count["query_args"]["amount"] = float("inf")
    invalid_plans.append(invalid_count)

    mixed_shape = _finance_plan("rank")
    mixed_shape["resource"] = {"name": "财务档案"}

    extra_control = _finance_plan("rank")
    extra_control["unexpected_control"] = "forged"
    invalid_plans.append(extra_control)

    for plan_factory in (
        _settlement_confirmation_plan,
        _settlement_proof_confirmed_unsettled_plan,
    ):
        settlement_extra_arg = plan_factory()
        settlement_extra_arg["query_args"]["rank"] = 1
        invalid_plans.append(settlement_extra_arg)

    for plan in invalid_plans:
        result = _call(plan)
        assert result["code"] == "invalid_mystand_query_arguments"

    assert calls == []

    result = _call(mixed_shape)
    assert result["ok"] is True
    assert calls[0][0] == _finance_plan("rank")


def test_handler_injects_trusted_query_text_and_session_identity_stays_out_of_body(
    monkeypatch,
):
    calls = []

    def fake_post(payload, session):
        calls.append((payload, session))
        return json.dumps(
            {
                "ok": True,
                "facts": [{"predicate": "owner.name", "value": "测试姓名"}],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(bridge, "_post_internal", fake_post)
    result = _call(_valid_plan())

    assert result["ok"] is True
    assert len(calls) == 1
    payload, session = calls[0]
    assert payload == {
        **_valid_plan(),
        "queryText": "查复地金融岛17栋1单元801的业主姓名和电话",
    }
    assert session == {
        "platform": "api_server",
        "user_id": "ZYJ005",
        "message_id": "msg-1",
        "session_id": "session-1",
    }
    assert {
        "owner",
        "owner_user",
        "user",
        "user_id",
        "authorization_id",
        "resource_uid",
        "source_id",
    }.isdisjoint(payload)


def test_handler_rejects_model_supplied_identity_ids_and_query_text(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args: calls.append(args) or json.dumps({"ok": True}),
    )

    for forbidden in (
        {"owner": "ZYJ999"},
        {"user_id": "ZYJ999"},
        {"authorization_id": "AUTH-forged"},
        {"resource_uid": "forged"},
        {"source_id": "forged"},
        {"queryText": "伪造问题"},
    ):
        result = _call({**_valid_plan(), **forbidden})
        assert result["code"] == "invalid_mystand_query_arguments"

    nested = _valid_plan()
    nested["resource"] = {**nested["resource"], "resource_uid": "forged"}
    assert _call(nested)["code"] == "invalid_mystand_query_arguments"
    assert calls == []


def test_handler_accepts_person_entity_without_guessed_resource_title(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda payload, session: calls.append((payload, session))
        or json.dumps({"ok": True}),
    )
    plan = {
        "operation": "read",
        "entities": [
            {"kind": "person", "value": "汤总", "role": "subject"},
        ],
        "fact_needs": ["owner.family", "owner.interests"],
        "mode": "facts",
    }
    result = _call(
        plan,
        user_message="汤总家里是什么情况，平时喜欢什么？",
    )
    assert result["ok"] is True
    assert calls[0][0] == {
        **plan,
        "queryText": "汤总家里是什么情况，平时喜欢什么？",
    }


def test_handler_rejects_invalid_semantic_enums_and_duplicate_fact_needs(
    monkeypatch,
):
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *_args: json.dumps({"ok": True}),
    )

    invalid_type = _valid_plan()
    invalid_type["resource"] = {
        "name": "资料",
        "type_hint": "unknown",
    }
    assert _call(invalid_type)["code"] == "invalid_mystand_query_arguments"

    invalid_entity = _valid_plan()
    invalid_entity["entities"] = [{"kind": "authorization", "value": "AUTH-x"}]
    assert _call(invalid_entity)["code"] == "invalid_mystand_query_arguments"

    invalid_fact = _valid_plan()
    invalid_fact["fact_needs"] = ["owner.name", "owner.name"]
    assert _call(invalid_fact)["code"] == "invalid_mystand_query_arguments"

    invalid_fact_type = _valid_plan()
    invalid_fact_type["fact_needs"] = [{"predicate": "owner.name"}]
    assert _call(invalid_fact_type)["code"] == "invalid_mystand_query_arguments"


def test_handler_requires_authenticated_api_session_and_trusted_user_message(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        bridge,
        "_post_internal",
        lambda *args: calls.append(args) or json.dumps({"ok": True}),
    )

    assert _call(_valid_plan(), platform="telegram")["code"] == (
        "mystand_session_required"
    )
    assert _call(_valid_plan(), user_id="")["code"] == "mystand_session_required"
    assert _call(_valid_plan(), user_message="")["code"] == (
        "trusted_query_text_required"
    )
    assert calls == []


def test_internal_post_uses_only_trusted_session_headers(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b'{"ok":true,"status":"matched"}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)
    payload = {
        **_valid_plan(),
        "queryText": "可信原始问题",
    }
    result = json.loads(
        bridge._post_internal(
            payload,
            {
                "user_id": "ZYJ005",
                "message_id": "msg-1",
                "session_id": "session-1",
            },
        )
    )

    request = captured["request"]
    headers = {key.lower(): value for key, value in request.header_items()}
    assert result == {"ok": True, "status": "matched"}
    assert captured["timeout"] == 20
    assert request.full_url.endswith("/api/xiaoban/internal/query")
    assert headers["x-xiaoban-user-id"] == "ZYJ005"
    assert headers["x-xiaoban-message-id"] == "msg-1"
    assert headers["x-xiaoban-session-id"] == "session-1"
    assert json.loads(request.data.decode("utf-8")) == payload


def test_internal_post_rejects_oversized_result(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return b"x" * (bridge._MAX_RESPONSE_BYTES + 1)

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(
        bridge.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    result = json.loads(
        bridge._post_internal(
            {"operation": "read"},
            {"user_id": "ZYJ005"},
        )
    )

    assert result["status"] == 413
    assert result["code"] == "mystand_query_result_too_large"


def test_http_409_synthesizes_safe_clarification_and_candidates(monkeypatch):
    upstream = {
        "ok": False,
        "code": "resource_query_ambiguous",
        "clarification": "找到两个楼盘，请补充完整名称。",
        "candidates": [
            {
                "safeLabel": "复地金融岛楼盘MD",
                "resourceType": "property-md",
                "resourceUid": "resource-secret-uid",
                "authorizationId": "AUTH-secret",
                "ownerUser": "ZYJ999",
                "sourceId": "source-secret",
            },
            {
                "displayName": "复地金融岛楼盘数据",
                "typeHint": "property-data",
                "internalId": "internal-secret",
            },
        ],
        "details": {
            "resourceUid": "nested-resource-secret",
            "owner": "nested-owner-secret",
        },
    }

    def raise_conflict(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            409,
            "Conflict",
            {},
            io.BytesIO(json.dumps(upstream).encode("utf-8")),
        )

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", raise_conflict)
    result = json.loads(
        bridge._post_internal(
            {"operation": "read"},
            {"user_id": "ZYJ005"},
        )
    )
    serialized = json.dumps(result, ensure_ascii=False)

    assert result == {
        "ok": False,
        "status": 409,
        "code": "resource_query_ambiguous",
        "error": "找到多项可能资料，需要补充信息。",
        "clarification": "找到两个楼盘，请补充完整名称。",
        "candidates": [
            {"label": "复地金融岛楼盘MD", "type": "property-md"},
            {"label": "复地金融岛楼盘数据", "type": "property-data"},
        ],
    }
    for secret in (
        "resource-secret-uid",
        "AUTH-secret",
        "ZYJ999",
        "source-secret",
        "internal-secret",
        "nested-resource-secret",
        "nested-owner-secret",
    ):
        assert secret not in serialized


def test_http_409_without_candidates_does_not_invent_multiple_matches(
    monkeypatch,
):
    upstream = {
        "ok": False,
        "code": "resource_needs_clarification",
        "clarification": "没有找到唯一且可供小伴读取的资料，请补充资料名称。",
        "candidates": [],
    }

    def raise_conflict(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            409,
            "Conflict",
            {},
            io.BytesIO(json.dumps(upstream).encode("utf-8")),
        )

    monkeypatch.setattr(
        bridge,
        "_api_base_url",
        lambda: "http://127.0.0.1:18081",
    )
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", raise_conflict)

    result = json.loads(
        bridge._post_internal(
            {"operation": "read"},
            {"user_id": "ZYJ005"},
        )
    )

    assert result["status"] == 409
    assert result.get("candidates", []) == []
    assert result["error"] == (
        "没有找到唯一且可供小伴读取的资料，请补充资料名称。"
    )
    assert "多项" not in result["error"]


def test_http_404_limits_clarification_and_drops_internal_identifiers(monkeypatch):
    upstream = {
        "code": "resource_query_not_found",
        "details": {
            "clarification": "请选择 AUTH-secret 后继续。",
            "candidates": [
                {
                    "name": "候选资料",
                    "type": "note",
                    "uid": "hidden-uid",
                }
            ],
        },
    }

    def raise_not_found(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            io.BytesIO(json.dumps(upstream).encode("utf-8")),
        )

    monkeypatch.setattr(bridge, "_api_base_url", lambda: "http://127.0.0.1:18081")
    monkeypatch.setattr(bridge, "_internal_token", lambda: "internal-token")
    monkeypatch.setattr(bridge.urllib.request, "urlopen", raise_not_found)
    result = json.loads(
        bridge._post_internal(
            {"operation": "read"},
            {"user_id": "ZYJ005"},
        )
    )

    assert result["status"] == 404
    assert result["code"] == "resource_query_not_found"
    assert "clarification" not in result
    assert result["candidates"] == [{"name": "候选资料", "type": "note"}]
    assert "AUTH-secret" not in json.dumps(result, ensure_ascii=False)
    assert "hidden-uid" not in json.dumps(result, ensure_ascii=False)
