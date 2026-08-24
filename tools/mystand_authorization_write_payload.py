"""Shared model-facing schema and normalization for My Stand write payloads."""

from __future__ import annotations

from copy import deepcopy

_NODE_FIELD_ORDER = (
    "id",
    "label",
    "type",
    "summary",
    "body",
    "x",
    "y",
    "color",
)
_FLAT_ADD_NODE_FIELD_ORDER = (
    "nodeId",
    "label",
    "type",
    "summary",
    "body",
    "x",
    "y",
    "color",
)
_NODE_CHANGE_FIELD_ORDER = (
    "label",
    "type",
    "summary",
    "body",
    "x",
    "y",
    "color",
)
_NESTED_NODE_ALIASES = {
    "nodeId": "id",
    "name": "label",
    "nodeType": "type",
    "content": "body",
}
_NODE_CHANGE_ALIASES = {
    "name": "label",
    "nodeType": "type",
    "content": "body",
}

_NODE_PROPERTIES = {
    "id": {
        "type": "string",
        "description": "Optional stable node id. Omit to let My Stand derive one.",
        "minLength": 1,
        "maxLength": 160,
    },
    "label": {
        "type": "string",
        "minLength": 1,
        "maxLength": 40,
    },
    "type": {
        "type": "string",
        "enum": ["positive", "negative", "skill"],
    },
    "summary": {
        "type": "string",
        "maxLength": 240,
    },
    "body": {
        "type": "string",
        "maxLength": 12_000,
    },
    "x": {
        "type": "number",
        "minimum": 0,
        "maximum": 6_000,
    },
    "y": {
        "type": "number",
        "minimum": 0,
        "maximum": 4_000,
    },
    "color": {
        "type": "string",
        "pattern": "^#[0-9A-Fa-f]{6}$",
    },
}


class AuthorizationWritePayloadError(ValueError):
    """Fail-closed error returned before a model payload reaches My Stand."""

    def __init__(self, message: str, *, code: str = "invalid_write_payload"):
        super().__init__(message)
        self.code = code
        self.status = 400


def build_authorization_write_payload_schema() -> dict:
    """Return the shared payload schema without top-level combinators."""

    node_properties = deepcopy(_NODE_PROPERTIES)
    node_properties.update({
        "nodeId": {
            "type": "string",
            "description": "Compatibility alias normalized to node.id.",
        },
        "name": {
            "type": "string",
            "description": "Compatibility alias normalized to node.label.",
        },
        "nodeType": {
            "type": "string",
            "description": "Compatibility alias normalized to node.type.",
        },
        "content": {
            "type": "string",
            "description": "Compatibility alias normalized to node.body.",
        },
    })
    change_properties = {
        key: deepcopy(_NODE_PROPERTIES[key])
        for key in _NODE_CHANGE_FIELD_ORDER
    }
    change_properties.update({
        "name": {
            "type": "string",
            "description": "Compatibility alias normalized to changes.label.",
        },
        "nodeType": {
            "type": "string",
            "description": "Compatibility alias normalized to changes.type.",
        },
        "content": {
            "type": "string",
            "description": "Compatibility alias normalized to changes.body.",
        },
    })
    change_properties.update({
        "contractNo": {"type": "string"},
        "signDate": {"type": "string"},
        "signType": {"type": "string"},
        "signAddress": {"type": "string"},
        "settlementPerformance": {"type": "number"},
        "performanceLoss": {"type": "number"},
        "completed": {"type": "boolean"},
        "allocationDeduction": {"type": "number"},
        "deductionItem": {"type": "string"},
        "taxTotal": {"type": "number"},
        "notes": {"type": "string"},
    })
    return {
        "type": "object",
        "description": (
            "Action-specific payload. For knowledge-graph.add-node, prefer the "
            "canonical {node:{id?,label,type,summary?,body?,x?,y?,color?}} shape. "
            "The exact legacy flat fields "
            "{nodeId?,label,type,summary?,body?,x?,y?,color?} are accepted only "
            "for add-node and normalized to node.id plus the canonical node. "
            "Inside node, the exact compatibility aliases "
            "{nodeId,name,nodeType,content} map to {id,label,type,body}. "
            "Never mix canonical and flat fields. For knowledge-graph.update-node "
            "use exactly {nodeId,changes:{label?,type?,summary?,body?,x?,y?,color?}}; "
            "changes also accepts only name→label, nodeType→type, and content→body. "
            "Canonical fields and their same-meaning aliases cannot be mixed. "
            "knowledge-graph.delete accepts an empty payload and requires the separate "
            "preview/confirmation flow; My Stand retains its audit receipt and resource tombstone. "
            "Other permanent destructive actions are not available to website Xiaoban. "
            "business-archive.archive accepts an empty payload and moves the record "
            "to the website's recoverable archive area. Never "
            "include graphId; My Stand resolves the graph from the "
            "server-authorized resource. note.append-content needs only {content}; "
            "My Stand resolves the note id from the same authorized resource. "
            "property-note.append-text-block also needs only {content}; My Stand "
            "resolves the property-note archive and active document from the same "
            "authorized resource. For structure-aware property-note work, use "
            "property-note.edit-blocks with {operations:[...]}. Read the resource first "
            "and use the exact document/block ids and beforeText from writeCapability.structure. "
            "One approved call may insert text blocks, replace existing block text, and delete "
            "existing blocks atomically; My Stand resolves archiveId itself."
        ),
        "properties": {
            "node": {
                "type": "object",
                "description": "Canonical knowledge-graph.add-node node.",
                "properties": node_properties,
                "additionalProperties": False,
            },
            "nodeId": {
                "type": "string",
                "description": (
                    "Update target node id, or the add-node flat compatibility "
                    "alias normalized to node.id."
                ),
                "minLength": 1,
                "maxLength": 160,
            },
            **{
                key: deepcopy(_NODE_PROPERTIES[key])
                for key in _FLAT_ADD_NODE_FIELD_ORDER
                if key != "nodeId"
            },
            "changes": {
                "type": "object",
                "description": (
                    "Canonical update fields for knowledge-graph.update-node, "
                    "or the existing finance row changes object."
                ),
                "properties": change_properties,
                "minProperties": 1,
                "additionalProperties": False,
            },
            "edge": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                    "strength": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "noteId": {"type": "string"},
            "content": {"type": "string"},
            "archiveId": {"type": "string"},
            "documentId": {"type": "string"},
            "operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 40,
                "description": (
                    "property-note.edit-blocks operations. insert-text-block uses "
                    "{op,documentId,afterBlockId?,blockType,text}; replace-block-text "
                    "uses {op,documentId,blockId,beforeText,afterText}; delete-block uses "
                    "{op,documentId,blockId,beforeText}. beforeText must exactly equal the "
                    "latest structure snapshot so punctuation-level differences remain visible."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["insert-text-block", "replace-block-text", "delete-block"],
                        },
                        "documentId": {"type": "string"},
                        "blockId": {"type": "string"},
                        "afterBlockId": {"type": "string"},
                        "blockType": {
                            "type": "string",
                            "enum": [
                                "h1", "h2", "h3", "h4", "h5", "paragraph", "bullet",
                                "ordered", "check", "quote", "divider", "code", "callout",
                            ],
                        },
                        "beforeText": {"type": "string"},
                        "afterText": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["op", "documentId"],
                    "additionalProperties": False,
                },
            },
            "subjectType": {"type": "string"},
            "subjectId": {"type": "string"},
            "sectionKey": {"type": "string"},
            "fieldKey": {"type": "string"},
            "mode": {
                "type": "string",
                "enum": ["append", "replace"],
            },
            "brokerUser": {"type": "string"},
            "year": {"type": "integer"},
            "recordId": {"type": "string"},
        },
        "additionalProperties": False,
    }


def _reject_unknown_fields(value: dict, allowed: tuple[str, ...], label: str) -> None:
    unknown = [key for key in value if key not in allowed]
    if unknown:
        raise AuthorizationWritePayloadError(
            f"{label}包含不允许的字段。",
            code="write_payload_fields_not_allowed",
        )


def _require_fields(value: dict, required: tuple[str, ...], label: str) -> None:
    if any(key not in value for key in required):
        raise AuthorizationWritePayloadError(f"{label}缺少必要字段。")


def _ordered_fields(value: dict, field_order: tuple[str, ...]) -> dict:
    return {key: value[key] for key in field_order if key in value}


def _normalize_aliases(
    value: dict,
    field_order: tuple[str, ...],
    aliases: dict[str, str],
    label: str,
) -> dict:
    if any(alias in value and canonical in value for alias, canonical in aliases.items()):
        raise AuthorizationWritePayloadError(
            f"{label}不能混用同义字段。",
            code="write_payload_fields_not_allowed",
        )
    normalized = _ordered_fields(value, field_order)
    for alias, canonical in aliases.items():
        if alias in value:
            normalized[canonical] = value[alias]
    return _ordered_fields(normalized, field_order)


def normalize_authorization_write_payload(action: str, payload: dict) -> dict:
    """Normalize only the proven KG compatibility form and reject ambiguity."""

    if not isinstance(payload, dict):
        raise AuthorizationWritePayloadError("payload 必须是动作对应的对象。")
    if action in {"business-archive.archive", "knowledge-graph.delete"}:
        label = "业务档案可恢复归档请求" if action == "business-archive.archive" else "知识图谱删除请求"
        _reject_unknown_fields(payload, (), label)
        return {}
    if action == "note.append-content":
        _reject_unknown_fields(
            payload,
            ("content", "noteId", "mode"),
            "追加笔记请求",
        )
        _require_fields(payload, ("content",), "追加笔记请求")
        if "mode" in payload and payload["mode"] != "append":
            raise AuthorizationWritePayloadError(
                "追加笔记请求的 mode 只能是 append。",
                code="write_payload_fields_not_allowed",
            )
        return {"content": payload["content"]}
    if action == "property-note.append-text-block":
        _reject_unknown_fields(
            payload,
            ("content", "archiveId", "documentId", "mode"),
            "追加房源笔记请求",
        )
        _require_fields(payload, ("content",), "追加房源笔记请求")
        if "mode" in payload and payload["mode"] != "append":
            raise AuthorizationWritePayloadError(
                "追加房源笔记请求的 mode 只能是 append。",
                code="write_payload_fields_not_allowed",
            )
        return {"content": payload["content"]}
    if action == "property-note.edit-blocks":
        _reject_unknown_fields(payload, ("operations", "archiveId"), "房源笔记编辑请求")
        _require_fields(payload, ("operations",), "房源笔记编辑请求")
        operations = payload["operations"]
        if not isinstance(operations, list) or not operations or len(operations) > 40:
            raise AuthorizationWritePayloadError("房源笔记编辑需要 1 到 40 个操作。")
        normalized = []
        for index, operation in enumerate(operations):
            label = f"房源笔记操作 {index + 1}"
            if not isinstance(operation, dict):
                raise AuthorizationWritePayloadError(f"{label}必须是对象。")
            _require_fields(operation, ("op", "documentId"), label)
            op = operation["op"]
            if op == "insert-text-block":
                _reject_unknown_fields(
                    operation,
                    ("op", "documentId", "afterBlockId", "blockType", "text"),
                    label,
                )
                _require_fields(operation, ("blockType", "text"), label)
                normalized.append({
                    "op": op,
                    "documentId": operation["documentId"],
                    **({"afterBlockId": operation["afterBlockId"]} if "afterBlockId" in operation else {}),
                    "blockType": operation["blockType"],
                    "text": operation["text"],
                })
            elif op == "replace-block-text":
                _reject_unknown_fields(
                    operation,
                    ("op", "documentId", "blockId", "beforeText", "afterText"),
                    label,
                )
                _require_fields(operation, ("blockId", "beforeText", "afterText"), label)
                normalized.append({
                    "op": op,
                    "documentId": operation["documentId"],
                    "blockId": operation["blockId"],
                    "beforeText": operation["beforeText"],
                    "afterText": operation["afterText"],
                })
            elif op == "delete-block":
                _reject_unknown_fields(
                    operation,
                    ("op", "documentId", "blockId", "beforeText"),
                    label,
                )
                _require_fields(operation, ("blockId", "beforeText"), label)
                normalized.append({
                    "op": op,
                    "documentId": operation["documentId"],
                    "blockId": operation["blockId"],
                    "beforeText": operation["beforeText"],
                })
            else:
                raise AuthorizationWritePayloadError("房源笔记编辑操作不受支持。")
        return {"operations": normalized}
    if action == "knowledge-graph.add-node":
        if "node" in payload:
            _reject_unknown_fields(payload, ("node",), "新增图谱节点请求")
            node = payload["node"]
            if not isinstance(node, dict):
                raise AuthorizationWritePayloadError("node 必须是对象。")
            _reject_unknown_fields(
                node,
                (*_NODE_FIELD_ORDER, *_NESTED_NODE_ALIASES),
                "图谱节点",
            )
            normalized_node = _normalize_aliases(
                node,
                _NODE_FIELD_ORDER,
                _NESTED_NODE_ALIASES,
                "图谱节点",
            )
            _require_fields(normalized_node, ("label", "type"), "图谱节点")
            return {"node": normalized_node}

        _reject_unknown_fields(
            payload,
            _FLAT_ADD_NODE_FIELD_ORDER,
            "新增图谱节点请求",
        )
        _require_fields(payload, ("label", "type"), "新增图谱节点请求")
        node = {
            **({"id": payload["nodeId"]} if "nodeId" in payload else {}),
            **_ordered_fields(payload, _NODE_FIELD_ORDER[1:]),
        }
        return {"node": node}

    if action == "knowledge-graph.update-node":
        _reject_unknown_fields(
            payload,
            ("nodeId", "changes"),
            "更新图谱节点请求",
        )
        _require_fields(payload, ("nodeId", "changes"), "更新图谱节点请求")
        changes = payload["changes"]
        if not isinstance(changes, dict) or not changes:
            raise AuthorizationWritePayloadError("changes 必须是非空对象。")
        _reject_unknown_fields(
            changes,
            (*_NODE_CHANGE_FIELD_ORDER, *_NODE_CHANGE_ALIASES),
            "图谱节点变更",
        )
        return {
            "nodeId": payload["nodeId"],
            "changes": _normalize_aliases(
                changes,
                _NODE_CHANGE_FIELD_ORDER,
                _NODE_CHANGE_ALIASES,
                "图谱节点变更",
            ),
        }

    return payload
