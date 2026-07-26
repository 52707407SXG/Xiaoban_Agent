"""渠道适配层：各渠道只负责收消息、身份映射和提交任务。

微信、飞书从已验证的平台事件和服务端账号绑定解析身份；未绑定、
绑定冲突或缺少 conversationId/messageId 一律 fail closed。
CLI 没有服务端 My Stand 身份绑定，因此不存在独立业务成功路径。
渠道不做业务判断、权限判断或索引查询，只构造 CommandEnvelope 并
进入同一个 Trusted Action Runtime。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from xiaoban.trusted_runtime.completion_guard import (
    DENIED_MESSAGE,
    NO_EVIDENCE_MESSAGE,
    check_mystand_final_answer,
)
from xiaoban.trusted_runtime.turns import classify_interaction
from xiaoban.trusted_runtime.types import (
    CommandEnvelope,
    CompletionDecision,
    TrustedIdentity,
    INTERACTION_WORK,
)

PLATFORM_WEB = "web"
PLATFORM_WECHAT = "weixin"
PLATFORM_FEISHU = "feishu"
PLATFORM_CLI = "cli"


def envelope_from_wechat_event(event: Mapping[str, Any]) -> CommandEnvelope:
    """微信 connector 模拟事件 → 统一输入。

    ``boundAccountId`` 必须来自服务端账号绑定，不接受消息正文自报。
    """
    return CommandEnvelope(
        request_id=str(event.get("requestId") or event.get("messageId") or ""),
        platform=PLATFORM_WECHAT,
        conversation_id=str(event.get("conversationId") or ""),
        message_id=str(event.get("messageId") or ""),
        external_user_ref=str(event.get("fromUser") or ""),
        text=str(event.get("text") or ""),
        received_at=str(event.get("receivedAt") or ""),
        metadata={"boundAccountId": str(event.get("boundAccountId") or "")},
    )


def envelope_from_feishu_event(event: Mapping[str, Any]) -> CommandEnvelope:
    """飞书 connector 模拟事件 → 统一输入。"""
    return CommandEnvelope(
        request_id=str(event.get("requestId") or event.get("messageId") or ""),
        platform=PLATFORM_FEISHU,
        conversation_id=str(event.get("chatId") or ""),
        message_id=str(event.get("messageId") or ""),
        external_user_ref=str(event.get("openId") or ""),
        text=str(event.get("text") or ""),
        received_at=str(event.get("receivedAt") or ""),
        metadata={"boundAccountId": str(event.get("boundAccountId") or "")},
    )


def identity_from_envelope(envelope: CommandEnvelope) -> Optional[TrustedIdentity]:
    """只信服务端绑定；没有绑定时返回 None（fail closed）。"""
    bound = str(envelope.metadata.get("boundAccountId") or "")
    if not bound:
        return None
    return TrustedIdentity(
        account_id=bound,
        data_scope="mystand",
        source="platform_binding",
    )


def evaluate_channel_answer(
    envelope: CommandEnvelope,
    *,
    final_text: str,
    user_message: Any,
    conversation_history: Any = None,
    result: Any = None,
) -> CompletionDecision:
    """所有渠道共享同一个 Runtime 与 CompletionGuard。

    身份只来自服务端绑定（identity_from_envelope），渠道适配器不得
    读取 result 自报身份补 envelope；未绑定一律 deny。
    """
    if envelope.platform == PLATFORM_CLI:
        # CLI 只是可选入口：业务请求只能得到非业务拒绝，零业务事实。
        if (
            classify_interaction(user_message or envelope.text, conversation_history)
            == INTERACTION_WORK
        ):
            return CompletionDecision(
                False, NO_EVIDENCE_MESSAGE, "blocked_cli_no_server_identity"
            )
        return CompletionDecision(True, str(final_text or ""), "not_mystand")
    identity = identity_from_envelope(envelope)
    if (
        identity is None
        or not envelope.conversation_id
        or not envelope.message_id
    ):
        return CompletionDecision(False, DENIED_MESSAGE, "blocked_unbound_channel")
    return check_mystand_final_answer(
        final_text,
        user_message=user_message,
        conversation_history=conversation_history,
        result=result,
        channel=envelope.platform,
        account_id=identity.account_id,
        request_id=envelope.request_id,
        message_id=envelope.message_id,
    )
