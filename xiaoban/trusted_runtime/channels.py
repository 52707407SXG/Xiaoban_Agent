"""渠道适配层：各渠道只负责收消息、身份映射和提交任务。

微信、飞书从已验证的平台事件和服务端账号绑定解析身份；
CLI 没有服务端 My Stand 身份绑定，因此不存在独立业务成功路径。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from xiaoban.trusted_runtime.completion_guard import check_mystand_final_answer
from xiaoban.trusted_runtime.types import (
    CommandEnvelope,
    CompletionDecision,
    TrustedIdentity,
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
    """所有渠道共享同一个 Runtime 与 CompletionGuard。"""
    if envelope.platform == PLATFORM_CLI:
        # CLI 只是可选入口：没有服务端身份绑定就没有业务成功路径。
        return CompletionDecision(True, str(final_text or ""), "not_mystand")
    return check_mystand_final_answer(
        final_text,
        user_message=user_message,
        conversation_history=conversation_history,
        result=result,
        channel=envelope.platform,
        account_id=str(envelope.metadata.get("boundAccountId") or ""),
        request_id=envelope.request_id,
        message_id=envelope.message_id,
    )
