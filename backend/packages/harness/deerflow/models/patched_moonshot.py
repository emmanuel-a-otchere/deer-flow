"""Patched ChatOpenAI for Moonshot that preserves reasoning_content.

Moonshot's thinking models (e.g. kimi-k2.6) require that when thinking is enabled,
the ``reasoning_content`` field from previous assistant messages must be echoed back
in subsequent requests, similar to how DeepSeek and Gemini (thought_signature) work.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class PatchedMoonshotChatModel(ChatOpenAI):
    """ChatOpenAI with ``reasoning_content`` preservation for Moonshot thinking models.

    When using Moonshot thinking models, the API expects ``reasoning_content`` to be
    present on assistant messages in multi-turn conversations. This patched version
    restores that field from ``AIMessage.additional_kwargs["reasoning_content"]``
    into the serialised request payload.
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get request payload with ``reasoning_content`` preserved."""
        # Capture original messages
        original_messages = self._convert_input(input_).to_messages()

        # Obtain base payload from parent
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        payload_messages = payload.get("messages", [])

        if len(payload_messages) == len(original_messages):
            for payload_msg, orig_msg in zip(payload_messages, original_messages):
                if payload_msg.get("role") == "assistant" and isinstance(orig_msg, AIMessage):
                    _restore_reasoning_content(payload_msg, orig_msg)
        else:
            # Fallback: match assistant-role entries positionally against AIMessages.
            ai_messages = [m for m in original_messages if isinstance(m, AIMessage)]
            assistant_payloads = [(i, m) for i, m in enumerate(payload_messages) if m.get("role") == "assistant"]
            for (_, payload_msg), ai_msg in zip(assistant_payloads, ai_messages):
                _restore_reasoning_content(payload_msg, ai_msg)

        return payload


def _restore_reasoning_content(payload_msg: dict, orig_msg: AIMessage) -> None:
    """Re-inject ``reasoning_content`` into *payload_msg*."""
    reasoning_content = orig_msg.additional_kwargs.get("reasoning_content")
    if reasoning_content:
        payload_msg["reasoning_content"] = reasoning_content
