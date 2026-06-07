"""Base LLM provider interface."""

import asyncio
import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import json_repair
from loguru import logger


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize to an OpenAI-style tool_call payload."""
        tool_call = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function"]["provider_specific_fields"] = self.function_provider_specific_fields
        return tool_call


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.
    thinking_blocks: list[dict] | None = None  # Anthropic extended thinking
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation parameters for LLM calls.

    Stored on the provider so every call site inherits the same defaults
    without having to pass temperature / max_tokens / reasoning_effort
    through every layer.  Individual call sites can still override by
    passing explicit keyword arguments to chat() / chat_with_retry().
    """

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "disconnected",
        "server error",
        "temporarily unavailable",
    )
    _IMAGE_UNSUPPORTED_MARKERS = (
        "image_url is only supported",
        "does not support image",
        "images are not supported",
        "image input is not supported",
        "image_url is not supported",
        "unsupported image input",
    )
    _TOOL_PAYLOAD_REJECTED_MARKERS = (
        "unsupported server-side tools",
        "oversized tool descriptions",
        "tool_use/tool_result",
        "mismatched tool",
        "improperly formed",
        "tool_choice",
        "unsupported tool",
        "function calling",
    )

    _SENTINEL = object()
    _CLOUD_PROBE_EXEC_PATTERNS = (
        r"\bnvidia-smi\b",
        r"\bnvcc\b",
        r"\bCUDA_VISIBLE_DEVICES\b",
        r"\bwhoami\s*&&\s*hostname\s*&&\s*pwd\s*&&\s*date\b",
        r"\bssh backend\b",
        r"\bpython(?:3)?\s+--version\b.*\bnvidia-smi\b",
    )

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace empty text content that causes provider 400 errors.

        Empty content can appear when MCP tools return nothing. Most providers
        reject empty-string content or empty text blocks in list content.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = None if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue

            if isinstance(content, list):
                filtered = [
                    item for item in content
                    if not (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    )
                ]
                if len(filtered) != len(content):
                    clean = dict(msg)
                    if filtered:
                        clean["content"] = filtered
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = None
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            if isinstance(content, dict):
                clean = dict(msg)
                clean["content"] = [content]
                result.append(clean)
                continue

            result.append(msg)
        return result

    @staticmethod
    def _sanitize_request_messages(
        messages: list[dict[str, Any]],
        allowed_keys: frozenset[str],
    ) -> list[dict[str, Any]]:
        """Keep only provider-safe message keys and normalize assistant content."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in allowed_keys}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            tool_choice: Tool selection strategy ("auto", "required", or specific tool dict).
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return any(marker in err for marker in cls._TRANSIENT_ERROR_MARKERS)

    @classmethod
    def _is_image_unsupported_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return any(marker in err for marker in cls._IMAGE_UNSUPPORTED_MARKERS)

    @classmethod
    def _is_tool_payload_rejected_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return any(marker in err for marker in cls._TOOL_PAYLOAD_REJECTED_MARKERS)

    @staticmethod
    def _strip_image_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Replace image_url blocks with text placeholder. Returns None if no images found."""
        found = False
        result = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "image_url":
                        new_content.append({"type": "text", "text": "[image omitted]"})
                        found = True
                    else:
                        new_content.append(b)
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result if found else None

    @staticmethod
    def _text_tool_protocol_messages(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Represent tools as text for OpenAI-compatible gateways without tool calling.

        Some Claude/OpenAI gateways accept ordinary chat completions but reject the
        function-calling fields. This fallback keeps the agent loop usable by asking
        the model to emit a small JSON envelope that RoboClaw parses back into
        ``ToolCallRequest`` objects.
        """
        compact_tools = []
        for tool in tools:
            fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
            name = fn.get("name")
            if not name:
                continue
            description = str(fn.get("description") or "")
            if len(description) > 500:
                description = description[:500] + "..."
            compact_tools.append({
                "name": name,
                "description": description,
                "parameters": fn.get("parameters") or {},
            })

        compact_tools = LLMProvider._consult_first_text_tools(compact_tools)
        tool_catalog = json.dumps(compact_tools, ensure_ascii=False)
        if len(tool_catalog) > 24_000:
            # Keep all names, trim schemas. This is less precise, but still lets
            # chat-only gateways choose a tool instead of failing outright.
            tool_catalog = json.dumps(
                [
                    {
                        "name": item["name"],
                        "description": item["description"],
                        "parameters": "schema omitted because the tool catalog is large; infer arguments from user intent",
                    }
                    for item in compact_tools
                ],
                ensure_ascii=False,
            )

        tool_choice_text = json.dumps(tool_choice, ensure_ascii=False) if isinstance(tool_choice, dict) else str(tool_choice or "auto")
        protocol = (
            "Native tool/function calling is unavailable for this provider. "
            "Use this text tool protocol instead.\n\n"
            "When evo_studio_agent_consult is available, delegate Evo Studio "
            "training/data/cloud tasks to that single backend consult tool. Do "
            "not request local exec for cloud, SSH, GPU, or VLA runtime checks.\n\n"
            "If you need to call tools, reply ONLY with JSON and no markdown:\n"
            '{"tool_calls":[{"name":"tool_name","arguments":{}}]}\n\n'
            "If no tool is needed, reply normally with the final answer.\n"
            f"tool_choice: {tool_choice_text}\n"
            f"available_tools: {tool_catalog}"
        )
        return [{"role": "system", "content": protocol}, *LLMProvider._messages_as_plain_text(messages)]

    @staticmethod
    def _consult_first_text_tools(compact_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prefer a tiny consult surface for chat-only gateways.

        OpenAI-compatible relays often reject long tool schemas or mixed
        tool-use histories. When RoboClaw has an OpenClaw-style consult tool,
        the relay only needs to call that one stable surface; the backend
        agent then runs the real skills/tools.
        """
        consult_names = {"evo_studio_agent_consult", "openclaw_agent_consult"}
        consult_tools = [item for item in compact_tools if item.get("name") in consult_names]
        if not consult_tools:
            return compact_tools
        # Keep only the consult surface. Native tool-capable providers still get
        # the full schema list; this branch is only for rejected/chat-only relays.
        return consult_tools

    @staticmethod
    def _messages_as_plain_text(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop provider-specific tool fields while preserving useful history."""
        safe: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            content = message.get("content")

            if role == "tool":
                name = str(message.get("name") or "tool")
                output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                safe.append({
                    "role": "user",
                    "content": f"Tool result from {name}: {output_text or '(empty)'}",
                })
                continue

            clean: dict[str, Any] = {"role": role if role in {"system", "user", "assistant"} else "user"}
            if isinstance(content, str):
                clean["content"] = content
            elif isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") in {"text", "input_text", "output_text"}:
                        text = block.get("text")
                        if text:
                            parts.append(str(text))
                    elif block.get("type") == "image_url":
                        parts.append("[image omitted]")
                clean["content"] = "\n".join(parts) if parts else ""
            elif content is None:
                clean["content"] = ""
            else:
                clean["content"] = json.dumps(content, ensure_ascii=False)
            if clean["role"] == "assistant":
                clean["content"] = LLMProvider._sanitize_plain_assistant_content(str(clean.get("content") or ""))

            tool_calls = message.get("tool_calls") or []
            if isinstance(tool_calls, list) and tool_calls:
                call_lines = []
                cloud_probe_inserted = False
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                    name = fn.get("name") or tool_call.get("name") or "tool"
                    arguments = fn.get("arguments") or tool_call.get("arguments") or "{}"
                    if LLMProvider._is_cloud_probe_exec(name, arguments):
                        if cloud_probe_inserted:
                            continue
                        call_lines.append('- evo_studio_cloud_train({"action":"backend_probe"})')
                        cloud_probe_inserted = True
                        continue
                    if cloud_probe_inserted and str(name) == "exec":
                        continue
                    call_lines.append(f"- {name}({arguments})")
                if call_lines:
                    existing = clean.get("content") or ""
                    clean["content"] = (existing + "\n" if existing else "") + "Previous assistant tool requests:\n" + "\n".join(call_lines)
            safe.append(clean)
        return LLMProvider._sanitize_empty_content(safe)

    @staticmethod
    def _is_cloud_probe_exec(name: Any, arguments: Any) -> bool:
        if str(name) != "exec":
            return False
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {}
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            parsed = {}
        command = str(parsed.get("command") or "").lower()
        return any(re.search(pattern, command) for pattern in LLMProvider._CLOUD_PROBE_EXEC_PATTERNS)

    @staticmethod
    def _sanitize_plain_assistant_content(content: str) -> str:
        if "Previous assistant tool requests:" not in content or "exec(" not in content:
            return content
        lines = content.splitlines()
        result: list[str] = []
        cloud_probe_inserted = False
        for line in lines:
            lower = line.lower()
            if re.search(r"^\s*-\s*exec\(", lower):
                if cloud_probe_inserted:
                    continue
                if any(re.search(pattern, lower) for pattern in LLMProvider._CLOUD_PROBE_EXEC_PATTERNS):
                    result.append('- evo_studio_cloud_train({"action":"backend_probe"})')
                    cloud_probe_inserted = True
                    continue
                result.append(line)
                continue
            result.append(line)
        return "\n".join(result)

    @staticmethod
    def _parse_text_tool_protocol(content: str | None) -> tuple[str | None, list[ToolCallRequest]]:
        """Parse a JSON text-tool envelope emitted by a chat-only model."""
        text = (content or "").strip()
        if not text:
            return content, []

        candidate = text
        if "```" in candidate:
            parts = candidate.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    candidate = stripped
                    break
        elif "{" in candidate and "}" in candidate:
            candidate = candidate[candidate.find("{"): candidate.rfind("}") + 1]

        try:
            parsed = json_repair.loads(candidate)
        except Exception:
            plain_call = LLMProvider._parse_plain_safe_tool_request(text)
            if plain_call:
                return None, [plain_call]
            return content, []

        if not isinstance(parsed, dict):
            return content, []

        raw_calls = parsed.get("tool_calls") or parsed.get("tools") or []
        if not isinstance(raw_calls, list):
            return content, []

        calls: list[ToolCallRequest] = []
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = fn.get("arguments") or raw.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json_repair.loads(arguments)
                except Exception:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            calls.append(ToolCallRequest(
                id=f"texttool_{uuid.uuid4().hex[:9]}",
                name=name,
                arguments=arguments,
            ))

        if calls:
            return None, calls
        plain_call = LLMProvider._parse_plain_safe_tool_request(text)
        if plain_call:
            return None, [plain_call]
        return content, []

    @staticmethod
    def _parse_plain_safe_tool_request(content: str) -> ToolCallRequest | None:
        """Recover safe tool calls if a chat-only model prints them as text.

        Some relays/models ignore the JSON text-tool protocol and literally emit
        ``Previous assistant tool requests: - evo_studio_agent_consult({...})`` or
        ``evo_studio_cloud_train({"action":"current"})``. Treat only stable consult
        calls and read-only cloud-training actions as executable; never recover
        mutating actions like start/stop from prose.
        """
        safe_cloud_actions = {
            "backend_probe",
            "balance",
            "current",
            "provider_balance",
            "runtime_match",
            "source_preflight",
            "status",
        }
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith((
                "-",
                "evo_studio_agent_consult",
                "openclaw_agent_consult",
                "evo_studio_cloud_train",
            )):
                continue
            for name in ("evo_studio_agent_consult", "openclaw_agent_consult", "evo_studio_cloud_train"):
                marker = f"{name}("
                start = stripped.find(marker)
                if start < 0:
                    continue
                raw = stripped[start + len(marker):]
                if raw.endswith(")"):
                    raw = raw[:-1]
                try:
                    arguments = json_repair.loads(raw)
                except Exception:
                    return None
                if not isinstance(arguments, dict):
                    arguments = {"task": str(arguments)}
                if name == "evo_studio_cloud_train":
                    action = str(arguments.get("action") or "").strip()
                    if action not in safe_cloud_actions:
                        return None
                return ToolCallRequest(
                    id=f"texttool_{uuid.uuid4().hex[:9]}",
                    name=name,
                    arguments=arguments,
                )
        return None

    async def _safe_chat(self, **kwargs: Any) -> LLMResponse:
        """Call chat() and convert unexpected exceptions to error responses."""
        try:
            return await self.chat(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call chat() with retry on transient provider failures.

        Parameters default to ``self.generation`` when not explicitly passed,
        so callers no longer need to thread temperature / max_tokens /
        reasoning_effort through every layer.
        """
        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        kw: dict[str, Any] = dict(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )

        for attempt, delay in enumerate(self._CHAT_RETRY_DELAYS, start=1):
            response = await self._safe_chat(**kw)

            if response.finish_reason != "error":
                return response

            if not self._is_transient_error(response.content):
                if self._is_image_unsupported_error(response.content):
                    stripped = self._strip_image_content(messages)
                    if stripped is not None:
                        logger.warning("Model does not support image input, retrying without images")
                        return await self._safe_chat(**{**kw, "messages": stripped})
                return response

            logger.warning(
                "LLM transient error (attempt {}/{}), retrying in {}s: {}",
                attempt, len(self._CHAT_RETRY_DELAYS), delay,
                (response.content or "")[:120].lower(),
            )
            await asyncio.sleep(delay)

        return await self._safe_chat(**kw)

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
