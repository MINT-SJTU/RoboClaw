"""Minimal LLM tool-call loop for RoboClaw Next."""

from __future__ import annotations

from typing import Any

from roboclaw_next.llm.openai_compatible import LLMProvider
from roboclaw_next.tools import ToolExecutionContext, ToolRegistry


async def run_tool_call_loop(
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tool_registry: ToolRegistry,
    *,
    max_iterations: int = 4,
) -> str | None:
    """Run LLM -> tool call -> tool result -> LLM until a final answer is produced."""

    current_messages = list(messages)
    # 这里需要 max_iterations，是为了防止 Agent 进入无限工具调用循环。
    for _ in range(max_iterations):
        # `await` 的语义是“当前协程暂停在这里，等待这个异步调用完成”：
        # chat_with_retry 返回结果后，才会赋值给 response，并继续执行下面的代码。
        # 如果希望请求先在后台跑、当前代码继续往下执行，需要显式使用：
        # task = asyncio.create_task(provider.chat_with_retry(...))
        # 然后在真正需要结果的位置再写：response = await task
        response = await provider.chat_with_retry(
            current_messages,
            tools=tool_registry.definitions(),
        )
        if not response.has_tool_calls:
            current_messages.append({"role": "assistant", "content": response.content})
            return response.content

        current_messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [tool_call.to_openai_tool_call() for tool_call in response.tool_calls],
            }
        )
        for tool_call in response.tool_calls:
            context = ToolExecutionContext(tool_call_id=tool_call.id)
            result = await tool_registry.invoke(tool_call.name, tool_call.arguments, context)
            current_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": result.as_text(),
                }
            )

    return None
