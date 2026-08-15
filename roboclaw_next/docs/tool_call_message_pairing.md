# Tool Call Message Pairing

## 核心知识点

在 OpenAI-compatible tool calling 中，工具调用消息必须成对出现：

```text
assistant message with tool_calls
    -> tool message with matching tool_call_id
```

也就是说，工具结果不能直接孤立地塞进对话历史。每一条 `role="tool"` 消息，都必须对应前面某一条 `role="assistant"` 消息里的 `tool_calls`。

## 为什么需要成对

LLM 返回 tool call 时，它实际上是在说：

```text
assistant: 我要调用这个工具，并给出调用参数。
```

Runtime 执行工具后，需要把结果放回上下文：

```text
tool: 这是刚才那个工具调用的执行结果。
```

为了让模型知道“这个结果对应哪个工具调用”，`tool` 消息必须带上同一个 `tool_call_id`。

## 正确消息结构

```python
messages.append(
    {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "demo__add",
                    "arguments": "{\"a\": 19, \"b\": 23}",
                },
            }
        ],
    }
)

messages.append(
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "demo__add",
        "content": "42",
    }
)
```

## 错误结构

```python
messages.append(
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "name": "demo__add",
        "content": "42",
    }
)
```

这条 `tool` 消息前面没有对应的 assistant `tool_calls` 消息，很多 OpenAI-compatible API 会认为上下文不合法。
这条规则是 Agent Loop 的基础约束之一。后续无论工具来自本地函数、机器人能力、MCP server，还是用户自定义 skill，只要最后走 OpenAI-compatible tool calling，都必须维持这个消息配对关系。
