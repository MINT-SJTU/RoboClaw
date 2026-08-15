1. MCP Server
   将具体函数、进程或外部服务注册为 MCP tool。

2. MCPClientRuntime
   启动并连接 MCP server，建立 MCP client session。

3. MCPClientRuntime
   通过 session.list_tools() 从 MCP server 发现可用工具。

4. MCPToolAdapter
   将 MCP tool definition 包装成 Agent Runtime 可管理的工具对象。

5. ToolRegistry
   维护工具名到工具对象的映射，并向 Agent Loop 提供工具定义和调用入口。

6. Agent Loop
   将 messages 和 tool definitions 一起发送给 LLM。

7. LLM
   根据上下文和工具定义，返回普通回答或 tool_calls。

8. Agent Loop
   解析 tool_calls，并通过 ToolRegistry.invoke(...) 执行对应工具。

9. MCPToolAdapter
   通过 MCPClientRuntime.call_tool(...) 请求 MCP server 执行真实工具。

10. MCP Server
    执行具体函数、进程或外部服务，并返回工具执行结果。

11. Agent Loop
    将工具结果写回 messages，并再次请求 LLM 生成最终回答或继续调用工具。