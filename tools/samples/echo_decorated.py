"""Echo tool using the @tool decorator — minimal boilerplate."""

from agentos_tool import tool, ToolResult


@tool(
    name="echo-py",
    description="Echo tool (Python) — returns the input message",
    semantic_description=(
        "A simple echo tool that repeats back whatever message "
        "it receives. Demonstrates the @tool decorator."
    ),
)
class EchoTool:
    class Input:
        message: str
        times: int = 1

    def handle(self, input) -> ToolResult:
        repeated = (input.message + "\n") * input.times
        return ToolResult.ok(f"echo-py: {repeated.strip()}")
