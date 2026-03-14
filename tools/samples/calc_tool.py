"""Calculator tool — evaluates math expressions safely inside WASM."""

import math

from agentos_tool import tool, ToolResult


@tool(
    name="calc",
    description="Evaluate a math expression safely. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e, abs, round, min, max.",
    semantic_description=(
        "A calculator tool that evaluates mathematical expressions. "
        "Use when the user needs arithmetic, scientific calculations, "
        "unit conversions, or any numeric computation."
    ),
)
class CalcTool:
    class Input:
        expression: str

    def handle(self, input) -> ToolResult:
        safe_names = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "pi": math.pi,
            "e": math.e,
            "inf": math.inf,
        }

        try:
            result = eval(input.expression, {"__builtins__": {}}, safe_names)
            return ToolResult.ok(f"{input.expression} = {result}")
        except Exception as ex:
            return ToolResult.err(f"Error evaluating '{input.expression}': {ex}")
