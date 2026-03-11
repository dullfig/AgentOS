"""AgentOS Python Runtime — executes Python tool source inside WASM.

This is the single Python WASM component that ships with the platform.
Individual tools are plain .py files; this runtime loads and executes them.

The agentos_tool module is bundled into this component and available to
all tool source via `from agentos_tool import tool, ToolResult`.
"""

import sys
import wit_world

# Make agentos_tool importable by tools running inside this runtime.
# It's bundled alongside app.py in the WASM component.
import agentos_tool
sys.modules["agentos_tool"] = agentos_tool


def _convert_metadata(m) -> wit_world.ToolMetadata:
    """Convert an agentos_tool.ToolMetadata to a wit_world.ToolMetadata."""
    return wit_world.ToolMetadata(
        name=m.name,
        description=m.description,
        semantic_description=m.semantic_description,
        request_tag=m.request_tag,
        request_schema=m.request_schema,
        response_schema=m.response_schema,
        input_json_schema=m.input_json_schema,
    )


def _convert_result(r) -> wit_world.ToolResult:
    """Convert an agentos_tool.ToolResult to a wit_world.ToolResult."""
    return wit_world.ToolResult(success=r.success, payload=r.payload)


class WitWorld(wit_world.WitWorld):
    def _load_tool(self, source: str) -> dict:
        """Execute tool source and return its module namespace."""
        namespace = {"__builtins__": __builtins__}
        exec(source, namespace)
        return namespace

    def get_metadata(self, source: str) -> wit_world.ToolMetadata:
        """Load tool source and call its get_metadata() function."""
        try:
            ns = self._load_tool(source)
            if "get_metadata" not in ns:
                return wit_world.ToolMetadata(
                    name="unknown",
                    description="Error: tool source has no get_metadata() function",
                    semantic_description="",
                    request_tag="UnknownRequest",
                    request_schema="",
                    response_schema="",
                    input_json_schema="{}",
                )
            result = ns["get_metadata"]()
            # Convert if it's from agentos_tool (not wit_world)
            if not isinstance(result, wit_world.ToolMetadata):
                return _convert_metadata(result)
            return result
        except Exception as e:
            return wit_world.ToolMetadata(
                name="error",
                description=f"Error loading tool: {e}",
                semantic_description="",
                request_tag="ErrorRequest",
                request_schema="",
                response_schema="",
                input_json_schema="{}",
            )

    def handle(self, source: str, request_xml: str) -> wit_world.ToolResult:
        """Load tool source and call its handle(request_xml) function."""
        try:
            ns = self._load_tool(source)
            if "handle" not in ns:
                return wit_world.ToolResult(
                    success=False,
                    payload="Error: tool source has no handle() function",
                )
            result = ns["handle"](request_xml)
            # Convert if it's from agentos_tool (not wit_world)
            if not isinstance(result, wit_world.ToolResult):
                return _convert_result(result)
            return result
        except Exception as e:
            return wit_world.ToolResult(
                success=False,
                payload=f"Runtime error: {e}",
            )
