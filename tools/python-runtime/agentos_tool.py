"""AgentOS Tool SDK — write tools with pure Python, no boilerplate.

Usage:
    from agentos_tool import tool, ToolResult

    @tool(name="echo", description="Echo tool")
    class EchoTool:
        class Input:
            message: str
            count: int = 1

        def handle(self, input):
            return ToolResult(ok=f"echo: {input.message}")

The @tool decorator auto-generates:
  - get_metadata() with JSON schema derived from Input annotations
  - handle(request_xml) that parses XML into Input, calls your handle(), wraps result
  - Request tag from tool name (e.g., "echo" -> "EchoRequest")
"""

import re
import json
from dataclasses import dataclass, fields, field, make_dataclass


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    payload: str

    @classmethod
    def ok(cls, result: str) -> "ToolResult":
        return cls(success=True, payload=result)

    @classmethod
    def err(cls, error: str) -> "ToolResult":
        return cls(success=False, payload=error)


@dataclass
class ToolMetadata:
    """Metadata for a tool (matches WIT tool-metadata record)."""
    name: str
    description: str
    semantic_description: str
    request_tag: str
    request_schema: str
    response_schema: str
    input_json_schema: str


# Python type -> JSON Schema type mapping
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _extract_tag(xml: str, tag: str):
    """Extract text between <tag> and </tag>."""
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", xml, re.DOTALL)
    return match.group(1) if match else None


def _xml_unescape(s: str) -> str:
    """Unescape XML entities."""
    return (s.replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&amp;", "&"))


def _annotations_to_schema(cls):
    """Convert a class's type annotations to JSON Schema + field info.

    Returns (json_schema_dict, field_specs) where field_specs is a list of
    (name, type, default_or_MISSING) tuples.
    """
    hints = getattr(cls, "__annotations__", {})
    if not hints:
        return {"type": "object", "properties": {}}, []

    properties = {}
    required = []
    field_specs = []

    for name, typ in hints.items():
        # Get default if any
        default = getattr(cls, name, _MISSING)

        json_type = _TYPE_MAP.get(typ, "string")

        # Extract doc from comment (not available at runtime, use field name)
        prop = {"type": json_type, "description": name.replace("_", " ")}
        properties[name] = prop

        if default is _MISSING:
            required.append(name)
            field_specs.append((name, typ, _MISSING))
        else:
            prop["default"] = default
            field_specs.append((name, typ, default))

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema, field_specs


class _MISSING:
    """Sentinel for missing default values."""
    pass


def _build_input_class(input_cls, field_specs):
    """Build a proper dataclass from the Input class annotations."""
    dc_fields = []
    for name, typ, default in field_specs:
        if default is _MISSING:
            dc_fields.append((name, typ))
        else:
            dc_fields.append((name, typ, field(default=default)))

    return make_dataclass("Input", dc_fields)


def _parse_value(value_str: str, typ: type):
    """Parse a string value into the target type."""
    if typ == str:
        return _xml_unescape(value_str)
    elif typ == int:
        return int(value_str)
    elif typ == float:
        return float(value_str)
    elif typ == bool:
        return value_str.lower() in ("true", "1", "yes")
    return value_str


def tool(name: str, description: str, semantic_description: str = ""):
    """Decorator that turns a class into an AgentOS tool.

    The class must have:
      - An `Input` inner class with type-annotated fields
      - A `handle(self, input) -> ToolResult` method
    """

    def decorator(cls):
        # Extract Input class
        input_cls = getattr(cls, "Input", None)
        if input_cls is None:
            raise TypeError(f"@tool class {cls.__name__} must have an Input inner class")

        handle_method = getattr(cls, "handle", None)
        if handle_method is None:
            raise TypeError(f"@tool class {cls.__name__} must have a handle() method")

        # Build schema from Input annotations
        json_schema, field_specs = _annotations_to_schema(input_cls)
        json_schema_str = json.dumps(json_schema)

        # Build request tag: "echo" -> "EchoRequest"
        parts = name.replace("-", "_").split("_")
        camel = "".join(p.capitalize() for p in parts)
        request_tag = f"{camel}Request"

        # Build the proper Input dataclass
        InputDC = _build_input_class(input_cls, field_specs)

        # Create the tool instance
        instance = cls()

        # Module-level get_metadata function
        def get_metadata() -> ToolMetadata:
            return ToolMetadata(
                name=name,
                description=description,
                semantic_description=semantic_description or description,
                request_tag=request_tag,
                request_schema="",
                response_schema="",
                input_json_schema=json_schema_str,
            )

        # Module-level handle function
        def handle(request_xml: str) -> ToolResult:
            try:
                # Parse XML fields into Input dataclass
                kwargs = {}
                for fname, ftype, fdefault in field_specs:
                    # Try kebab-case and snake_case tag names
                    value_str = _extract_tag(request_xml, fname.replace("_", "-"))
                    if value_str is None:
                        value_str = _extract_tag(request_xml, fname)
                    if value_str is not None:
                        kwargs[fname] = _parse_value(value_str, ftype)
                    elif fdefault is not _MISSING:
                        kwargs[fname] = fdefault
                    else:
                        return ToolResult.err(f"missing required field: {fname}")

                input_obj = InputDC(**kwargs)
                result = instance.handle(input_obj)

                # Allow returning a plain string as success
                if isinstance(result, str):
                    return ToolResult.ok(result)
                return result

            except Exception as e:
                return ToolResult.err(f"tool error: {e}")

        # Inject into the caller's module globals
        # (the python-runtime exec()'s the source and looks for these)
        import sys
        frame = sys._getframe(1)
        frame.f_globals["get_metadata"] = get_metadata
        frame.f_globals["handle"] = handle

        # Also attach to the class for direct use
        cls._get_metadata = staticmethod(get_metadata)
        cls._handle = staticmethod(handle)

        return cls

    return decorator
