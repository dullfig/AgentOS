"""Echo tool — a pure Python tool for the AgentOS Python runtime.

No WASM compilation needed. This file is loaded and executed by
the python-runtime component at call time.

Required exports:
  get_metadata() -> ToolMetadata dataclass
  handle(request_xml: str) -> ToolResult dataclass
"""

import re
from dataclasses import dataclass


@dataclass
class ToolMetadata:
    name: str
    description: str
    semantic_description: str
    request_tag: str
    request_schema: str
    response_schema: str
    input_json_schema: str


@dataclass
class ToolResult:
    success: bool
    payload: str


def get_metadata() -> ToolMetadata:
    return ToolMetadata(
        name="echo-py",
        description="Echo tool (Python) — returns the input message",
        semantic_description=(
            "A simple echo tool that repeats back whatever message "
            "it receives. Demonstrates pure Python tool authoring."
        ),
        request_tag="EchoRequest",
        request_schema="",
        response_schema="",
        input_json_schema=(
            '{"type":"object","properties":{"message":{"type":"string",'
            '"description":"The message to echo back"}},"required":["message"]}'
        ),
    )


def handle(request_xml: str) -> ToolResult:
    match = re.search(r"<message>(.*?)</message>", request_xml, re.DOTALL)
    message = match.group(1) if match else "(no message)"
    return ToolResult(success=True, payload=f"echo-py: {message}")
