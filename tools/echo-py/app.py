"""Echo tool — Python WASM component implementing the AgentOS tool contract."""

import re
import wit_world


class WitWorld(wit_world.WitWorld):
    def get_metadata(self) -> wit_world.ToolMetadata:
        return wit_world.ToolMetadata(
            name="echo-py",
            description="Echo tool (Python) — returns the input message",
            semantic_description=(
                "A simple echo tool written in Python that repeats back whatever "
                "message it receives. Demonstrates Python WASM tool authoring."
            ),
            request_tag="EchoRequest",
            request_schema=(
                '<xs:schema>'
                '  <xs:element name="EchoRequest">'
                '    <xs:complexType>'
                '      <xs:sequence>'
                '        <xs:element name="message" type="xs:string"/>'
                '      </xs:sequence>'
                '    </xs:complexType>'
                '  </xs:element>'
                '</xs:schema>'
            ),
            response_schema=(
                '<xs:schema>'
                '  <xs:element name="ToolResponse">'
                '    <xs:complexType>'
                '      <xs:sequence>'
                '        <xs:element name="success" type="xs:boolean"/>'
                '        <xs:element name="result" type="xs:string" minOccurs="0"/>'
                '        <xs:element name="error" type="xs:string" minOccurs="0"/>'
                '      </xs:sequence>'
                '    </xs:complexType>'
                '  </xs:element>'
                '</xs:schema>'
            ),
            input_json_schema=(
                '{"type":"object","properties":{"message":{"type":"string",'
                '"description":"The message to echo back"}},"required":["message"]}'
            ),
        )

    def handle(self, request_xml: str) -> wit_world.ToolResult:
        match = re.search(r"<message>(.*?)</message>", request_xml, re.DOTALL)
        message = match.group(1) if match else "(no message)"
        return wit_world.ToolResult(success=True, payload=f"echo-py: {message}")
