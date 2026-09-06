"""Exercise the installed AgentScope Toolkit, not a mocked framework adapter."""

import json
from importlib.metadata import version

import pytest

pytest.importorskip("agentscope")

from agentrun.integration.agentscope.tool_adapter import AgentScopeToolAdapter
from agentrun.integration.utils.canonical import CanonicalTool

V2 = int(version("agentscope").split(".")[0]) >= 2


def echo_tool(result):
    return CanonicalTool(
        name="echo",
        description="Echo a value",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string", "enum": ["hello"]}},
            "required": ["value"],
        },
        func=lambda value: result,
    )


@pytest.mark.skipif(not V2, reason="AgentScope 2.x Toolkit API")
@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["hello", {"value": "hello"}])
async def test_v2_register_schema_and_execute(result):
    from agentscope.message import ToolCallBlock
    from agentscope.state import AgentState
    from agentscope.tool import FunctionTool, Toolkit

    canonical = echo_tool(result)
    converted = AgentScopeToolAdapter().from_canonical([canonical])
    assert isinstance(converted[0], FunctionTool)
    toolkit = Toolkit(tools=converted)
    schemas = await toolkit.get_tool_schemas()
    schema = next(
        s["function"] for s in schemas if s["function"]["name"] == "echo"
    )
    assert schema["parameters"] == canonical.parameters
    outputs = [
        output
        async for output in toolkit.call_tool(
            ToolCallBlock(id="call-1", name="echo", input='{"value": "hello"}'),
            AgentState(),
        )
    ]
    text = outputs[-1].content[0].text
    assert (json.loads(text) if isinstance(result, dict) else text) == result


@pytest.mark.skipif(not V2, reason="AgentScope 2.x Toolkit API")
@pytest.mark.asyncio
async def test_agentcore_tool_conversion_preserves_prefix_and_call():
    from agentcore.integrations.common import Tool
    from agentscope.message import ToolCallBlock
    from agentscope.state import AgentState
    from agentscope.tool import Toolkit

    from agentrun.compat.agentcore import _AgentCoreToolResource
    from agentrun.integration.utils.tool import CommonToolSet

    calls = []

    def call(arguments):
        calls.append(arguments)
        return "hello"

    async def async_call(arguments):
        return call(arguments)

    canonical = echo_tool("hello")
    tool = Tool(
        "echo", "Echo", canonical.parameters, async_call
    ).with_sync_call(call)
    toolset = CommonToolSet.from_agentrun_tool(
        _AgentCoreToolResource(tools=[tool])
    )
    toolkit = Toolkit(tools=toolset.to_agentscope(prefix="test_"))
    schemas = await toolkit.get_tool_schemas()
    name = next(
        s["function"]["name"]
        for s in schemas
        if "echo" in s["function"]["name"]
    )
    assert name.startswith("test_")
    outputs = [
        output
        async for output in toolkit.call_tool(
            ToolCallBlock(id="call-1", name=name, input='{"value": "hello"}'),
            AgentState(),
        )
    ]
    assert outputs[-1].content[0].text == "hello"
    assert calls == [{"value": "hello"}]


@pytest.mark.skipif(V2, reason="AgentScope 1.x Toolkit API")
@pytest.mark.asyncio
async def test_v1_preserves_function_registration_and_response():
    from agentscope.tool import Toolkit, ToolResponse

    converted = AgentScopeToolAdapter().from_canonical(
        [echo_tool({"value": "hello"})]
    )
    assert callable(converted[0])
    toolkit = Toolkit()
    toolkit.register_tool_function(converted[0])
    assert toolkit.get_json_schemas()[0]["function"]["name"] == "echo"
    stream = await toolkit.call_tool_function({
        "type": "tool_use",
        "id": "call-1",
        "name": "echo",
        "input": {"value": "hello"},
    })
    outputs = [output async for output in stream]
    assert isinstance(outputs[-1], ToolResponse)
    assert json.loads(outputs[-1].content[0]["text"]) == {"value": "hello"}
