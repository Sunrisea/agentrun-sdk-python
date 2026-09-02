import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from agentcore.errors import MCPServerNotFoundError, ModelConnectionNotFoundError

from agentrun.integration.builtin.model import model
from agentrun.integration.builtin.tool_resource import tool_resource
from agentrun.integration.utils.skill_loader import skill_tools
from agentrun.integration.utils.tool import CommonToolSet
from agentrun.model import (
    BackendType,
    ModelProxy,
    ModelService,
    ProviderSettings,
)
from agentrun.tool.tool import Tool as ToolResource


class FakeModelClient:
    descriptor = SimpleNamespace(model_name="qwen3.8-max", provider_type="qwen")

    def completion(self, messages, **kwargs):
        return {
            "id": "response-id",
            "object": "chat.completion",
            "created": 1,
            "model": self.descriptor.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
        }

    def stream(self, messages, **kwargs):
        yield {
            "id": "response-id",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": self.descriptor.model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": "hello"},
                "finish_reason": None,
            }],
        }

    def responses_stream(self, input, **kwargs):
        yield {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "item-id",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
            "logprobs": [],
        }

    def responses(self, input, **kwargs):
        return {
            "id": "response-id",
            "created_at": 1,
            "model": self.descriptor.model_name,
            "object": "response",
            "output": [],
            "status": "completed",
        }


class FakeModelCore:

    def __init__(self):
        self.client = FakeModelClient()
        self.model_calls = []

    def model(self, resource_name, *, model=None):
        self.model_calls.append((resource_name, model))
        return self.client


def test_model_string_uses_agentcore_when_enabled(monkeypatch):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    expected = MagicMock()

    with patch(
        "agentrun.compat.agentcore.managed_model", return_value=expected
    ) as managed_model:
        result = model("test-mc", model="qwen3.8-max")

    assert result is expected
    managed_model.assert_called_once_with(
        "test-mc",
        model_name="qwen3.8-max",
        legacy_config=None,
        legacy_backend_type=None,
    )


def test_model_proxy_keeps_agentrun_route(monkeypatch):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    proxy = ModelProxy(model_proxy_name="proxy")

    with (
        patch(
            "agentrun.model.client.ModelClient.get", return_value=proxy
        ) as legacy_get,
        patch("agentrun.compat.agentcore.managed_model") as managed,
    ):
        model("proxy", backend_type=BackendType.PROXY)

    legacy_get.assert_called_once()
    managed.assert_not_called()


def test_explicit_model_service_keeps_direct_route(monkeypatch):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    service = ModelService(
        model_service_name="direct",
        provider_settings=ProviderSettings(
            base_url="https://model.example/v1",
            api_key="key",
            model_names=["model"],
        ),
    )

    with patch("agentrun.compat.agentcore.managed_model") as managed:
        result = model(service)

    assert result.model_obj is service
    managed.assert_not_called()


def test_mcp_string_uses_agentcore_when_enabled(monkeypatch):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    expected = MagicMock()

    with patch(
        "agentrun.compat.agentcore.managed_mcp_tools", return_value=expected
    ) as managed_mcp:
        result = tool_resource("test-mcp")

    assert result is expected
    managed_mcp.assert_called_once_with("test-mcp")


def test_explicit_tool_resource_keeps_agentrun_route(monkeypatch):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    resource = ToolResource(tool_name="direct-mcp")
    expected = MagicMock()

    module = importlib.import_module("agentrun.integration.builtin.tool_resource")
    with patch.object(
        module.CommonToolSet,
        "from_agentrun_tool",
        return_value=expected,
    ) as legacy:
        result = tool_resource(resource)

    assert result is expected
    legacy.assert_called_once_with(resource, config=None)


def test_missing_agentcore_mcp_falls_back_to_agentrun(monkeypatch):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    resource = ToolResource(tool_name="legacy-tool")
    expected = MagicMock()
    module = importlib.import_module("agentrun.integration.builtin.tool_resource")

    with (
        patch(
            "agentrun.compat.agentcore.managed_mcp_tools",
            side_effect=MCPServerNotFoundError("missing"),
        ),
        patch.object(module.ToolClient, "get", return_value=resource) as legacy_get,
        patch.object(
            module.CommonToolSet,
            "from_agentrun_tool",
            return_value=expected,
        ),
    ):
        result = tool_resource("legacy-tool")

    assert result is expected
    legacy_get.assert_called_once_with(name="legacy-tool", config=None)


def test_agentcore_mcp_errors_do_not_fall_back_to_agentrun(monkeypatch):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    module = importlib.import_module("agentrun.integration.builtin.tool_resource")

    with (
        patch(
            "agentrun.compat.agentcore.managed_mcp_tools",
            side_effect=RuntimeError("AgentCore unavailable"),
        ),
        patch.object(module.ToolClient, "get") as legacy_get,
        pytest.raises(RuntimeError, match="AgentCore unavailable"),
    ):
        tool_resource("managed-tool")

    legacy_get.assert_not_called()


def test_remote_skills_use_agentcore_when_enabled(monkeypatch):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    expected = MagicMock()

    with patch(
        "agentrun.compat.agentcore.managed_skill_tools", return_value=expected
    ) as managed_skills:
        result = skill_tools(
            ["skill-a", "skill-b"],
            skills_dir=".skills",
            command_timeout=120,
        )

    assert result is expected
    managed_skills.assert_called_once_with(
        ["skill-a", "skill-b"],
        skills_dir=".skills",
        command_approval=None,
        command_timeout=120,
    )


def test_local_skills_keep_agentrun_route(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    with patch("agentrun.compat.agentcore.managed_skill_tools") as managed:
        result = skill_tools(skills_dir=str(skills_dir))

    assert isinstance(result, CommonToolSet)
    managed.assert_not_called()


def test_agentcore_mode_is_exact(monkeypatch):
    from agentrun.compat.agentcore import is_agentcore_runtime

    monkeypatch.setenv("AGENTRUN_RUNTIME", "legacy")
    assert not is_agentcore_runtime()

    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    assert is_agentcore_runtime()


def test_managed_model_preserves_litellm_response_shape(monkeypatch):
    import agentrun.compat.agentcore as compat

    core = FakeModelCore()
    monkeypatch.setattr(compat, "_core", core)

    result = compat.managed_model("test-mc", model_name="qwen3.8-max")
    response = result.completions(messages=[{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "hello"
    assert core.model_calls == [("test-mc", "qwen3.8-max")]


def test_managed_model_preserves_litellm_stream_shape(monkeypatch):
    import agentrun.compat.agentcore as compat

    monkeypatch.setattr(compat, "_core", FakeModelCore())

    result = compat.managed_model("test-mc", model_name="qwen3.8-max")
    chunks = list(
        result.completions(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    )

    assert chunks[0].choices[0].delta.content == "hello"


def test_managed_model_preserves_responses_stream_object_shape(monkeypatch):
    import agentrun.compat.agentcore as compat

    monkeypatch.setattr(compat, "_core", FakeModelCore())

    result = compat.managed_model("test-mc", model_name="qwen3.8-max")
    events = list(result.responses(input="hi", stream=True))

    assert events[0].type == "response.output_text.delta"
    assert events[0].delta == "hello"


def test_managed_model_preserves_responses_object_shape(monkeypatch):
    import agentrun.compat.agentcore as compat

    monkeypatch.setattr(compat, "_core", FakeModelCore())

    result = compat.managed_model("test-mc", model_name="qwen3.8-max")
    response = result.responses(input="hi")

    assert response.id == "response-id"
    assert response.status == "completed"


@pytest.mark.asyncio
async def test_managed_model_preserves_async_completion_shapes(monkeypatch):
    import agentrun.compat.agentcore as compat

    monkeypatch.setattr(compat, "_core", FakeModelCore())
    result = compat.managed_model("test-mc", model_name="qwen3.8-max")

    response = await result.model_obj.acompletion(
        messages=[{"role": "user", "content": "hi"}]
    )
    stream = await result.model_obj.acompletion(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = [chunk async for chunk in stream]

    assert response.choices[0].message.content == "hello"
    assert chunks[0].choices[0].delta.content == "hello"


@pytest.mark.asyncio
async def test_managed_model_preserves_async_responses_stream_shape(
    monkeypatch,
):
    import agentrun.compat.agentcore as compat

    monkeypatch.setattr(compat, "_core", FakeModelCore())
    result = compat.managed_model("test-mc", model_name="qwen3.8-max")

    stream = await result.model_obj.aresponses(input="hi", stream=True)
    events = [event async for event in stream]

    assert events[0].type == "response.output_text.delta"
    assert events[0].delta == "hello"


@pytest.mark.parametrize(
    ("method", "module_name"),
    [
        ("to_langchain", "langchain"),
        ("to_langgraph", "langgraph"),
        ("to_google_adk", "google_adk"),
        ("to_agentscope", "agentscope"),
        ("to_pydantic_ai", "pydantic_ai"),
        ("to_crewai", "crewai"),
    ],
)
def test_managed_model_framework_conversion_uses_agentcore_adapter(
    monkeypatch, method, module_name
):
    import agentrun.compat.agentcore as compat

    core = FakeModelCore()
    monkeypatch.setattr(compat, "_core", core)
    if method == "to_agentscope":
        monkeypatch.setattr(compat.sys, "version_info", (3, 11))
    adapter = MagicMock()
    monkeypatch.setattr(compat.importlib, "import_module", lambda _: adapter)

    result = compat.managed_model("test-mc", model_name="qwen3.8-max")
    getattr(result, method)()

    adapter.model.assert_called_once_with(
        "test-mc",
        model_name="qwen3.8-max",
    )
    assert core.model_calls == []


def test_missing_framework_model_falls_back_to_agentrun_adapter(monkeypatch):
    import agentrun.compat.agentcore as compat

    adapter = MagicMock()
    adapter.model.side_effect = ModelConnectionNotFoundError("missing")
    legacy_resource = MagicMock()
    converted = MagicMock()

    with (
        patch(
            "agentrun.model.client.ModelClient.get",
            return_value=legacy_resource,
        ) as legacy_get,
        patch.object(
            compat.CommonModel,
            "to_langchain",
            return_value=converted,
        ),
    ):
        monkeypatch.setattr(compat.importlib, "import_module", lambda _: adapter)
        result = compat.managed_model("legacy-model")
        assert result.to_langchain() is converted

    legacy_get.assert_called_once_with(
        "legacy-model",
        backend_type=None,
        config=None,
    )


def test_agentscope_on_python310_keeps_agentrun_model_route(monkeypatch):
    import agentrun.compat.agentcore as compat

    core = FakeModelCore()
    legacy_resource = MagicMock()
    converted = MagicMock()
    monkeypatch.setattr(compat, "_core", core)
    monkeypatch.setattr(compat.sys, "version_info", (3, 10))

    with (
        patch(
            "agentrun.model.client.ModelClient.get",
            return_value=legacy_resource,
        ) as legacy_get,
        patch.object(
            compat.CommonModel,
            "to_agentscope",
            return_value=converted,
        ) as legacy_convert,
    ):
        result = compat.managed_model("test-mc", model_name="qwen3.8-max")
        assert result.to_agentscope() is converted

    legacy_get.assert_called_once_with(
        "test-mc",
        backend_type=None,
        config=None,
    )
    legacy_convert.assert_called_once_with()
    assert core.model_calls == []


def test_managed_model_clients_are_cached_by_effective_model(monkeypatch):
    import agentrun.compat.agentcore as compat

    core = FakeModelCore()
    monkeypatch.setattr(compat, "_core", core)
    result = compat.managed_model("test-mc", model_name="qwen3.8-max")

    result.completions(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen3.8-plus",
    )
    result.completions(
        messages=[{"role": "user", "content": "hi again"}],
        model="qwen3.8-plus",
    )
    info = result.get_model_info()

    assert info.model == "qwen3.8-max"
    assert info.provider == "qwen"
    assert core.model_calls == [
        ("test-mc", "qwen3.8-plus"),
        ("test-mc", "qwen3.8-max"),
    ]


def test_missing_agentcore_model_falls_back_to_agentrun(monkeypatch):
    import agentrun.compat.agentcore as compat

    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    core = MagicMock()
    core.model.side_effect = ModelConnectionNotFoundError("missing")
    legacy = MagicMock()
    legacy.completion.return_value = "legacy-response"
    monkeypatch.setattr(compat, "_core", core)

    with patch(
        "agentrun.model.client.ModelClient.get", return_value=legacy
    ) as legacy_get:
        result = model("legacy-model")
        response = result.completions(messages=[])

    assert response == "legacy-response"
    legacy_get.assert_called_once_with(
        "legacy-model",
        backend_type=None,
        config=None,
    )


def test_missing_agentcore_model_preserves_explicit_service_route(monkeypatch):
    import agentrun.compat.agentcore as compat

    monkeypatch.setenv("AGENTRUN_RUNTIME", "agentcore")
    core = MagicMock()
    core.model.side_effect = ModelConnectionNotFoundError("missing")
    legacy = MagicMock()
    monkeypatch.setattr(compat, "_core", core)

    with patch(
        "agentrun.model.client.ModelClient.get", return_value=legacy
    ) as legacy_get:
        result = model("legacy-model", backend_type=BackendType.SERVICE)
        result.completions(messages=[])

    legacy_get.assert_called_once_with(
        "legacy-model",
        backend_type=BackendType.SERVICE,
        config=None,
    )


def test_managed_model_embedding_keeps_legacy_agentrun_route(monkeypatch):
    import agentrun.compat.agentcore as compat
    from agentrun.utils.config import Config

    core = FakeModelCore()
    legacy = MagicMock()
    legacy.embedding.return_value = {"data": []}
    config = Config()
    monkeypatch.setattr(compat, "_core", core)

    with patch(
        "agentrun.model.client.ModelClient.get", return_value=legacy
    ) as legacy_get:
        result = compat.managed_model(
            "embedding-service",
            legacy_config=config,
        )
        response = result.model_obj.embedding(input=["hello"])

    assert response == {"data": []}
    assert core.model_calls == []
    legacy_get.assert_called_once_with(
        "embedding-service",
        backend_type=None,
        config=config,
    )


def test_failed_agentcore_model_call_does_not_fallback_to_agentrun(monkeypatch):
    import agentrun.compat.agentcore as compat

    core = MagicMock()
    core.model.side_effect = RuntimeError("AgentCore unavailable")
    monkeypatch.setattr(compat, "_core", core)

    with patch("agentrun.model.client.ModelClient.get") as legacy_get:
        result = compat.managed_model("test-mc")
        with pytest.raises(RuntimeError, match="AgentCore unavailable"):
            result.completions(messages=[])

    legacy_get.assert_not_called()


def test_managed_mcp_tools_preserve_tool_schema_and_call(monkeypatch):
    import agentrun.compat.agentcore as compat

    mcp = MagicMock()
    mcp.list_tools.return_value = [
        SimpleNamespace(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    mcp.call_tool.return_value = {"result": "ok"}
    core = MagicMock()
    core.mcp.return_value = mcp
    monkeypatch.setattr(compat, "_core", core)

    toolset = compat.managed_mcp_tools("test-mcp")
    tool = toolset.tools()[0]
    result = tool.func(query="agentcore")

    assert tool.parameters["required"] == ["query"]
    assert result == {"result": "ok"}
    mcp.call_tool.assert_called_once_with("search", {"query": "agentcore"})


def test_managed_skill_command_preserves_agentcore_default_cwd(monkeypatch):
    import agentrun.compat.agentcore as compat

    skills_dir = "skills"
    observed = {}

    def execute(arguments):
        observed.update(arguments)
        return "ok"

    execute_tool = SimpleNamespace(
        name="execute_command",
        description="Execute a command",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string"},
            },
            "required": ["command"],
        },
        invoke=execute,
    )
    core = MagicMock()
    core.skills.managed.return_value = SimpleNamespace(name="test-skill")
    monkeypatch.setattr(compat, "_get_skill_core", lambda _: core)
    agentcore = ModuleType("agentcore")
    agentcore_skill = ModuleType("agentcore.skill")
    agentcore_skill.skill_tools = lambda *args, **kwargs: [execute_tool]
    monkeypatch.setitem(sys.modules, "agentcore", agentcore)
    monkeypatch.setitem(sys.modules, "agentcore.skill", agentcore_skill)

    toolset = compat.managed_skill_tools(
        ["test-skill"],
        skills_dir=skills_dir,
        command_approval=None,
        command_timeout=300,
    )
    result = toolset.tools()[0].func(command="pwd")

    assert result == "ok"
    assert observed == {"command": "pwd"}


def test_agentcore_clients_are_created_once_and_closed_once(
    monkeypatch, tmp_path
):
    import agentcore

    import agentrun.compat.agentcore as compat

    default_core = MagicMock()
    skill_core = MagicMock()
    agentcore_type = MagicMock()
    agentcore_type.auto.side_effect = [default_core, skill_core]
    monkeypatch.setattr(agentcore, "AgentCore", agentcore_type)
    monkeypatch.setattr(compat, "_core", None)
    monkeypatch.setattr(compat, "_skill_cores", {})

    assert compat._get_core() is default_core
    assert compat._get_core() is default_core
    assert compat._get_skill_core(str(tmp_path)) is skill_core
    assert compat._get_skill_core(str(tmp_path)) is skill_core

    compat._close_cores()

    assert agentcore_type.auto.call_count == 2
    default_core.close.assert_called_once_with()
    skill_core.close.assert_called_once_with()
