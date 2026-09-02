"""AgentCore backend used by the AgentRun compatibility mode."""

from __future__ import annotations

import asyncio
import atexit
import importlib
import os
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from agentcore.errors import ModelConnectionNotFoundError

from agentrun.integration.utils.model import CommonModel
from agentrun.integration.utils.tool import CommonToolSet
from agentrun.model import BackendType, ModelProxy
from agentrun.model.api.data import BaseInfo
from agentrun.utils.config import Config
from agentrun.utils.log import logger

_RUNTIME_ENV = "AGENTRUN_RUNTIME"
_AGENTCORE_RUNTIME = "agentcore"
_core: Any = None
_skill_cores: dict[str, Any] = {}
_core_lock = Lock()


def is_agentcore_runtime() -> bool:
    """Return whether AgentRun should route supported calls to AgentCore."""
    return os.getenv(_RUNTIME_ENV, "").strip().lower() == _AGENTCORE_RUNTIME


def managed_model(
    resource_name: str,
    *,
    model_name: str | None = None,
    legacy_config: Config | None = None,
    legacy_backend_type: BackendType | None = None,
) -> CommonModel:
    """Build the legacy CommonModel facade over an AgentCore model."""
    logger.info(
        "AgentCore compatibility route selected: capability=model resource=%s",
        resource_name,
    )
    resource = _AgentCoreModelResource(
        resource_name,
        model_name=model_name,
        legacy_config=legacy_config,
        legacy_backend_type=legacy_backend_type,
    )
    return _AgentCoreCommonModel(resource)


def managed_mcp_tools(name: str) -> CommonToolSet:
    """Resolve an AgentCore MCP and expose the AgentRun CommonToolSet shape."""
    logger.info(
        "AgentCore compatibility route selected: capability=mcp resource=%s",
        name,
    )
    client = _get_core().mcp(name)
    return CommonToolSet.from_agentrun_tool(_AgentCoreToolResource(client))


def managed_skill_tools(
    names: Sequence[str],
    *,
    skills_dir: str,
    command_approval: Callable[[str, str], bool] | None,
    command_timeout: int,
) -> CommonToolSet:
    """Resolve AgentCore Skills while retaining the AgentRun tool protocol."""
    logger.info(
        "AgentCore compatibility route selected: capability=skill resources=%s",
        ",".join(names),
    )
    core = _get_skill_core(skills_dir)
    skills = [core.skills.managed(name) for name in names]

    from agentcore.skill import skill_tools

    tools = skill_tools(
        skills,
        command_approval=command_approval,
        command_timeout=command_timeout,
    )
    return CommonToolSet.from_agentrun_tool(
        _AgentCoreToolResource(tools=tools)
    )


def _get_core() -> Any:
    global _core

    if _core is not None:
        return _core
    with _core_lock:
        if _core is None:
            from agentcore import AgentCore

            _core = AgentCore.auto()
        return _core


def _get_skill_core(skills_dir: str) -> Any:
    workspace = str((Path(skills_dir) / ".agentcore-managed").absolute())
    core = _skill_cores.get(workspace)
    if core is not None:
        return core
    with _core_lock:
        core = _skill_cores.get(workspace)
        if core is None:
            from agentcore import AgentCore

            core = AgentCore.auto(skill_workspace_dir=workspace)
            _skill_cores[workspace] = core
        return core


def _close_cores() -> None:
    cores = [
        core for core in [_core, *_skill_cores.values()] if core is not None
    ]
    for core in {id(value): value for value in cores}.values():
        core.close()


atexit.register(_close_cores)


class _AgentCoreCommonModel(CommonModel):

    def __init__(self, resource: _AgentCoreModelResource) -> None:
        super().__init__(
            model_obj=resource,  # type: ignore[arg-type]
            backend_type=BackendType.SERVICE,
            specific_model=resource.model_name,
            config=resource.legacy_config,
        )
        self._resource = resource

    def to_langchain(self) -> Any:
        return self._framework_model("langchain")

    def to_langgraph(self) -> Any:
        return self._framework_model("langgraph")

    def to_google_adk(self) -> Any:
        return self._framework_model("google_adk")

    def to_agentscope(self) -> Any:
        if sys.version_info < (3, 11):
            logger.info(
                "AgentCore compatibility keeps AgentRun route: "
                "capability=agentscope resource=%s",
                self._resource.resource_name,
            )
            return self._resource._legacy_common_model().to_agentscope()
        return self._framework_model("agentscope")

    def to_pydantic_ai(self) -> Any:
        return self._framework_model("pydantic_ai")

    def to_crewai(self) -> Any:
        return self._framework_model("crewai")

    def _framework_model(self, framework: str) -> Any:
        if self._resource._use_legacy:
            return getattr(
                self._resource._legacy_common_model(),
                f"to_{framework}",
            )()
        module = importlib.import_module(f"agentcore.integrations.{framework}")
        try:
            result = module.model(
                self._resource.resource_name,
                model_name=self._resource.model_name,
            )
        except ModelConnectionNotFoundError:
            if self._resource._agentcore_bound:
                raise
            logger.info(
                "AgentCore model connection not found; keeping AgentRun route: "
                "resource=%s",
                self._resource.resource_name,
            )
            self._resource._use_legacy = True
            return getattr(
                self._resource._legacy_common_model(),
                f"to_{framework}",
            )()
        self._resource._agentcore_bound = True
        return result


class _AgentCoreModelResource:

    def __init__(
        self,
        resource_name: str,
        *,
        model_name: str | None,
        legacy_config: Config | None,
        legacy_backend_type: BackendType | None,
    ) -> None:
        self.resource_name = resource_name
        self.model_name = model_name
        self.legacy_config = legacy_config
        self.legacy_backend_type = legacy_backend_type
        self._clients: dict[str | None, Any] = {}
        self._legacy_model: Any = None
        self._agentcore_bound = False
        self._use_legacy = False

    def model_info(self, config: Config | None = None) -> BaseInfo:
        client = self._client_for(None)
        if client is None:
            return self._get_legacy_model().model_info(config=config)
        descriptor = client.descriptor
        return BaseInfo(
            model=descriptor.model_name,
            provider=descriptor.provider_type or None,
        )

    def completions(self, *args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            "completions() is deprecated, use completion() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.completion(*args, **kwargs)

    def completion(
        self,
        messages: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        custom_llm_provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        stream = bool(kwargs.pop("stream", False))
        client = self._client_for(model)
        if client is None:
            return self._get_legacy_model().completion(
                messages=messages or [],
                model=model,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                **kwargs,
            )
        if stream:
            return _chat_stream(client.stream(messages or [], **kwargs))
        return _chat_response(client.completion(messages or [], **kwargs))

    async def acompletion(
        self,
        messages: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
        custom_llm_provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        result = await asyncio.to_thread(
            self.completion,
            messages,
            model,
            custom_llm_provider,
            **kwargs,
        )
        if kwargs.get("stream"):
            return _async_from_sync(result)
        return result

    def responses(
        self,
        input: Any,
        model: str | None = None,
        custom_llm_provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        stream = bool(kwargs.pop("stream", False))
        client = self._client_for(model)
        if client is None:
            return self._get_legacy_model().responses(
                input=input,
                model=model,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                **kwargs,
            )
        if stream:
            return _responses_stream(client.responses_stream(input, **kwargs))
        return _responses_response(client.responses(input, **kwargs))

    async def aresponses(
        self,
        input: Any,
        model: str | None = None,
        custom_llm_provider: str | None = None,
        **kwargs: Any,
    ) -> Any:
        result = await asyncio.to_thread(
            self.responses,
            input,
            model,
            custom_llm_provider,
            **kwargs,
        )
        if kwargs.get("stream"):
            return _async_from_sync(result)
        return result

    def embedding(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_legacy_model().embedding(*args, **kwargs)

    def aembedding(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_legacy_model().aembedding(*args, **kwargs)

    def _client_for(self, model_name: str | None) -> Any | None:
        if self._use_legacy:
            return None
        effective_model = model_name if model_name is not None else self.model_name
        client = self._clients.get(effective_model)
        if client is None:
            try:
                client = _get_core().model(
                    self.resource_name,
                    model=effective_model,
                )
            except ModelConnectionNotFoundError:
                if self._agentcore_bound:
                    raise
                logger.info(
                    "AgentCore model connection not found; keeping AgentRun route: "
                    "resource=%s",
                    self.resource_name,
                )
                self._use_legacy = True
                return None
            self._agentcore_bound = True
            self._clients[effective_model] = client
        return client

    def _get_legacy_model(self) -> Any:
        if self._legacy_model is None:
            from agentrun.model.client import ModelClient

            self._legacy_model = ModelClient(config=self.legacy_config).get(
                self.resource_name,
                backend_type=self.legacy_backend_type,
                config=self.legacy_config,
            )
        return self._legacy_model

    def _legacy_common_model(self) -> CommonModel:
        model = self._get_legacy_model()
        backend_type = (
            BackendType.PROXY if isinstance(model, ModelProxy) else BackendType.SERVICE
        )
        return CommonModel(
            model_obj=model,
            backend_type=backend_type,
            specific_model=self.model_name,
            config=self.legacy_config,
        )


class _AgentCoreToolResource:

    def __init__(
        self,
        client: Any = None,
        tools: Sequence[Any] | None = None,
    ):
        self._client = client
        self._tools = list(tools) if tools is not None else None

    def list_tools(self, config: Config | None = None) -> Sequence[Any]:
        del config
        if self._tools is not None:
            return self._tools
        return self._client.list_tools()

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        config: Config | None = None,
    ) -> Any:
        del config
        if self._tools is None:
            return self._client.call_tool(name, arguments or {})
        tool = next(item for item in self._tools if item.name == name)
        return tool.invoke(arguments or {})


def _chat_response(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    from litellm.types.utils import ModelResponse

    return ModelResponse(**dict(value))


def _chat_stream(values: Iterator[Mapping[str, Any]]) -> Iterator[Any]:
    from litellm.types.utils import ModelResponseStream

    for value in values:
        yield ModelResponseStream(**dict(value))


def _responses_response(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    from litellm.types.llms.openai import ResponsesAPIResponse

    return ResponsesAPIResponse(**dict(value))


def _responses_stream(values: Iterator[Mapping[str, Any]]) -> Iterator[Any]:
    from openai.types.responses import ResponseStreamEvent
    from pydantic import TypeAdapter

    adapter = TypeAdapter(ResponseStreamEvent)
    for value in values:
        yield adapter.validate_python(dict(value))


async def _async_from_sync(values: Iterator[Any]):
    iterator = iter(values)
    while True:
        has_value, value = await asyncio.to_thread(_next_value, iterator)
        if not has_value:
            return
        yield value


def _next_value(iterator: Iterator[Any]) -> tuple[bool, Any]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None
