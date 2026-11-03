"""Manual public-network acceptance. Not collected by pytest; uses no LLM API."""

import asyncio
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mcp import ClientSession
from mcp.client import stdio
from backend.agent_runtime.contracts import LLMMessage, LLMResponse, LLMToolCall
from backend.mcp_host.bootstrap import bootstrap_host
from backend.tools.infrastructure.contracts import ToolContext
from backend.tools.infrastructure.dispatcher import ToolDispatcher

NAME = "mcp_web_page_fetch__fetch_web_page"


class FakeLLM:
    def __init__(self): self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            assert NAME in [tool.name for tool in request.tools]
            return LLMResponse(None, "tool_calls", [LLMToolCall("public-1", NAME,
                '{"url":"https://example.com/"}', {"url": "https://example.com/"})])
        payload = json.loads(next(m.content for m in reversed(request.messages) if m.role == "tool"))
        assert payload["status"] == "success", payload
        data = payload["data"]
        assert data["content"] and data["content_type"] == "text/html"
        assert set(data) == {"title", "url", "content", "truncated", "content_type"}
        return LLMResponse("公网网页读取成功", "stop")


async def main():
    processes, forced, methods = [], [], []
    create, terminate, send = stdio._create_platform_compatible_process, stdio._terminate_process_tree, ClientSession.send_request
    async def observed(*args, **kwargs):
        process = await create(*args, **kwargs); processes.append(process); return process
    async def killed(*args, **kwargs):
        forced.append(True); return await terminate(*args, **kwargs)
    async def traced(self, request, *args, **kwargs):
        methods.append(request.root.method); return await send(self, request, *args, **kwargs)
    stdio._create_platform_compatible_process = observed
    stdio._terminate_process_tree = killed
    ClientSession.send_request = traced
    llm = FakeLLM()
    async with bootstrap_host(llm=llm) as host:
        assert host.statuses["web_page_fetch"] == "registered"
        result = await host.runtime.run([LLMMessage("user", "读取 example.com")], ToolContext())
        assert result.final_answer == "公网网页读取成功"
        data = json.loads(next(m.content for m in result.messages if m.role == "tool"))["data"]
        dispatcher = ToolDispatcher(host.registry)
        failures = []
        for url in ("file:///not-allowed", "http://localhost/?canary=private", "https://10.0.0.1/"):
            failed = await dispatcher.dispatch(NAME, {"url": url}, ToolContext())
            assert failed.error.code == "MCP_TOOL_ERROR"
            failures.append(failed.error.code)
        again = await dispatcher.dispatch(NAME, {"url": "https://example.com/"}, ToolContext())
        assert again.status == "success", again
        assert methods.count("tools/list") == 1
    report = {"mcp": version("mcp"), "jsonschema": version("jsonschema"),
        "title": data["title"], "content_characters": len(data["content"]),
        "fake_llm_calls": llm.calls, "failures": failures, "methods": methods,
        "returncodes": [p.returncode for p in processes], "forced_terminations": len(forced),
        "owner_tasks_finished": all(c.task.done() for c in host.manager.connections.values())}
    assert report["returncodes"] == [0] and not forced
    print(json.dumps(report, ensure_ascii=False, indent=2))


asyncio.run(main())
