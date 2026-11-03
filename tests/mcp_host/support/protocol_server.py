"""Test-only paginated/malformed/crashing/stubborn server. Never imported by production."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

mode = sys.argv[1] if len(sys.argv) > 1 else "pages"
server = Server("host-test-server")
lists = 0


@server.list_tools()
async def list_tools(request: types.ListToolsRequest):
    global lists
    lists += 1
    cursor = request.params.cursor if request.params else None
    name = "first" if cursor is None else "last"
    return types.ListToolsResult(
        tools=[types.Tool(name=name, inputSchema={"type": "object"},
                          outputSchema={"type": "object", "required": ["ok"]})],
        nextCursor="page-2" if cursor is None else None,
    )


@server.call_tool()
async def call_tool(name, arguments):
    if mode == "crash":
        os._exit(7)
    if mode == "hang":
        await asyncio.Event().wait()
    if mode == "malformed":
        print("private-malformed-canary", flush=True)
        # The next valid response still traverses the official SDK reader.
    if mode == "bad_output":
        return types.CallToolResult(content=[], structuredContent={"secret-canary": True})
    return types.CallToolResult(content=[], structuredContent={"ok": True, "lists": lists, "remote": name})


async def main():
    if mode == "no_initialize":
        await asyncio.Event().wait()
    if mode == "stubborn":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        Path(sys.argv[2]).write_text(str(child.pid), encoding="ascii")
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
    if mode == "stubborn":
        await asyncio.Event().wait()


asyncio.run(main())
