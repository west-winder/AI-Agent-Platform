"""Manual Phase 3 acceptance: real public HTTPS via the production stdio entry.

Not a pytest test: it deliberately depends on external network availability.
No production component is replaced. Only SDK client process lifecycle is observed.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys
import tempfile

from mcp import ClientSession, StdioServerParameters
from mcp.client import stdio

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("https://example.com/", "https://www.python.org/", "https://www.iana.org/help/example-domains")
CODES = ("INVALID_URL", "TARGET_NOT_ALLOWED", "DNS_ERROR", "TIMEOUT", "NETWORK_ERROR",
         "HTTP_ERROR", "REDIRECT_ERROR", "UNSUPPORTED_CONTENT_TYPE", "UNSUPPORTED_CONTENT_ENCODING",
         "RESPONSE_TOO_LARGE", "INVALID_RESPONSE", "INTERNAL_ERROR")
LAST_REPORT = {}


def error_code(result):
    text = " ".join(getattr(item, "text", "") for item in result.content)
    return next((code for code in CODES if code + ":" in text), "SDK_ERROR")


async def acceptance():
    global LAST_REPORT
    report = {
        "recorded_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "versions": {name: metadata.version(name) for name in ("mcp", "httpx", "httpcore", "pydantic")},
        "server_entry": "python -B -m web_page_fetch",
        "server_components_replaced": False,
        "public_attempts": [], "negative_cases": [],
    }
    LAST_REPORT = report
    report["stage"] = "version_check"
    assert report["versions"] == dict(mcp="1.26.0", httpx="0.28.1", httpcore="1.0.9", pydantic="2.13.4")
    processes, forced, notifications = [], [], []
    create, terminate = stdio._create_platform_compatible_process, stdio._terminate_process_tree
    async def observed_create(*args, **kwargs):
        process = await create(*args, **kwargs)
        processes.append(process)
        return process
    async def observed_terminate(*args, **kwargs):
        forced.append(True)
        await terminate(*args, **kwargs)
    async def on_log(params):
        notifications.append(str(params))
    stdio._create_platform_compatible_process = observed_create
    stdio._terminate_process_tree = observed_terminate
    winner = None
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", dir=ROOT / ".venv") as stderr:
            params = StdioServerParameters(command=sys.executable, args=["-B", "-m", "web_page_fetch"],
                                           cwd=ROOT, env={"PYTHONUTF8": "1"})
            async with asyncio.timeout(150):
                async with stdio.stdio_client(params, errlog=stderr) as (read, write):
                    async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=40),
                                             logging_callback=on_log) as client:
                        report["stage"] = "initialize"
                        init = await client.initialize()
                        report["initialize"] = {"server_name": init.serverInfo.name,
                                                "protocol_version": init.protocolVersion}
                        assert init.serverInfo.name == "Web Page Fetch MCP Server"
                        tools = (await client.list_tools()).tools
                        report["tools"] = [tool.name for tool in tools]
                        assert report["tools"] == ["fetch_web_page"]
                        tool = tools[0]
                        assert tool.inputSchema["required"] == ["url"]
                        assert tool.inputSchema["properties"]["url"]["type"] == "string"
                        report["input_schema_checked"] = True
                        for target in TARGETS:
                            report["stage"] = "public_call"
                            start = asyncio.get_running_loop().time()
                            result = await client.call_tool("fetch_web_page", {"url": target})
                            attempt = {"target": target, "isError": result.isError,
                                       "seconds": round(asyncio.get_running_loop().time() - start, 3)}
                            if result.isError:
                                attempt["error_code"] = error_code(result)
                                report["public_attempts"].append(attempt)
                                continue
                            data = result.structuredContent
                            report["stage"] = "public_result_validation"
                            report["public_result_shape"] = {
                                "fields": sorted(data) if isinstance(data, dict) else None,
                                "title_is_example_domain": isinstance(data, dict) and data.get("title") == "Example Domain",
                            }
                            assert set(data) == {"title", "url", "content", "truncated", "content_type"}
                            assert all(isinstance(data[k], str) for k in ("title", "url", "content", "content_type"))
                            assert isinstance(data["truncated"], bool)
                            assert data["content_type"] == "text/html"
                            assert data["url"].startswith("https://") and "@" not in data["url"]
                            assert data["title"].strip() and data["content"].strip()
                            if target == "https://example.com/":
                                assert data["title"] == "Example Domain"
                                assert "domain" in data["content"].lower()
                            attempt["structuredContent_preview"] = dict(data, content=data["content"][:180])
                            attempt["content_characters"] = len(data["content"])
                            attempt["preview_only"] = len(data["content"]) > 180
                            attempt["production_dns_and_verified_https_completed"] = True
                            report["public_attempts"].append(attempt)
                            winner = target
                            break
                        for label, url, expected in (
                            ("invalid_url", "file:///phase3-canary", "INVALID_URL"),
                            ("localhost", "http://localhost/?phase3-query-canary", "TARGET_NOT_ALLOWED"),
                            ("private_ip", "https://10.23.45.67/", "TARGET_NOT_ALLOWED"),
                        ):
                            report["stage"] = label
                            result = await client.call_tool("fetch_web_page", {"url": url})
                            code = error_code(result)
                            assert result.isError and code == expected
                            assert url not in str(result) and "phase3-query-canary" not in str(result)
                            assert "10.23.45.67" not in str(result)
                            report["negative_cases"].append({"case": label, "isError": True, "error_code": code})
                        assert [tool.name for tool in (await client.list_tools()).tools] == ["fetch_web_page"]
                        report["tools_list_after_failures"] = True
                        if winner:
                            report["stage"] = "success_after_failures"
                            again = await client.call_tool("fetch_web_page", {"url": winner})
                            assert not again.isError and again.structuredContent["content"]
                            report["public_success_after_failures"] = True
                        else:
                            report["public_success_after_failures"] = False
            stderr.seek(0)
            logs = stderr.read()
        report["stage"] = "stderr_validation"
        allowed_logs = {f"fetch_failed code={code}" for code in CODES}
        report["stderr_diagnostics"] = {
            "lines": len(logs.splitlines()),
            "allowed_lines": sum(line in allowed_logs for line in logs.splitlines()),
            "contains_warning": "Warning" in logs,
            "contains_traceback": "Traceback" in logs,
            "contains_url": "https://" in logs or "http://" in logs,
            "warning_categories": re.findall(r"\b([A-Za-z]+Warning):", logs),
            "warning_source_files": [Path(match).name for match in re.findall(r"([^\s]+\.py):[0-9]+:", logs)],
        }
        assert all(line in allowed_logs for line in logs.splitlines())
        report["stage"] = "notification_validation"
        report["mcp_log_notifications"] = len(notifications)
        assert not notifications
        report["stderr"] = {"only_fixed_error_codes": True, "lines": logs.splitlines()}
        report["stage"] = "exit_validation"
        report["server_exit_codes"] = [process.returncode for process in processes]
        report["forced_termination_calls"] = len(forced)
        assert len(processes) == 1 and processes[0].returncode == 0 and not forced
        report["public_fetch_passed"] = winner is not None
        report["protocol_and_negative_cases_passed"] = True
        report["stage"] = "completed"
        return report
    finally:
        stdio._create_platform_compatible_process = create
        stdio._terminate_process_tree = terminate


if __name__ == "__main__":
    try:
        result = asyncio.run(acceptance())
    except BaseException as exc:
        # Never print exception messages/tracebacks from network or protocol failures.
        def exception_types(error):
            children = getattr(error, "exceptions", ())
            return {"type": type(error).__name__, "children": [exception_types(child) for child in children]}
        result = dict(LAST_REPORT, acceptance_harness_failed=True, exception_types=exception_types(exc))
    (ROOT / "phase3-acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("public_fetch_passed") else 1)
