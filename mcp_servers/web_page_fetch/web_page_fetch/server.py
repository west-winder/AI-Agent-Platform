"""MCP adapter: one tool; business and transport logic live elsewhere."""

import logging
import sys

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .contracts import FetchWebPageResult
from .errors import ErrorCode, FetchError
from .fetcher import WebPageFetcher

_LOG = logging.getLogger("web_page_fetch.safe")


class _SafeHandler(logging.Handler):
    """Allow only an exact event and enum; never format arbitrary LogRecords."""

    def emit(self, record):
        if (record.name == _LOG.name and record.msg == "fetch_failed"
                and not record.args and isinstance(getattr(record, "code", None), ErrorCode)):
            sys.stderr.write(f"fetch_failed code={record.code.value}\n")
            sys.stderr.flush()


def configure_safe_logging() -> None:
    """Call before and after SDK construction, only in this standalone process."""
    root = logging.getLogger()
    root.handlers.clear()
    for item in root.manager.loggerDict.values():
        if isinstance(item, logging.Logger):
            item.handlers.clear()
            item.propagate = True
    root.addHandler(_SafeHandler())
    root.setLevel(logging.WARNING)
    _LOG.setLevel(logging.WARNING)
    # warnings.showwarning otherwise writes directly to stderr, bypassing handlers.
    logging.captureWarnings(True)


def create_server(fetcher: WebPageFetcher | None = None) -> FastMCP:
    service = fetcher if fetcher is not None else WebPageFetcher()
    mcp = FastMCP("Web Page Fetch MCP Server", log_level="WARNING")

    @mcp.tool(structured_output=True)
    async def fetch_web_page(url: str) -> FetchWebPageResult:
        """Fetch readable text from a public HTTP/HTTPS HTML page without JavaScript."""
        code = None
        try:
            return await service.fetch(url)
        except FetchError as exc:
            code = exc.code
        except Exception:
            code = ErrorCode.INTERNAL_ERROR
        # Raise OUTSIDE the except block: raw exceptions are not the ToolError context.
        _LOG.warning("fetch_failed", extra={"code": code})
        raise ToolError(str(FetchError(code))) from None

    return mcp
