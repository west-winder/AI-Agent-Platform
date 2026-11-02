"""Deterministic real stdio server; no network and no SSRF switches."""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web_page_fetch.contracts import FetchWebPageResult
from web_page_fetch.errors import ErrorCode, FetchError
from web_page_fetch.server import configure_safe_logging, create_server

CANARY = "raw_secret_canary_10.0.0.1"


class FakeFetcher:
    async def fetch(self, url):
        # Exercise third-party logging, traceback and both exception chain forms.
        logging.getLogger("mcp.server.lowlevel.server").debug("Received message: %s", url)
        logging.getLogger("httpx").info("GET %s", url)
        try:
            raise OSError(CANARY)
        except OSError:
            logging.getLogger("httpcore.connection").exception("raw %s", CANARY)
        if url.endswith("/failure"):
            try:
                raise OSError(CANARY)
            except OSError as exc:
                raise FetchError(ErrorCode.NETWORK_ERROR) from exc
        if url.endswith("/unknown"):
            raise RuntimeError(CANARY)
        return FetchWebPageResult(title="Fixture 标题", url="https://example.com/", content="Fixture 正文",
                                  truncated=False, content_type="text/html")


if __name__ == "__main__":
    configure_safe_logging()
    server = create_server(FakeFetcher())
    configure_safe_logging()
    # Even DEBUG-enabled third-party loggers must not bypass the safe handler.
    for name in ("mcp.server.lowlevel.server", "httpx", "httpcore.connection"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    server.run(transport="stdio")
