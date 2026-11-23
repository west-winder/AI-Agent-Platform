"""Test child process that closes stdio during a tool call."""

import os
from pathlib import Path
import sys

# The client launches this script with the existing Server directory as cwd.
sys.path.insert(0, str(Path.cwd()))

from web_page_fetch.server import configure_safe_logging, create_server


class DisconnectingFetcher:
    async def fetch(self, url):
        # End the child before it can return a tool result.
        os._exit(7)


if __name__ == "__main__":
    configure_safe_logging()
    server = create_server(DisconnectingFetcher())
    configure_safe_logging()
    server.run(transport="stdio")
