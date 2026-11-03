"""Suppress SDK-owned free text without changing the application's root setup."""

import logging


class _SDKFilter(logging.Filter):
    def filter(self, record):
        path = record.pathname.replace("\\", "/")
        return "/site-packages/mcp/" not in path and record.name not in {
            "client", "client.stdio.win32", "mcp.client.stdio",
        }


class SDKLogGuard:
    def __init__(self):
        self.filter = _SDKFilter()
        self.targets = []

    def __enter__(self):
        root = logging.getLogger()
        self.targets = [root, *root.handlers, *[
            logging.getLogger(name) for name in ("client", "client.stdio.win32", "mcp.client.stdio")
        ]]
        for target in self.targets:
            target.addFilter(self.filter)
        return self

    def __exit__(self, *args):
        for target in self.targets:
            target.removeFilter(self.filter)
