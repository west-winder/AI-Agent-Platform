"""Launch with the dedicated interpreter: python -m web_page_fetch."""

from .server import configure_safe_logging, create_server


def main() -> None:
    configure_safe_logging()
    server = create_server()
    configure_safe_logging()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
