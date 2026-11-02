"""Python warnings must not bypass the standalone process's stderr policy."""

from pathlib import Path
import subprocess
import sys


def test_startup_and_runtime_warnings_use_safe_log_sink():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", "-c", """
import logging
import warnings
from web_page_fetch.server import configure_safe_logging, create_server
from web_page_fetch.errors import ErrorCode
configure_safe_logging()
warnings.simplefilter('always')
create_server()
configure_safe_logging()
warnings.warn('raw_warning_canary_10.23.45.67_https://private.invalid/?token=secret', RuntimeWarning)
logging.getLogger('web_page_fetch.safe').warning('fetch_failed', extra={'code': ErrorCode.INVALID_URL})
"""], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == "fetch_failed code=INVALID_URL\n"
