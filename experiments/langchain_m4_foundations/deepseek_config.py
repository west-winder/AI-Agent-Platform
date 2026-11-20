import os
from pathlib import Path

from dotenv import dotenv_values


REPO_ENV = Path(__file__).resolve().parents[2] / ".env"
CONFIG_NAMES = ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEFAULT_MODEL")


def read_deepseek_config() -> dict[str, str]:
    # dotenv_values 只返回映射。进程中已设置的值优先，包括显式设置的空值。
    file_config = dotenv_values(REPO_ENV, interpolate=False)
    return {
        name: (os.environ.get(name, file_config.get(name)) or "").strip()
        for name in CONFIG_NAMES
    }
