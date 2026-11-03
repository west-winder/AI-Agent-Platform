"""Static, trusted process configuration. Paths are relative to the repository."""

import json
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "backend/config/mcp_servers.json"


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    server_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    enabled: bool = True
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    cwd: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    startup_timeout_seconds: float = Field(default=15.0, gt=0, le=300, allow_inf_nan=False)
    call_timeout_seconds: float = Field(default=40.0, gt=0, le=300, allow_inf_nan=False)

    def resolved(self, root: Path = REPO_ROOT) -> "ServerConfig":
        # Executables are explicit paths, not PATH/shell expressions.
        command, cwd = Path(self.command), Path(self.cwd)
        return self.model_copy(update={
            "command": str((root / command).resolve()),
            "cwd": str((root / cwd).resolve()),
        })


class ConfigEntry(BaseModel):
    """An identifiable invalid server can fail independently of its neighbours."""

    server_id: str
    config: ServerConfig | None = None
    enabled: bool = True


def load_config(path: Path = DEFAULT_CONFIG, *, root: Path = REPO_ROOT) -> list[ConfigEntry]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate configuration key")
            result[key] = value
        return result

    raw = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(raw, dict) or set(raw) != {"servers"} or not isinstance(raw["servers"], list):
        raise ValueError("Invalid MCP configuration document")
    entries, seen = [], set()
    for item in raw["servers"]:
        if not isinstance(item, dict) or not isinstance(item.get("server_id"), str):
            raise ValueError("MCP server identity is required")
        identity = item["server_id"]
        # Invalid identities and duplicate IDs are globally ambiguous.
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity) or identity in seen:
            raise ValueError("Invalid or duplicate MCP server identity")
        seen.add(identity)
        try:
            config = ServerConfig.model_validate(item).resolved(root)
        except ValueError:
            entries.append(ConfigEntry(server_id=identity, enabled=item.get("enabled") is not False))
        else:
            entries.append(ConfigEntry(server_id=identity, config=config, enabled=config.enabled))
    return entries
