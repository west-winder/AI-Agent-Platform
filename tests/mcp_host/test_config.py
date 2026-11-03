import json

import pytest
from pydantic import ValidationError

from backend.mcp_host.config import ServerConfig, load_config


def write(tmp_path, value):
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_paths_disabled_and_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_CANARY", "secret")
    entries = load_config(write(tmp_path, {"servers": [
        {"server_id": "one", "command": "child/python.exe", "cwd": "child", "env": {"SAFE": "yes"}},
        {"server_id": "two", "enabled": False, "command": "missing"},
    ]}), root=tmp_path)
    assert entries[0].config.command == str(tmp_path / "child/python.exe")
    assert entries[0].config.cwd == str(tmp_path / "child")
    assert entries[0].config.env == {"SAFE": "yes"}
    assert not entries[1].enabled


@pytest.mark.parametrize("value", [[], {}, {"servers": {}}, {"servers": [None]},
    {"servers": [{"server_id": "bad.id"}]},
    {"servers": [{"server_id": "x"}, {"server_id": "x"}]}])
def test_global_invalid(tmp_path, value):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, value))


def test_local_invalid_is_isolated(tmp_path):
    entries = load_config(write(tmp_path, {"servers": [
        {"server_id": "x", "command": "python", "call_timeout_seconds": -1},
        {"server_id": "y", "command": "python"}]}))
    assert entries[0].config is None and entries[1].config is not None


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "5", 301])
def test_budget(value):
    with pytest.raises(ValidationError):
        ServerConfig(server_id="x", command="python", startup_timeout_seconds=value)


def test_duplicate_json_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"servers": [], "servers": []}')
    with pytest.raises(ValueError):
        load_config(path)
