import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row


LAB_DB = "agent_platform_m5_lab"
LOOPBACK = {"localhost": "127.0.0.1", "127.0.0.1": "127.0.0.1", "::1": "::1"}


@contextmanager
def connect_lab() -> Iterator[psycopg.Connection]:
    conninfo = os.environ.get("M5_POSTGRES_URI")
    if not conninfo:
        raise ValueError("请设置 M5_POSTGRES_URI")
    try:
        params = conninfo_to_dict(conninfo)
    except psycopg.ProgrammingError:
        raise ValueError("M5_POSTGRES_URI 格式无效") from None

    host = LOOPBACK.get(params.get("host", ""))
    if (
        params.get("dbname") != LAB_DB
        or host is None
        or params.get("hostaddr", host) != host
        or "service" in params
    ):
        raise ValueError(f"M5 仅连接本机 PostgreSQL 实验库 {LAB_DB}")

    # 指定 hostaddr，避免环境中的 PGHOSTADDR 改写连接目标。
    try:
        connection = psycopg.connect(
            conninfo, hostaddr=host, connect_timeout=5,
            autocommit=True, prepare_threshold=0, row_factory=dict_row,
        )
    except psycopg.OperationalError:
        raise RuntimeError("无法连接 M5 本机实验库，请核对服务和连接配置") from None
    with connection:
        identity = connection.execute("SELECT current_database() AS db").fetchone()
        if identity != {"db": LAB_DB}:
            raise RuntimeError("实际连接的数据库与 M5 实验库不一致")
        yield connection


@contextmanager
def open_postgres_saver() -> Iterator[PostgresSaver]:
    # setup 由 Demo 或测试入口执行，恢复进程直接打开已建好的存储。
    with connect_lab() as connection:
        yield PostgresSaver(connection)
