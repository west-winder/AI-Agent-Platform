import os
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from .state import SendResult


class IdempotencyConflict(ValueError):
    """同一幂等键已用于不同的邮件内容。"""


class SQLiteMailStore:
    """模拟邮件服务，每一条 deliveries 记录代表一次实际模拟发送。"""

    def __init__(self, path: Path):
        self.path = path

    def setup(self) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS deliveries (
                    idempotency_key TEXT PRIMARY KEY NOT NULL,
                    recipient TEXT NOT NULL,
                    email_draft TEXT NOT NULL,
                    message_id TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS send_requests (
                    request_number INTEGER PRIMARY KEY,
                    idempotency_key TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    outcome TEXT NOT NULL CHECK (outcome IN ('sent', 'replayed', 'conflict'))
                );
            """)

    def send(self, key: str, recipient: str, email_draft: str) -> SendResult:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            # 先锁定写事务，保证读取、去重和新增发送记录是一个原子操作。
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT recipient, email_draft, message_id FROM deliveries WHERE idempotency_key = ?", (key,),
            ).fetchone()
            if existing is None:
                message_id = f"mock-mail:{uuid4().hex}"
                connection.execute(
                    "INSERT INTO deliveries (idempotency_key, recipient, email_draft, message_id) VALUES (?, ?, ?, ?)",
                    (key, recipient, email_draft, message_id),
                )
                outcome = "sent"
            else:
                saved_recipient, saved_draft, message_id = existing
                outcome = "replayed" if (saved_recipient, saved_draft) == (recipient, email_draft) else "conflict"

            connection.execute(
                "INSERT INTO send_requests (idempotency_key, pid, outcome) VALUES (?, ?, ?)",
                (key, os.getpid(), outcome),
            )

        # 事务提交后抛出冲突，保留请求记录并保留原有发送结果。
        if outcome == "conflict":
            raise IdempotencyConflict("相同幂等键对应不同的收件人或草稿")
        return {"message_id": message_id}

    def evidence(self, key: str) -> dict:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            requests = connection.execute(
                "SELECT * FROM send_requests WHERE idempotency_key = ? ORDER BY request_number", (key,),
            )
            deliveries = connection.execute("SELECT * FROM deliveries WHERE idempotency_key = ?", (key,))
            return {"requests": [dict(row) for row in requests], "deliveries": [dict(row) for row in deliveries]}
