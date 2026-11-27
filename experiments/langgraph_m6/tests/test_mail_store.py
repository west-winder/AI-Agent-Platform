from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from experiments.langgraph_m6.mail_store import IdempotencyConflict, SQLiteMailStore


@pytest.fixture
def store(tmp_path):
    mail = SQLiteMailStore(tmp_path / "mail.sqlite3")
    mail.setup()
    return mail


def test_b3_same_key_replays_saved_result_and_rejects_changed_request(store):
    saved = store.send("email", "one@example.test", "同一封邮件")
    reopened = SQLiteMailStore(store.path)
    assert reopened.send("email", "one@example.test", "同一封邮件") == saved
    for recipient, draft in [("two@example.test", "同一封邮件"), ("one@example.test", "修改后的邮件")]:
        with pytest.raises(IdempotencyConflict, match="不同的收件人或草稿"):
            reopened.send("email", recipient, draft)

    evidence = reopened.evidence("email")
    assert [row["outcome"] for row in evidence["requests"]] == ["sent", "replayed", "conflict", "conflict"]
    assert evidence["deliveries"] == [{
        "idempotency_key": "email", "recipient": "one@example.test", "email_draft": "同一封邮件", **saved,
    }]


def test_b3_concurrent_requests_create_one_delivery(store):
    ready = Barrier(4, timeout=10)

    def send(_: int):
        ready.wait()
        return store.send("same-key", "lab@example.test", "会议议程")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(send, range(4)))

    evidence = store.evidence("same-key")
    assert [row["outcome"] for row in evidence["requests"]] == ["sent", "replayed", "replayed", "replayed"]
    assert len(evidence["deliveries"]) == 1
    assert {result["message_id"] for result in results} == {evidence["deliveries"][0]["message_id"]}
