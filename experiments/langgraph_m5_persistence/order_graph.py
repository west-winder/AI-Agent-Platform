from collections.abc import Callable
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer
from psycopg import Connection, sql


class OrderInput(TypedDict):
    run_id: str
    item: str
    quantity: int


class OrderUpdate(TypedDict, total=False):
    idempotency_key: str
    order_id: int | None
    save_status: Literal["pending", "saved"]


class OrderState(OrderInput, OrderUpdate):
    """GenerateOrder 产生并保存业务键，SaveOrder 使用该键更新结果。"""


class OrderStore:
    def __init__(self, connection: Connection, *, idempotent: bool):
        self.connection = connection
        self.idempotent = idempotent
        self.table = sql.Identifier(
            "m5_orders_idempotent" if idempotent else "m5_orders_unprotected"
        )

    def setup(self) -> None:
        unique = sql.SQL(" UNIQUE" if self.idempotent else "")
        self.connection.execute(
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    idempotency_key TEXT NOT NULL{},
                    item TEXT NOT NULL,
                    quantity INTEGER NOT NULL
                )
            """).format(self.table, unique)
        )

    def save(self, key: str, item: str, quantity: int) -> int:
        insert = sql.SQL("INSERT INTO {} (idempotency_key, item, quantity) VALUES (%s, %s, %s)")
        statement = insert.format(self.table)
        if self.idempotent:
            # 由唯一约束串行化相同键的写入，仅在业务参数一致时返回已有 ID。
            statement += sql.SQL("""
                ON CONFLICT (idempotency_key) DO UPDATE
                    SET idempotency_key = EXCLUDED.idempotency_key
                    WHERE {table}.item = EXCLUDED.item AND {table}.quantity = EXCLUDED.quantity
            """).format(table=self.table)
        with self.connection.transaction():
            order = self.connection.execute(
                statement + sql.SQL(" RETURNING id"), (key, item, quantity)
            ).fetchone()
            if order is None:
                raise ValueError("同一幂等键对应不同订单参数")
        return order["id"]

    def find(self, key: str) -> list[dict]:
        return self.connection.execute(
            sql.SQL("SELECT id, item, quantity FROM {} WHERE idempotency_key = %s ORDER BY id")
            .format(self.table), (key,),
        ).fetchall()

    def delete(self, key: str) -> None:
        self.connection.execute(
            sql.SQL("DELETE FROM {} WHERE idempotency_key = %s").format(self.table), (key,),
        )


def order_key(run_id: str) -> str:
    return f"m5-order:{run_id}"


def generate_order(state: OrderState) -> OrderUpdate:
    return {
        "idempotency_key": order_key(state["run_id"]),
        "order_id": None,
        "save_status": "pending",
    }


def build_order_graph(
    store: OrderStore,
    checkpointer: Checkpointer,
    *,
    on_attempt: Callable[[str], None] | None = None,
    after_commit: Callable[[str, int], None] | None = None,
) -> CompiledStateGraph[OrderState, None, OrderInput, OrderState]:
    def save_order(state: OrderState) -> OrderUpdate:
        key = state["idempotency_key"]
        if on_attempt is not None:
            on_attempt(key)
        order_id = store.save(key, state["item"], state["quantity"])
        # store.save 已提交订单事务。故障注入发生在返回 LangGraph 更新之前。
        if after_commit is not None:
            after_commit(key, order_id)
        return {"order_id": order_id, "save_status": "saved"}

    graph = StateGraph(OrderState, input_schema=OrderInput)
    graph.add_node("GenerateOrder", generate_order)
    graph.add_node("SaveOrder", save_order)
    graph.add_edge(START, "GenerateOrder")
    graph.add_edge("GenerateOrder", "SaveOrder")
    graph.add_edge("SaveOrder", END)
    return graph.compile(checkpointer=checkpointer)
