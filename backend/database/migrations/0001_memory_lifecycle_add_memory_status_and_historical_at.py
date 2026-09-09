"""
Memory Lifecycle V1 - SQLite Schema Migration 0001

目标：

为 memories 表增加 Memory Lifecycle V1 所需的两个字段：

memory_status  VARCHAR(20)  NOT NULL  DEFAULT 'current'
historical_at  DATETIME     NULL

并为 memory_status 建立索引：

ix_memories_memory_status

数据迁移规则：

现有所有 Memory
    → memory_status = 'current'

historical_at
    → NULL

严禁：

1. 根据 Memory content 自动推断哪些旧数据应该 historical
2. 删除或重建 test.db
3. 丢失现有数据

安全性：

1. 执行前自动备份数据库文件
2. 幂等：字段/索引已存在则跳过
3. 执行后校验 schema 与数据

用法：

    python backend/database/migrations/
        0001_memory_lifecycle_add_memory_status_and_historical_at.py
        [db_path]

默认 db_path：

    ./test.db

示例（真实项目）：

    python backend/database/migrations/
        0001_memory_lifecycle_add_memory_status_and_historical_at.py
        F:/AI_Agent_Platform/test.db

注意：

本项目当前没有 Alembic。

本脚本是一次性、显式、可重复执行的
SQLite schema migration 记录。
"""

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


# ==================================================
# Migration 常量
# ==================================================

MEMORY_STATUS_CURRENT = "current"

COLUMN_MEMORY_STATUS = "memory_status"

COLUMN_HISTORICAL_AT = "historical_at"

TABLE_NAME = "memories"

INDEX_NAME = "ix_memories_memory_status"


# ==================================================
# 工具
# ==================================================

def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:

    cursor = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    )

    return cursor.fetchone() is not None


def column_names(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[str]:

    cursor = connection.execute(
        f"PRAGMA table_info({table_name})"
    )

    return [
        row[1]
        for row in cursor.fetchall()
    ]


def index_exists(
    connection: sqlite3.Connection,
    index_name: str,
) -> bool:

    cursor = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index'
          AND name = ?
        """,
        (index_name,),
    )

    return cursor.fetchone() is not None


def count_rows(
    connection: sqlite3.Connection,
) -> int:

    cursor = connection.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME}"
    )

    return int(
        cursor.fetchone()[0]
    )


def backup_database(
    db_path: Path,
) -> Path:

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = db_path.with_name(
        f"{db_path.name}.bak-{timestamp}"
    )

    shutil.copy2(
        db_path,
        backup_path,
    )

    return backup_path


# ==================================================
# Migration
# ==================================================

def migrate(
    db_path: Path,
) -> None:

    if not db_path.exists():

        raise SystemExit(
            f"[MIGRATION] 数据库不存在：{db_path}"
        )

    print("=" * 70)
    print("Memory Lifecycle V1 Migration 0001")
    print("=" * 70)
    print(f"[DB] {db_path}")

    # --------------------------------------------------
    # Before
    # --------------------------------------------------

    connection = sqlite3.connect(
        str(db_path)
    )

    try:

        if not table_exists(
            connection,
            TABLE_NAME,
        ):

            raise SystemExit(
                "[MIGRATION] memories 表不存在，"
                "无法执行 migration"
            )

        columns = column_names(
            connection,
            TABLE_NAME,
        )

        print(
            "[BEFORE] columns: "
            f"{columns}"
        )

        row_count_before = count_rows(
            connection
        )

        print(
            "[BEFORE] memories rows: "
            f"{row_count_before}"
        )

        # ----------------------------------------------
        # 备份
        #
        # 备份必须先于任何 schema 变更。
        #
        # 只在确实需要修改 schema 时备份。
        # ----------------------------------------------

        needs_backup = (
            COLUMN_MEMORY_STATUS not in columns
            or COLUMN_HISTORICAL_AT not in columns
        )

        if needs_backup:

            connection.close()

            backup_path = backup_database(
                db_path
            )

            print(
                "[BACKUP] "
                f"{backup_path}"
            )

            connection = sqlite3.connect(
                str(db_path)
            )

        else:

            print(
                "[BACKUP] 无需修改 schema，跳过备份"
            )

        # ----------------------------------------------
        # 1. memory_status
        # ----------------------------------------------

        if COLUMN_MEMORY_STATUS in columns:

            print(
                "[SKIP] memory_status 已存在"
            )

        else:

            print(
                "[RUN] ALTER TABLE memories "
                "ADD COLUMN memory_status"
            )

            connection.execute(
                """
                ALTER TABLE memories
                ADD COLUMN memory_status
                VARCHAR(20) NOT NULL
                DEFAULT 'current'
                """
            )

        # ----------------------------------------------
        # 2. historical_at
        # ----------------------------------------------

        if COLUMN_HISTORICAL_AT in columns:

            print(
                "[SKIP] historical_at 已存在"
            )

        else:

            print(
                "[RUN] ALTER TABLE memories "
                "ADD COLUMN historical_at"
            )

            connection.execute(
                """
                ALTER TABLE memories
                ADD COLUMN historical_at
                DATETIME
                """
            )

        # ----------------------------------------------
        # 3. 数据迁移
        #
        # 现有所有 Memory：
        #
        # memory_status = 'current'
        # historical_at = NULL
        #
        # 严禁根据 content 推断 historical。
        # ----------------------------------------------

        print(
            "[RUN] 现有 Memory 统一置为 current，"
            "historical_at 置 NULL"
        )

        cursor = connection.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET {COLUMN_MEMORY_STATUS} = ?,
                {COLUMN_HISTORICAL_AT} = NULL
            WHERE {COLUMN_MEMORY_STATUS} IS NULL
               OR {COLUMN_MEMORY_STATUS} = ''
               OR {COLUMN_HISTORICAL_AT} IS NOT NULL
            """,
            (MEMORY_STATUS_CURRENT,),
        )

        print(
            "[RUN] updated rows: "
            f"{cursor.rowcount}"
        )

        # ----------------------------------------------
        # 4. 索引
        # ----------------------------------------------

        if index_exists(
            connection,
            INDEX_NAME,
        ):

            print(
                f"[SKIP] {INDEX_NAME} 已存在"
            )

        else:

            print(
                f"[RUN] CREATE INDEX {INDEX_NAME}"
            )

            connection.execute(
                f"""
                CREATE INDEX {INDEX_NAME}
                ON {TABLE_NAME} ({COLUMN_MEMORY_STATUS})
                """
            )

        connection.commit()

        # ----------------------------------------------
        # 5. 校验
        # ----------------------------------------------

        columns_after = column_names(
            connection,
            TABLE_NAME,
        )

        print(
            "[AFTER] columns: "
            f"{columns_after}"
        )

        for required_column in (
            COLUMN_MEMORY_STATUS,
            COLUMN_HISTORICAL_AT,
        ):

            if required_column not in (
                columns_after
            ):

                raise SystemExit(
                    "[MIGRATION] 校验失败，缺少字段："
                    f"{required_column}"
                )

        if not index_exists(
            connection,
            INDEX_NAME,
        ):

            raise SystemExit(
                "[MIGRATION] 校验失败，缺少索引："
                f"{INDEX_NAME}"
            )

        row_count_after = count_rows(
            connection
        )

        if row_count_after != row_count_before:

            raise SystemExit(
                "[MIGRATION] 校验失败，"
                "行数发生变化："
                f"{row_count_before} -> "
                f"{row_count_after}"
            )

        cursor = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {TABLE_NAME}
            WHERE {COLUMN_MEMORY_STATUS} != ?
               OR {COLUMN_HISTORICAL_AT} IS NOT NULL
            """,
            (MEMORY_STATUS_CURRENT,),
        )

        invalid_rows = int(
            cursor.fetchone()[0]
        )

        if invalid_rows:

            raise SystemExit(
                "[MIGRATION] 校验失败，"
                "存在非 current / "
                "historical_at 非 NULL 的行："
                f"{invalid_rows}"
            )

        print(
            "[AFTER] memories rows: "
            f"{row_count_after}"
        )

        print(
            "[VERIFY] 所有现有 Memory "
            "均为 current 且 historical_at 为 NULL"
        )

    finally:

        connection.close()

    print("=" * 70)
    print("MIGRATION 0001 COMPLETED")
    print("=" * 70)


# ==================================================
# Entry
# ==================================================

def main() -> None:

    if len(sys.argv) > 1:

        db_path = Path(
            sys.argv[1]
        )

    else:

        db_path = Path(
            "test.db"
        )

    migrate(
        db_path.resolve()
    )


if __name__ == "__main__":

    main()
