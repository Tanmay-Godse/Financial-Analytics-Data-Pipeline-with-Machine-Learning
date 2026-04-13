from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


class DatabaseManager:
    """Thin SQLite wrapper for schema creation and analytics tables."""

    def __init__(self, database_path: Path, schema_path: Path) -> None:
        self.database_path = database_path
        self.schema_path = schema_path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_schema(self) -> None:
        schema_sql = self.schema_path.read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.executescript(schema_sql)

    def replace_table(self, table_name: str, frame: pd.DataFrame) -> None:
        with self.connect() as connection:
            connection.execute(f"DELETE FROM {table_name}")
            frame.to_sql(table_name, connection, if_exists="append", index=False)

    def append_record(self, table_name: str, record: dict[str, Any]) -> None:
        columns = ", ".join(record.keys())
        placeholders = ", ".join(["?"] * len(record))
        values = list(record.values())
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
                values,
            )

    def fetch_dataframe(self, query: str) -> pd.DataFrame:
        with self.connect() as connection:
            return pd.read_sql_query(query, connection)
