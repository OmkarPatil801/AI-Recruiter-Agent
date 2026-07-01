import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "recruiter.db"
SQL_INIT_PATH = Path(__file__).parent / "db" / "init.sql"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    sql = SQL_INIT_PATH.read_text()
    with get_connection() as conn:
        conn.executescript(sql)
