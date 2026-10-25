"""Almacenamiento SQLite persistente. Solo se guarda lo que el usuario indica."""
from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

def db_path() -> Path:
    return Path(os.environ.get("NYX_DB_PATH", "/data/nyx.db"))

@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    with connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)

def list_messages(limit: int = 50) -> list[dict]:
    with connect() as db:
        rows = db.execute(
            "SELECT id,role,content,created_at FROM messages ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(row) for row in reversed(rows)]

def add_message(role: str, content: str) -> dict:
    if role not in {"user", "assistant"}:
        raise ValueError("Rol no permitido")
    with connect() as db:
        cur = db.execute("INSERT INTO messages(role,content) VALUES (?,?)", (role,content))
        row = db.execute("SELECT id,role,content,created_at FROM messages WHERE id=?",
                         (cur.lastrowid,)).fetchone()
    return dict(row)

def list_tasks() -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT id,title,completed,created_at FROM tasks ORDER BY id DESC").fetchall()
    return [{**dict(row), "completed": bool(row["completed"])} for row in rows]

def add_task(title: str) -> dict:
    with connect() as db:
        cur = db.execute("INSERT INTO tasks(title) VALUES (?)", (title,))
        row = db.execute("SELECT id,title,completed,created_at FROM tasks WHERE id=?",
                         (cur.lastrowid,)).fetchone()
    return {**dict(row), "completed": False}

def update_task(task_id: int, completed: bool) -> dict | None:
    with connect() as db:
        cur = db.execute("UPDATE tasks SET completed=? WHERE id=?",
                         (int(completed), task_id))
        if not cur.rowcount:
            return None
        row = db.execute("SELECT id,title,completed,created_at FROM tasks WHERE id=?",
                         (task_id,)).fetchone()
    return {**dict(row), "completed": bool(row["completed"])}

def delete_task(task_id: int) -> bool:
    with connect() as db:
        return bool(db.execute("DELETE FROM tasks WHERE id=?", (task_id,)).rowcount)

def list_memories(limit: int = 25) -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT id,content,created_at FROM memories ORDER BY id DESC LIMIT ?",
                          (limit,)).fetchall()
    return [dict(row) for row in rows]

def add_memory(content: str) -> dict:
    with connect() as db:
        cur = db.execute("INSERT INTO memories(content) VALUES (?)", (content,))
        row = db.execute("SELECT id,content,created_at FROM memories WHERE id=?",
                         (cur.lastrowid,)).fetchone()
    return dict(row)

def delete_memory(memory_id: int) -> bool:
    with connect() as db:
        return bool(db.execute("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount)
