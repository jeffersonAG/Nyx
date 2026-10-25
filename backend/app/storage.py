"""Persistencia SQLite y migracion no destructiva de NYX 0.1 -> 0.2."""
from __future__ import annotations

import os
import sqlite3
from datetime import date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
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
    conn.execute("PRAGMA foreign_keys=ON")
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
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
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
            CREATE TABLE IF NOT EXISTS local_profile (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                timezone TEXT NOT NULL DEFAULT 'America/Costa_Rica',
                city TEXT NOT NULL DEFAULT '',
                country TEXT NOT NULL DEFAULT '',
                tone TEXT NOT NULL DEFAULT 'natural'
            );
            INSERT OR IGNORE INTO local_profile(id) VALUES (1);
        """)
        columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)")}
        if "session_id" not in columns:
            db.execute("ALTER TABLE messages ADD COLUMN session_id INTEGER REFERENCES sessions(id)")
        # Los mensajes de 0.1 sin session se agrupan en una bitacora de legado.
        legacy = db.execute("SELECT COUNT(*) FROM messages WHERE session_id IS NULL").fetchone()[0]
        if legacy:
            cur = db.execute("INSERT INTO sessions(title) VALUES (?)", ("Conversaciones anteriores (0.1)",))
            db.execute("UPDATE messages SET session_id=? WHERE session_id IS NULL", (cur.lastrowid,))
        db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")
        task_columns = {row["name"] for row in db.execute("PRAGMA table_info(tasks)")}
        if "due_date" not in task_columns:
            db.execute("ALTER TABLE tasks ADD COLUMN due_date TEXT")


def create_session(title: str = "Nueva conversación") -> dict:
    title = title.strip()[:90] or "Nueva conversación"
    with connect() as db:
        cur = db.execute("INSERT INTO sessions(title) VALUES (?)", (title,))
        row = db.execute("SELECT * FROM sessions WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_session(session_id: int) -> dict | None:
    with connect() as db:
        row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(query: str = "") -> list[dict]:
    pattern = "%" + query.strip()[:100] + "%"
    with connect() as db:
        rows = db.execute("""
            SELECT s.id, s.title, s.created_at, s.updated_at,
                (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS count,
                (SELECT m.content FROM messages m
                 WHERE m.session_id=s.id ORDER BY m.id DESC LIMIT 1) AS preview
            FROM sessions s
            WHERE s.title LIKE ? OR EXISTS(
                SELECT 1 FROM messages m WHERE m.session_id=s.id AND m.content LIKE ?
            )
            ORDER BY s.updated_at DESC, s.id DESC LIMIT 150
        """, (pattern, pattern)).fetchall()
    return [dict(row) for row in rows]


def rename_session(session_id: int, title: str) -> dict | None:
    with connect() as db:
        cur = db.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
        if not cur.rowcount:
            return None
        row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row)


def delete_session(session_id: int) -> bool:
    with connect() as db:
        db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        return bool(db.execute("DELETE FROM sessions WHERE id=?", (session_id,)).rowcount)


def list_messages(session_id: int | None = None, limit: int = 200) -> list[dict]:
    with connect() as db:
        if session_id is None:
            rows = db.execute(
                "SELECT id,session_id,role,content,created_at FROM messages "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT id,session_id,role,content,created_at FROM messages "
                "WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, limit)
            ).fetchall()
    return [dict(row) for row in reversed(rows)]


def add_message(role: str, content: str, session_id: int) -> dict:
    if role not in {"user", "assistant"}:
        raise ValueError("Rol no permitido")
    with connect() as db:
        if not db.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone():
            raise ValueError("Sesión inexistente")
        cur = db.execute(
            "INSERT INTO messages(role,content,session_id) VALUES (?,?,?)",
            (role, content, session_id)
        )
        db.execute(
            "UPDATE sessions SET updated_at=datetime('now') WHERE id=?", (session_id,)
        )
        row = db.execute(
            "SELECT id,session_id,role,content,created_at FROM messages WHERE id=?",
            (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def list_tasks() -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT id,title,completed,created_at,due_date FROM tasks ORDER BY id DESC").fetchall()
    return [{**dict(row), "completed": bool(row["completed"])} for row in rows]


def add_task(title: str, due_date: str | None = None) -> dict:
    if due_date is not None:
        try:
            due_date = date.fromisoformat(due_date).isoformat()
        except (ValueError, TypeError) as exc:
            raise ValueError("Fecha de tarea inválida; utiliza YYYY-MM-DD.") from exc
    with connect() as db:
        cur = db.execute("INSERT INTO tasks(title,due_date) VALUES (?,?)", (title, due_date))
        row = db.execute(
            "SELECT id,title,completed,created_at,due_date FROM tasks WHERE id=?", (cur.lastrowid,)
        ).fetchone()
    return {**dict(row), "completed": False}


def update_task(task_id: int, completed: bool) -> dict | None:
    with connect() as db:
        cur = db.execute("UPDATE tasks SET completed=? WHERE id=?", (int(completed), task_id))
        if not cur.rowcount:
            return None
        row = db.execute(
            "SELECT id,title,completed,created_at,due_date FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
    return {**dict(row), "completed": bool(row["completed"])}


def delete_task(task_id: int) -> bool:
    with connect() as db:
        return bool(db.execute("DELETE FROM tasks WHERE id=?", (task_id,)).rowcount)


def list_memories(limit: int = 100) -> list[dict]:
    with connect() as db:
        rows = db.execute(
            "SELECT id,content,created_at FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def add_memory(content: str) -> dict:
    with connect() as db:
        cur = db.execute("INSERT INTO memories(content) VALUES (?)", (content,))
        row = db.execute(
            "SELECT id,content,created_at FROM memories WHERE id=?", (cur.lastrowid,)
        ).fetchone()
    return dict(row)


def delete_memory(memory_id: int) -> bool:
    with connect() as db:
        return bool(db.execute("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount)


def get_profile() -> dict:
    with connect() as db:
        row = db.execute(
            "SELECT timezone,city,country,tone FROM local_profile WHERE id=1"
        ).fetchone()
    if row is None:
        raise RuntimeError("La base de datos no ha sido inicializada.")
    return dict(row)


def update_profile(*, timezone: str, city: str, country: str, tone: str) -> dict:
    if not isinstance(timezone, str) or len(timezone) > 70:
        raise ValueError("Zona horaria inválida.")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("No se encontró esa zona horaria IANA.") from exc
    if not isinstance(city, str) or len(city) > 90 or not isinstance(country, str) or len(country) > 90:
        raise ValueError("La ciudad y el país deben contener como máximo 90 caracteres.")
    if tone not in {"natural", "tecnico", "academico"}:
        raise ValueError("Tono conversacional no permitido.")
    with connect() as db:
        db.execute(
            "UPDATE local_profile SET timezone=?,city=?,country=?,tone=? WHERE id=1",
            (timezone, city.strip(), country.strip(), tone)
        )
    return get_profile()
