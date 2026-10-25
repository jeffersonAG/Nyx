"""SQLite de NYX 0.4: propiedad por cuenta y migraciones no destructivas.

Ningun endpoint expone registros sin un user_id autenticado. Las filas heredadas
con owner_id NULL permanecen invisibles hasta que el CLI de primer arranque las
adopta en la cuenta administradora.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def db_path() -> Path:
    return Path(os.environ.get("NYX_DB_PATH", "/data/nyx.db"))


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


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
                id INTEGER PRIMARY KEY CHECK (id=1),
                timezone TEXT NOT NULL DEFAULT 'America/Costa_Rica',
                city TEXT NOT NULL DEFAULT '',
                country TEXT NOT NULL DEFAULT '',
                tone TEXT NOT NULL DEFAULT 'natural'
            );
            INSERT OR IGNORE INTO local_profile(id) VALUES (1);
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                timezone TEXT NOT NULL DEFAULT 'America/Costa_Rica',
                city TEXT NOT NULL DEFAULT '',
                country TEXT NOT NULL DEFAULT '',
                tone TEXT NOT NULL DEFAULT 'natural'
            );
        """)
        msg_columns = {r["name"] for r in db.execute("PRAGMA table_info(messages)")}
        if "session_id" not in msg_columns:
            db.execute("ALTER TABLE messages ADD COLUMN session_id INTEGER REFERENCES sessions(id)")
        if db.execute("SELECT 1 FROM messages WHERE session_id IS NULL LIMIT 1").fetchone():
            cur = db.execute("INSERT INTO sessions(title) VALUES ('Conversaciones anteriores (0.1)')")
            db.execute("UPDATE messages SET session_id=? WHERE session_id IS NULL", (cur.lastrowid,))
        for table in ("sessions", "tasks", "memories"):
            columns = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
            if "owner_id" not in columns:
                db.execute(f"ALTER TABLE {table} ADD COLUMN owner_id INTEGER")
        task_columns = {r["name"] for r in db.execute("PRAGMA table_info(tasks)")}
        if "due_date" not in task_columns:
            db.execute("ALTER TABLE tasks ADD COLUMN due_date TEXT")
        db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id,id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner_id,updated_at)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner_id,id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_id,id)")
        profile_columns = {r["name"] for r in db.execute("PRAGMA table_info(user_profiles)")}
        if "online_enabled" not in profile_columns:
            db.execute("ALTER TABLE user_profiles ADD COLUMN online_enabled INTEGER NOT NULL DEFAULT 1")


def adopt_legacy_records(admin_id: int) -> None:
    """Se ejecuta una unica vez desde el CLI de bootstrap, NUNCA desde una solicitud HTTP."""
    with connect() as db:
        for table in ("sessions", "tasks", "memories"):
            db.execute(f"UPDATE {table} SET owner_id=? WHERE owner_id IS NULL", (admin_id,))
        db.execute("""
            INSERT OR IGNORE INTO user_profiles(user_id,timezone,city,country,tone)
            SELECT ?,timezone,city,country,tone FROM local_profile WHERE id=1
        """, (admin_id,))


def _owned_session(db: sqlite3.Connection, session_id: int, user_id: int) -> bool:
    return db.execute("SELECT 1 FROM sessions WHERE id=? AND owner_id=?",
                      (session_id, user_id)).fetchone() is not None


def create_session(title: str, user_id: int) -> dict:
    title = title.strip()[:90] or "Nueva conversación"
    with connect() as db:
        cur = db.execute("INSERT INTO sessions(title,owner_id) VALUES (?,?)", (title, user_id))
        row = db.execute("SELECT * FROM sessions WHERE id=?", (cur.lastrowid,)).fetchone()
    return dict(row)


def get_session(session_id: int, user_id: int) -> dict | None:
    with connect() as db:
        row = db.execute("SELECT * FROM sessions WHERE id=? AND owner_id=?",
                         (session_id, user_id)).fetchone()
    return dict(row) if row else None


def list_sessions(query: str, user_id: int) -> list[dict]:
    pattern = "%" + query.strip()[:100] + "%"
    with connect() as db:
        rows = db.execute("""
            SELECT s.id,s.title,s.created_at,s.updated_at,
                (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.id) AS count,
                (SELECT m.content FROM messages m WHERE m.session_id=s.id
                 ORDER BY m.id DESC LIMIT 1) AS preview
            FROM sessions s WHERE s.owner_id=? AND
                (s.title LIKE ? OR EXISTS(
                    SELECT 1 FROM messages m WHERE m.session_id=s.id AND m.content LIKE ?
                ))
            ORDER BY s.updated_at DESC,s.id DESC LIMIT 150
        """, (user_id, pattern, pattern)).fetchall()
    return [dict(row) for row in rows]


def rename_session(session_id: int, title: str, user_id: int) -> dict | None:
    with connect() as db:
        cur = db.execute("UPDATE sessions SET title=? WHERE id=? AND owner_id=?",
                         (title, session_id, user_id))
        if not cur.rowcount:
            return None
        row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row)


def delete_session(session_id: int, user_id: int) -> bool:
    with connect() as db:
        if not _owned_session(db, session_id, user_id):
            return False
        db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        return bool(db.execute("DELETE FROM sessions WHERE id=? AND owner_id=?",
                               (session_id, user_id)).rowcount)


def list_messages(session_id: int | None, limit: int, user_id: int) -> list[dict]:
    limit = max(1, min(500, limit))
    with connect() as db:
        if session_id is not None:
            if not _owned_session(db, session_id, user_id):
                return []
            rows = db.execute("""
                SELECT m.id,m.session_id,m.role,m.content,m.created_at
                FROM messages m JOIN sessions s ON s.id=m.session_id
                WHERE s.owner_id=? AND s.id=? ORDER BY m.id DESC LIMIT ?
            """, (user_id, session_id, limit)).fetchall()
        else:
            rows = db.execute("""
                SELECT m.id,m.session_id,m.role,m.content,m.created_at
                FROM messages m JOIN sessions s ON s.id=m.session_id
                WHERE s.owner_id=? ORDER BY m.id DESC LIMIT ?
            """, (user_id, limit)).fetchall()
    return [dict(row) for row in reversed(rows)]


def add_message(role: str, content: str, session_id: int, user_id: int) -> dict:
    if role not in {"user", "assistant"}:
        raise ValueError("Rol no permitido")
    with connect() as db:
        if not _owned_session(db, session_id, user_id):
            raise ValueError("La conversación no existe.")
        cur = db.execute("INSERT INTO messages(role,content,session_id) VALUES (?,?,?)",
                         (role, content, session_id))
        db.execute("UPDATE sessions SET updated_at=datetime('now') WHERE id=?", (session_id,))
        row = db.execute("SELECT id,session_id,role,content,created_at FROM messages WHERE id=?",
                         (cur.lastrowid,)).fetchone()
    return dict(row)


def list_tasks(user_id: int) -> list[dict]:
    with connect() as db:
        rows = db.execute("""SELECT id,title,completed,created_at,due_date
            FROM tasks WHERE owner_id=? ORDER BY id DESC""", (user_id,)).fetchall()
    return [{**dict(row), "completed": bool(row["completed"])} for row in rows]


def add_task(title: str, user_id: int, due_date: str | None = None) -> dict:
    if due_date is not None:
        try:
            due_date = date.fromisoformat(due_date).isoformat()
        except (ValueError, TypeError) as exc:
            raise ValueError("Fecha inválida; utiliza YYYY-MM-DD.") from exc
    with connect() as db:
        cur = db.execute("INSERT INTO tasks(title,due_date,owner_id) VALUES (?,?,?)",
                         (title, due_date, user_id))
        row = db.execute("SELECT id,title,completed,created_at,due_date FROM tasks WHERE id=?",
                         (cur.lastrowid,)).fetchone()
    return {**dict(row), "completed": False}


def update_task(task_id: int, completed: bool, user_id: int) -> dict | None:
    with connect() as db:
        cur = db.execute("UPDATE tasks SET completed=? WHERE id=? AND owner_id=?",
                         (int(completed), task_id, user_id))
        if not cur.rowcount:
            return None
        row = db.execute("SELECT id,title,completed,created_at,due_date FROM tasks WHERE id=?",
                         (task_id,)).fetchone()
    return {**dict(row), "completed": bool(row["completed"])}


def delete_task(task_id: int, user_id: int) -> bool:
    with connect() as db:
        return bool(db.execute("DELETE FROM tasks WHERE id=? AND owner_id=?",
                               (task_id, user_id)).rowcount)


def list_memories(limit: int, user_id: int) -> list[dict]:
    limit = max(1, min(100, limit))
    with connect() as db:
        rows = db.execute("SELECT id,content,created_at FROM memories WHERE owner_id=? "
                          "ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(row) for row in rows]


def add_memory(content: str, user_id: int) -> dict:
    with connect() as db:
        cur = db.execute("INSERT INTO memories(content,owner_id) VALUES (?,?)",
                         (content, user_id))
        row = db.execute("SELECT id,content,created_at FROM memories WHERE id=?",
                         (cur.lastrowid,)).fetchone()
    return dict(row)


def delete_memory(memory_id: int, user_id: int) -> bool:
    with connect() as db:
        return bool(db.execute("DELETE FROM memories WHERE id=? AND owner_id=?",
                               (memory_id, user_id)).rowcount)


def get_profile(user_id: int) -> dict:
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO user_profiles(user_id) VALUES (?)", (user_id,))
        row = db.execute("SELECT timezone,city,country,tone FROM user_profiles WHERE user_id=?",
                         (user_id,)).fetchone()
    return dict(row)


def update_profile(*, user_id: int, timezone: str, city: str,
                   country: str, tone: str) -> dict:
    if not isinstance(timezone, str) or not 1 <= len(timezone) <= 70:
        raise ValueError("Zona horaria inválida.")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("Zona horaria IANA desconocida.") from exc
    if not isinstance(city, str) or len(city) > 90 or not isinstance(country, str) or len(country) > 90:
        raise ValueError("La ciudad y país deben contener hasta 90 caracteres.")
    if tone not in {"natural", "tecnico", "academico"}:
        raise ValueError("Tono no permitido.")
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO user_profiles(user_id) VALUES (?)", (user_id,))
        db.execute("""UPDATE user_profiles SET timezone=?,city=?,country=?,tone=?
            WHERE user_id=?""", (timezone, city.strip(), country.strip(), tone, user_id))
    return get_profile(user_id)


def get_online_enabled(user_id: int) -> bool:
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO user_profiles(user_id) VALUES (?)", (user_id,))
        row = db.execute("SELECT online_enabled FROM user_profiles WHERE user_id=?", (user_id,)).fetchone()
    return bool(row["online_enabled"]) if row else True


def set_online_enabled(user_id: int, enabled: bool) -> bool:
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO user_profiles(user_id) VALUES (?)", (user_id,))
        db.execute("UPDATE user_profiles SET online_enabled=? WHERE user_id=?",
                   (int(enabled), user_id))
    return get_online_enabled(user_id)
