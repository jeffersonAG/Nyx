"""NYX 0.4: autenticación local con Argon2id, sesiones opacas y privilegios temporales.

Nunca se aceptan credenciales de administrador desde el repositorio o archivos
públicos. La primera cuenta se configura exclusivamente por CLI local.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from dataclasses import dataclass

from argon2 import PasswordHasher, exceptions as argon_errors

from . import storage

HASHER = PasswordHasher(time_cost=3, memory_cost=32768, parallelism=2)
SESSION_SECONDS = 8 * 60 * 60
ELEVATION_SECONDS = 5 * 60
FAILURE_WINDOW = 15 * 60
MAX_FAILURES = 5
COOKIE_NAME = "nyx_session"
USER_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


@dataclass(frozen=True)
class Principal:
    id: int
    username: str
    role: str
    must_change: bool
    csrf: str
    token_hash: str
    elevated: bool


def init_auth_db() -> None:
    storage.init_db()
    with storage.connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','standard')),
                must_change INTEGER NOT NULL DEFAULT 1,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS login_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                csrf_token TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                elevated_until INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON login_sessions(user_id);
            CREATE TABLE IF NOT EXISTS login_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                login_name TEXT NOT NULL COLLATE NOCASE,
                occurred_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_login_failures ON login_failures(login_name,occurred_at);
        """)


def validate_username(username: str) -> str:
    if not isinstance(username, str) or not USER_NAME_RE.fullmatch(username):
        raise ValueError("Usuario inválido: utiliza 3 a 32 letras, números, puntos, guiones o _.")
    return username


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def account_count() -> int:
    with storage.connect() as db:
        return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def bootstrap_admin(username: str, initial_password: str) -> None:
    """Admite clave provisional débil SOLO en setup local; exige cambiarla al ingresar."""
    username = validate_username(username)
    if not isinstance(initial_password, str) or not 6 <= len(initial_password) <= 256:
        raise ValueError("La contraseña inicial debe tener entre 6 y 256 caracteres.")
    init_auth_db()
    hashed = HASHER.hash(initial_password)
    with storage.connect() as db:
        if db.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            raise ValueError("Ya existe una cuenta. No se permiten inicializaciones repetidas.")
        cur = db.execute(
            "INSERT INTO users(username,password_hash,role,must_change) VALUES (?,?,'admin',1)",
            (username, hashed)
        )
        admin_id = cur.lastrowid
    storage.adopt_legacy_records(admin_id)


def create_standard(username: str, password: str) -> dict:
    username = validate_username(username)
    check_new_password(password)
    with storage.connect() as db:
        if db.execute("SELECT 1 FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone():
            raise ValueError("Ya existe una cuenta con ese nombre.")
        cur = db.execute(
            "INSERT INTO users(username,password_hash,role,must_change) VALUES (?,?,'standard',1)",
            (username, HASHER.hash(password))
        )
        return {"id": cur.lastrowid, "username": username, "role": "standard",
                "must_change": True}


def list_users() -> list[dict]:
    with storage.connect() as db:
        rows = db.execute("SELECT id,username,role,must_change,enabled,created_at "
                          "FROM users ORDER BY id ASC").fetchall()
    return [{**dict(row), "must_change": bool(row["must_change"]),
             "enabled": bool(row["enabled"])} for row in rows]


def check_new_password(password: str) -> None:
    if not isinstance(password, str) or not 12 <= len(password) <= 256:
        raise ValueError("La contraseña definitiva debe tener entre 12 y 256 caracteres.")


def _verify(hash_value: str, candidate: str) -> bool:
    try:
        return bool(HASHER.verify(hash_value, candidate))
    except (argon_errors.VerificationError, argon_errors.InvalidHashError, TypeError):
        return False


def is_limited(username: str, now: int | None = None) -> bool:
    now = now if now is not None else int(time.time())
    with storage.connect() as db:
        total = db.execute(
            "SELECT COUNT(*) FROM login_failures WHERE login_name=? COLLATE NOCASE "
            "AND occurred_at >= ?", (username, now - FAILURE_WINDOW)
        ).fetchone()[0]
    return total >= MAX_FAILURES


def register_failure(username: str) -> bool:
    """Registra un intento fallido; devuelve True al alcanzar el umbral por primera vez."""
    username = username[:32]
    now = int(time.time())
    with storage.connect() as db:
        db.execute("DELETE FROM login_failures WHERE occurred_at < ?", (now - FAILURE_WINDOW,))
        total = db.execute(
            "SELECT COUNT(*) FROM login_failures WHERE login_name=? COLLATE NOCASE "
            "AND occurred_at >= ?", (username, now - FAILURE_WINDOW)
        ).fetchone()[0]
        db.execute("INSERT INTO login_failures(login_name,occurred_at) VALUES (?,?)", (username, now))
    return total + 1 == MAX_FAILURES


def authenticate(username: str, password: str) -> dict | None:
    if not isinstance(username, str) or not isinstance(password, str) or len(username) > 32 or len(password) > 256:
        return None
    if is_limited(username):
        return None
    with storage.connect() as db:
        row = db.execute("SELECT id,username,password_hash,role,must_change,enabled "
                         "FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
    # Evita diferencias grandes de tiempo cuando la cuenta no existe.
    dummy = "$argon2id$v=19$m=32768,t=3,p=2$EAEBAQEBAQEBAQEBAQEBAQ$EKMBUFVF06qlYSBAEQ8oC+NqHW5oIWMI+dPIjO13cR8"
    valid = _verify(row["password_hash"] if row else dummy, password)
    if not row or not row["enabled"] or not valid:
        register_failure(username)
        return None
    with storage.connect() as db:
        db.execute("DELETE FROM login_failures WHERE login_name=? COLLATE NOCASE", (username,))
    return {"id": row["id"], "username": row["username"], "role": row["role"],
            "must_change": bool(row["must_change"])}


def create_session(user_id: int) -> tuple[str, Principal]:
    token = secrets.token_urlsafe(40)
    csrf = secrets.token_urlsafe(32)
    now = int(time.time())
    token_hash = _hash_token(token)
    with storage.connect() as db:
        db.execute("DELETE FROM login_sessions WHERE expires_at < ?", (now,))
        db.execute("INSERT INTO login_sessions(token_hash,user_id,csrf_token,expires_at) "
                   "VALUES (?,?,?,?)", (token_hash, user_id, csrf, now + SESSION_SECONDS))
    principal = principal_for_token(token)
    if principal is None:
        raise RuntimeError("No se pudo crear la sesión.")
    return token, principal


def principal_for_token(token: str | None) -> Principal | None:
    if not token or not isinstance(token, str) or len(token) > 160:
        return None
    with storage.connect() as db:
        row = db.execute("""
            SELECT s.token_hash,s.csrf_token,s.expires_at,s.elevated_until,
                   u.id,u.username,u.role,u.must_change,u.enabled
            FROM login_sessions s JOIN users u ON u.id=s.user_id
            WHERE s.token_hash=?
        """, (_hash_token(token),)).fetchone()
    now = int(time.time())
    if row is None or not row["enabled"] or row["expires_at"] < now:
        return None
    return Principal(row["id"], row["username"], row["role"], bool(row["must_change"]),
                     row["csrf_token"], row["token_hash"],
                     row["role"] == "admin" and row["elevated_until"] > now)


def clear_session(token: str | None) -> None:
    if token:
        with storage.connect() as db:
            db.execute("DELETE FROM login_sessions WHERE token_hash=?", (_hash_token(token),))


def elevate(principal: Principal, password: str) -> bool:
    if principal.role != "admin" or principal.must_change:
        return False
    with storage.connect() as db:
        user = db.execute("SELECT password_hash FROM users WHERE id=?", (principal.id,)).fetchone()
    if not user or not _verify(user["password_hash"], password):
        return False
    with storage.connect() as db:
        db.execute("UPDATE login_sessions SET elevated_until=? WHERE token_hash=?",
                   (int(time.time()) + ELEVATION_SECONDS, principal.token_hash))
    return True


def revoke_elevation(principal: Principal) -> None:
    with storage.connect() as db:
        db.execute("UPDATE login_sessions SET elevated_until=0 WHERE token_hash=?",
                   (principal.token_hash,))


def change_password(principal: Principal, old: str, new: str) -> None:
    check_new_password(new)
    with storage.connect() as db:
        user = db.execute("SELECT password_hash FROM users WHERE id=?", (principal.id,)).fetchone()
        if not user or not _verify(user["password_hash"], old):
            raise ValueError("La contraseña actual no es correcta.")
        if hmac.compare_digest(old, new):
            raise ValueError("La nueva contraseña debe ser diferente.")
        db.execute("UPDATE users SET password_hash=?,must_change=0 WHERE id=?",
                   (HASHER.hash(new), principal.id))
        db.execute("DELETE FROM login_sessions WHERE user_id=?", (principal.id,))
