"""Security Sentinel: eventos de seguridad locales, revisión y correo diferido.

Una alerta es una señal para evaluación humana; no representa una acusación.
El contenido del mensaje se minimiza y se redactan credenciales detectables.
"""
from __future__ import annotations

import asyncio
import os
import re
import smtplib
import ssl
import time
from email.message import EmailMessage

from . import storage

MAX_EXCERPT = 1200
MAX_ATTEMPTS = 10

_SENSITIVE = re.compile(
    r"\b(?:creador|administrador|admin|dueño|dueno|j\.?\s*g\.?\s*r\.?)\b.{0,180}"
    r"\b(?:contrase(?:ña|na)|password|token|clave|credencial(?:es)?|"
    r"cuenta bancaria|datos bancarios|documentos? privados?|"
    r"informaci[oó]n (?:privada|confidencial)|correo personal|"
    r"direcci[oó]n personal|tel[eé]fono personal)\b"
    r"|\b(?:contrase(?:ña|na)|password|token|clave|credencial(?:es)?|"
    r"cuenta bancaria|datos bancarios|documentos? privados?|"
    r"informaci[oó]n (?:privada|confidencial))\b.{0,180}"
    r"\b(?:creador|administrador|admin|dueño|dueno|j\.?\s*g\.?\s*r\.?)\b",
    re.IGNORECASE | re.DOTALL
)
_REDACT = [
    re.compile(r"(?i)\b(password|contrase(?:ña|na)|token|api[_ -]?key|secret|clave)"
               r"(\s*(?:=|:)\s*)([^\s,;]+)"),
    re.compile(r"(?i)\b(password|contrase(?:ña|na)|clave)\s+(es|is)\s+([^\s,;]+)"),
]


def sanitize_message(content: str) -> str:
    result = str(content or "")[:MAX_EXCERPT].replace("\x00", "")
    for pattern in _REDACT:
        result = pattern.sub(lambda m: m.group(1) + " [VALOR OCULTO]", result)
    return result


def sensitive_creator_request(text: str) -> bool:
    """Detecta intentos de obtener datos personales, no preguntas públicas genéricas.

    Se normalizan acentos y mayúsculas antes de aplicar reglas.
    Toda detección requiere una referencia al propietario y a información
    sensible; no se envían notificaciones por menciones aisladas.
    """
    import unicodedata
    normalized = unicodedata.normalize("NFKD", text[:4000])
    normalized = normalized.encode("ascii", "ignore").decode().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    owner = re.search(r"\b(?:creador|administrador|admin|dueno|propietario|jgr|j\.g\.r\.)\b", normalized)
    sensitive = re.search(
        r"\b(?:informacion personal|datos personales|informacion privada|"
        r"informacion confidencial|documentos privados|secretos|"
        r"contrasena|password|credenciales|clave privada|token|"
        r"cuenta bancaria|datos bancarios|correo personal|"
        r"direccion personal|telefono personal|identidad privada)\b", normalized
    )
    intent = re.search(
        r"\b(?:dime|decirme|dame|mu[eé]strame|muestrame|"
        r"revela|revelar|proporciona|proporcionar|comparte|"
        r"compartir|necesito|quiero|cual es|puedes)\b", normalized
    )
    if not owner or not sensitive or not intent:
        return False
    return abs(owner.start() - sensitive.start()) <= 240


def init_events_db() -> None:
    with storage.connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_id INTEGER,
                actor_username TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('medium','high')),
                reason TEXT NOT NULL,
                message_excerpt TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','false_positive','confirmed')),
                mail_state TEXT NOT NULL DEFAULT 'none'
                    CHECK (mail_state IN ('none','queued','sent')),
                mail_attempts INTEGER NOT NULL DEFAULT 0,
                mail_next_retry INTEGER NOT NULL DEFAULT 0,
                mail_last_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                sent_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_security_mail
                ON security_events(mail_state,mail_next_retry);
            CREATE INDEX IF NOT EXISTS idx_security_actor
                ON security_events(actor_username,category,id);
        """)


def record_event(*, actor_username: str, category: str, reason: str,
                 message: str = "", actor_id: int | None = None,
                 severity: str = "medium", immediate: bool = False) -> int:
    if category not in {"restricted_access", "sensitive_request", "login_failures"}:
        raise ValueError("Categoría de alerta no válida")
    if severity not in {"medium", "high"}:
        raise ValueError("Severidad inválida")
    actor_username = str(actor_username)[:32]
    reason = str(reason)[:220]
    msg = sanitize_message(message)
    with storage.connect() as db:
        count = db.execute("""
            SELECT COUNT(*) FROM security_events
            WHERE actor_username=? AND category=?
              AND datetime(created_at) >= datetime('now','-10 minutes')
        """, (actor_username, category)).fetchone()[0]
        should_notify = bool(immediate or severity == "high" or count + 1 == 3)
        cursor = db.execute("""
            INSERT INTO security_events(
                actor_id,actor_username,category,severity,reason,message_excerpt,mail_state
            ) VALUES (?,?,?,?,?,?,?)
        """, (actor_id, actor_username, category, severity, reason, msg,
              "queued" if should_notify else "none"))
        return cursor.lastrowid


def list_events(limit: int = 80) -> list[dict]:
    with storage.connect() as db:
        rows = db.execute("""
            SELECT id,actor_username,category,severity,reason,message_excerpt,
                   status,mail_state,created_at,sent_at,mail_attempts
            FROM security_events ORDER BY id DESC LIMIT ?
        """, (max(1, min(200, limit)),)).fetchall()
    return [dict(row) for row in rows]


def review_event(event_id: int, status: str) -> dict | None:
    if status not in {"pending", "false_positive", "confirmed"}:
        raise ValueError("Clasificación no permitida.")
    with storage.connect() as db:
        changed = db.execute(
            "UPDATE security_events SET status=? WHERE id=?", (status, event_id)
        )
        if not changed.rowcount:
            return None
        row = db.execute("""
            SELECT id,actor_username,category,severity,reason,message_excerpt,
                   status,mail_state,created_at,sent_at,mail_attempts
            FROM security_events WHERE id=?
        """, (event_id,)).fetchone()
    return dict(row)


def smtp_config() -> dict:
    cfg = {
        "host": os.environ.get("NYX_SMTP_HOST", "").strip(),
        "port": os.environ.get("NYX_SMTP_PORT", "587").strip(),
        "user": os.environ.get("NYX_SMTP_USER", "").strip(),
        "password": os.environ.get("NYX_SMTP_PASSWORD", ""),
        "sender": os.environ.get("NYX_SMTP_FROM", "").strip(),
        "recipient": os.environ.get("NYX_ALERT_EMAIL_TO", "").strip(),
        "security": os.environ.get("NYX_SMTP_SECURITY", "starttls").strip().lower(),
    }
    return cfg


def mail_configured() -> bool:
    cfg = smtp_config()
    return bool(cfg["host"] and cfg["sender"] and cfg["recipient"]
                and cfg["security"] in {"starttls", "ssl"}
                and (not cfg["user"] or cfg["password"]))


def mail_status() -> dict:
    cfg = smtp_config()
    recipient = cfg["recipient"]
    if "@" in recipient:
        name, domain = recipient.rsplit("@", 1)
        destination = (name[:1] + "***@" + domain) if name else "***@" + domain
    else:
        destination = ""
    with storage.connect() as db:
        pending = db.execute(
            "SELECT COUNT(*) FROM security_events WHERE mail_state='queued'"
        ).fetchone()[0]
    return {"configured": mail_configured(), "recipient_masked": destination,
            "queued": pending, "requires_network": True}


def _send(to: str, subject: str, body: str) -> None:
    cfg = smtp_config()
    if not mail_configured():
        raise RuntimeError("El servicio SMTP no está configurado.")
    try:
        port = int(cfg["port"])
    except ValueError as exc:
        raise RuntimeError("Puerto SMTP inválido.") from exc
    msg = EmailMessage()
    msg["From"] = cfg["sender"]
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    context = ssl.create_default_context()
    if cfg["security"] == "ssl":
        with smtplib.SMTP_SSL(cfg["host"], port, timeout=15, context=context) as server:
            if cfg["user"]:
                server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            if cfg["user"]:
                server.login(cfg["user"], cfg["password"])
            server.send_message(msg)


def send_test_mail() -> None:
    cfg = smtp_config()
    _send(cfg["recipient"], "[NYX] Prueba de Security Sentinel",
          "El canal de alertas ha superado la prueba de envío.\n"
          "Esta notificación no corresponde a un incidente real.")


def _event_mail_body(row) -> str:
    return (
        f"Security Sentinel — evento SEC-{row['id']:06d}\n\n"
        "Esta alerta requiere revisión humana. No es una acusación.\n\n"
        f"Usuario autenticado/declarado: {row['actor_username']}\n"
        f"Fecha UTC: {row['created_at']}\n"
        f"Severidad: {row['severity']}\n"
        f"Categoría: {row['category']}\n"
        f"Motivo: {row['reason']}\n"
        f"Consulta que activó la alerta (se ocultan credenciales detectables):\n"
        f"{row['message_excerpt'] or '[Sin contenido de conversación]'}\n\n"
        "Revisa el incidente en el panel administrativo de NYX y "
        "clasifícalo como confirmado o falso positivo.\n"
        "No respondas a este correo con contraseñas ni secretos.\n"
    )


def process_queue(max_items: int = 5) -> int:
    if not mail_configured():
        return 0
    now = int(time.time())
    with storage.connect() as db:
        rows = db.execute("""
            SELECT * FROM security_events
            WHERE mail_state='queued' AND mail_next_retry <= ?
            ORDER BY id ASC LIMIT ?
        """, (now, max(1, min(10, max_items)))).fetchall()
    successes = 0
    for row in rows:
        try:
            cfg = smtp_config()
            _send(cfg["recipient"], f"[NYX SECURITY] Revisar evento SEC-{row['id']:06d}",
                  _event_mail_body(row))
            with storage.connect() as db:
                db.execute("""UPDATE security_events
                    SET mail_state='sent',sent_at=datetime('now'),mail_last_error=''
                    WHERE id=? AND mail_state='queued'""", (row["id"],))
            successes += 1
        except (OSError, smtplib.SMTPException, RuntimeError, ValueError):
            attempts = int(row["mail_attempts"]) + 1
            # No almacenar información sensible de errores SMTP en la base de datos.
            retry = now + min(3600, 60 * (2 ** min(attempts, 6)))
            with storage.connect() as db:
                db.execute("""UPDATE security_events
                    SET mail_attempts=?,mail_next_retry=?,mail_last_error='Entrega no disponible'
                    WHERE id=?""", (attempts, retry, row["id"]))
    return successes


async def delivery_loop() -> None:
    while True:
        try:
            await asyncio.to_thread(process_queue)
        except Exception:
            # El fallo del transporte de correo no debe detener el API.
            pass
        await asyncio.sleep(90)
