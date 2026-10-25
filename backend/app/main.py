"""NYX 0.4: API privada, control de acceso por cuenta y Security Sentinel."""
from __future__ import annotations

import asyncio
import hmac
import io
import json
import os
import re
import unicodedata
import wave
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import cognition, security, sentinel, storage, network, brain

app = FastAPI(title="NYX 0.5.0", docs_url=None, redoc_url=None, openapi_url=None)
OLLAMA_URL = os.environ.get("NYX_OLLAMA_URL", "http://ollama:11434").rstrip("/")
MODEL = os.environ.get("NYX_MODEL", "nyx:0.4")
VOICE_PATH = Path(os.environ.get("NYX_VOICE_PATH", "/voices"))
TTS_MODEL = VOICE_PATH / "es_ES-davefx-medium.onnx"
STT_MODEL = VOICE_PATH / "vosk-model-small-es-0.42"


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=256)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class PasswordConfirmation(BaseModel):
    password: str = Field(min_length=1, max_length=256)


class UserCreate(Credentials):
    pass


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    session_id: int | None = Field(default=None, ge=1)


class SessionRequest(BaseModel):
    title: str = Field(default="Nueva conversación", min_length=1, max_length=90)


class TaskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    due_date: str | None = Field(default=None, max_length=10)


class ProfileRequest(BaseModel):
    timezone: str = Field(default="America/Costa_Rica", min_length=1, max_length=70)
    city: str = Field(default="", max_length=90)
    country: str = Field(default="", max_length=90)
    tone: str = Field(default="natural", pattern="^(natural|tecnico|academico)$")


class ItemRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class TaskUpdate(BaseModel):
    completed: bool


class CalendarCreate(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1200)
    kind: str = Field(default="task", pattern="^(task|meeting|reservation|reminder)$")
    start_at: str = Field(min_length=16, max_length=16)
    end_at: str = Field(min_length=16, max_length=16)
    reminder_minutes: int = Field(default=15, ge=0, le=10080)


class CalendarUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1200)
    kind: str | None = Field(default=None, pattern="^(task|meeting|reservation|reminder)$")
    start_at: str | None = Field(default=None, min_length=16, max_length=16)
    end_at: str | None = Field(default=None, min_length=16, max_length=16)
    reminder_minutes: int | None = Field(default=None, ge=0, le=10080)


class NetworkModeRequest(BaseModel):
    enabled: bool


class EventReview(BaseModel):
    status: str = Field(pattern="^(pending|false_positive|confirmed)$")


class VoiceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1400)
    mode: str = Field(default="conversation", pattern="^(conversation|system)$")


class BrainCaseRequest(BaseModel):
    kind: str = Field(pattern="^(experience|procedure|hypothesis)$")
    title: str = Field(min_length=3, max_length=150)
    situation: str = Field(min_length=5, max_length=900)
    action_taken: str = Field(min_length=3, max_length=1300)
    result: str = Field(default="", max_length=900)
    outcome: str = Field(default="unknown", pattern="^(unknown|success|failure|partial)$")
    evidence: str = Field(default="", max_length=900)


class BrainCaseReview(BaseModel):
    outcome: str = Field(pattern="^(unknown|success|failure|partial)$")
    evidence: str = Field(default="", max_length=900)


class BrainLinkRequest(BaseModel):
    from_case_id: int = Field(ge=1)
    to_case_id: int = Field(ge=1)
    relation: str = Field(pattern="^(related_to|supports|contradicts|prerequisite_for)$")
    evidence: str = Field(min_length=12, max_length=750)
    source_url: str = Field(default="", max_length=500)


class BrainFeedbackRequest(BaseModel):
    assistant_message_id: int = Field(ge=1)
    rating: str = Field(pattern="^(helpful|unhelpful)$")
    note: str = Field(default="", max_length=400)


_delivery_task: asyncio.Task | None = None


@app.on_event("startup")
async def setup_db() -> None:
    global _delivery_task
    security.init_auth_db()
    brain.init_db()
    sentinel.init_events_db()
    if os.environ.get("NYX_DISABLE_MAIL_WORKER") != "1":
        _delivery_task = asyncio.create_task(sentinel.delivery_loop())


@app.on_event("shutdown")
async def stop_worker() -> None:
    global _delivery_task
    if _delivery_task:
        _delivery_task.cancel()
        with suppress(asyncio.CancelledError):
            await _delivery_task
        _delivery_task = None


def creator_question(text: str) -> bool:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return any(re.search(pattern, normalized) for pattern in (
        r"\bquien\s+(?:es\s+)?tu\s+creador\b",
        r"\bquien\s+te\s+(?:creo|desarrollo|programo|hizo)\b",
        r"\bquien\s+(?:es\s+)?el\s+creador\s+de\s+nyx\b",
        r"\bquien\s+(?:creo|desarrollo|programo|hizo)\s+(?:a\s+)?nyx\b",
    ))


def _principal(request: Request) -> security.Principal:
    user = getattr(request.state, "principal", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Debes iniciar sesión.")
    return user


def require_admin(user: security.Principal = Depends(_principal)) -> security.Principal:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="No tienes acceso a esa función.")
    return user


def require_elevated(user: security.Principal = Depends(require_admin)) -> security.Principal:
    if not user.elevated:
        raise HTTPException(status_code=403, detail="Autoriza temporalmente los cambios administrativos.")
    return user


def _origin_allowed(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True
    try:
        supplied = urlsplit(origin)
        host = request.headers.get("host", "")
        return (supplied.scheme in {"http", "https"} and
                supplied.netloc.lower() == host.lower())
    except ValueError:
        return False


@app.middleware("http")
async def authentication_boundary(request: Request, call_next):
    route = request.url.path
    if not route.startswith("/api/"):
        return await call_next(request)
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not _origin_allowed(request):
        return JSONResponse({"detail": "Origen de solicitud no permitido."}, status_code=403)
    if route == "/api/auth/login":
        return await call_next(request)
    cookie = request.cookies.get(security.COOKIE_NAME)
    user = security.principal_for_token(cookie)
    if user is None:
        return JSONResponse({"detail": "Debes iniciar sesión."}, status_code=401,
                            headers={"Cache-Control": "no-store"})
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        csrf = request.headers.get("x-nyx-csrf", "")
        if not csrf or not hmac.compare_digest(csrf, user.csrf):
            return JSONResponse({"detail": "Solicitud rechazada: protección de sesión."},
                                status_code=403, headers={"Cache-Control": "no-store"})
    if user.must_change and route not in {
        "/api/auth/me", "/api/auth/logout", "/api/auth/change-password"
    }:
        return JSONResponse({"detail": "Debes cambiar la contraseña provisional."},
                            status_code=403, headers={"Cache-Control": "no-store"})
    if (user.role != "admin" and
        (route.startswith("/api/security/") or
         route.startswith("/api/auth/users") or
         route in {"/api/auth/elevate", "/api/auth/revoke-elevation"} or
         (route == "/api/profile" and request.method != "GET"))):
        sentinel.record_event(
            actor_username=user.username, actor_id=user.id, category="restricted_access",
            reason="Intento de utilizar una función administrativa sin autorización.",
            message=f"{request.method} {route}", severity="medium"
        )
        return JSONResponse({"detail": "No tienes permisos para esta función."},
                            status_code=403, headers={"Cache-Control": "no-store"})
    request.state.principal = user
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def auth_payload(user: security.Principal) -> dict:
    return {"id": user.id, "username": user.username, "role": user.role,
            "must_change": user.must_change, "elevated": user.elevated,
            "csrf": user.csrf}


@app.post("/api/auth/login")
def login(request: Request, body: Credentials, response: Response):
    if security.account_count() == 0:
        raise HTTPException(status_code=503,
                            detail="Inicializa la cuenta administrativa desde la terminal local.")
    account = security.authenticate(body.username, body.password)
    if account is None:
        if security.is_limited(body.username):
            # Registro de umbral (no se incluye ninguna contraseña o dato personal).
            with storage.connect() as db:
                recent = db.execute("""
                    SELECT COUNT(*) FROM security_events WHERE category='login_failures'
                    AND actor_username=? AND datetime(created_at) >= datetime('now','-15 minutes')
                """, (body.username,)).fetchone()[0]
            if not recent:
                sentinel.record_event(
                    actor_username=body.username, category="login_failures", severity="high",
                    reason="Cinco intentos fallidos de inicio de sesión en quince minutos.",
                    message="Intentos fallidos de autenticación", immediate=True
                )
        raise HTTPException(status_code=401, detail="Credenciales incorrectas o cuenta temporalmente bloqueada.")
    token, principal = security.create_session(account["id"])
    response.set_cookie(
        security.COOKIE_NAME, token, httponly=True, samesite="strict", path="/",
        max_age=security.SESSION_SECONDS,
        secure=(request.url.scheme == "https" or os.environ.get("NYX_COOKIE_SECURE") == "1")
    )
    response.headers["Cache-Control"] = "no-store"
    return auth_payload(principal)


@app.get("/api/auth/me")
def me(user: security.Principal = Depends(_principal)):
    return auth_payload(user)


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, user: security.Principal = Depends(_principal)):
    security.clear_session(request.cookies.get(security.COOKIE_NAME))
    response.delete_cookie(security.COOKIE_NAME, path="/", samesite="strict")
    return {"ok": True}


@app.post("/api/auth/change-password")
def change_password(body: PasswordChange, response: Response,
                    user: security.Principal = Depends(_principal)):
    try:
        security.change_password(user, body.current_password, body.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response.delete_cookie(security.COOKIE_NAME, path="/", samesite="strict")
    return {"ok": True, "login_required": True}


@app.post("/api/auth/elevate")
def elevate(body: PasswordConfirmation, user: security.Principal = Depends(require_admin)):
    if not security.elevate(user, body.password):
        raise HTTPException(status_code=403, detail="No se pudo autorizar el acceso administrativo.")
    return {"elevated": True, "duration_seconds": security.ELEVATION_SECONDS}


@app.post("/api/auth/revoke-elevation")
def revoke(user: security.Principal = Depends(require_admin)):
    security.revoke_elevation(user)
    return {"elevated": False}


@app.get("/api/auth/users")
def users(user: security.Principal = Depends(require_admin)):
    return security.list_users()


@app.post("/api/auth/users", status_code=201)
def create_user(body: UserCreate, user: security.Principal = Depends(require_elevated)):
    try:
        return security.create_standard(body.username, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/security/events")
def security_events(user: security.Principal = Depends(require_admin)):
    return sentinel.list_events()


@app.patch("/api/security/events/{event_id}")
def security_review(event_id: int, body: EventReview,
                    user: security.Principal = Depends(require_elevated)):
    entry = sentinel.review_event(event_id, body.status)
    if entry is None:
        raise HTTPException(status_code=404, detail="Incidente inexistente.")
    return entry


@app.get("/api/security/mail-status")
def email_status(user: security.Principal = Depends(require_admin)):
    return sentinel.mail_status()


@app.post("/api/security/test-mail")
def test_alert_email(user: security.Principal = Depends(require_elevated)):
    if not sentinel.mail_configured():
        raise HTTPException(status_code=409, detail="Configura el correo SMTP localmente.")
    try:
        sentinel.send_test_mail()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="No se pudo enviar la prueba.") from exc
    return {"sent": True}


@app.get("/api/network/status")
async def network_status(user: security.Principal = Depends(_principal)):
    return await network.connectivity_status(user.id)


@app.put("/api/network/mode")
def set_network_mode(body: NetworkModeRequest,
                     user: security.Principal = Depends(require_elevated)):
    # Cambiar el modo de red es una operación de configuración privilegiada.
    return {"enabled": storage.set_online_enabled(user.id, body.enabled)}


@app.get("/api/security/unreviewed")
def unreviewed_security_events(user: security.Principal = Depends(require_admin)):
    with storage.connect() as db:
        pending = db.execute("SELECT COUNT(*) FROM security_events WHERE status='pending'").fetchone()[0]
        queued = db.execute(
            "SELECT COUNT(*) FROM security_events WHERE mail_state='queued'"
        ).fetchone()[0]
    return {"pending": pending, "queued": queued}


@app.get("/api/status")
async def status(user: security.Principal = Depends(_principal)):
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            res = await client.get(f"{OLLAMA_URL}/api/tags")
            res.raise_for_status()
            models = [x.get("name", "") for x in res.json().get("models", [])]
        return {"online": MODEL in models, "ollama": True, "model": MODEL, "version": "0.5.0"}
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        return {"online": False, "ollama": False, "model": MODEL, "version": "0.5.0"}


@app.get("/api/sessions")
def sessions(q: str = "", user: security.Principal = Depends(_principal)):
    return storage.list_sessions(q, user.id)


@app.post("/api/sessions", status_code=201)
def create_session(request: SessionRequest, user: security.Principal = Depends(_principal)):
    return storage.create_session(request.title, user.id)


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: int, request: SessionRequest,
                   user: security.Principal = Depends(_principal)):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="El título está vacío.")
    item = storage.rename_session(session_id, title, user.id)
    if item is None:
        raise HTTPException(status_code=404, detail="Sesión inexistente.")
    return item


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: int, user: security.Principal = Depends(_principal)):
    if not storage.delete_session(session_id, user.id):
        raise HTTPException(status_code=404, detail="Sesión inexistente.")


@app.get("/api/sessions/{session_id}/messages")
def session_messages(session_id: int, user: security.Principal = Depends(_principal)):
    if storage.get_session(session_id, user.id) is None:
        raise HTTPException(status_code=404, detail="Sesión inexistente.")
    return storage.list_messages(session_id, limit=500, user_id=user.id)


@app.get("/api/history")
def history(user: security.Principal = Depends(_principal)):
    return storage.list_messages(None, limit=200, user_id=user.id)


@app.post("/api/chat")
async def chat(request: ChatRequest, user: security.Principal = Depends(_principal)):
    message = request.text.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Mensaje vacío.")
    if request.session_id is None:
        session = storage.create_session(message[:65], user.id)
    else:
        session = storage.get_session(request.session_id, user.id)
        if session is None:
            raise HTTPException(status_code=404, detail="La conversación no existe.")
    sid = session["id"]
    storage.add_message("user", message, sid, user.id)
    if user.role != "admin" and sentinel.sensitive_creator_request(message):
        sentinel.record_event(
            actor_id=user.id, actor_username=user.username, category="sensitive_request",
            severity="high", immediate=True,
            reason="Solicitud de información sensible sobre el propietario.",
            message=message
        )
        reply = "No puedo proporcionar información privada ni credenciales de otras personas."
        events = []
    elif creator_question(message):
        reply = ("Esa información es privada." if user.role != "admin"
                 else "Estás utilizando la cuenta administradora de NYX.")
        events = []
    else:
        try:
            reply, events = await cognition.generate_reply(
                message, sid, OLLAMA_URL, MODEL, user.id
            )
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=503,
                detail=f"No se pudo consultar {MODEL}. Comprueba Ollama y el modelo.") from exc
    saved = storage.add_message("assistant", reply, sid, user.id)
    if events:
        # Never claim a task solved in the real world: only record tool execution.
        try:
            brain.record_tool_episode(user.id, saved["id"], message, reply, events)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "No se pudo registrar experiencia de herramienta (sin datos sensibles)."
            )
    return {"reply": reply, "message": saved, "session_id": sid, "tool_events": events}



@app.get("/api/brain/status")
def brain_status(user: security.Principal = Depends(_principal)):
    return {**brain.metrics(user.id),
            "embedding_model": os.environ.get("NYX_BRAIN_EMBED_MODEL", "").strip() or None,
            "semantic_index_optional": True, "automatic_weight_training": False}


@app.get("/api/brain/cases")
def brain_cases(user: security.Principal = Depends(_principal)):
    return brain.list_cases(user.id, 250)


@app.post("/api/brain/cases", status_code=201)
def brain_add_case(request: BrainCaseRequest,
                   user: security.Principal = Depends(_principal)):
    try:
        return brain.create_case(user.id, **request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/api/brain/cases/{case_id}")
def brain_review_case(case_id: int, request: BrainCaseReview,
                      user: security.Principal = Depends(_principal)):
    try:
        found = brain.review_case(user.id, case_id, request.outcome, request.evidence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if found is None:
        raise HTTPException(status_code=404, detail="Experiencia inexistente.")
    return found


@app.delete("/api/brain/cases/{case_id}", status_code=204)
def brain_remove_case(case_id: int, user: security.Principal = Depends(_principal)):
    if not brain.delete_case(user.id, case_id):
        raise HTTPException(status_code=404, detail="Experiencia inexistente.")


@app.get("/api/brain/relations")
def brain_relations(user: security.Principal = Depends(_principal)):
    return brain.list_relations(user.id, 250)


@app.post("/api/brain/relations", status_code=201)
def brain_add_relation(request: BrainLinkRequest,
                       user: security.Principal = Depends(_principal)):
    try:
        return brain.add_relation(user.id, **request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/brain/relations/{link_id}", status_code=204)
def brain_remove_relation(link_id: int,
                          user: security.Principal = Depends(_principal)):
    if not brain.delete_relation(user.id, link_id):
        raise HTTPException(status_code=404, detail="Relación inexistente.")


@app.get("/api/brain/suggestions/{case_id}")
def brain_link_suggestions(case_id: int,
                           user: security.Principal = Depends(_principal)):
    if not brain.get_case(user.id, case_id):
        raise HTTPException(status_code=404, detail="Experiencia inexistente.")
    return brain.propose_links(user.id, case_id)


@app.get("/api/brain/feedback")
def brain_all_feedback(user: security.Principal = Depends(_principal)):
    return [{k: row[k] for k in ("assistant_message_id","rating","note")}
            for row in brain.list_feedback(user.id, 300)]


@app.post("/api/brain/feedback")
def brain_rate_reply(request: BrainFeedbackRequest,
                     user: security.Principal = Depends(_principal)):
    try:
        found = brain.record_feedback(user.id, **request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {k: found[k] for k in ("assistant_message_id","rating","note")}


@app.get("/api/brain/recall")
async def brain_recall(q: str = "", user: security.Principal = Depends(_principal)):
    if not 3 <= len(q.strip()) <= 300:
        raise HTTPException(status_code=422, detail="Consulta de 3 a 300 caracteres.")
    return await brain.retrieve(q, user.id, OLLAMA_URL, max_results=8)


@app.post("/api/brain/index")
async def brain_index_next(user: security.Principal = Depends(_principal)):
    # Explicit action. No background model downloads and no externally shared data.
    return await brain.index_next(user.id, OLLAMA_URL, batch_size=8)


@app.get("/api/tasks")
def tasks(user: security.Principal = Depends(_principal)):
    return storage.list_tasks(user.id)


@app.post("/api/tasks", status_code=201)
def create_task(request: TaskRequest, user: security.Principal = Depends(_principal)):
    title = request.text.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Título vacío.")
    try:
        return storage.add_task(title, user.id, request.due_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/api/tasks/{task_id}")
def change_task(task_id: int, request: TaskUpdate, user: security.Principal = Depends(_principal)):
    task = storage.update_task(task_id, request.completed, user.id)
    if task is None:
        raise HTTPException(status_code=404, detail="Tarea inexistente.")
    return task


@app.delete("/api/tasks/{task_id}", status_code=204)
def remove_task(task_id: int, user: security.Principal = Depends(_principal)):
    if not storage.delete_task(task_id, user.id):
        raise HTTPException(status_code=404, detail="Tarea inexistente.")


@app.get("/api/calendar")
def calendar(user: security.Principal = Depends(_principal)):
    return storage.list_calendar_events(user.id)


@app.post("/api/calendar", status_code=201)
def create_calendar(request: CalendarCreate, user: security.Principal = Depends(_principal)):
    try:
        return storage.add_calendar_event(user.id, **request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/api/calendar/{event_id}")
def edit_calendar(event_id: int, request: CalendarUpdate,
                  user: security.Principal = Depends(_principal)):
    try:
        changes = request.model_dump(exclude_unset=True, exclude_none=True)
        event = storage.update_calendar_event(event_id, user.id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if event is None:
        raise HTTPException(status_code=404, detail="Evento inexistente.")
    return event


@app.delete("/api/calendar/{event_id}", status_code=204)
def remove_calendar(event_id: int, user: security.Principal = Depends(_principal)):
    if not storage.delete_calendar_event(event_id, user.id):
        raise HTTPException(status_code=404, detail="Evento inexistente.")


@app.get("/api/calendar/reminders")
def calendar_reminders(user: security.Principal = Depends(_principal)):
    return storage.due_calendar_reminders(user.id)


@app.post("/api/calendar/{event_id}/acknowledge")
def acknowledge_calendar(event_id: int, user: security.Principal = Depends(_principal)):
    if not storage.acknowledge_calendar_reminder(event_id, user.id):
        raise HTTPException(status_code=404, detail="Recordatorio inexistente o atendido.")
    return {"acknowledged": True}


@app.get("/api/knowledge")
def knowledge(user: security.Principal = Depends(_principal)):
    return storage.list_research(user.id,limit=100)


@app.delete("/api/knowledge/{record_id}", status_code=204)
def delete_knowledge(record_id: int, user: security.Principal = Depends(_principal)):
    if not storage.delete_research(record_id,user.id):
        raise HTTPException(status_code=404, detail="Investigación inexistente.")


@app.get("/api/memory")
def memory(user: security.Principal = Depends(_principal)):
    return storage.list_memories(100, user.id)


@app.post("/api/memory", status_code=201)
def create_memory(request: ItemRequest, user: security.Principal = Depends(_principal)):
    content = request.text.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Nota vacía.")
    return storage.add_memory(content, user.id)


@app.delete("/api/memory/{memory_id}", status_code=204)
def remove_memory(memory_id: int, user: security.Principal = Depends(_principal)):
    if not storage.delete_memory(memory_id, user.id):
        raise HTTPException(status_code=404, detail="Nota inexistente.")


@app.get("/api/profile")
def read_profile(user: security.Principal = Depends(_principal)):
    return storage.get_profile(user.id)


@app.put("/api/profile")
def save_profile(request: ProfileRequest, user: security.Principal = Depends(require_elevated)):
    try:
        return storage.update_profile(user_id=user.id, **request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/context")
def local_context(user: security.Principal = Depends(_principal)):
    return cognition.current_context(user.id)


@app.get("/api/voice/status")
def voice_status(user: security.Principal = Depends(_principal)):
    return {"tts_ready": TTS_MODEL.is_file() and TTS_MODEL.with_suffix(".onnx.json").is_file(),
            "stt_ready": STT_MODEL.is_dir(),
            "tts": "Piper es_ES-davefx-medium", "stt": "Vosk small es-0.42"}


@lru_cache(maxsize=1)
def piper_voice():
    from piper import PiperVoice
    return PiperVoice.load(str(TTS_MODEL), use_cuda=False)


@app.post("/api/voice/speak")
def voice_speak(request: VoiceRequest, user: security.Principal = Depends(_principal)):
    if not TTS_MODEL.is_file() or not TTS_MODEL.with_suffix(".onnx.json").is_file():
        raise HTTPException(status_code=409, detail="Instala la voz Piper mediante app.voice_setup.")
    content = request.text.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Texto vacío.")
    buffer = io.BytesIO()
    try:
        with wave.open(buffer, "wb") as wav:
            piper_voice().synthesize_wav(content, wav)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Error al sintetizar la voz local.") from exc
    return Response(buffer.getvalue(), media_type="audio/wav", headers={"Cache-Control": "no-store"})


@lru_cache(maxsize=1)
def vosk_model():
    from vosk import Model
    return Model(str(STT_MODEL))


@app.post("/api/voice/transcribe")
async def voice_transcribe(request: Request, user: security.Principal = Depends(_principal)):
    if not STT_MODEL.is_dir():
        raise HTTPException(status_code=409, detail="Instala el modelo local Vosk.")
    content = await request.body()
    if len(content) > 2_000_000:
        raise HTTPException(status_code=413, detail="Audio demasiado largo (máximo 2 MB).")
    try:
        with wave.open(io.BytesIO(content), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                raise HTTPException(status_code=422, detail="Se requiere WAV PCM 16kHz mono 16-bit.")
            from vosk import KaldiRecognizer
            rec = KaldiRecognizer(vosk_model(), 16000)
            while True:
                frames = wav.readframes(4000)
                if not frames:
                    break
                rec.AcceptWaveform(frames)
            result = json.loads(rec.FinalResult())
    except (wave.Error, EOFError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Archivo de audio inválido.") from exc
    return {"text": result.get("text", "").strip()}
