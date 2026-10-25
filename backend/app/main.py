"""API local NYX 0.3. No publicar en red sin autenticacion."""
from __future__ import annotations

import io
import json
import os
import re
import unicodedata
import wave
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from . import storage, cognition

app = FastAPI(title="NYX 0.3", docs_url="/api/docs", openapi_url="/api/openapi.json")
OLLAMA_URL = os.environ.get("NYX_OLLAMA_URL", "http://ollama:11434").rstrip("/")
MODEL = os.environ.get("NYX_MODEL", "nyx:0.3")
VOICE_PATH = Path(os.environ.get("NYX_VOICE_PATH", "/voices"))
TTS_MODEL = VOICE_PATH / "es_ES-davefx-medium.onnx"
STT_MODEL = VOICE_PATH / "vosk-model-small-es-0.42"


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


class VoiceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1400)
    mode: str = Field(default="conversation", pattern="^(conversation|system)$")


@app.on_event("startup")
def setup_db() -> None:
    storage.init_db()


def creator_question(text: str) -> bool:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    patterns = (
        r"\bquien\s+(?:es\s+)?tu\s+creador\b",
        r"\bquien\s+te\s+(?:creo|desarrollo|programo|hizo)\b",
        r"\bquien\s+(?:es\s+)?el\s+creador\s+de\s+nyx\b",
        r"\bquien\s+(?:creo|desarrollo|programo|hizo)\s+(?:a\s+)?nyx\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


@app.get("/api/status")
async def status():
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            res = await client.get(f"{OLLAMA_URL}/api/tags")
            res.raise_for_status()
            models = [x.get("name", "") for x in res.json().get("models", [])]
        return {"online": MODEL in models, "ollama": True, "model": MODEL, "version": "0.3"}
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        return {"online": False, "ollama": False, "model": MODEL, "version": "0.3"}


@app.get("/api/sessions")
def sessions(q: str = ""):
    return storage.list_sessions(q)


@app.post("/api/sessions", status_code=201)
def create_session(request: SessionRequest):
    return storage.create_session(request.title)


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: int, request: SessionRequest):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="El título está vacío")
    item = storage.rename_session(session_id, title)
    if item is None:
        raise HTTPException(status_code=404, detail="Sesión inexistente")
    return item


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: int):
    if not storage.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Sesión inexistente")


@app.get("/api/sessions/{session_id}/messages")
def session_messages(session_id: int):
    if storage.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Sesión inexistente")
    return storage.list_messages(session_id, limit=500)


@app.get("/api/history")
def history():
    return storage.list_messages(limit=200)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.text.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Mensaje vacío")
    if request.session_id is None:
        session = storage.create_session(message[:65])
    else:
        session = storage.get_session(request.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="La conversación ya no existe")
    sid = session["id"]
    storage.add_message("user", message, sid)
    if creator_question(message):
        reply = "Mi creador es J.G.R."
        tool_events = []
    else:
        try:
            reply, tool_events = await cognition.generate_reply(message, sid, OLLAMA_URL, MODEL)
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"No se pudo consultar {MODEL}. Comprueba Ollama y el modelo configurado."
            ) from exc
    saved = storage.add_message("assistant", reply, sid)
    return {"reply": reply, "message": saved, "session_id": sid,
            "tool_events": tool_events}


@app.get("/api/tasks")
def tasks():
    return storage.list_tasks()


@app.post("/api/tasks", status_code=201)
def create_task(request: TaskRequest):
    title = request.text.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Título vacío")
    try:
        return storage.add_task(title, request.due_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/api/tasks/{task_id}")
def change_task(task_id: int, request: TaskUpdate):
    task = storage.update_task(task_id, request.completed)
    if task is None:
        raise HTTPException(status_code=404, detail="Tarea inexistente")
    return task


@app.delete("/api/tasks/{task_id}", status_code=204)
def remove_task(task_id: int):
    if not storage.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Tarea inexistente")


@app.get("/api/memory")
def memory():
    return storage.list_memories()


@app.post("/api/memory", status_code=201)
def create_memory(request: ItemRequest):
    content = request.text.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Nota vacía")
    return storage.add_memory(content)


@app.delete("/api/memory/{memory_id}", status_code=204)
def remove_memory(memory_id: int):
    if not storage.delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Nota inexistente")


@app.get("/api/profile")
def read_profile():
    return storage.get_profile()


@app.put("/api/profile")
def save_profile(request: ProfileRequest):
    try:
        return storage.update_profile(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/context")
def local_context():
    return cognition.current_context()


@app.get("/api/voice/status")
def voice_status():
    return {"tts_ready": TTS_MODEL.is_file() and TTS_MODEL.with_suffix(".onnx.json").is_file(),
            "stt_ready": STT_MODEL.is_dir(),
            "tts": "Piper es_ES-davefx-medium",
            "stt": "Vosk small es-0.42"}


@lru_cache(maxsize=1)
def piper_voice():
    from piper import PiperVoice
    return PiperVoice.load(str(TTS_MODEL), use_cuda=False)


@app.post("/api/voice/speak")
def voice_speak(request: VoiceRequest):
    if not TTS_MODEL.is_file() or not TTS_MODEL.with_suffix(".onnx.json").is_file():
        raise HTTPException(status_code=409,
            detail="Instala primero la voz Piper: docker compose exec api python -m app.voice_setup --tts")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Texto vacío")
    buffer = io.BytesIO()
    try:
        with wave.open(buffer, "wb") as wav:
            piper_voice().synthesize_wav(text, wav)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Error al sintetizar la voz local") from exc
    return Response(buffer.getvalue(), media_type="audio/wav",
                    headers={"Cache-Control": "no-store"})


@lru_cache(maxsize=1)
def vosk_model():
    from vosk import Model
    return Model(str(STT_MODEL))


@app.post("/api/voice/transcribe")
async def voice_transcribe(request: Request):
    if not STT_MODEL.is_dir():
        raise HTTPException(status_code=409,
            detail="Instala primero Vosk: docker compose exec api python -m app.voice_setup --stt")
    content = await request.body()
    if len(content) > 2_000_000:
        raise HTTPException(status_code=413, detail="Audio demasiado largo (máximo 2 MB)")
    try:
        with wave.open(io.BytesIO(content), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                raise HTTPException(status_code=422, detail="Se requiere WAV PCM 16kHz mono 16-bit")
            from vosk import KaldiRecognizer
            rec = KaldiRecognizer(vosk_model(), 16000)
            while True:
                frames = wav.readframes(4000)
                if not frames:
                    break
                rec.AcceptWaveform(frames)
            result = json.loads(rec.FinalResult())
    except (wave.Error, EOFError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Archivo de audio inválido") from exc
    return {"text": result.get("text", "").strip()}
