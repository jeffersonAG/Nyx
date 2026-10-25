"""Microservicio privado del modelo propio. Nunca publicar puerto al host.

Separado del servidor principal para no cargar PyTorch en la API de NYX.
La API principal seguirá usando Qwen salvo elección experimental explícita.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .checkpoint import load_checkpoint
from .inference import generate_text

CHECKPOINT = Path(os.getenv("NYX_NATIVE_CHECKPOINT", "/models/native.pt"))
app = FastAPI(title="NYX Native Lab", docs_url=None, redoc_url=None,
              openapi_url=None)


class Prompt(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    max_new_tokens: int = Field(default=80, ge=1, le=256)
    temperature: float = Field(default=0.9, ge=0.1, le=2.0)
    top_k: int = Field(default=40, ge=1, le=259)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


@app.get("/health")
def health():
    try:
        bundle = load_checkpoint(CHECKPOINT)
        return {
            "ready": True,
            "architecture": bundle["architecture"],
            "step": bundle["step"],
            "context": bundle["config"]["context"],
            "validation_loss": bundle["validation_loss"],
            "trained_from_scratch": True,
        }
    except (FileNotFoundError, ValueError, RuntimeError, OSError, KeyError):
        return {"ready": False, "reason": "No hay checkpoint válido todavía."}


@app.post("/generate")
async def generate(request: Prompt):
    try:
        return await asyncio.to_thread(
            generate_text, CHECKPOINT, request.text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k, seed=request.seed,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409,
                            detail="Entrena primero el modelo propio.") from exc
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise HTTPException(status_code=422,
                            detail="No fue posible inferir con este checkpoint.") from exc
