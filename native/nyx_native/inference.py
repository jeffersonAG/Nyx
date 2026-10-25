"""Inferencia aislada del modelo nativo: no ejecuta herramientas ni comandos."""
from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import restore_model
from .tokenizer import encode, decode


def generate_text(checkpoint: Path, prompt: str, *,
                  max_new_tokens: int = 80, temperature: float = 0.9,
                  top_k: int = 40, seed: int | None = None) -> dict:
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 3000:
        raise ValueError("El prompt debe contener entre 1 y 3000 caracteres.")
    model, saved = restore_model(checkpoint)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    prepared = encode("Usuario: " + prompt.strip() + "\nNYX:", add_bos=True)
    prepared = prepared[-model.config.context:]
    source = torch.tensor([prepared], dtype=torch.long)
    sampled = model.generate(source, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k, seed=seed)
    continuation = sampled[0, source.shape[1]:].tolist()
    return {
        "text": decode(continuation),
        "step": saved["step"],
        "parameters": model.parameter_count,
        "architecture": saved["architecture"],
        "experiment": True,
        "note": "Modelo entrenado desde cero; las respuestas no se consideran fiables.",
    }
