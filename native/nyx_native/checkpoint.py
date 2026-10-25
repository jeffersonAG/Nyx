"""Comprobantes reproducibles: estado de pesos, optimizador, corpus y evaluación.

No utiliza torch.load sin weights_only; no ejecuta contenido del archivo.
Los archivos se guardan atómicamente en volumen de datos, no en Git.
"""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .model import ModelConfig, NativeTransformer


def save_checkpoint(path: Path, *, model: NativeTransformer,
                    optimizer: torch.optim.Optimizer, step: int,
                    corpus_digest: str, validation_loss: float,
                    seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "architecture": "nyx-native-transformer-v1",
        "config": model.config.as_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "corpus_sha256": corpus_digest,
        "validation_loss": validation_loss,
        "seed": seed,
    }
    with NamedTemporaryFile(prefix="nyx-", suffix=".tmp",
                            dir=path.parent, delete=False) as temp:
        tmp_name = temp.name
    try:
        torch.save(bundle, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_checkpoint(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("No existe un checkpoint entrenado.")
    # weights_only=True bloquea la deserialización de objetos Python arbitrarios.
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("architecture") != "nyx-native-transformer-v1":
        raise ValueError("Formato o arquitectura de checkpoint incompatible.")
    if not isinstance(bundle.get("config"), dict):
        raise ValueError("Faltan los parámetros de arquitectura.")
    cfg = ModelConfig(**bundle["config"])
    cfg.validate()
    if not isinstance(bundle.get("model"), dict):
        raise ValueError("Checkpoint sin pesos válidos.")
    if not 0 <= int(bundle.get("step", -1)) <= 10_000_000:
        raise ValueError("Paso de entrenamiento inválido.")
    return bundle


def restore_model(path: Path) -> tuple[NativeTransformer, dict]:
    bundle = load_checkpoint(path)
    model = NativeTransformer(ModelConfig(**bundle["config"]))
    model.load_state_dict(bundle["model"], strict=True)
    model.eval()
    return model, bundle
