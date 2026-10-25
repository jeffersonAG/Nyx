"""Corpus explícito y separado de las conversaciones personales de NYX."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import encode


@dataclass(frozen=True)
class Corpus:
    train: torch.Tensor
    validation: torch.Tensor
    digest: str
    documents: int


def load_corpus(directory: Path, *, context: int, max_bytes: int = 5_000_000
                ) -> Corpus:
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("La ruta del corpus debe ser un directorio.")
    files = sorted(p for p in root.glob("*.txt")
                   if p.is_file() and not p.is_symlink())
    if len(files) < 2:
        raise ValueError("Se necesitan al menos dos archivos .txt (uno reservado para validación).")
    documents = []
    digest = hashlib.sha256()
    total_size = 0
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("No se aceptan rutas fuera del corpus.")
        data = path.read_bytes()
        total_size += len(data)
        if total_size > max_bytes:
            raise ValueError("El corpus excede el límite de tamaño configurado.")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("El corpus debe usar UTF-8.") from exc
        if len(text.strip()) < 50:
            raise ValueError(f"Archivo demasiado pequeño: {path.name}")
        documents.append(encode(text.strip() + "\n", add_eos=True))
        digest.update(path.name.encode("utf-8") + b"\0" + data + b"\0")
    # Separación por documento, evitando fuga de fragmentos idénticos.
    validation_count = max(1, len(documents) // 5)
    train_docs = documents[:-validation_count]
    valid_docs = documents[-validation_count:]
    train_ids = [token for doc in train_docs for token in doc]
    valid_ids = [token for doc in valid_docs for token in doc]
    if len(train_ids) <= context + 1 or len(valid_ids) <= context + 1:
        raise ValueError("Los conjuntos de entrenamiento y validación deben superar "
                         "el contexto. Añade más texto o reduce --context.")
    return Corpus(
        train=torch.tensor(train_ids, dtype=torch.long),
        validation=torch.tensor(valid_ids, dtype=torch.long),
        digest=digest.hexdigest(),
        documents=len(documents),
    )


def sample_batch(series: torch.Tensor, batch_size: int, context: int,
                 generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = len(series) - context - 1
    if maximum < 0:
        raise ValueError("Corpus insuficiente para el contexto.")
    offsets = torch.randint(0, maximum + 1, (batch_size,), generator=generator)
    x = torch.stack([series[start:start + context] for start in offsets.tolist()])
    y = torch.stack([series[start + 1:start + context + 1]
                     for start in offsets.tolist()])
    return x, y
