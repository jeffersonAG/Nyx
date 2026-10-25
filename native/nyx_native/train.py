"""Entrenamiento real de una red ORIGINAL en CPU con autograd y AdamW.

No entrena silenciosamente a partir de conversaciones personales. El operador
elige cada archivo de corpus y ejecuta manualmente este comando.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import load_corpus, sample_batch
from .model import ModelConfig, NativeTransformer


@torch.no_grad()
def evaluate(model: NativeTransformer, data: torch.Tensor, *,
             batch_size: int, batches: int, rng: torch.Generator) -> float:
    model.eval()
    scores = []
    for _ in range(batches):
        x, y = sample_batch(data, batch_size, model.config.context, rng)
        _, loss = model(x, y)
        scores.append(float(loss.item()))
    model.train()
    return sum(scores) / len(scores)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena desde cero la red neuronal propia de NYX (PyTorch, CPU).")
    parser.add_argument("--data-dir", type=Path, default=Path("/samples"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/models/native.pt"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-from", type=Path, default=None,
        help="Adaptar pesos propios a otro corpus; siempre guardar en otro checkpoint.")
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> dict:
    if not 1 <= args.steps <= 50_000:
        raise ValueError("--steps debe estar entre 1 y 50000.")
    if not 1 <= args.batch_size <= 32:
        raise ValueError("--batch-size debe estar entre 1 y 32.")
    if not 1 <= args.eval_every <= 2000 or not 1 <= args.eval_batches <= 30:
        raise ValueError("Parámetros de evaluación fuera de rango.")
    if not 1 <= args.cpu_threads <= 12:
        raise ValueError("--cpu-threads fuera de rango.")
    if not 1e-6 <= args.learning_rate <= .01 or not 0 <= args.weight_decay <= 1:
        raise ValueError("Hiperparámetros de optimización fuera de rango.")
    if args.resume and args.continue_from:
        raise ValueError("--resume y --continue-from son modos incompatibles.")
    if args.continue_from and args.continue_from.resolve() == args.checkpoint.resolve():
        raise ValueError("Para adaptación usa un checkpoint de salida NUEVO, "
                         "sin sobrescribir los pesos originales.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = ModelConfig(context=args.context, d_model=args.d_model,
                         n_heads=args.heads, n_layers=args.layers)
    config.validate()
    corpus = load_corpus(args.data_dir, context=config.context)
    model = NativeTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    initial_step = 0
    training_mode = "scratch"
    parent_sha256 = None
    if args.continue_from:
        previous = load_checkpoint(args.continue_from)
        if previous["config"] != config.as_dict():
            raise ValueError("El checkpoint original utiliza otra arquitectura.")
        model.load_state_dict(previous["model"], strict=True)
        parent_sha256 = hashlib.sha256(args.continue_from.read_bytes()).hexdigest()
        training_mode = "adaptation"
    if args.resume:
        saved = load_checkpoint(args.checkpoint)
        if saved["config"] != config.as_dict():
            raise ValueError("El checkpoint utiliza otra arquitectura.")
        if saved["corpus_sha256"] != corpus.digest:
            raise ValueError("El corpus cambió. No se admite reanudar con datos diferentes.")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        training_mode = saved.get("training_mode", "scratch")
        parent_sha256 = saved.get("parent_sha256")
        initial_step = int(saved["step"])
        if initial_step >= args.steps:
            raise ValueError("El número de pasos debe superar el checkpoint anterior.")

    train_rng = torch.Generator().manual_seed(args.seed + initial_step)

    def held_out_loss() -> float:
        # Las mismas ventanas de validación en cada medición: comparación justa.
        return evaluate(model, corpus.validation, batch_size=args.batch_size,
                        batches=args.eval_batches,
                        rng=torch.Generator().manual_seed(args.seed + 999))

    validation_baseline = held_out_loss()
    def schedule(step: int) -> float:
        # Reanudar no cambia el máximo de pasos definido por el operador.
        warmup = min(30, max(1, args.steps // 10))
        if step <= warmup:
            return args.learning_rate * step / warmup
        position = (step - warmup) / max(1, args.steps - warmup)
        return args.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * position)))

    last_training = float("nan")
    last_validation = float("nan")
    for step in range(initial_step + 1, args.steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = schedule(step)
        x, y = sample_batch(corpus.train, args.batch_size, config.context, train_rng)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()  # RETROPROPAGACIÓN REAL sobre la red propia.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # MODIFICACIÓN REAL de los pesos neuronales.
        last_training = float(loss.detach().item())
        if not math.isfinite(last_training):
            raise RuntimeError("Pérdida no finita; entrenamiento detenido sin sobrescribir pesos.")
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            last_validation = held_out_loss()
            if not math.isfinite(last_validation):
                raise RuntimeError("Pérdida de validación no finita.")
            save_checkpoint(args.checkpoint, model=model, optimizer=optimizer,
                            step=step, corpus_digest=corpus.digest,
                            validation_loss=last_validation, seed=args.seed,
                            validation_baseline=validation_baseline,
                            training_mode=training_mode,
                            parent_sha256=parent_sha256)
            print(json.dumps({
                "step": step, "train_loss": round(last_training, 5),
                "validation_loss": round(last_validation, 5),
                "parameters": model.parameter_count, "documents": corpus.documents,
                "mode": training_mode,
                "validation_baseline": round(validation_baseline, 5),
                "saved": str(args.checkpoint),
            }, ensure_ascii=False), flush=True)
    return {
        "step": args.steps, "train_loss": last_training,
        "training_mode": training_mode,
        "validation_baseline": validation_baseline,
        "validation_loss": last_validation, "parameter_count": model.parameter_count,
        "corpus_sha256": corpus.digest, "trained_from_scratch": True,
        "checkpoint": str(args.checkpoint),
    }


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
