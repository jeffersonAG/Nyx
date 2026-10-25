"""Convierte ejemplos APROBADOS explícitamente en un corpus para la red propia.

Entrada privada JSONL, no acceso automático a SQLite ni a conversaciones.
Validación reservada por ejemplo/documento y sin frases duplicadas.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def curate(input_file: Path, output_dir: Path, *,
           shards: int = 5, seed: int = 2026) -> dict:
    if not 2 <= shards <= 20:
        raise ValueError("--shards debe estar entre 2 y 20.")
    if not input_file.is_file() or input_file.is_symlink():
        raise ValueError("El archivo JSONL de entrada debe existir y no ser un enlace.")
    if input_file.stat().st_size > 2_000_000:
        raise ValueError("El conjunto JSONL supera 2 MB.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("El directorio de salida contiene archivos; "
                         "utiliza otro nombre para preservar el corpus anterior.")
    items = []
    seen = set()
    skipped = 0
    for index, line in enumerate(input_file.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido en la línea {index}.") from exc
        if not isinstance(record, dict):
            raise ValueError(f"La línea {index} debe contener un objeto JSON.")
        if record.get("approved") is not True:
            skipped += 1
            continue
        question = record.get("question")
        answer = record.get("answer")
        evidence = record.get("evidence", "")
        if not isinstance(question,str) or not 5 <= len(question.strip()) <= 500:
            raise ValueError(f"Pregunta inválida en línea {index}.")
        if not isinstance(answer,str) or not 5 <= len(answer.strip()) <= 1500:
            raise ValueError(f"Respuesta inválida en línea {index}.")
        if not isinstance(evidence,str) or len(evidence.strip()) > 500:
            raise ValueError(f"Evidencia inválida en línea {index}.")
        # No asumir que el autor marcó correctamente hechos no comprobados.
        key = re.sub(r"\s+", " ", question.casefold().strip())
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        items.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "evidence": evidence.strip(),
        })
    if len(items) < shards * 2:
        raise ValueError(f"Se necesitan al menos {shards * 2} ejemplos únicos "
                         "aprobados para dividir entrenamiento y validación.")
    random.Random(seed).shuffle(items)
    groups = [[] for _ in range(shards)]
    for i,item in enumerate(items):
        groups[i % shards].append(item)
    output_dir.mkdir(parents=True,exist_ok=False)
    paths=[]
    try:
        for i,group in enumerate(groups):
            text=[]
            for item in group:
                block = f"Usuario: {item['question']}\nNYX: {item['answer']}"
                if item["evidence"]:
                    block += f"\nEvidencia aportada por el usuario: {item['evidence']}"
                text.append(block)
            path=output_dir / f"{i + 1:02d}_aprobados.txt"
            path.write_text("\n\n".join(text)+"\n",encoding="utf-8")
            paths.append(str(path))
    except Exception:
        for path in paths:
            Path(path).unlink(missing_ok=True)
        output_dir.rmdir()
        raise
    return {
        "approved_examples": len(items),
        "skipped_unapproved_or_duplicates": skipped,
        "documents": len(paths),
        "reserved_validation_document": paths[-1],
        "training_documents": paths[:-1],
        "note": "Aprobación humana no equivale a evidencia científica independiente.",
    }


def main() -> None:
    parser=argparse.ArgumentParser(description="Crea corpus curado desde JSONL aprobado.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args=parser.parse_args()
    print(json.dumps(curate(args.input,args.output,shards=args.shards,seed=args.seed),
                     indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
