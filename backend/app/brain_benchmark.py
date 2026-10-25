"""NYX Brain: reproducible synthetic retrieval benchmark.

Runs ONLY in a temporary database. This is a sanity check of Python retrieval,
not proof of model intelligence or neural learning. Never touches /data/nyx.db.
Invocation from the project root:
    docker compose exec api python -m app.brain_benchmark
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

from . import brain, security, storage

CASES = [
    ("Falla WiFi en antena", "El enlace inalámbrico deja de funcionar",
     "Comprobar alimentación PoE y cable", "Se restableció señal", "success"),
    ("Primera capa PLA", "La impresora 3D Neptune no adhiere el filamento a la cama",
     "Limpiar superficie PEI y calibrar altura", "Mejor adhesión", "partial"),
    ("Revisión de router", "Enrutamiento entre VLANs falla en el switch",
     "Revisar tablas y reglas de firewall", "No mejoró", "failure"),
    ("Circuito de corriente directa", "Resistencia y capacitor de prototipo RC",
     "Medir tensión y constante de tiempo", "Resultados medidos", "success"),
    ("Gestión de respaldo", "No se encuentra la copia de seguridad del NAS",
     "Validar snapshots y almacenamiento", "Respaldo localizado", "success"),
]
QUESTIONS = [
    ("Problemas de antena inalámbrica y alimentación", 0),
    ("La impresora dejó de adherir plástico a la cama", 1),
    ("Qué hicimos ante un fallo de enrutamiento VLAN", 2),
    ("Mediciones en circuito RC con capacitor", 3),
    ("Recuperar copia de almacenamiento NAS", 4),
]


async def evaluate() -> dict:
    previous_db = os.environ.get("NYX_DB_PATH")
    previous_embeddings = os.environ.get("NYX_BRAIN_EMBED_MODEL")
    try:
        with tempfile.TemporaryDirectory(prefix="nyx-brain-bench-") as directory:
            os.environ["NYX_DB_PATH"] = str(Path(directory) / "synthetic.db")
            os.environ.pop("NYX_BRAIN_EMBED_MODEL", None)
            security.init_auth_db()
            brain.init_db()
            security.bootstrap_admin("bench_admin", "TemporaryBenchmarkPass123")
            with storage.connect() as db:
                user_id = db.execute("SELECT id FROM users WHERE role='admin'").fetchone()[0]
            rows = [brain.create_case(user_id, kind="experience", title=title,
                situation=situation, action_taken=action, result=result,
                outcome=outcome, evidence="Caso de demostración sintética")
                for title, situation, action, result, outcome in CASES]
            started = perf_counter()
            hits = []
            modes = set()
            for question, expected_index in QUESTIONS:
                result = await brain.retrieve(question, user_id, "http://ollama", model="",
                                              max_results=3)
                modes.add(result["mode"])
                expected_id = rows[expected_index]["id"]
                rank = next((i + 1 for i, candidate in enumerate(result["items"])
                             if candidate["kind"] == "case" and
                                candidate["id"] == expected_id), None)
                hits.append({"question": question, "expected_case": expected_id,
                             "rank": rank, "hit_at_3": rank is not None})
            elapsed_ms = round((perf_counter() - started) * 1000, 2)
            return {"benchmark": "synthetic_lexical_retrieval_v1",
                    "test_questions": len(hits),
                    "hit_at_3": sum(hit["hit_at_3"] for hit in hits),
                    "rate_at_3": round(sum(hit["hit_at_3"] for hit in hits) / len(hits), 3),
                    "elapsed_ms": elapsed_ms, "modes": sorted(modes),
                    "details": hits,
                    "scope": "Datos sintéticos. No mide precisión de Qwen ni capacidad científica. "
                             "No utiliza el perfil ni las memorias reales."}
    finally:
        if previous_db is None:
            os.environ.pop("NYX_DB_PATH", None)
        else:
            os.environ["NYX_DB_PATH"] = previous_db
        if previous_embeddings is not None:
            os.environ["NYX_BRAIN_EMBED_MODEL"] = previous_embeddings


def main():
    print(json.dumps(asyncio.run(evaluate()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
