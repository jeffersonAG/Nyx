"""Respuestas verificables a consultas frecuentes sobre módulos propios de NYX.

Sin mediación del modelo: evita mensajes inventados de 'no tengo permisos'.
Toda operación está restringida por user_id en consultas SQLite.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

from . import storage


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _current_date(user_id: int) -> str:
    timezone = storage.get_profile(user_id)["timezone"]
    return datetime.now(ZoneInfo(timezone)).date().isoformat()


def module_reply(text: str, user_id: int) -> str | None:
    norm = normalize(text)
    if len(norm) > 500:
        return None
    if re.search(r"\b(?:tareas?|pendientes?|actividades?)\b", norm) and re.search(
        r"\b(?:cuant[ao]s?|numero|total|lista|listado|muestra|muestrame|"
        r"consulta|consultar|revisa|ver|tengo|pendientes?|programad[ao]s?|"
        r"organiza|organizar|cuales|dime|hoy)\b", norm
    ):
        tasks = storage.list_tasks(user_id)
        today = _current_date(user_id)
        today_only = bool(re.search(r"\b(?:hoy|para hoy|de hoy)\b", norm))
        pending_only = bool(re.search(r"\b(?:pendientes?|sin completar|por hacer)\b", norm))
        relevant = [task for task in tasks if (not today_only or task["due_date"] == today)
                    and (not pending_only or not task["completed"])]
        counts = bool(re.search(r"\b(?:cuant[ao]s?|numero|total)\b", norm))
        if counts:
            adjective = "pendientes " if pending_only else ""
            qualifier = "con fecha de hoy " if today_only else ""
            return f"Tienes {len(relevant)} tareas {adjective}{qualifier}registradas en tu cuenta."
        if not relevant:
            extra = " con fecha de hoy" if today_only else ""
            return f"No tienes tareas{extra} que coincidan con esa consulta."
        lines = [f"Encontré {len(relevant)} tareas en tu cuenta:"]
        for task in relevant[:30]:
            status = "completada" if task["completed"] else "pendiente"
            due = f" · fecha {task['due_date']}" if task["due_date"] else ""
            lines.append(f"• #{task['id']} {task['title']} ({status}{due})")
        if len(relevant) > 30:
            lines.append(f"…y {len(relevant) - 30} tareas adicionales en el módulo Tareas.")
        return "\n".join(lines)

    if re.search(r"\b(?:memoria|recuerdos?|notas?)\b", norm) and re.search(
        r"\b(?:cuant[ao]s?|muestra|muestrame|consulta|recuerdas|"
        r"que sabes|lista|listar|tengo|dime|guardad[ao]s?)\b", norm
    ):
        notes = storage.list_memories(30, user_id)
        if re.search(r"\b(?:cuant[ao]s?|total)\b", norm):
            return f"Tienes {len(storage.list_memories(100, user_id))} notas de memoria entre las 100 más recientes."
        if not notes:
            return "No tienes notas guardadas en tu memoria personal."
        return "Estas son tus notas guardadas:\n" + "\n".join(
            f"• #{item['id']} {item['content']}" for item in notes[:20]
        )

    match = re.search(
        r"\b(?:busca|buscar|encuentra|consulta|revisa)\s+(?:en\s+)?"
        r"(?:mi\s+)?(?:bitacora|historial|conversaciones?)\s+"
        r"(?:sobre|de|la\s+palabra|el\s+tema)\s+(.+?)[.!?]?$", norm
    )
    if match:
        query = match.group(1).strip()
        sessions = storage.list_sessions(query, user_id)
        if not sessions:
            return f"No encontré conversaciones en tu bitácora relacionadas con «{query}»."
        return "Encontré estas conversaciones:\n" + "\n".join(
            f"• #{s['id']} {s['title']}" for s in sessions[:12]
        )

    if re.search(r"\b(?:perfil|configuracion)\b", norm) and re.search(
        r"\b(?:consulta|ver|muestra|dime|cual|revisa)\b", norm
    ) and not re.search(r"\b(?:cambia|modifica|actualiza|establece)\b", norm):
        profile = storage.get_profile(user_id)
        city = profile["city"] or "no configurada"
        country = profile["country"] or "no configurado"
        return (f"Tu perfil local indica ciudad habitual: {city}; país: {country}; "
                f"zona horaria: {profile['timezone']}; estilo: {profile['tone']}.")
    return None
