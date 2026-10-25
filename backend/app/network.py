"""Consultas de red restringidas y voluntarias de NYX.

El backend únicamente contacta servicios HTTPS de una lista fija.
Nunca transmite conversaciones, memorias, contraseñas ni cookies.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any
from urllib.parse import quote

import httpx

from . import storage

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WIKI_URL = "https://es.wikipedia.org/w/api.php"
TIMEOUT = httpx.Timeout(8.0, connect=4.0)
AGENT = "NYX-Local/0.4 (weather and public encyclopedia requests; user initiated)"

CONDITIONS = {
    0: "cielo despejado", 1: "mayormente despejado", 2: "parcialmente nublado",
    3: "nublado", 45: "neblina", 48: "neblina con escarcha",
    51: "llovizna ligera", 53: "llovizna moderada", 55: "llovizna intensa",
    56: "llovizna helada ligera", 57: "llovizna helada intensa",
    61: "lluvia ligera", 63: "lluvia moderada", 65: "lluvia intensa",
    66: "lluvia helada ligera", 67: "lluvia helada intensa",
    71: "nieve ligera", 73: "nieve moderada", 75: "nieve intensa",
    80: "chubascos ligeros", 81: "chubascos moderados", 82: "chubascos intensos",
    95: "tormenta eléctrica", 96: "tormenta con granizo", 99: "tormenta fuerte con granizo",
}

NEILY_ALIASES = {"neily", "ciudad neily", "ciudad neilly", "villa neily"}
# Coordenadas aproximadas de la ciudad de Neily (Costa Rica), NO dirección del usuario.
NEILY_CENTER = {"name": "Ciudad Neily", "country": "Costa Rica",
                "latitude": 8.64903, "longitude": -82.9439}


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()


def online_enabled(user_id: int) -> bool:
    return storage.get_online_enabled(user_id)


async def connectivity_status(user_id: int) -> dict[str, Any]:
    enabled = online_enabled(user_id)
    if not enabled:
        return {"enabled": False, "connected": False, "mode": "offline",
                "message": "Modo local seleccionado."}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4, connect=2),
                                     headers={"User-Agent": AGENT}, follow_redirects=False) as client:
            response = await client.get(GEO_URL, params={"name": "San José", "count": 1})
            response.raise_for_status()
        return {"enabled": True, "connected": True, "mode": "online",
                "message": "Consulta HTTPS a Open-Meteo disponible."}
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        return {"enabled": True, "connected": False, "mode": "offline",
                "message": "Sin acceso al servicio de red; NYX continúa localmente."}


def extract_weather_location(text: str, user_id: int) -> str | None:
    normalized = normalize(text)
    if not re.search(r"\b(clima|temperatura|tiempo meteorologico|pronostico)\b", normalized):
        return None
    # No captar interrogaciones ambiguas como "qué es el clima tropical".
    intent = re.search(r"\b(actual|ahora|hoy|pronostico|que tiempo|cual es el clima|dime el clima|clima en|temperatura en)\b", normalized)
    if not intent:
        return None
    match = re.search(r"\b(?:en|de|para)\s+([a-záéíóúüñA-ZÁÉÍÓÚÜÑ][\w\sáéíóúüñÁÉÍÓÚÜÑ,-]{1,80})",
                      text, re.I)
    if match:
        raw = match.group(1).strip(" \t.,?!¡¿")
        raw = re.split(r"\s+(?:ahora|actualmente|hoy|por favor)\b", raw, maxsplit=1, flags=re.I)[0]
        # Puede ser "en ciudad neily" y también "en Ciudad Neily, Costa Rica".
        if 2 <= len(raw) <= 80:
            return raw
    return storage.get_profile(user_id)["city"].strip() or ""


async def _resolve_location(city: str, country: str) -> dict | None:
    clean = normalize(city).strip()
    if clean in NEILY_ALIASES or clean.startswith("ciudad neily,"):
        return NEILY_CENTER.copy()
    params = {"name": city[:80], "count": 6, "language": "es"}
    if normalize(country) in {"costa rica", "cr"}:
        params["countryCode"] = "CR"
    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": AGENT},
                                 follow_redirects=False) as client:
        response = await client.get(GEO_URL, params=params)
        response.raise_for_status()
        matches = response.json().get("results") or []
    if not matches:
        return None
    first = matches[0]
    return {
        "name": first.get("name", city),
        "country": first.get("country", country),
        "latitude": float(first["latitude"]),
        "longitude": float(first["longitude"]),
    }


async def current_weather(city: str, country: str = "") -> dict[str, Any]:
    if not isinstance(city, str) or not city.strip() or len(city) > 80:
        return {"error": "Indica una ciudad para consultar el clima."}
    try:
        coordinates = await _resolve_location(city.strip(), country)
        if coordinates is None:
            return {"error": "No encontré esa ciudad en el servicio geográfico. Comprueba su nombre."}
        async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": AGENT},
                                     follow_redirects=False) as client:
            response = await client.get(WEATHER_URL, params={
                "latitude": coordinates["latitude"],
                "longitude": coordinates["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "timezone": "auto",
            })
            response.raise_for_status()
            result = response.json()
        current = result.get("current") or {}
        if "temperature_2m" not in current:
            return {"error": "El proveedor meteorológico no devolvió mediciones actuales."}
        code = int(current.get("weather_code", -1))
        return {
            "city": coordinates["name"], "country": coordinates["country"],
            "temperature_c": current["temperature_2m"],
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_kmh": current.get("wind_speed_10m"),
            "condition": CONDITIONS.get(code, "condiciones variables"),
            "observed_at": current.get("time", "hora no disponible"),
            "timezone": result.get("timezone", "zona local"),
            "source": "Open-Meteo Forecast API",
            "source_url": "https://open-meteo.com/en/docs",
            "location_is_current": False,
        }
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, TypeError):
        return {"error": "No pude obtener datos meteorológicos actualizados. Puedes continuar usando NYX sin Internet."}


def weather_response(data: dict[str, Any]) -> str:
    if data.get("error"):
        return str(data["error"])
    bits = [f"En {data['city']}, {data['country']}, el clima registrado a las "
            f"{data['observed_at']} ({data['timezone']}) es de "
            f"{data['temperature_c']} °C, con {data['condition']}."]
    if data.get("feels_like_c") is not None:
        bits.append(f"Sensación térmica: {data['feels_like_c']} °C.")
    if data.get("humidity_pct") is not None:
        bits.append(f"Humedad relativa: {data['humidity_pct']} %.")
    if data.get("precipitation_mm") is not None:
        bits.append(f"Precipitación actual: {data['precipitation_mm']} mm.")
    if data.get("wind_kmh") is not None:
        bits.append(f"Viento: {data['wind_kmh']} km/h.")
    bits.append("Fuente: Open-Meteo (https://open-meteo.com/en/docs). "
                "La localización corresponde a la ciudad consultada, no a tu posición GPS.")
    return " ".join(bits)


def extract_wiki_query(text: str) -> str | None:
    match = re.match(r"^\s*(?:(?:nyx\s*[:,]\s*)?)"
                     r"(?:busca|investiga|consulta)\s+(?:en\s+(?:internet|la\s+red|wikipedia)\s+)"
                     r"(?:sobre\s+|acerca\s+de\s+)?(.+?)\s*[?!.]?\s*$",
                     text, re.I | re.S)
    return match.group(1).strip()[:130] if match else None


async def search_encyclopedia(query: str) -> dict:
    """Búsqueda pública sin exfiltrar el historial; contenido no confiable."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False,
                                     headers={"User-Agent": AGENT}) as client:
            response = await client.get(WIKI_URL, params={
                "action": "query", "list": "search", "srsearch": query[:130],
                "srlimit": 3, "format": "json", "utf8": 1,
            })
            response.raise_for_status()
            results = response.json().get("query", {}).get("search", [])
        return {"results": [
            {"title": str(item.get("title", ""))[:120],
             "url": "https://es.wikipedia.org/wiki/" + quote(str(item.get("title", "")).replace(" ", "_"))}
            for item in results[:3]
        ], "provider": "Wikipedia"}
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        return {"error": "La búsqueda pública no está disponible en este momento."}


def encyclopedia_response(data: dict) -> str:
    if data.get("error"):
        return data["error"]
    results = data.get("results") or []
    if not results:
        return "No encontré resultados en Wikipedia. Prueba otra consulta."
    parts = ["Estos son los resultados encontrados en Wikipedia. "
             "Aún no he analizado ni verificado el contenido completo:"]
    for item in results:
        parts.append(f"• {item['title']}: {item['url']}")
    return "\n".join(parts)
