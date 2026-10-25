"""Consultas públicas de NYX 0.5: divisas y búsqueda explícita por Internet.

No acepta URL arbitrarias ni ejecuta texto remoto. La salida de proveedores
externos es información no confiable, no instrucciones para el asistente.
Las solicitudes externas sólo se realizan cuando el usuario las solicita.
"""
from __future__ import annotations

import html
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from xml.etree import ElementTree
from typing import Any
from urllib.parse import urlparse

import httpx

from . import network

RATES_API = "https://api.frankfurter.dev/v2/rate"
BING_RSS_API = "https://www.bing.com/search"
MAX_QUERY = 135
CURRENCY_WORDS = {
    "usd": "USD", "dolar": "USD", "dolares": "USD",
    "crc": "CRC", "colon": "CRC", "colones": "CRC",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP", "libra": "GBP", "libras": "GBP",
    "mxn": "MXN", "peso mexicano": "MXN",
    "jpy": "JPY", "yen": "JPY",
    "cad": "CAD", "dolar canadiense": "CAD",
    "brl": "BRL", "real brasileno": "BRL",
}
CODE = re.compile(r"^[A-Z]{3}$")


def norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def extract_currency_request(text: str) -> tuple[str, str, Decimal | None] | None:
    n = norm(text)[:320]
    if not re.search(r"\b(?:tipo de cambio|tasa de cambio|cotizacion|convertir|conversion|"
                     r"cuanto (?:vale|valen|equivale|equivalen|es|son)|dolar(?:es)?|euro(?:s)?|divisas?|usd|crc|eur)\b", n):
        return None
    # Pide tasas, no cualquier mención casual de una divisa.
    if not re.search(r"\b(?:cambio|cotizacion|convertir|conversion|precio|vale|equivale|"
                     r"cuanto|a colones|en colones|a dolares|en dolares|a euros|a crc|a usd|a eur)\b", n):
        return None
    names = []
    for key, currency in CURRENCY_WORDS.items():
        for m in re.finditer(r"(?<![a-z])" + re.escape(key) + r"(?![a-z])", n):
            names.append((m.start(), currency))
    names.sort()
    distinct = []
    for _, currency in names:
        if currency not in distinct:
            distinct.append(currency)
    # Consultas sin par explícito usan USD/CRC: referencia habitual de CR.
    base, quote = ("USD", "CRC") if not distinct else (
        (distinct[0], distinct[1]) if len(distinct) >= 2 else
        ("USD", "CRC") if distinct[0] in {"USD", "CRC"} else
        (distinct[0], "CRC")
    )
    # 'colones a dólares' invierte el par explícito.
    amount = None
    m = re.search(r"\b(?:convierte|convertir|cuanto (?:vale|valen|equivale|equivalen|es|son))\s+"
                  r"(?:son\s+)?([0-9]+(?:[.,][0-9]+)?)", n)
    if m:
        try:
            amount = Decimal(m.group(1).replace(",", "."))
            if amount <= 0 or amount > 1_000_000_000:
                amount = None
        except InvalidOperation:
            pass
    return base, quote, amount


async def exchange_rate(base: str, quote: str) -> dict[str, Any]:
    if not CODE.fullmatch(base) or not CODE.fullmatch(quote):
        return {"error": "Indica códigos monetarios ISO válidos."}
    if base == quote:
        return {"base": base, "quote": quote, "rate": 1.0,
                "date": "sin actualización externa", "source": "Identidad monetaria"}
    try:
        async with httpx.AsyncClient(timeout=network.TIMEOUT,
                follow_redirects=False, headers={"User-Agent": network.AGENT}) as client:
            r = await client.get(f"{RATES_API}/{base.lower()}/{quote.lower()}")
            r.raise_for_status()
            value = r.json()
        rate = Decimal(str(value["rate"]))
        if not rate.is_finite() or rate <= 0:
            raise ValueError("Tasa inválida")
        return {"base": base, "quote": quote, "rate": float(rate),
                "date": str(value["date"])[:20],
                "source": "Frankfurter v2 (tipos de referencia de proveedores)",
                "url": "https://frankfurter.dev/"}
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, TypeError,
            InvalidOperation):
        return {"error": "No pude verificar el tipo de cambio en la fuente pública. "
                "Comprueba la conexión o los códigos de moneda."}


def exchange_response(data: dict[str, Any], amount: Decimal | None = None) -> str:
    if data.get("error"):
        return str(data["error"])
    base, quote, rate = data["base"], data["quote"], Decimal(str(data["rate"]))
    answer = [f"Tipo de cambio de referencia: 1 {base} = {rate:,.4f} {quote} "
              f"(dato de {data['date']})."]
    if amount is not None:
        answer.append(f"{amount:,.2f} {base} equivalen aproximadamente a "
                      f"{(amount * rate):,.2f} {quote}.")
    answer.append(f"Fuente: {data['source']} "
                  "(https://frankfurter.dev/). No es necesariamente la tasa de "
                  "compra/venta de un banco ni una cotización intradía.")
    return " ".join(answer)


def extract_web_query(text: str) -> str | None:
    # No navegar por defecto: evita enviar mensajes personales a buscadores.
    cleaned = text.strip()
    patterns = [
        r"^(?:nyx[,!:]?\s*)?(?:busca|buscar|investiga|investigar|consulta|consultar|"
        r"averigua)\s+(?:en\s+)?(?:internet|la\s+web|la\s+red|la\s+web\s+publica)"
        r"\s*(?:sobre|acerca de|de)?\s*(.+)$",
        r"^(?:nyx[,!:]?\s*)?(?:busca|investiga|investigar|consulta|averigua)\s+"
        r"(?:informacion|información|fuentes|noticias)\s+(?:sobre|de)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned, flags=re.I | re.S)
        if match:
            query = re.sub(r"\s+", " ", match.group(1)).strip(" .?!¡¿")
            return query[:MAX_QUERY] if len(query) >= 2 else None
    return None


def _clean_excerpt(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", value))).strip()[:650]


async def _bing_search(query: str) -> list[dict[str, str]]:
    async with httpx.AsyncClient(timeout=network.TIMEOUT, follow_redirects=False,
                                headers={"User-Agent": network.AGENT}) as client:
        r = await client.get(BING_RSS_API, params={"q": query, "format": "rss"})
        r.raise_for_status()
        if len(r.content) > 500_000:
            raise ValueError("Documento demasiado grande")
        root = ElementTree.fromstring(r.content)
    items = []
    for item in root.findall(".//item")[:5]:
        title = _clean_excerpt(item.findtext("title") or "")
        link = (item.findtext("link") or "").strip()
        excerpt = _clean_excerpt(item.findtext("description") or "")
        parts = urlparse(link)
        if parts.scheme == "https" and parts.hostname and title:
            items.append({"title": title, "url": link[:500], "excerpt": excerpt})
    return items


async def _wikipedia_excerpts(query: str) -> list[dict[str, str]]:
    results = await network.search_encyclopedia(query)
    if results.get("error"):
        return []
    listings = results.get("results", [])[:3]
    if not listings:
        return []
    names = "|".join(item["title"] for item in listings)
    async with httpx.AsyncClient(timeout=network.TIMEOUT,
                                 follow_redirects=False,
                                 headers={"User-Agent": network.AGENT}) as client:
        r = await client.get(network.WIKI_URL, params={
            "action": "query", "prop": "extracts", "exintro": 1,
            "explaintext": 1, "exchars": 900, "format": "json", "titles": names,
        })
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    by_title = {p.get("title"): _clean_excerpt(p.get("extract", ""))
                for p in pages.values()}
    return [{"title": item["title"], "url": item["url"],
             "excerpt": by_title.get(item["title"], "")} for item in listings]


async def web_search(query: str, wikipedia_only: bool = False) -> dict[str, Any]:
    query = re.sub(r"\s+", " ", query).strip()[:MAX_QUERY]
    if len(query) < 2:
        return {"error": "Indica qué tema quieres investigar."}
    if not wikipedia_only:
        try:
            items = await _bing_search(query)
            if items:
                return {"results": items, "provider": "Bing RSS (resultados públicos)",
                        "query": query}
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError,
                ElementTree.ParseError):
            pass
    try:
        items = await _wikipedia_excerpts(query)
        if items:
            return {"results": items, "provider": "Wikipedia (resumen público)",
                    "query": query}
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError,
            KeyError, TypeError):
        pass
    return {"error": "No pude recuperar resultados verificables de las fuentes públicas. "
            "Comprueba el acceso a Internet y vuelve a intentarlo."}


def search_response(data: dict[str, Any]) -> str:
    if data.get("error"):
        return str(data["error"])
    results = data.get("results", [])
    if not results:
        return "No encontré resultados verificables para esa consulta."
    lines = [f"Consulté fuentes públicas sobre «{data['query']}». "
             f"Proveedor: {data['provider']}. Estos son los resultados; "
             "los extractos no sustituyen la revisión de las fuentes originales:"]
    for item in results[:5]:
        lines.append(f"• {item['title']}: {item.get('excerpt') or 'Sin extracto disponible.'}\n"
                     f"  Fuente: {item['url']}")
    return "\n".join(lines)
