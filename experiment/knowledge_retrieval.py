"""Computer-science retrieval shared by the MCP server and tests."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SmartRouterResearch/0.1 (MMLU-Pro open-book experiment)"


def search_wikipedia(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Return short Wikipedia extracts for the best matching pages."""
    params = urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": query,
        "gsrlimit": max(1, min(limit, 5)), "prop": "extracts|info",
        "exintro": 1, "explaintext": 1, "inprop": "url", "format": "json",
        "formatversion": 2,
    })
    request = urllib.request.Request(
        f"{WIKIPEDIA_API}?{params}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.loads(response.read())
    pages = body.get("query", {}).get("pages", [])
    pages.sort(key=lambda page: page.get("index", 10**9))
    return [{
        "title": page.get("title", ""),
        "url": page.get("fullurl", ""),
        "text": page.get("extract", "")[:1800],
    } for page in pages[:limit] if page.get("extract")]


def format_context(passages: list[dict[str, str]]) -> str:
    if not passages:
        return "No relevant reference material was found."
    return "\n\n".join(
        f"[{i}] {item['title']} ({item['url']})\n{item['text']}"
        for i, item in enumerate(passages, 1)
    )
