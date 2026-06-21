"""Web tools for the AI agent: live web search and page reading.

These give the model real, up-to-date knowledge instead of relying only on
its training data. No paid search API is required — DuckDuckGo is free.
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup

try:
    # `ddgs` is the maintained package name.
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback to the older name
    from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 6) -> list[dict]:
    """Search the live web with DuckDuckGo. No API key required.

    Returns a list of dicts with `title`, `url` and `snippet`.
    """
    results: list[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href") or r.get("url", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return results


def read_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a web page and return its readable text content."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SmartAI/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]
