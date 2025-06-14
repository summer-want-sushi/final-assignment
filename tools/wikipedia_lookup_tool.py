import requests


def wikipedia_lookup(query: str) -> str:
    """Return the summary for the given Wikipedia query."""
    try:
        lowered = query.lower()
        if "infobox" in lowered or "table" in lowered:
            return "Structured infobox/table parsing not implemented in this version."

        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        summary = data.get("extract", "").strip()
        if not summary:
            return "No summary found."
        return summary
    except Exception as e:
        return f"Tool error: {e}"
