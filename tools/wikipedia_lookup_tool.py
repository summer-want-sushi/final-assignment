import re
import wikipedia
import wikipediaapi


def lookup_wikipedia(query: str) -> str:
    """Search Wikipedia and return a relevant paragraph or structured data."""
    try:
        if not query:
            return "No relevant Wikipedia content found."

        search_results = wikipedia.search(query)
        if not search_results:
            return "No relevant Wikipedia content found."

        page_title = search_results[0]
        try:
            page = wikipedia.page(page_title)
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0])
            except Exception:
                return "No relevant Wikipedia content found."
        except wikipedia.exceptions.PageError:
            return "No relevant Wikipedia content found."

        lowered = query.lower()
        if "infobox" in lowered or "table" in lowered:
            return "Structured infobox/table parsing not implemented in this version."


        summary = page.summary or page.content
        paragraphs = [p.strip() for p in summary.split("\n") if p.strip()]
        if not paragraphs:
            return "No relevant Wikipedia content found."

        query_words = set(re.findall(r"\w+", query.lower()))
        for para in paragraphs:
            if any(word in para.lower() for word in query_words):
                return para

        return paragraphs[0]
    except Exception:
        return "No relevant Wikipedia content found."
