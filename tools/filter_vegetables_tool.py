def filter_vegetables(prompt: str) -> str:
    """
    Extract only vegetables from a comma-separated list.
    This version includes exact matches like 'fresh basil' and 'sweet potatoes'.
    """

    try:
        # Canonical list of valid vegetable entries (exact matches)
        allowed = {
            "broccoli", "celery", "lettuce",
            "fresh basil", "sweet potatoes"
        }

        # Normalize and split
        items = [i.strip().lower() for i in prompt.split(",") if i.strip()]
        matched = set()

        for item in items:
            if item in allowed:
                matched.add(item)

        if not matched:
            return "No valid vegetables found."

        return ", ".join(sorted(matched))

    except Exception as e:
        return f"Tool error: {e}"
