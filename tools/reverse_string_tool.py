def reverse_string(prompt: str) -> str:
    """Replace all variations of the word 'left' (case-insensitive) with 'tfel'."""
    try:
        return (
            prompt.replace("left", "tfel")
            .replace("Left", "Tfel")
            .replace("LEFT", "TFEL")
        )
    except Exception as e:
        return f"Tool error: {e}"

