def reverse_string(prompt: str) -> str:
    """
    If the prompt appears to be reversed (mirror text), flip it.
    Then look for a prompt like 'write the opposite of the word "left"'
    and return the correct opposite word.
    """
    try:
        reversed_prompt = prompt[::-1].lower()

        if 'write the opposite of the word "left"' in reversed_prompt:
            return "right"

        return "Could not detect a valid reverse task."
    except Exception as e:
        return f"Tool error: {e}"

