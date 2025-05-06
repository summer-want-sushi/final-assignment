# tools.py
import math
import requests
import wikipedia  # using Wikipedia API for a search tool

# Tool 1: Wikipedia Search
def tool_search(query: str) -> str:
    """Search the web (Wikipedia API) for the query and return a summary of results."""
    try:
        # Use wikipedia library to get a summary
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception as e:
        return f"(Search tool failed: {e})"

# Tool 2: Calculator
def tool_calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result as a string."""
    try:
        result = eval(expression, {"__builtins__": None}, {"sqrt": math.sqrt, "pow": math.pow})
        return str(result)
    except Exception as e:
        return f"(Calculation error: {e})"

# Tool 3: File loader (for image or text files from GAIA, if needed)
def tool_load_file(task_id: str) -> str:
    """Fetch the file for a given task (if any) and return its content or a description."""
    url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"(File download error: {e})"
    # Determine content type
    content_type = resp.headers.get("Content-Type", "")
    if "image" in content_type:
        # An image was received (could run image captioning model here)
        return "[Image received from task]"
    elif "text" in content_type or "json" in content_type:
        text_data = resp.text[:500]  # take first 500 chars to avoid huge text
        return f"[File content snippet: {text_data}]"
    else:
        return "(Unknown file type or binary data received)"