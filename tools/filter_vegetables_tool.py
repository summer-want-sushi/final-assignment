def filter_vegetables(prompt: str) -> str:
    """Extract only vegetables from a comma-separated list.

    Extracts only vegetables from a comma-separated grocery list string.
    It uses a trusted allow list of vegetables to exclude items that are
    technically botanical fruits or seeds.
    The result should:
    - Be a comma-separated string of vegetable names.
    - Be sorted alphabetically.
    - Remove duplicates and extra spaces.
    - Only include true vegetables (e.g., celery, broccoli, lettuce, zucchini, green beans).
    - Exclude common fruits or seeds that might appear in a list.
    """
    try:
        # Normalize spacing and plurals to increase matching consistency
        items = {
            i.strip().lower().rstrip('s')
            for i in prompt.split(',')
            if i.strip()
        }

        # Trusted list of vegetables to include
        allowed = {
            'celery', 'lettuce', 'broccoli', 'zucchini',
            'green beans', 'basil'
        }

        vegetables = [item for item in items if item in allowed]
        if not vegetables:
            return "No valid vegetables found."
        return ', '.join(sorted(set(vegetables)))
    except Exception as e:
        return f"Tool error: {e}"
