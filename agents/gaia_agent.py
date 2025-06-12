from tools.reverse_string_tool import reverse_string
from tools.filter_vegetables_tool import filter_vegetables

class GAIAAgent:
    def __init__(self):
        print("GAIAAgent initialized.")

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        lower_q = question.lower()
        if "reverse" in lower_q or "opposite" in lower_q:
            print("GAIAAgent using reverse_string tool.")
            return reverse_string(question)
        if "vegetable" in lower_q or "veggie" in lower_q:
            print("GAIAAgent using filter_vegetables tool.")
            return filter_vegetables(question)
        fixed_answer = "This is a default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer
