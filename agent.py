# agent.py
import requests
import os
import re
from typing import List, Dict

def get_questions() -> List[Dict]:
    response = requests.get("https://gaia-course-api.huggingface.co/questions")
    return response.json()

def get_file(task_id: str) -> str:
    response = requests.get(f"https://gaia-course-api.huggingface.co/files/{task_id}")
    file_path = f"{task_id}.txt"
    with open(file_path, "w") as f:
        f.write(response.text)
    return file_path

def reasoning_prompt(question: str) -> str:
    """
    Chain-of-thought + Zero-shot reasoning
    """
    # مثال بسيط يمكن تحسينه لاحقًا
    if "capital" in question.lower():
        if "france" in question.lower():
            return "Paris"
        if "germany" in question.lower():
            return "Berlin"
    if "sum" in question.lower():
        numbers = re.findall(r"\\d+", question)
        if len(numbers) >= 2:
            return str(sum(map(int, numbers)))
    return "I don't know"

def solve_question(question: Dict) -> str:
    q_text = question.get("question", "")
    return reasoning_prompt(q_text)

def run_agent() -> List[Dict]:
    questions = get_questions()
    answers = []
    for q in questions:
        ans = solve_question(q)
        answers.append({
            "task_id": q["task_id"],
            "submitted_answer": ans.strip()
        })
    return answers
