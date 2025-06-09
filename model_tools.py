# model_tools.py

import ollama
import requests
from bs4 import BeautifulSoup

# ---- LLM Task Extractor ----

def extract_task(user_input: str) -> str:
    """
    Use local Ollama LLM to classify user query into Hugging Face task.
    """
    prompt = f"""
You are an AI agent helping a developer select the right ML model.
Given this request: "{user_input}"

Reply with only the corresponding Hugging Face task like:
- text-classification
- summarization
- translation
- image-classification
- etc.
Only reply with the task name, and nothing else.
    """

    response = ollama.chat(
        model="mistral",  # Replace with llama3, phi3, etc. if needed
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content'].strip().lower()

# ---- Hugging Face Scraper ----

def scrape_huggingface_models(task: str, max_results=5) -> list[dict]:
    """
    Scrapes Hugging Face for top models for a given task.
    """
    url = f"https://huggingface.co/models?pipeline_tag={task}&sort=downloads"

    try:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        model_cards = soup.find_all("article", class_="model-card")[:max_results]

        results = []
        for card in model_cards:
            name_tag = card.find("a", class_="model-link")
            model_name = name_tag.text.strip() if name_tag else "unknown"

            task_div = card.find("div", class_="task-tag")
            task_name = task_div.text.strip() if task_div else task

            arch = "encoder-decoder" if "bart" in model_name.lower() or "t5" in model_name.lower() else "unknown"

            results.append({
                "model_name": model_name,
                "task": task_name,
                "architecture": arch
            })

        return results

    except Exception as e:
        print(f"Scraping error: {e}")
        return []
