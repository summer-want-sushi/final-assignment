import os
import logging

import gradio as gr
import requests
import pandas as pd
import openai
from openai import OpenAI

from smolagents import CodeAgent, DuckDuckGoSearchTool, tool
from smolagents.models import OpenAIServerModel

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_API_URL   = "https://agents-course-unit4-scoring.hf.space"
OPENAI_MODEL_ID   = os.getenv("OPENAI_MODEL_ID", "gpt-4.1")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your Space secrets.")

# --- Configure OpenAI SDK (for tools if needed) ---
openai.api_key = "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"
client = OpenAI()

# --- Tools ---

@tool
def summarize_query(query: str) -> str:
    """
    Reframes an unclear search query to improve relevance.

    Args:
        query (str): The original search query.

    Returns:
        str: A concise, improved version.
    """
    return f"Summarize and reframe: {query}"

@tool
def wikipedia_search(page: str) -> str:
    """
    Fetches the summary extract of an English Wikipedia page.

    Args:
        page (str): e.g. 'Mercedes_Sosa_discography'

    Returns:
        str: The page’s extract text.
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("extract", "")
    except Exception as e:
        logger.exception("Wikipedia lookup failed")
        return f"Wikipedia error: {e}"

search_tool    = DuckDuckGoSearchTool()
wiki_tool      = wikipedia_search
summarize_tool = summarize_query

# --- ReACT Prompt ---

instruction_prompt = """
You are a ReACT agent with three tools: 
 • DuckDuckGoSearchTool(query: str)
 • wikipedia_search(page: str)
 • summarize_query(query: str)

Internally, for each question:
1. Thought: decide which tool to call.
2. Action: call the chosen tool.
3. Observation: record the result.
4. If empty/irrelevant:
   Thought: retry with summarize_query + DuckDuckGoSearchTool.
   Record new Observation.
5. Thought: integrate observations.

Finally, output your answer with the following template: 
FINAL ANSWER: [YOUR FINAL ANSWER]. 
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

"""

# --- Build the Agent with OpenAIServerModel ---

model = OpenAIServerModel(
    model_id=OPENAI_MODEL_ID,
    api_key=OPENAI_API_KEY
)

smart_agent = CodeAgent(
    tools=[search_tool, wiki_tool, summarize_tool],
    model=model
)

# --- Gradio Wrapper ---

class BasicAgent:
    def __init__(self):
        logger.info("Initialized SmolAgent with OpenAI GPT-4.1")

    def __call__(self, question: str) -> str:
        if not question.strip():
            return "AGENT ERROR: empty question"
        prompt = instruction_prompt.strip() + "\n\nQUESTION: " + question.strip()
        try:
            return smart_agent.run(prompt)
        except Exception as e:
            logger.exception("Agent run error")
            return f"AGENT ERROR: {e}"

# --- Submission Logic ---

def run_and_submit_all(profile: gr.OAuthProfile | None):
    if not profile:
        return "Please log in to Hugging Face.", None

    username   = profile.username
    space_id   = os.getenv("SPACE_ID", "")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    agent      = BasicAgent()

    # fetch
    try:
        resp = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15)
        resp.raise_for_status()
        questions = resp.json() or []
    except Exception as e:
        logger.exception("Failed fetch")
        return f"Error fetching questions: {e}", None

    logs, payload = [], []
    for item in questions:
        tid = item.get("task_id")
        q   = item.get("question")
        if not tid or not q:
            continue
        ans = agent(q)
        logs.append({"Task ID": tid, "Question": q, "Submitted Answer": ans})
        payload.append({"task_id": tid, "submitted_answer": ans})

    if not payload:
        return "Agent did not produce any answers.", pd.DataFrame(logs)

    # submit
    try:
        post = requests.post(
            f"{DEFAULT_API_URL}/submit",
            json={"username": username, "agent_code": agent_code, "answers": payload},
            timeout=60
        )
        post.raise_for_status()
        result = post.json()
        status = (
            f"Submission Successful!\n"
            f"User: {result.get('username')}\n"
            f"Score: {result.get('score','N/A')}%\n"
            f"({result.get('correct_count','?')}/"
            f"{result.get('total_attempted','?')})\n"
            f"Message: {result.get('message','')}"
        )
        return status, pd.DataFrame(logs)
    except Exception as e:
        logger.exception("Submit failed")
        return f"Submission Failed: {e}", pd.DataFrame(logs)

# --- Gradio App ---

with gr.Blocks() as demo:
    gr.Markdown("# SmolAgent GAIA Runner 🚀")
    gr.Markdown("""
**Instructions:**  
1. Clone this space.  
2. In Settings → Secrets, add `OPENAI_API_KEY` and (optionally) `OPENAI_MODEL_ID`.  
3. Log in to Hugging Face.  
4. Click **Run Evaluation & Submit All Answers**.
""")
    gr.LoginButton()
    btn = gr.Button("Run Evaluation & Submit All Answers")
    out_status = gr.Textbox(label="Status", lines=5, interactive=False)
    out_table  = gr.DataFrame(label="Questions & Answers", wrap=True)
    btn.click(run_and_submit_all, outputs=[out_status, out_table])

if __name__ == "__main__":
    demo.launch(debug=True, share=False)
