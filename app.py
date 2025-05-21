import os
import gradio as gr
import requests
import inspect
import pandas as pd
from smolagents import DuckDuckGoSearchTool,GoogleSearchTool, HfApiModel, PythonInterpreterTool, VisitWebpageTool, CodeAgent,Tool, LiteLLMModel
import hashlib
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TransformersEngine
import wikipedia
from tooling import WikipediaPageFetcher,MathModelQuerer, YoutubeTranscriptFetcher, CodeModelQuerer
from langchain_community.agent_toolkits.load_tools import load_tools
import time
import torch


# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------

cache = {}



class WebSearchTool(DuckDuckGoSearchTool):
    name = "web_search_ddg"
    description = "Search the web using DuckDuckGo"
web_search_ddf = WebSearchTool()
google_search = GoogleSearchTool(provider="serper")
python_interpreter = PythonInterpreterTool(authorized_imports = [
    # standard library
    'os',                      # For file path manipulation, checking existence, deletion
    'glob',                    # Find files matching specific patterns
    'pathlib',                 # Alternative for path manipulation
    'sys',
    'math',
    'random',
    'datetime',
    'time',
    'json',
    'csv',
    're',
    'collections',
    'itertools',
    'functools',
    'io',
    'base64',
    'hashlib',
    'pathlib',
    'glob',

    # Third-Party Libraries (ensure they are installed in the execution env)
    'pandas',         # Data manipulation and analysis
    'numpy',          # Numerical operations
    'scipy',          # Scientific and technical computing (stats, optimize, etc.)
    'sklearn',        # Machine learning

])
visit_webpage_tool = VisitWebpageTool()
wiki_tool = WikipediaPageFetcher()
yt_transcript_fetcher = YoutubeTranscriptFetcher()
# math_model_querer = MathModelQuerer()
# code_model_querer = CodeModelQuerer()

# batch of tools fromm Langchain. Credits DataDiva88
lc_ddg_search = Tool.from_langchain(load_tools(["ddg-search"])[0])
lc_wikipedia = Tool.from_langchain(load_tools(["wikipedia"])[0])
lc_arxiv = Tool.from_langchain(load_tools(["arxiv"])[0])
lc_pubmed = Tool.from_langchain(load_tools(["pubmed"])[0])
lc_stackechange = Tool.from_langchain(load_tools(["stackexchange"])[0])


def load_cached_answer(question_id: str) -> str:
    if question_id in cache.keys():
        return cache[question_id]
    else:
        return None


def cache_answer(question_id: str, answer: str):
    cache[question_id] = answer


# --- Model Setup ---
#MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'  # 'meta-llama/Llama-3.2-3B-Instruct'


# "Qwen/Qwen2.5-VL-3B-Instruct"#'meta-llama/Llama-2-7b-hf'#'meta-llama/Llama-3.1-8B-Instruct'#'TinyLlama/TinyLlama-1.1B-Chat-v1.0'#'mistralai/Mistral-7B-Instruct-v0.2'#'microsoft/DialoGPT-small'# 'EleutherAI/gpt-neo-2.7B'#'distilbert/distilgpt2'#'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'#'mistralai/Mistral-7B-Instruct-v0.2'


def load_model(model_name):
    """Download and load the model and tokenizer."""
    try:
        print(f"Loading model {MODEL_NAME}...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model {MODEL_NAME} loaded successfully.")

        transformers_engine = TransformersEngine(pipeline("text-generation", model=model, tokenizer=tokenizer))

        return transformers_engine, model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# Load the model and tokenizer locally
# model, tokenizer = load_model()

#model_id = "meta-llama/Llama-3.1-8B-Instruct"  # "microsoft/phi-2"# not working out of the box"google/gemma-2-2b-it" #toobig"Qwen/Qwen1.5-7B-Chat"#working but stupid: "meta-llama/Llama-3.2-3B-Instruct"
model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", temperature=0.2, max_tokens=512)
#from smolagents import TransformersModel
# model = TransformersModel(
#     model_id=model_id,
#     max_new_tokens=256)

# model = HfApiModel()
lc_ddg_search = Tool.from_langchain(load_tools(["ddg-search"])[0])
lc_wikipedia = Tool.from_langchain(load_tools(["wikipedia"])[0])
lc_arxiv = Tool.from_langchain(load_tools(["arxiv"])[0])
lc_pubmed = Tool.from_langchain(load_tools(["pubmed"])[0])


class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.agent = CodeAgent(
            model=model,
            tools=[google_search,web_search_ddf, python_interpreter, visit_webpage_tool, wiki_tool,lc_wikipedia,lc_arxiv,lc_pubmed,lc_stackechange],
            max_steps=10,
            verbosity_level=1,
            grammar=None,
            planning_interval=3,
            add_base_tools=True,
            additional_authorized_imports=['requests', 'wikipedia', 'pandas','datetime']

        )

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        answer = self.agent.run(question)
        return answer


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:

        
        time.sleep(60)

        
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            cached = load_cached_answer(task_id)
            if cached:
                submitted_answer = cached
                print(f"Loaded cached answer for task {task_id}")
            else:
                submitted_answer = agent(question_text)
                cache_answer(task_id, submitted_answer)
                print(f"Generated and cached answer for task {task_id}")

            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
