import os
import gradio as gr
import requests
import pandas as pd
import openai

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Secure API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Smart Agent Logic ---
class SmartAgent:
    def __init__(self):
        print("SmartAgent initialized using OpenAI.")
        
    def __call__(self, question: str) -> str:
        print(f"Question received: {question[:100]}")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": question}],
                temperature=0.2,
                max_tokens=100
            )
            answer = response["choices"][0]["message"]["content"].strip()
            print(f"Answer: {answer}")
            return self.clean_answer(answer)
        except Exception as e:
            print(f"Error: {e}")
            return "ERROR"

    def clean_answer(self, answer: str) -> str:
        return answer.strip().replace("FINAL ANSWER:", "").replace("Answer:", "").strip()

# --- Evaluation and Submission Logic ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = profile.username
        print(f"Logged in as: {username}")
    else:
        return "Please log in to Hugging Face using the button above.", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        print(f"Fetched {len(questions_data)} questions.")
    except Exception as e:
        return f"Failed to fetch questions: {e}", None

    agent = SmartAgent()
    results_log = []
    answers_payload = []

    for item in questions_data:
        task_id = item.get("task_id")
        question = item.get("question")
        if not task_id or not question:
            continue
        try:
            answer = agent(question)
            answers_payload.append({"task_id": task_id, "submitted_answer": answer})
            results_log.append({"Task ID": task_id, "Question": question, "Submitted Answer": answer})
        except Exception as e:
            results_log.append({"Task ID": task_id, "Question": question, "Submitted Answer": f"ERROR: {e}"})

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload
    }

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        summary = (
            f"✅ Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score')}%\n"
            f"Correct: {result_data.get('correct_count')} / {result_data.get('total_attempted')}\n"
            f"Message: {result_data.get('message', '')}"
        )
        return summary, pd.DataFrame(results_log)
    except Exception as e:
        return f"❌ Submission failed: {e}", pd.DataFrame(results_log)

# --- UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 GAIA Smart Agent Evaluation")
    gr.Markdown(
        """
        1. Login to Hugging Face.
        2. Click "Run Evaluation" to evaluate your OpenAI-powered agent.
        3. View your score on the leaderboard (requires public repo).
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Status", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Agent Answers")

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    

    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)