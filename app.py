import os
import gradio as gr
import requests
import inspect
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your LangGraph agent
from graph.graph_builder import graph
from langchain_core.messages import HumanMessage

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Your LangGraph Agent Definition ---
# ----- THIS IS WHERE YOU BUILD YOUR AGENT ------
class BasicAgent:
    def __init__(self):
        """Initialize the LangGraph agent"""
        print("LangGraph Agent initialized with multimodal, search, math, and YouTube tools.")
        
        # Verify environment variables
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # The graph is already compiled and ready to use
        self.graph = graph
        print("✅ Agent ready with tools: multimodal, search, math, YouTube")
    
    def __call__(self, question: str) -> str:
        """
        Process a question using the LangGraph agent and return just the answer
        
        Args:
            question: The question to answer
            
        Returns:
            str: The final answer (formatted for evaluation)
        """
        print(f"🤖 Processing question: {question[:50]}...")
        
        try:
            # Create initial state with the question
            initial_state = {"messages": [HumanMessage(content=question)]}
            
            # Run the LangGraph agent
            result = self.graph.invoke(initial_state)
            
            # Extract the final message content
            final_message = result["messages"][-1]
            answer = final_message.content
            
            # Clean up the answer for evaluation (remove any extra formatting)
            # The evaluation system expects just the answer, no explanations
            if isinstance(answer, str):
                answer = answer.strip()
                
                # Remove common prefixes that might interfere with evaluation
                prefixes_to_remove = [
                    "The answer is: ",
                    "Answer: ",
                    "The result is: ",
                    "Result: ",
                    "The final answer is: ",
                ]
                
                for prefix in prefixes_to_remove:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()
                        break
            
            print(f"✅ Agent answer: {answer}")
            return answer
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

# Keep the rest of the file unchanged (run_and_submit_all function and Gradio interface)
def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent (using your LangGraph agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    
    # In the case of an app running as a hugging Face space, this link points toward your codebase
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
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
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
    gr.Markdown("# LangGraph Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        This space uses a LangGraph agent with multimodal, search, math, and YouTube tools powered by OpenRouter.
        
        1.  Log in to your Hugging Face account using the button below.
        2.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        **Agent Capabilities:**
        - 🎨 **Multimodal**: Analyze images, extract text (OCR), process audio transcripts
        - 🔍 **Search**: Web search using multiple providers (DuckDuckGo, Tavily, SerpAPI)
        - 🧮 **Math**: Basic arithmetic, complex calculations, percentages, factorials
        - 📺 **YouTube**: Extract captions, get video information
        
        ---
        **Note:** Processing all questions may take some time as the agent carefully analyzes each question and uses appropriate tools.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for LangGraph Agent Evaluation...")
    demo.launch(debug=True, share=False)
