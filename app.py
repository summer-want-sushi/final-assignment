import os

import agent
import gradio as gr
import logic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def run_and_submit_all(
    profile: gr.OAuthProfile | None,
) -> tuple[str, pd.DataFrame | None]:
    """Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.

    Args:
        profile: An optional gr.OAuthProfile object containing user information
            if the user is logged in. If None, the user is not logged in.

    Returns:
        tuple[str, pd.DataFrame | None]: A tuple containing:
            - A string representing the status of the run and submission process.
              This could be a success message, an error message, or a message
              indicating that no answers were produced.
            - A pandas DataFrame containing the results log. This DataFrame will
              be displayed in the Gradio interface. It can be None if an error
              occurred before the agent was run.
    """
    # 0. Get user details
    space_id = os.getenv("SPACE_ID")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)
    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    # 1. Instantiate Agent
    try:
        gaia_agent = agent.GaiaAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    # 2. Fetch Questions
    try:
        questions_data = logic.fetch_all_questions()
    except Exception as e:
        return str(e), None

    # 3. Run the Agent
    results_log, answers_payload = logic.run_agent(gaia_agent, questions_data)
    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare & Submit Answers
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
    print(
        f"Agent finished. Submitting {len(answers_payload)} answers for user '"
        f"{username}'..."
    )
    return logic.submit_answers(submission_data, results_log)


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as gaia_ui:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's 
        logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses 
        your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your 
        agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is 
        the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to 
        encourage you to develop your own, more robust solution. For instance for the 
        delay process of the submit button, a solution could be to cache the answers 
        and submit in a separate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all, inputs=None, outputs=[status_output, results_table]
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
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/"
            f"{space_id_startup}/tree/main"
        )
    else:
        print(
            "ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL "
            "cannot be determined."
        )

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    gaia_ui.launch(debug=True, share=True)
