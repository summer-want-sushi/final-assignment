# main.py
import requests
from agent import CustomAgent
from config import HF_USERNAME, QUESTIONS_ENDPOINT, SUBMIT_ENDPOINT, DEFAULT_MODEL

def get_questions():
    """Retrieve the list of evaluation questions from the GAIA Unit4 API."""
    resp = requests.get(QUESTIONS_ENDPOINT, timeout=15)
    resp.raise_for_status()
    questions = resp.json()
    if not isinstance(questions, list):
        raise ValueError("Unexpected response format for questions.")
    return questions

def submit_answers(username, answers_payload):
    """Submit the answers to the GAIA API and return the result data."""
    submission = {
        "username": username.strip(),
        "agent_code": f"https://huggingface.co/spaces/{username}/Final_Assignment_Template/tree/main",
        "answers": answers_payload
    }
    resp = requests.post(SUBMIT_ENDPOINT, json=submission, timeout=60)
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    # Initialize our custom agent (you can change model or settings if needed)
    agent = CustomAgent(model_name=DEFAULT_MODEL, use_gpu=False)
    print("Agent initialized with model:", DEFAULT_MODEL)
    # Fetch evaluation questions
    try:
        questions = get_questions()
    except Exception as e:
        print("Error fetching questions:", e)
        exit(1)
    print(f"Retrieved {len(questions)} questions for evaluation.")
    # Run the agent on each question
    answers_payload = []
    for item in questions:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or not question_text:
            continue  # skip if malformed
        print(f"\n=== Question {task_id} ===")
        print("Q:", question_text)
        try:
            ans = agent.answer(question_text)
        except Exception as err:
            ans = "(Agent failed to produce an answer)"
            print("Error during agent reasoning:", err)
        print("A:", ans)
        answers_payload.append({"task_id": task_id, "submitted_answer": ans})
    # All answers ready, submit them for scoring
    try:
        result = submit_answers(HF_USERNAME, answers_payload)
    except Exception as e:
        print("Submission failed:", e)
        exit(1)
    # Print the results
    score = result.get('score', 'N/A')
    correct = result.get('correct_count', '?')
    total = result.get('total_attempted', '?')
    message = result.get('message', '')
    print(f"\nSubmission complete! Score: {score}% ({correct}/{total} correct)")
    if message:
        print("Message from server:", message)