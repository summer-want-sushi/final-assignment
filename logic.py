import os
from typing import Any, Dict, List, Tuple
import pandas as pd
import requests

def fetch_all_questions() -> List[Dict[str, Any]]:
    """Fetch all questions from the GAIA benchmark API."""
    try:
        # The actual endpoint will be provided by the GAIA benchmark
        api_url = os.getenv("GAIA_API_URL", "")
        if not api_url:
            raise ValueError("GAIA_API_URL environment variable not set")
            
        response = requests.get(f"{api_url}/questions")
        response.raise_for_status()
        
        questions = response.json()
        return questions
    except Exception as e:
        raise Exception(f"Failed to fetch questions: {str(e)}")

def run_agent(agent: Any, questions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the agent on all questions and collect results.
    
    Args:
        agent: The GaiaAgent instance
        questions: List of question data from the API
        
    Returns:
        Tuple containing:
        - List of result logs for display
        - List of answer payloads for submission
    """
    results_log = []
    answers_payload = []
    
    for question in questions:
        question_id = question.get("id", "unknown")
        question_text = question.get("question", "")
        
        try:
            # Get answer from agent
            answer = agent.get_answer(question)
            
            # Log result
            result_entry = {
                "Question ID": question_id,
                "Question": question_text,
                "Answer": answer if answer else "No answer provided",
                "Status": "Success" if answer else "Failed"
            }
            results_log.append(result_entry)
            
            # Prepare submission payload if answer was generated
            if answer:
                answer_entry = {
                    "question_id": question_id,
                    "answer": answer
                }
                answers_payload.append(answer_entry)
                
        except Exception as e:
            # Log error
            result_entry = {
                "Question ID": question_id,
                "Question": question_text,
                "Answer": f"Error: {str(e)}",
                "Status": "Failed"
            }
            results_log.append(result_entry)
    
    return results_log, answers_payload

def submit_answers(submission_data: Dict[str, Any], results_log: List[Dict[str, Any]]) -> Tuple[str, pd.DataFrame]:
    """Submit answers to the GAIA benchmark API.
    
    Args:
        submission_data: Dictionary containing submission details
        results_log: List of result logs for display
        
    Returns:
        Tuple containing:
        - Status message string
        - DataFrame of results for display
    """
    try:
        # The actual endpoint will be provided by the GAIA benchmark
        api_url = os.getenv("GAIA_API_URL", "")
        if not api_url:
            raise ValueError("GAIA_API_URL environment variable not set")
            
        # Submit answers
        response = requests.post(
            f"{api_url}/submit",
            json=submission_data
        )
        response.raise_for_status()
        
        # Create DataFrame for display
        results_df = pd.DataFrame(results_log)
        
        # Return success message and results
        return "Answers submitted successfully!", results_df
        
    except Exception as e:
        # If submission fails, still show results but with error message
        results_df = pd.DataFrame(results_log)
        return f"Error submitting answers: {str(e)}", results_df 