import os
import gradio as gr
import requests
import inspect
import pandas as pd
from agent import AmbiguityClassifier
import json

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    """A langgraph agent that detects and classifies ambiguities in user stories."""
    def __init__(self):
        print("BasicAgent initialized.")
        self.analizar_historia = AmbiguityClassifier()
      
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        try:
            resultado = self.analizar_historia(question)
            
            # Formatear la respuesta
            respuesta = []
            if resultado["tiene_ambiguedad"]:
                respuesta.append("Se encontraron las siguientes ambigüedades:")
                
                if resultado["ambiguedad_lexica"]:
                    respuesta.append("\nAmbigüedades léxicas:")
                    for amb in resultado["ambiguedad_lexica"]:
                        respuesta.append(f"- {amb}")
                
                if resultado["ambiguedad_sintactica"]:
                    respuesta.append("\nAmbigüedades sintácticas:")
                    for amb in resultado["ambiguedad_sintactica"]:
                        respuesta.append(f"- {amb}")
                
                respuesta.append(f"\nScore de ambigüedad: {resultado['score_ambiguedad']}")
                respuesta.append("\nSugerencias de mejora:")
                for sug in resultado["sugerencias"]:
                    respuesta.append(f"- {sug}")
            else:
                respuesta.append("No se encontraron ambigüedades en la historia de usuario.")
                respuesta.append(f"Score de ambigüedad: {resultado['score_ambiguedad']}")
            
            return "\n".join(respuesta)
        except Exception as e:
            error_msg = f"Error analizando la historia: {str(e)}"
            print(error_msg)
            return error_msg

def run_and_submit_all( profile: gr.OAuthProfile | None):
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

# Inicializar el clasificador
classifier = AmbiguityClassifier()

def analyze_user_story(user_story: str) -> str:
    """Analiza una historia de usuario y retorna los resultados formateados."""
    if not user_story.strip():
        return "Por favor, ingrese una historia de usuario para analizar."
    
    # Analizar la historia
    result = classifier(user_story)
    
    # Formatear resultados
    output = []
    output.append(f"📝 Historia analizada:\n{user_story}\n")
    output.append(f"🎯 Score de ambigüedad: {result['score_ambiguedad']}")
    
    if result['ambiguedad_lexica']:
        output.append("\n📚 Ambigüedades léxicas encontradas:")
        for amb in result['ambiguedad_lexica']:
            output.append(f"• {amb}")
    
    if result['ambiguedad_sintactica']:
        output.append("\n🔍 Ambigüedades sintácticas encontradas:")
        for amb in result['ambiguedad_sintactica']:
            output.append(f"• {amb}")
    
    if result['sugerencias']:
        output.append("\n💡 Sugerencias de mejora:")
        for sug in result['sugerencias']:
            output.append(f"• {sug}")
    
    return "\n".join(output)

def analyze_multiple_stories(user_stories: str) -> str:
    """Analiza múltiples historias de usuario separadas por líneas."""
    if not user_stories.strip():
        return "Por favor, ingrese al menos una historia de usuario para analizar."
    
    stories = [s.strip() for s in user_stories.split('\n') if s.strip()]
    all_results = []
    
    for i, story in enumerate(stories, 1):
        result = classifier(story)
        story_result = {
            "historia": story,
            "score": result['score_ambiguedad'],
            "ambiguedades_lexicas": result['ambiguedad_lexica'],
            "ambiguedades_sintacticas": result['ambiguedad_sintactica'],
            "sugerencias": result['sugerencias']
        }
        all_results.append(story_result)
    
    return json.dumps(all_results, indent=2, ensure_ascii=False)

# Crear la interfaz
with gr.Blocks(title="Detector de Ambigüedades en Historias de Usuario") as demo:
    gr.Markdown("""
    # 🔍 Detector de Ambigüedades en Historias de Usuario
    
    Esta herramienta analiza historias de usuario en busca de ambigüedades léxicas y sintácticas, 
    proporcionando sugerencias para mejorarlas.
    
    ## 📝 Instrucciones:
    1. Ingrese una historia de usuario en el campo de texto
    2. Haga clic en "Analizar"
    3. Revise los resultados y las sugerencias de mejora
    """)
    
    with gr.Tab("Análisis Individual"):
        input_text = gr.Textbox(
            label="Historia de Usuario",
            placeholder="Como usuario quiero...",
            lines=3
        )
        analyze_btn = gr.Button("Analizar")
        output = gr.Textbox(
            label="Resultados del Análisis",
            lines=10
        )
        analyze_btn.click(
            analyze_user_story,
            inputs=[input_text],
            outputs=[output]
        )
    
    with gr.Tab("Análisis Múltiple"):
        input_stories = gr.Textbox(
            label="Historias de Usuario (una por línea)",
            placeholder="Como usuario quiero...\nComo administrador necesito...",
            lines=5
        )
        analyze_multi_btn = gr.Button("Analizar Todas")
        output_json = gr.JSON(label="Resultados del Análisis")
        analyze_multi_btn.click(
            analyze_multiple_stories,
            inputs=[input_stories],
            outputs=[output_json]
        )
    
    gr.Markdown("""
    ## 🚀 Ejemplos de Uso
    
    Pruebe con estas historias de usuario:
    - Como usuario quiero un sistema rápido y eficiente para gestionar mis tareas
    - El sistema debe permitir exportar varios tipos de archivos
    - Como administrador necesito acceder fácilmente a los reportes
    """)

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
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