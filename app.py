import gradio as gr
from datasets import load_dataset
from transformers import pipeline
import pandas as pd

def run_evaluation():
    # Load your custom dataset
    dataset = load_dataset("JAIKRISHVK/qa_dataset")
    questions = dataset["train"]

    # Load model
    model = pipeline("text2text-generation", model="google/flan-t5-base")

    results = []
    for item in questions:
        question = item.get("Question") or item.get("question") or ""
        try:
            output = model(question, max_length=50)
            answer = output[0].get("generated_text", "").strip()
        except Exception as e:
            answer = f"Error: {e}"
        results.append({"Question": question, "Answer": answer})

    # Convert to DataFrame
    df = pd.DataFrame(results)
    # Save to CSV file
    file_path = "/tmp/generated_answers.csv"
    df.to_csv(file_path, index=False)

    return df, file_path

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## AI Question Answer Evaluation")
    submit_btn = gr.Button("RUN EVALUATION & SUBMIT ALL ANSWERS")
    output_table = gr.DataFrame()
    download_file = gr.File(label="Download CSV")

    submit_btn.click(fn=run_evaluation, outputs=[output_table, download_file])

demo.launch()
