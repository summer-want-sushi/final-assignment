import gradio as gr
from duckduckgo_search import DDGS
from transformers import pipeline
from PIL import Image
import requests
from bs4 import BeautifulSoup
import re
import torch
from io import BytesIO

# Pipelines
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Utils
def search_web(query, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        return "\n\n".join([f"**{r['title']}**\n{r['body']}\n{r['href']}" for r in results])

def explain_image(img):
    return caption_pipeline(img)[0]['generated_text']

def extract_text_from_url(url):
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        # Remove scripts/styles
        for script in soup(["script", "style"]): script.extract()
        text = soup.get_text(separator=' ')
        clean_text = re.sub(r'\s+', ' ', text)
        return clean_text[:3000]  # Limit to 3000 characters
    except Exception as e:
        return f"Failed to extract text: {str(e)}"

def summarize_url(url):
    text = extract_text_from_url(url)
    if len(text) > 100:
        summary = summarizer(text[:1024])[0]['summary_text']
        return summary
    return "Not enough text to summarize."

# Main Agent Function
def ai_agent(input_text, image=None, url=None):
    results = []

    # Process Image
    if image:
        results.append("🖼️ **Image Explanation:**\n" + explain_image(image))

    # Process URL
    if url:
        if "youtube.com" in url or "youtu.be" in url:
            results.append("📹 **Video URL detected.** Currently only summaries of page content are available.")
        results.append("🔗 **Webpage Summary:**\n" + summarize_url(url))

    # Web search for complex questions
    if input_text:
        if len(input_text.split()) > 10:  # assume complex
            web_results = search_web(input_text)
            results.append("🔍 **Web Search Results:**\n" + web_results)
        else:
            results.append("🧠 **Answer:**\n" + search_web(input_text))

    return "\n\n---\n\n".join(results)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌐🧠 Multi-Modal AI Agent (Web + Image + URL)")
    with gr.Row():
        input_text = gr.Textbox(label="Ask a Question", lines=2, placeholder="E.g. What are the latest AI trends?")
        image = gr.Image(type="pil", label="Upload an Image (optional)")
        url = gr.Textbox(label="Provide a URL (optional)", placeholder="https://example.com")
    submit = gr.Button("Get Answer")
    output = gr.Markdown()

    submit.click(fn=ai_agent, inputs=[input_text, image, url], outputs=output)

demo.launch()