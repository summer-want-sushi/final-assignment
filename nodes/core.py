# nodes/core.py
from states.state import AgentState
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Using OpenAI-compatible API for OpenRouter
from tools.langchain_tools import (
    extract_text, 
    analyze_image_tool, 
    analyze_audio_tool,
    add, 
    subtract, 
    multiply, 
    divide,
    search_tool,
    extract_youtube_transcript,
    get_youtube_info,
    calculate_expression,
    factorial,
    square_root,
    percentage,
    average
)

load_dotenv()

# Read your API key from the environment variable
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# Initialize OpenRouter ChatOpenAI with OpenRouter-specific configuration
chat = ChatOpenAI(
    model="google/gemini-2.5-flash-preview-05-20",  # Free multimodal model
    # Alternative models you can use:
    # model="mistralai/mistral-7b-instruct:free",  # Fast, free text model
    # model="google/gemma-2-9b-it:free",  # Google's free model
    # model="qwen/qwen-2.5-72b-instruct:free",  # High-quality free model
    
    temperature=0,
    max_retries=2,
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    default_headers={
        "HTTP-Referer": "https://your-app.com",  # Optional: for analytics
        "X-Title": "LangGraph Agent",  # Optional: for analytics
    }
)

# Core tools list (matching original structure)
tools = [
    extract_text,
    analyze_image_tool,
    analyze_audio_tool,
    extract_youtube_transcript,
    add,
    subtract,
    multiply,
    divide,
    search_tool
]

# Extended tools list (if you want more capabilities)
extended_tools = tools + [
    get_youtube_info,
    calculate_expression,
    factorial,
    square_root,
    percentage,
    average
]

# Use core tools by default (matching original), but you can switch to extended_tools
chat_with_tools = chat.bind_tools(tools)

def assistant(state: AgentState):
    """
    Assistant node - maintains the exact same system prompt for evaluation compatibility
    """
    sys_msg = (
        "You are a helpful assistant with access to tools. Understand user requests accurately. "
        "Use your tools when needed to answer effectively. Strictly follow all user instructions and constraints. "
        "Pay attention: your output needs to contain only the final answer without any reasoning since it will be "
        "strictly evaluated against a dataset which contains only the specific response. "
        "Your final output needs to be just the string or integer containing the answer, not an array or technical stuff."
    )
    
    return {
        "messages": [chat_with_tools.invoke([sys_msg] + state["messages"])]
    }
