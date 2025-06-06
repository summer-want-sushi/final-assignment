# tools/langchain_tools.py (Updated)
"""
LangChain-compatible tool wrappers for our existing tools
"""

from langchain_core.tools import tool
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables FIRST, before any tool imports
load_dotenv()

from .multimodal_tools import MultimodalTools, analyze_transcript as _analyze_transcript
from .search_tools import SearchTools
from .math_tools import MathTools
from .youtube_tools import YouTubeTools

# Initialize tool instances (now env vars are available)
multimodal_tools = MultimodalTools()
search_tools = SearchTools()
youtube_tools = YouTubeTools()

# Rest of the file remains the same...
@tool
def extract_text(image_path: str) -> str:
    """Extract text from an image using OCR"""
    return multimodal_tools.extract_text_from_image(image_path)

@tool
def analyze_image_tool(image_path: str, question: str = "Describe this image in detail") -> str:
    """Analyze an image and answer questions about it"""
    return multimodal_tools.analyze_image(image_path, question)

@tool
def analyze_audio_tool(transcript: str, question: str = "Summarize this audio content") -> str:
    """Analyze audio content via transcript"""
    return multimodal_tools.analyze_audio_transcript(transcript, question)

@tool
def search_tool(query: str, max_results: int = 5) -> str:
    """Search the web for information"""
    results = search_tools.search(query, max_results)
    if not results:
        return "No search results found"
    
    # Format results for the LLM
    formatted_results = []
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        content = result.get('content', 'No content')
        url = result.get('url', 'No URL')
        formatted_results.append(f"{i}. {title}\n{content[:200]}...\nSource: {url}\n")
    
    return "\n".join(formatted_results)

@tool
def extract_youtube_transcript(url: str, language_code: str = 'en') -> str:
    """Extract transcript/captions from a YouTube video"""
    captions = youtube_tools.get_captions(url, language_code)
    if captions:
        return captions
    else:
        return "No captions available for this video"

@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return MathTools.add(a, b)

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers"""
    return MathTools.subtract(a, b)

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return MathTools.multiply(a, b)

@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers"""
    result = MathTools.divide(a, b)
    return str(result)

@tool
def get_youtube_info(url: str) -> str:
    """Get information about a YouTube video"""
    info = youtube_tools.get_video_info(url)
    if info:
        return f"Title: {info.get('title', 'Unknown')}\nAuthor: {info.get('author', 'Unknown')}\nDuration: {info.get('length', 0)} seconds\nViews: {info.get('views', 0):,}"
    else:
        return "Could not retrieve video information"

@tool
def calculate_expression(expression: str) -> str:
    """Calculate a mathematical expression safely"""
    from .math_tools import calculate_expression as calc_expr
    return str(calc_expr(expression))

@tool
def factorial(n: int) -> str:
    """Calculate factorial of a number"""
    result = MathTools.factorial(n)
    return str(result)

@tool
def square_root(n: float) -> str:
    """Calculate square root of a number"""
    result = MathTools.square_root(n)
    return str(result)

@tool
def percentage(part: float, whole: float) -> str:
    """Calculate percentage"""
    result = MathTools.percentage(part, whole)
    return str(result)

@tool
def average(numbers: str) -> str:
    """Calculate average of numbers (provide as comma-separated string)"""
    try:
        number_list = [float(x.strip()) for x in numbers.split(',')]
        result = MathTools.average(number_list)
        return str(result)
    except Exception as e:
        return f"Error parsing numbers: {str(e)}"
