# tools/__init__.py
"""
Tools package for AI agents
Provides multimodal, search, math, and YouTube capabilities
"""

from .multimodal_tools import MultimodalTools, analyze_image, extract_text, analyze_transcript
from .search_tools import SearchTools, search_web, search_news
from .math_tools import MathTools, add, subtract, multiply, divide, power, factorial, square_root, percentage, average, calculate_expression
from .youtube_tools import YouTubeTools, get_video_info, download_video, download_audio, get_captions, get_playlist_info

__all__ = [
    # Multimodal tools
    'MultimodalTools',
    'analyze_image',
    'extract_text', 
    'analyze_transcript',
    
    # Search tools
    'SearchTools',
    'search_web',
    'search_news',
    
    # Math tools
    'MathTools',
    'add',
    'subtract', 
    'multiply',
    'divide',
    'power',
    'factorial',
    'square_root',
    'percentage',
    'average',
    'calculate_expression',
    
    # YouTube tools
    'YouTubeTools',
    'get_video_info',
    'download_video',
    'download_audio',
    'get_captions',
    'get_playlist_info'
]

__version__ = "1.0.0"
