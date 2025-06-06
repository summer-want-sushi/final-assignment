# tools/utils.py
import base64
import os
import mimetypes
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 encoding"""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        raise

def get_mime_type(file_path: str) -> str:
    """Get MIME type for file"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def validate_file_exists(file_path: str) -> bool:
    """Validate that file exists and is readable"""
    return os.path.exists(file_path) and os.path.isfile(file_path)

def get_env_var(var_name: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default"""
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is required")
    return value
