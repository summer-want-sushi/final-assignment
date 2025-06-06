# tools/multimodal_tools.py
import requests
import json
from typing import Optional, Dict, Any
from .utils import encode_image_to_base64, validate_file_exists, get_env_var, logger

class MultimodalTools:
    """Free multimodal AI tools using OpenRouter and other free services"""
    
    def __init__(self, openrouter_key: Optional[str] = None):
        self.openrouter_key = openrouter_key or get_env_var("OPENROUTER_API_KEY", None)
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app.com",  # Optional: for analytics
            "X-Title": "Multimodal Tools"  # Optional: for analytics
        }
        
        # Available free multimodal models
        self.vision_model = "moonshotai/kimi-vl-a3b-thinking:free"
        self.text_model = "meta-llama/llama-4-maverick:free"
    
    def _make_openrouter_request(self, payload: Dict[str, Any]) -> str:
        """Make request to OpenRouter API with error handling"""
        try:
            response = requests.post(
                self.openrouter_url, 
                headers=self.headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Unexpected response format: {result}")
                return "Error: Invalid response format"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {str(e)}")
            return f"Error making API request: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"Unexpected error: {str(e)}"
    
    def analyze_image(self, image_path: str, question: str = "Describe this image in detail") -> str:
        """
        Analyze image content using multimodal AI
        
        Args:
            image_path: Path to image file
            question: Question about the image
            
        Returns:
            AI analysis of the image
        """
        if not validate_file_exists(image_path):
            return f"Error: Image file not found at {image_path}"
        
        try:
            encoded_image = encode_image_to_base64(image_path)
            
            payload = {
                "model": self.vision_model,
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                            }
                        ]
                    }
                ],
                "temperature": 0,
                "max_tokens": 1024
            }
            
            return self._make_openrouter_request(payload)
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR via multimodal AI
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text from image
        """
        ocr_prompt = """Extract all visible text from this image. 
        Return only the text content without any additional commentary or formatting. 
        If no text is visible, return 'No text found'."""
        
        return self.analyze_image(image_path, ocr_prompt)
    
    def analyze_audio_transcript(self, transcript: str, question: str = "Summarize this audio content") -> str:
        """
        Analyze audio content via transcript
        
        Args:
            transcript: Audio transcript text
            question: Question about the audio content
            
        Returns:
            AI analysis of the audio content
        """
        if not transcript.strip():
            return "Error: Empty transcript provided"
        
        try:
            payload = {
                "model": self.text_model,
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Audio transcript: {transcript}\n\nQuestion: {question}"
                    }
                ],
                "temperature": 0,
                "max_tokens": 1024
            }
            
            return self._make_openrouter_request(payload)
            
        except Exception as e:
            error_msg = f"Error analyzing audio transcript: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def describe_image(self, image_path: str) -> str:
        """Get a detailed description of an image"""
        return self.analyze_image(
            image_path, 
            "Provide a detailed, objective description of this image including objects, people, colors, setting, and any notable details."
        )
    
    def answer_visual_question(self, image_path: str, question: str) -> str:
        """Answer a specific question about an image"""
        return self.analyze_image(image_path, question)

# Convenience functions for direct use
def analyze_image(image_path: str, question: str = "Describe this image in detail") -> str:
    """Standalone function to analyze an image"""
    tools = MultimodalTools()
    return tools.analyze_image(image_path, question)

def extract_text(image_path: str) -> str:
    """Standalone function to extract text from an image"""
    tools = MultimodalTools()
    return tools.extract_text_from_image(image_path)

def analyze_transcript(transcript: str, question: str = "Summarize this content") -> str:
    """Standalone function to analyze audio transcript"""
    tools = MultimodalTools()
    return tools.analyze_audio_transcript(transcript, question)
