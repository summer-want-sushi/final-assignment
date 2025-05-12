from typing import Any, Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from duckduckgo_search import DDGS
import re
import math

class WebSearchTool:
    def __init__(self):
        self.search = DDGS()
        
    def run(self, query: str, max_results: int = 3) -> str:
        """Perform a web search and return formatted results."""
        try:
            results = list(self.search.text(query, max_results=max_results))
            formatted_results = []
            for r in results:
                formatted_results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['link']}\n")
            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"

class Calculator:
    def run(self, expression: str) -> str:
        """Evaluate mathematical expressions safely."""
        try:
            # Remove any characters that aren't numbers, operators, or parentheses
            cleaned = re.sub(r'[^0-9+\-*/().\ ]', '', expression)
            # Evaluate the expression
            result = eval(cleaned, {"__builtins__": {}}, {"math": math})
            return str(result)
        except Exception as e:
            return f"Error in calculation: {str(e)}"

class GaiaAgent:
    def __init__(self):
        # Initialize Qwen-7B model
        self.model_name = "Qwen/Qwen-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        # Initialize tools
        self.tools = {
            "web_search": WebSearchTool(),
            "calculator": Calculator()
        }
        
        # System prompt template
        self.system_prompt = """You are a helpful AI assistant with access to the following tools:
1. web_search: Search the internet for current information
2. calculator: Perform mathematical calculations

To use a tool, respond with: <tool>tool_name|input</tool>
For example: <tool>calculator|2 + 2</tool> or <tool>web_search|latest news about AI</tool>

If you don't need any tools to answer, just provide your response directly.
Always explain your reasoning before using tools or providing final answers."""

    def _generate_response(self, prompt: str, max_length: int = 2048) -> str:
        """Generate a response using the Qwen model."""
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the assistant's response
            response = response.split(prompt)[-1].strip()
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _extract_tool_calls(self, response: str) -> List[Dict[str, str]]:
        """Extract tool calls from the response."""
        tool_pattern = r'<tool>(.*?)\|(.*?)</tool>'
        matches = re.finditer(tool_pattern, response)
        tool_calls = []
        
        for match in matches:
            tool_name = match.group(1).strip()
            tool_input = match.group(2).strip()
            tool_calls.append({"name": tool_name, "input": tool_input})
            
        return tool_calls

    def _execute_tool_call(self, tool_call: Dict[str, str]) -> str:
        """Execute a single tool call and return the result."""
        tool_name = tool_call["name"]
        tool_input = tool_call["input"]
        
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            result = self.tools[tool_name].run(tool_input)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def process_question(self, question: str) -> str:
        """Process a single question and return the answer."""
        # Construct the full prompt
        full_prompt = f"{self.system_prompt}\n\nQuestion: {question}\n\nAnswer:"
        
        # Get initial response
        response = self._generate_response(full_prompt)
        
        # Extract and execute any tool calls
        tool_calls = self._extract_tool_calls(response)
        
        if tool_calls:
            # Execute each tool call and collect results
            tool_results = []
            for tool_call in tool_calls:
                result = self._execute_tool_call(tool_call)
                tool_results.append(f"Tool {tool_call['name']} result: {result}")
            
            # Generate final response with tool results
            tool_results_str = "\n".join(tool_results)
            final_prompt = f"{full_prompt}\n{response}\n\nTool Results:\n{tool_results_str}\n\nFinal Answer:"
            final_response = self._generate_response(final_prompt)
            
            return final_response
        
        return response

    def get_answer(self, question_data: Dict[str, Any]) -> Optional[str]:
        """Process a question from the GAIA benchmark and return an answer."""
        try:
            # Extract the actual question from the question data
            question = question_data.get("question", "")
            if not question:
                return None
                
            # Process the question and get the answer
            answer = self.process_question(question)
            
            return answer
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            return None 