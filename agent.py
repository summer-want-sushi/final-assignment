# agent.py
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from tools import tool_search, tool_calculate, tool_load_file

class CustomAgent:
    def __init__(self, model_name="google/flan-t5-xl", use_gpu=False):
        """Initialize the agent with an LLM (planner) and set up tools and prompt templates."""
        # Load the language model pipeline for text generation (the 'planner' LLM)
        device = 0 if use_gpu else -1
        self.llm = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=device)
        # Define the system prompt describing the agent and its tools
        self.tool_descriptions = (
            "Available tools:\n"
            "1. search(query) - searches for information about 'query' and returns a summary.\n"
            "2. calculate(expression) - evaluates a mathematical expression and returns the result.\n"
            "3. load_file(task_id) - loads an attached file for the task if any (returns a description or content snippet).\n"
        )
        self.system_message = (
            "You are an AI agent that can use tools to answer questions. "
            "You have the following tools:\n"
            f"{self.tool_descriptions}\n"
            "Follow this format:\n"
            "Thought: (think about the problem step by step)\n"
            "Action: (choose one of the tools and specify input)\n"
            "Observation: (result of the tool will be given)\n"
            "… [this Thought->Action->Observation cycle can repeat] …\n"
            "Thought: (when you have enough information, think final step)\n"
            "Final Answer: (provide the answer to the user's question)\n"
            "Make sure to output the final answer directly with no extra text.\n"
        )

    def answer(self, question: str) -> str:
        """Generate an answer for the given question by reasoning and using tools as needed."""
        # Initialize the dialogue history with system instructions and user question
        dialog = f"{self.system_message}\nUser Question: {question}\n"
        # We will accumulate the agent's reasoning in this string as we loop
        agent_thoughts = ""
        for step in range(1, 10):  # limit to 10 steps to avoid infinite loops
            # Prompt the LLM with the conversation so far (system + history + current accumulated reasoning)
            prompt = f"{dialog}{agent_thoughts}\nThought:"
            response = self.llm(prompt, max_new_tokens=200, do_sample=False, return_text=True)[0]['generated_text']
            # The LLM is expected to continue from "Thought:" and produce something like:
            # "Thought: ...\nAction: tool_name(...)\n" or "Thought: ...\nFinal Answer: ...\n"
            agent_output = response.strip()
            # Append the LLM output to agent_thoughts
            agent_thoughts += agent_output + "\n"
            # Parse the LLM output to see if an action was proposed or a final answer given
            if "Action:" in agent_output:
                # Extract the tool name and argument from the action line
                try:
                    action_line = agent_output.split("Action:")[1].strip()
                    # e.g. action_line = "search(World War 2)" or "calculate(12*7)"
                    tool_name, arg = action_line.split("(")
                    tool_name = tool_name.strip()
                    arg = arg.rstrip(")")  # remove closing parenthesis
                except Exception as e:
                    return "(Parsing Error: Invalid action format)"
                # Execute the appropriate tool
                if tool_name.lower() == "search":
                    result = tool_search(arg.strip().strip('"\''))
                elif tool_name.lower() == "calculate":
                    result = tool_calculate(arg)
                elif tool_name.lower() == "load_file":
                    result = tool_load_file(arg.strip().strip('"\''))
                else:
                    result = f"(Unknown tool: {tool_name})"
                # Add the observation to the conversation for the next loop iteration
                agent_thoughts += f"Observation: {result}\n"
            elif "Final Answer:" in agent_output:
                # The agent is presenting a final answer – extract and return it
                answer_text = agent_output.split("Final Answer:")[1].strip()
                return answer_text  # return without any "FINAL ANSWER" prefix
            else:
                # If neither Action nor Final Answer is found (LLM didn't follow format), break
                break
        # If loop ends without Final Answer, return whatever the agent last said or a fallback
        return "(No conclusive answer)"