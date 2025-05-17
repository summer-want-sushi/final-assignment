# agent.py

import os
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool

class BasicAgent:
    def __init__(self):
        print("Initializing BasicAgent...")

        # Define tools
        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        add_tool = FunctionTool.from_defaults(fn=add)
        multiply_tool = FunctionTool.from_defaults(fn=multiply)

        # Load web search tool
        tool_spec = DuckDuckGoSearchToolSpec()
        search_tools = tool_spec.to_tool_list()

        all_tools = [add_tool, multiply_tool] + search_tools

        # Load LLM
        hf_token = os.getenv("HF_TOKEN")
        self.llm = HuggingFaceInferenceAPI(
            model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=hf_token,
        )

        # Create the agent
        self.agent = ReActAgent(
            name="universal_agent",
            description="An assistant that can search the web and do math.",
            system_prompt="You are a helpful assistant with access to tools.",
            tools=all_tools,
            llm=self.llm,
        )

        # Setup workflow
        self.workflow = AgentWorkflow(
            agents=[self.agent],
            root_agent="universal_agent",
        )

    async def __call__(self, question: str) -> str:
        print(f"[BasicAgent] Received question: {question}")
        response = await self.workflow.run(user_msg=question)
        return str(response)
