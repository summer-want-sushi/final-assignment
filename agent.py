import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek



load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
@tool 
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b
@tool 
def add (a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b
@tool 
def subtract (a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b
@tool 
def divide (a: int, b: int) -> int:
    """Divides two numbers."""
    return a / b
@tool
def modulo (a: int, b: int) -> int:
    """Returns the remainder of two numbers."""
    return a % b
@tool
def wiki_search(query:str)->str:
    "Using Wikipedia, search for a query and return the first result."
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}
@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}


# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()
sys_msg = SystemMessage(content=system_prompt)

tools = [
    multiply,
    add,
    subtract,
    divide,
    modulo,
    wiki_search,
    arvix_search,
]
def build_graph():
    llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=DEEPSEEK_API_KEY,
)
    llm_with_tools = llm.bind_tools(tools)
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools",ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    return builder.compile()
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # Build the graph
    graph = build_graph()
    png_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
