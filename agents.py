import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool

from supabase.client import create_client, Client


# Load environment variables
load_dotenv()


# ---- Basic Arithmetic Utilities ---- #
@tool
def multiply(a: int, b: int) -> int:
    """Returns the product of two integers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Returns the sum of two integers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Returns the difference between two integers."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Performs division and handles zero division errors."""
    if b == 0:
        raise ValueError("Division by zero is undefined.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Returns the remainder after division."""
    return a % b


# ---- Search Tools ---- #
@tool
def search_wikipedia(query: str) -> str:
    """Returns up to 2 documents related to a query from Wikipedia."""
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return {"wiki_results": "\n\n---\n\n".join(
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
        for doc in docs
    )}

@tool
def search_web(query: str) -> str:
    """Fetches up to 3 web results using Tavily."""
    results = TavilySearchResults(max_results=3).invoke(query=query)
    return {"web_results": "\n\n---\n\n".join(
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}'
        for doc in results
    )}

@tool
def search_arxiv(query: str) -> str:
    """Retrieves up to 3 papers related to the query from ArXiv."""
    results = ArxivLoader(query=query, load_max_docs=3).load()
    return {"arvix_results": "\n\n---\n\n".join(
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}'
        for doc in results
    )}


system_message = SystemMessage(content="""You are a helpful assistant tasked with answering questions using a set of tools. Now, I will ask you a question. Report your thoughts, and finish your answer with the following template:

FINAL ANSWER: [YOUR FINAL ANSWER]

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings. 
- If you are asked for a number, don't use a comma in the number and avoid units like $ or % unless specified otherwise.
- If you are asked for a string, avoid using articles and abbreviations (e.g. for cities), and write digits in plain text unless specified otherwise.
- If you are asked for a comma-separated list, apply the above rules depending on whether each item is a number or string.

Your answer should start only with "Responce: ", followed by your result.""")

toolset = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    search_wikipedia,
    search_web,
    search_arxiv,
]


# ---- Graph Construction ---- #
def create_agent_flow(provider: str = "groq"):
    """Constructs the LangGraph conversational flow with tool support."""

    if provider == "google":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
            url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
            temperature=0
        ))
    else:
        raise ValueError("Unsupported provider. Choose from: 'google', 'groq', 'huggingface'.")

    llm_toolchain = llm.bind_tools(toolset)

    # Assistant node behavior
    def assistant_node(state: MessagesState):
        response = llm_toolchain.invoke(state["messages"])
        return {"messages": [response]}

    
    # Build the conversational graph
    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", ToolNode(toolset))
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()